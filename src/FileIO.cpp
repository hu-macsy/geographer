/*
 * IO.cpp
 *
 *  Created on: 15.02.2017
 *      Author: moritzl
 */

#include "FileIO.h"
#include "quadtree/QuadTreeCartesianEuclid.h"

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>
#include <scai/common/Math.hpp>
#include <scai/common/Settings.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>
#include <scai/tracing.hpp>

#include <assert.h>
#include <cmath>
#include <set>
#include <climits>
#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>
#include <tuple>


using scai::lama::CSRStorage;
using scai::hmemo::HArray;

namespace ITI {

const IndexType fileTypeVersionNumber= 3;

//-------------------------------------------------------------------------------------------------
/*Given the adjacency matrix it writes it in the file "filename" using the METIS format. In the
 * METIS format the first line has two numbers, first is the number on vertices and the second
 * is the number of edges. Then, row i has numbers e1, e2, e3, ... notating the edges:
 * (i, e1), (i, e2), (i, e3), ....
 *
 */

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeGraph (
    const CSRSparseMatrix<ValueType> &adjM,
    const std::string filename,
    const bool edgeWeights,
    const bool binary) {
    SCAI_REGION( "FileIO.writeGraph" )

    const scai::dmemo::CommunicatorPtr comm = adjM.getRowDistributionPtr()->getCommunicatorPtr();
    const IndexType thisPE = comm->getRank();
    const IndexType numPEs = comm->getSize();

    const IndexType root =0;
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    const IndexType globalN = distPtr->getGlobalSize();
    const IndexType localN = distPtr->getLocalSize();
    const IndexType globalM = adjM.getNumValues();
    assert(globalN==adjM.getNumRows());

    SCAI_ASSERT_ERROR( distPtr->isBlockDistributed(comm), 
        "Graph must have a block or general block distribution for this version of writeGraph" );

    std::ofstream fNew;
    std::string newFile = filename;
    if( binary ){
        fNew.open(newFile, std::ios::binary | std::ios::out);
    }else{
        fNew.open(newFile, std::ios::out);
    }
    if(not fNew.is_open()){
        throw std::runtime_error("File "+ newFile+ " could not be opened");
    }

    //used only for binary
    typedef unsigned long int ULONG;

    //write header
    if(comm->getRank()==root) {
        if( binary){
            const ULONG fileTypeVersionNumber = 3; //not sure why 4...
            fNew.write((char*)(&fileTypeVersionNumber), sizeof( ULONG ));
            fNew.write((char*)(&globalN), sizeof( ULONG ));
            ULONG M = globalM/2;
            fNew.write((char*)(&M), sizeof( ULONG ));
        }else{
            // first line is the number of nodes and edges
            fNew << globalN <<" "<< globalM/2; //graph is undirected
            if(edgeWeights) {
                fNew << " 001";
            }
            fNew<< std::endl;
        }
    }

    //wait root to write header
    comm->synchronize();
    fNew.close();

    //find in which order are the rows distributed.
    //We may have a general block distribution but that does not necessarily mean that
    //the first rows are in PE 0, the next in PE 1 etc. It can be that rows from 0 to X 
    //are in  PE 5, rows X+1 till Y in PE 2 etc.

    const IndexType myFirstGlobalIndex = distPtr->local2Global(0);

    std::vector<IndexType> allFirstIndices(numPEs,0);
    allFirstIndices[thisPE] = myFirstGlobalIndex;
    comm->sumImpl(allFirstIndices.data(), allFirstIndices.data(), numPEs, scai::common::TypeTraits<IndexType>::stype);

    std::vector<IndexType> peIndices(numPEs);
    std::iota( peIndices.begin(), peIndices.end(), 0);
    // sort PE indices according to their starting global index
    std::sort(peIndices.begin(), peIndices.end(), [&allFirstIndices](IndexType a, IndexType b) {
        return allFirstIndices[a] < allFirstIndices[b];
    });

    const IndexType myTurn = peIndices[thisPE];

    //write the node degrees for binary format
    if(binary){
        std::vector<ULONG> localNodeDegrees(localN);
        const scai::lama::CSRStorage<ValueType>& localAdjM = adjM.getLocalStorage();
        const scai::hmemo::ReadAccess<IndexType> rLocalIA( localAdjM.getIA() );
        for( IndexType i=0; i<rLocalIA.size()-1; i++){
            localNodeDegrees[i] = (ULONG) (rLocalIA[i+1]-rLocalIA[i]);
        }
        fNew.open(newFile, std::ios::binary | std::ios::app);
        for(IndexType p=0; p<numPEs; p++){
            if( myTurn==p){
                fNew.write( (char*)(localNodeDegrees.data()), localN*sizeof(ULONG));
            }
            comm->synchronize();
        }
        PRINT0("node degrees written");
    }

    //
    //prepare the actual edges
    //

    std::stringstream ssBuffer;
    std::vector<ULONG> binaryBuffer();

    {
        const scai::lama::CSRStorage<ValueType>& localAdjM = adjM.getLocalStorage();
        const scai::hmemo::ReadAccess<IndexType> rLocalIA( localAdjM.getIA() );
        const scai::hmemo::ReadAccess<IndexType> rLocalJA( localAdjM.getJA() );
        const scai::hmemo::ReadAccess<ValueType> rLocalVal( localAdjM.getValues() );

        if(binary){

        }else{
            for(IndexType i=0; i<rLocalIA.size()-1; i++) {       // for all local nodes
                for(IndexType j= rLocalIA[i]; j<rLocalIA[i+1]; j++) {    // for all the edges of a node
                    if(!edgeWeights) {
                        ssBuffer << rLocalJA[j]+1 << " ";
                    } else {
                        ssBuffer << rLocalJA[j]+1 << " "<< rLocalVal[j]<< " ";
                    }
                }
                ssBuffer << std::endl;
            }
        }
    }

    //
    //write in file per PE
    //

    IndexType globalCurrentLine = 0;

    for(IndexType p=0; p<numPEs; p++){
        SCAI_REGION("FileIO.writeGraph.writeInFile");
        if( myTurn==p){
            SCAI_ASSERT_EQ_ERROR( globalCurrentLine, myFirstGlobalIndex, "Wrong line number for PE " << thisPE );
            fNew.open(newFile, std::ios::app); //append
            fNew << ssBuffer.str();
            fNew.close();
        }else{
            //all non-writing PEs do nothing
        }

        globalCurrentLine += localN;
        comm->bcast( &globalCurrentLine, 1, thisPE ); //just for the assertion
        //^^ should also works as a barrier?
        //without this, the file is not created properly; bcast does not act as a barrier?
        comm->synchronize();
    }
    fNew.close();
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeGraphDistributed (const CSRSparseMatrix<ValueType> &adjM, const std::string filename) {
    SCAI_REGION("FileIO.writeGraphDistributed")

    std::string fileTo;
    {
        const scai::dmemo::CommunicatorPtr comm = adjM.getRowDistributionPtr()->getCommunicatorPtr();
        fileTo = filename + std::to_string(comm->getRank());
    }
    std::ofstream f(fileTo);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    // notice that numValues is twice the number of edges of the graph
    assert(((int) adjM.getNumValues())%2 == 0); // even number of edges

    IndexType localNumNodes= adjM.getLocalNumRows();
    f<< localNumNodes <<" "<< adjM.getLocalNumValues()/2 << std::endl; // first line is number of nodes and edges

    // get local part
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    for(IndexType i=0; i< ia.size()-1; i++) {                 // for all local nodes
        for(IndexType j=ia[i]; j<ia[i+1]; j++) {            // for all the edges of a node
            f << ja[j]+1 << " ";
        }
        f << std::endl;
    }
    f.close();
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeGraphAsEdgeList (const CSRSparseMatrix<ValueType> &graph, const std::string filename) {
    SCAI_REGION("FileIO.writeAsEdgeList");

    const scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();

    //edgeList contains the local part of the edges, indexed by the global indices for each node ID
    [[maybe_unused]] IndexType maxDegree;
    const std::vector<std::tuple<IndexType,IndexType,ValueType>> edgeList = 
        GraphUtils<IndexType,ValueType>::localCSR2GlobalEdgeList(graph,  maxDegree );

     //sanity checks
    const IndexType localNumEdges = edgeList.size();
    //see GraphUtils::localCSR2GlobalEdgeList(
    SCAI_ASSERT_EQ_ERROR(localNumEdges, graph.getLocalNumValues(), "Local number of edges seems wrong" );

    const IndexType numEdges = graph.getNumValues()/2;
    const IndexType globalEsgeListSize = comm->sum( edgeList.size() );
    SCAI_ASSERT_EQ_ERROR( numEdges, globalEsgeListSize, "Global number of edges seems wrong" );

    typedef unsigned long int ULONG;
    //TODO: we can also directly use the edgeList...
    std::vector<std::vector<ULONG>> nodeList(2, std::vector<ULONG>(localNumEdges));

    for( int v=0; v<localNumEdges; v++){
        nodeList[0][v] = (ULONG) std::get<0>(edgeList[v]);
        nodeList[1][v] = (ULONG) std::get<1>(edgeList[v]);
    }

    //get prefix sum of local ranges.
    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();
    std::vector<IndexType> localRanges(numPEs+1,0);
    localRanges[thisPE+1] = localNumEdges;
    comm->sumImpl( localRanges.data(), localRanges.data(), numPEs+1, scai::common::TypeTraits<IndexType>::stype );
    for( int p=1; p<=numPEs; p++){
        localRanges[p] += localRanges[p-1];
    }

    //
    //write into file
    //

    std::ofstream outfile;
    
    const ULONG numNodes = graph.getNumRows();
    const ULONG numEdgesUL = (ULONG) numEdges;

    for(IndexType p=0; p<numPEs; p++) { // numPE rounds, in each round only one PE writes its part
        if(thisPE==p ) {
            if( p==0 ) {
                outfile.open(filename.c_str(), std::ios::binary | std::ios::out);
                //write header
                const ULONG fileTypeVersionNumber = 3; //not sure why 4...
                outfile.write((char*)(&fileTypeVersionNumber), sizeof( ULONG ));
                outfile.write((char*)(&numNodes), sizeof( ULONG ));
                outfile.write((char*)(&numEdgesUL), sizeof( ULONG ));
            } else {
                // if not the first PE then append to file
                outfile.open(filename.c_str(), std::ios::binary | std::ios::app);
            }

            for( IndexType i=0; i<localNumEdges; i++) {
                outfile.write( (char *)(&nodeList[0][i]), sizeof(ULONG) );
                outfile.write( (char *)(&nodeList[1][i]), sizeof(ULONG) );
            }

            SCAI_ASSERT_EQ_ERROR( outfile.tellp(), (localRanges[thisPE+1]*2+3)*sizeof(ULONG) , "While writing edge list in parallel: Position in file " << filename << " for PE " << thisPE << " is not correct." );
            outfile.close();
            std::cout<< "PE " << thisPE << " wrote its part " << std::endl;
        }
        comm->synchronize();
    }

}//writeGraphAsEdgeList

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeVTKCentral (const CSRSparseMatrix<ValueType> &adjM, const std::vector<DenseVector<ValueType>> &coords, const DenseVector<IndexType> &part, const std::string filename) {
    SCAI_REGION( "FileIO.writeVTKCentral" )

    const IndexType N = adjM.getNumRows();
    const IndexType dimensions = coords.size();

    std::ofstream f(filename);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    //------------------------------------------------------
    // write header

    f << "# vtk DataFile Version 2.0" << std::endl;
    f << "Saved graph and partition" << std::endl;
    f << "ASCII" << std::endl;
    f << "DATASET UNSTRUCTURED_GRID" << std::endl;

    //------------------------------------------------------
    // write 3D coordinates
    // TODO: use ReadAccess or copy to a vector<vector<>> of size [N,d] and not [d,N]

    f << "POINTS " << N << " double" << std::endl;
    for(int i=0; i<N; i++) {
        for(int d=0; d<dimensions; d++) {
            f<< coords[d].getLocalValues()[i] << " ";
        }
        f << std::endl;
    }

    //------------------------------------------------------
    // write the partition

    f <<  std::endl << "POINT_DATA " << N << std::endl;
    // below is the number or variables, aka how many different partitions we have in this file
    f << "SCALARS Partition float" <<  std::endl; // TODO: in this version, hardcoded to 1
    // for(inv v=0; v<number_of_variables; v++){}

    f << "LOOKUP_TABLE default" << std::endl;
    int k = part.max();
    for( int i=0; i<N; i++) {
        f << (double) part.getLocalValues()[i]/(double)k << std::endl;
    }
    f << std::endl;

    f.close();
}
//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and their dimension, writes them in file "filename".
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeCoords (const std::vector<DenseVector<ValueType>> &coords, const std::string filename) {
    SCAI_REGION( "FileIO.writeCoords" );

    const IndexType dimension = coords.size();
    const IndexType n = coords[0].size();
    const scai::dmemo::DistributionPtr dist = coords[0].getDistributionPtr();
    assert(dist->getGlobalSize() == n);
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( n ));
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

    /* If the input is replicated, we can write it directly from the root processor.
     * If it is not, we need to create a replicated copy.
     */
    //TODO: instead of replicate, gather to root PE

    std::vector<DenseVector<ValueType>> maybeCopy;
    const std::vector<DenseVector<ValueType>> &targetReference = dist->isReplicated() ? coords : maybeCopy;

    if (!dist->isReplicated()) {
        maybeCopy.resize(dimension);
        for (IndexType d = 0; d < dimension; d++) {
            maybeCopy[d] = scai::lama::distribute<DenseVector<ValueType>>(coords[d], noDist);
        }

    }
    assert(targetReference[0].getDistributionPtr()->isReplicated());
    assert(targetReference[0].size() == n);

    if (comm->getRank() == 0) {
        std::ofstream filehandle(filename);
        filehandle.precision(15);
        if (filehandle.fail()) {
            throw std::runtime_error("Could not write to file " + filename);
        }
        for (IndexType i = 0; i < n; i++) {
            for (IndexType d = 0; d < dimension; d++) {
                filehandle << targetReference[d].getLocalValues()[i] << " ";
            }
            filehandle << std::endl;
        }
    }
}
//-------------------------------------------------------------------------------------------------
/*
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeCoordsParallel(
    const std::vector<DenseVector<ValueType>> &coords,
    const std::string outFilename,
    const bool overwriteExisting) {

    const IndexType dimension = coords.size();

    scai::dmemo::DistributionPtr coordDist = coords[0].getDistributionPtr();
    const IndexType globalN = coordDist->getGlobalSize();
    const IndexType localN = coordDist->getLocalSize();
    const scai::dmemo::CommunicatorPtr comm = coords[0].getDistributionPtr()->getCommunicatorPtr();
    const IndexType numPEs = comm->getSize();

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());

    PRINT( *comm << ": "<< beginLocalRange << " - " << endLocalRange );
    SCAI_ASSERT_EQ_ERROR( localN, endLocalRange-beginLocalRange, "Local ranges do not agree");

    //
    // copy coords to a local vector<vector>
    //

    std::vector< std::vector<double>> localPartOfCoords( localN, std::vector<double>( dimension, 0.0) );

    for(IndexType d=0; d<dimension; d++) {
        scai::hmemo::ReadAccess<ValueType> localCoords( coords[d].getLocalValues() );
        for( IndexType i=0; i<localN; i++) {
            localPartOfCoords[i][d] = localCoords[i];
        }
    }

    //
    //  one PE at a time, write to file TODO: would seekp work?
    //

    std::ofstream outfile;

    for(IndexType p=0; p<numPEs; p++) { // numPE rounds, in each round only one PE writes its part
        if( comm->getRank()==p ) {
            if( p==0 ) {
                if( overwriteExisting ){
                    outfile.open(outFilename.c_str(), std::ios::binary | std::ios::out);
                }else{
                    outfile.open(outFilename.c_str(), std::ios::binary | std::ios::app);
                }
            } else {
                // if not the first PE then append to file
                outfile.open(outFilename.c_str(), std::ios::binary | std::ios::app);
            }

            for( IndexType i=0; i<localN; i++) {
                for( IndexType d=0; d<dimension; d++) {
                    outfile.write( (char *)(&localPartOfCoords[i][d]), sizeof(double) );
                }
            }

            SCAI_ASSERT_EQ_ERROR( outfile.tellp(), endLocalRange*dimension*sizeof(double), "While writing coordinates in parallel: Position in file " << outFilename << " is not correct." );

            outfile.close();
        }
        comm->synchronize();
    }

}

//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates each PE writes its own part in file "filename".
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeCoordsDistributed(const std::vector<DenseVector<ValueType>> &coords, const IndexType dimensions, const std::string filename) {
    SCAI_REGION( "FileIO.writeCoordsDistributed" )

    const scai::dmemo::DistributionPtr distPtr = coords[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = distPtr->getCommunicatorPtr();

    if( !(dimensions==2 or dimensions==3) ) {
        PRINT0("Only implemented for dimensions 2 or 3 for now, not for " << dimensions <<". Aborting.");
        throw std::runtime_error("Wrong number of dimensions.");
    }

    std::string thisPEFilename = filename +'_'+ std::to_string(comm->getRank()) + ".xyz";
    std::ofstream f(thisPEFilename);
    if(f.fail())
        throw std::runtime_error("File "+ thisPEFilename+ " failed.");

    IndexType localN = distPtr->getLocalSize();

    scai::hmemo::ReadAccess<ValueType> coordAccess0( coords[0].getLocalValues() );
    scai::hmemo::ReadAccess<ValueType> coordAccess1( coords[1].getLocalValues() );
    // in case dimensions==2 this will be ignored
    scai::hmemo::ReadAccess<ValueType> coordAccess2( coords[dimensions-1].getLocalValues() );

    for(IndexType i=0; i<localN; i++) {
        f<< std::setprecision(15)<< coordAccess0[i] << " " << coordAccess1[i];
        if( dimensions==3 ) {
            f << " "<< coordAccess2[i];
        }
        f<< std::endl;
    }

}
//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and the nodeWeights writes them both in a file in the form:
 *
 *   cood1 coord2 ... coordD weight
 *
 * for D dimensions. Each line coresponds to one point/vertex.
 */

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeInputParallel (const std::vector<DenseVector<ValueType>> &coords, const scai::lama::DenseVector<ValueType> nodeWeights, const std::string filename) {
    SCAI_REGION( "FileIO.writeInputParallel" )

    const scai::dmemo::DistributionPtr coordDistPtr = coords[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDistPtr->getCommunicatorPtr();
    const IndexType localN = coordDistPtr->getLocalSize();
    const IndexType dimension = coords.size();
    const IndexType numPEs = comm->getSize();

    //
    // copy coords to a local vector<vector>
    //

    std::vector< std::vector<ValueType>> localPartOfCoords( localN, std::vector<ValueType>( dimension, 0.0) );

    for(IndexType d=0; d<dimension; d++) {
        scai::hmemo::ReadAccess<ValueType> localCoords( coords[d].getLocalValues() );
        for( IndexType i=0; i<localN; i++) {
            localPartOfCoords[i][d] = localCoords[i];
        }
    }

    scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );

    std::ofstream outfile;

    comm->synchronize();

    for(IndexType p=0; p<numPEs; p++) { // numPE rounds, in each round only one PE writes its part
        if( comm->getRank()==p ) {
            if( p==0 ) {
                outfile.open(filename.c_str(), std::ios::binary | std::ios::out);
            } else {
                // if not the first PE then append to file
                outfile.open(filename.c_str(), std::ios::binary | std::ios::app);
            }
            if( outfile.fail() ) {
                throw std::runtime_error("Could not write to file " + filename);
            }

            for( IndexType i=0; i<localN; i++) {
                for(IndexType d=0; d<dimension; d++) {
                    outfile << localPartOfCoords[i][d]<< " ";   // write coords
                }
                //outfile << localWeights[i] << std::endl;        //write node weight
            }

            outfile.close();
        }
        comm->synchronize();    //TODO: takes huge time here
    }

}

//-------------------------------------------------------------------------------------------------
//TODO: unit test
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writePartitionParallel(const DenseVector<IndexType> &part, const std::string filename) {
    SCAI_REGION( "FileIO.writePartitionParallel" );

    scai::dmemo::DistributionPtr dist = part.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

    const IndexType localN = dist->getLocalSize();
    const IndexType globalN = dist->getGlobalSize();
    const IndexType numPEs = comm->getSize();

    scai::hmemo::ReadAccess<IndexType> localPart( part.getLocalValues() );
    SCAI_ASSERT_EQ_ERROR( localPart.size(), localN, "Local sizes do not agree");

    std::ofstream outfile;

    for(IndexType p=0; p<numPEs; p++) { // numPE rounds, in each round only one PE writes its part

        if( comm->getRank()==p ) {
            if( p==0 ) {
                outfile.open(filename.c_str(), std::ios::binary | std::ios::out);
                outfile << "% " << globalN << std::endl;    // the first line has a comment with the number of nodes
            } else {
                // if not the first PE then append to file
                outfile.open(filename.c_str(), std::ios::binary | std::ios::app);
            }
            if( outfile.fail() ) {
                throw std::runtime_error("Could not write to file " + filename);
            }

            //write local part
            for( IndexType i=0; i<localN; i++) {
                outfile << localPart[i] << std::endl;
            }

            outfile.close();
            //PRINT("PE " << p << " wrote its part");
        }
        comm->synchronize();    //TODO: takes huge time here
    }
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
void FileIO<IndexType, ValueType>::writeDenseVectorParallel(const DenseVector<T> &dv, const std::string filename) {
    SCAI_REGION( "FileIO.writeDenseVectorParallel" );

    scai::dmemo::CommunicatorPtr comm = dv.getDistributionPtr()->getCommunicatorPtr();

    const IndexType globalN =dv.size();
    const IndexType numPEs = comm->getSize();

    const scai::dmemo::Distribution blockDist( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, globalN) );

    // create a copy of the input and distribute with a block distribution
    const scai::lama::DenseVector<T> dvBlock( dv, blockDist);

    const IndexType localN = blockDist.getLocalSize();

    scai::hmemo::ReadAccess<IndexType> localPart( dvBlock.getLocalValues() );
    SCAI_ASSERT_EQ_ERROR( localPart.size(), localN, "Local sizes do not agree");

    std::ofstream outfile;

    comm->synchronize();

    for(IndexType p=0; p<numPEs; p++) { // numPE rounds, in each round only one PE writes its part
        if( comm->getRank()==p ) {
            if( p==0 ) {
                outfile.open(filename.c_str(), std::ios::binary | std::ios::out);
            } else {
                // if not the first PE then append to file
                outfile.open(filename.c_str(), std::ios::binary | std::ios::app);
            }
            if( outfile.fail() ) {
                throw std::runtime_error("Could not write to file " + filename);
            }

            for( IndexType i=0; i<localN; i++) {
                outfile << localPart[i] << std::endl;
            }

            // because of the block distribution, the last PE maybe has less local values
            if( p==numPEs-1 ) {
                SCAI_ASSERT_EQ_ERROR( outfile.tellp(), globalN, "While writing DenseVector in parallel: Position in file " << filename << " is not correct." );
            } else {
                SCAI_ASSERT_EQ_ERROR( outfile.tellp(), localN*(comm->getRank()+1), "While writing DenseVector in parallel: Position in file " << filename << " is not correct for processor " << comm->getRank() );
            }

            //PRINT("PE " << p << " wrote its part");
            outfile.close();
        }
        comm->synchronize();    //TODO: takes huge time here
    }
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeDenseVectorCentral(DenseVector<IndexType> &part, const std::string filename) {

    const scai::dmemo::DistributionPtr dist = part.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

    const IndexType globalN = dist->getGlobalSize();

    //TODO: change to gather as this way, part is replicated in all PEs
    //comm->gatherImpl( localPart.get(),

    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( globalN ));
    part.redistribute( noDist );
    SCAI_ASSERT_EQ_ERROR( part.getLocalValues().size(), globalN, "Partition must be replicated");

    if( comm->getRank()==0 ) {

        std::ofstream f( filename );

        if( f.fail() ) {
            throw std::runtime_error("Could not write to file " + filename);
        }

        const scai::hmemo::ReadAccess<IndexType> rPart( part.getLocalValues() );
        for( IndexType i=0; i<globalN; i++) {
            f << rPart[i]<< std::endl;
        }
    }
    //redistribute back to original distribution
    part.redistribute( dist );
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and transforms
 * it to the adjacency matrix as a CSRSparseMatrix.
 */

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraph(const std::string filename,  const scai::dmemo::CommunicatorPtr comm, Format format) {

    //then call other function, handling formats with optional node weights
    std::vector<DenseVector<ValueType>> dummyWeightContainer;
    return readGraph(filename, dummyWeightContainer, comm, format);
}

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraph(const std::string filename, std::vector<DenseVector<ValueType>>& nodeWeights,  const scai::dmemo::CommunicatorPtr comm, Format format) {
    SCAI_REGION("FileIO.readGraph");

    if(format == Format::MATRIXMARKET) {
        return FileIO<IndexType, ValueType>::readGraphMatrixMarket(filename, comm);
    }

    std::string ending = filename.substr( filename.size()-3,  filename.size() );
    if ((format == Format::AUTO and (ending == "bgf" or ending == "bfg")) or format==Format::BINARY) {
        // if file has a .bgf ending then is a binary file
        return readGraphBinary( filename, comm);
    }

    if (format==Format::EDGELIST or format==Format::BINARYEDGELIST) {
        return readEdgeList(filename, comm, format==Format::BINARYEDGELIST);
    }

    if (format==Format::EDGELISTDIST) {
        return readEdgeListDistributed( filename, comm);
    }

    if (!(format == Format::METIS or format == Format::AUTO)) {
        throw std::logic_error("Format not yet implemented.");
    }

    /*
     * Now assuming METIS format
     */

    std::ifstream file(filename);

    typedef unsigned long long int ULLI;

    if (file.fail()) {
        throw std::runtime_error("Reading graph from " + filename + " failed.");
    } else {
        if( comm->getRank()==0 ) {
            std::cout<< "Reading from file "<< filename << std::endl;
        }
    }

    //define variables
    std::string line;
    ULLI globalN, globalM;
    IndexType numberNodeWeights = 0;
    bool hasEdgeWeights = false;
    std::vector<ValueType> edgeWeights;//possibly of size 0

    //read first line to get header information
    std::getline(file, line);
    while( line[0]== '%') {
        std::getline(file, line);
    }
    std::stringstream ss( line );
    std::string item;

    {
        //node count and edge count are mandatory. If these fail, std::stoi will raise an error. TODO: maybe wrap into proper error message
        std::getline(ss, item, ' ');
        globalN = std::stoll(item);
        std::getline(ss, item, ' ');
        globalM = std::stoll(item);

        if( globalN<=0 or globalM<0 ) {
            throw std::runtime_error("Negative input, maybe int value is not big enough: globalN= "
                                     + std::to_string(globalN) + " , globalM= " + std::to_string(globalM));
        }

        bool readWeightInfo = !std::getline(ss, item, ' ').fail();
        if (readWeightInfo && item.size() > 0) {
            //three bits, describing presence of edge weights, vertex weights and vertex sizes
            int bitmask = std::stoi(item);
            hasEdgeWeights = bitmask % 10;
            if ((bitmask / 10) % 10) {
                bool readNodeWeightCount = !std::getline(ss, item, ' ').fail();
                if (readNodeWeightCount && item.size() > 0) {
                    numberNodeWeights = std::stoi(item);
                } else {
                    numberNodeWeights = 1;
                }
            }
        }

        if (comm->getRank() == 0) {
            std::cout << "Expecting " << globalN << " nodes and " << globalM << " edges, ";
            if (!hasEdgeWeights && numberNodeWeights == 0) {
                std::cout << "with no edge or node weights."<< std::endl;
            }
            else if (hasEdgeWeights && numberNodeWeights == 0) {
                std::cout << "with edge weights, but no node weights."<< std::endl;
            }
            else if (!hasEdgeWeights && numberNodeWeights > 0) {
                std::cout << "with no edge weights, but " << numberNodeWeights << " node weights."<< std::endl;
            }
            else {
                std::cout << "with edge weights and " << numberNodeWeights << " weights per node."<< std::endl;
            }
        }
    }

    const ValueType avgDegree = ValueType(2*globalM) / globalN;

    //get distribution and local range
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( globalN ));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;
    SCAI_ASSERT_LE_ERROR(localN, std::ceil(ValueType(globalN) / comm->getSize()), "localN: " << localN << ", optSize: " << std::ceil(globalN / comm->getSize()));

    //std::cout << "Process " << comm->getRank() << " reading from " << beginLocalRange << " to " << endLocalRange << std::endl;

    //scroll to begin of local range. Neighbors of node i are in line i+1
    IndexType ll;
    for (ll = 0; ll < beginLocalRange; ll++) {
        std::getline(file, line);
        if( file.tellg()<0) {
            PRINT(*comm << " : "<<  ll);
            exit(1);
        }
    }

    std::vector<IndexType> ia(localN+1, 0);
    std::vector<IndexType> ja;
    std::vector<ValueType> values;
    std::vector<std::vector<ValueType> > nodeWeightStorage(numberNodeWeights);
    for (IndexType i = 0; i < numberNodeWeights; i++) {
        nodeWeightStorage[i].resize(localN);
    }

    //we don't know exactly how many edges we are going to have, but in a regular mesh the average degree times the local nodes is a good estimate.
    ULLI edgeEstimate = ULLI(localN*avgDegree*1.1);
    assert(edgeEstimate >= 0);
    ja.reserve(edgeEstimate);

    //std::cout << "Process " << comm->getRank() << " reserved memory for  " <<  edgeEstimate << " edges." << std::endl;

    //now read in local edges
    for (IndexType i = 0; i < localN; i++) {
        bool read = !std::getline(file, line).fail();

        if( !read) PRINT(*comm << ": " <<  i << " __ " << line << " || " << file.tellg() );
        //remove leading and trailing whitespace, since these can confuse the string splitter
        trim(line);
        assert(read);//if we have read past the end of the file, the node count was incorrect
        std::stringstream ss( line );
        std::string item;
        std::vector<IndexType> neighbors;
        std::vector<IndexType> weights;

        for (IndexType j = 0; j < numberNodeWeights; j++) {
            bool readWeight = !std::getline(ss, item, ' ').fail();
            if (readWeight) {
                if (item.size() == 0) {
                    //some whitespace at beginning of line
                    j--;
                    continue;
                } else {
                    nodeWeightStorage[j][i] = std::stoi(item);
                }
            } else {
                std::cout << "Could not parse " << item << std::endl;
            }
        }

        while (!std::getline(ss, item, ' ').fail()) {
            if (item.size() == 0) {
                //probably some whitespace at end of line
                continue;
            }
            IndexType neighbor = std::stoi(item)-1;//-1 because of METIS format
            if (neighbor >= globalN || neighbor < 0) {
                throw std::runtime_error(std::string(__FILE__) +", "+std::to_string(__LINE__) + ": Found illegal neighbor " + std::to_string(neighbor) + " in line " + std::to_string(i+beginLocalRange));
            }

            if (hasEdgeWeights) {
                bool readEdgeWeight = !std::getline(ss, item, ' ').fail();
                if (!readEdgeWeight) {
                    throw std::runtime_error("Edge weight for " + std::to_string(neighbor) + " not found in line " + std::to_string(beginLocalRange + i) + ".");
                }
                ValueType edgeWeight = std::stod(item);
                values.push_back(edgeWeight);
            }
            //std::cout << "Converted " << item << " to " << neighbor << std::endl;
            neighbors.push_back(neighbor);
        }

        //set Ia array
        ia[i+1] = ia[i] + neighbors.size();
        //copy neighbors to Ja array
        std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(ja));
        if (hasEdgeWeights) {
            assert(ja.size() == values.size());
        }
    }

    //std::cout << "Process " << comm->getRank() << " read " << ja.size() << " local edges. " << std::endl;


    nodeWeights.resize(numberNodeWeights);
    //std::cout << "Process " << comm->getRank() << " allocated memory for " << numberNodeWeights << " node weights. " << std::endl;
    for (IndexType i = 0; i < numberNodeWeights; i++) {
        nodeWeights[i] = DenseVector<ValueType>(dist, HArray<ValueType>(localN, nodeWeightStorage[i].data()));
    }

    //std::cout << "Process " << comm->getRank() << " converted node weights. " << std::endl;

    if (endLocalRange == globalN) {
        bool eof = std::getline(file, line).eof();
        if (!eof) {
            throw std::runtime_error(std::to_string(globalN) + " lines read, but file continues.");
        }
    }

    file.close();

    //std::cout << "Process " << comm->getRank() << " closed file. " << std::endl;


    if (!hasEdgeWeights) {
        assert(values.size() == 0);
        values.resize(ja.size(), 1);//unweighted edges
    }

    assert(ja.size() == ia[localN]);
    SCAI_ASSERT(comm->sum(localN) == globalN, "Sum " << comm->sum(localN) << " should be " << globalN);

    if (comm->sum(ja.size()) != 2*globalM) {
        throw std::runtime_error("Expected " + std::to_string(2*globalM) + " edges, got " + std::to_string(comm->sum(ja.size())));
    }

    //assign matrix
    scai::lama::CSRStorage<ValueType> myStorage(localN, globalN,
            HArray<IndexType>(ia.size(), ia.data()),
            HArray<IndexType>(ja.size(), ja.data()),
            HArray<ValueType>(values.size(), values.data()));

    //std::cout << "Process " << comm->getRank() << " created local storage " << std::endl;

    return scai::lama::distribute<scai::lama::CSRSparseMatrix<ValueType>>(myStorage, dist, noDist);
    //ThomasBranses ? return scai::lama::CSRSparseMatrix<ValueType>( dist, std::move(myStorage) ); // if no comm
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraphBinary(const std::string filename, const scai::dmemo::CommunicatorPtr comm) {
    SCAI_REGION("FileIO.readGraphBinary")

    typedef unsigned long int ULONG;

    // root PE reads header and broadcasts information to the other PEs
    IndexType headerSize = 3;   // as used in KaHiP::parallel_graph_io.cpp
    std::vector<ULONG> header(headerSize, 0);
    bool success=false;

    if( comm->getRank()==0 ) {
        std::cout <<  "Reading binary graph ..."  << std::endl;
        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if(file) {
            success = true;
            file.read((char*)(&header[0]), headerSize*sizeof(ULONG));
        }
        file.close();
    }

    if (not comm->any(success)) {
        throw std::runtime_error("Error while opening the file " + filename);
    }

    //broadcast the header info
    comm->bcast( header.data(), 3, 0 );

    ULONG version = header[0];
    ULONG globalN = header[1];
    ULONG M = header[2];

    PRINT0( "Binary read, version= " << version << ", N= " << globalN << ", M= " << M );

    if( version != fileTypeVersionNumber ) {
        throw std::runtime_error( "filetype version mismatch" );
    }

    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();

    //
    // set local range
    //
    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, thisPE, numPEs );
    const IndexType localN = endLocalRange - beginLocalRange;
    SCAI_ASSERT_LE_ERROR(localN, std::ceil(ValueType(globalN) / numPEs), "localN: " << localN << ", optSize: " << std::ceil(globalN / numPEs));

    // set like in KaHiP/parallel/prallel_src/app/configuration.h in configuration::standard
    //const IndexType binary_io_window_size = 64;

    const IndexType window_size = numPEs;// std::min( binary_io_window_size, numPEs );
    IndexType lowPE =0;
    IndexType highPE = window_size;

    std::vector<IndexType> ia;//(localN+1, 0);  localN is not known yet
    std::vector<IndexType> ja;
    std::vector<ValueType> values;


    while( lowPE<numPEs ) {
        if( thisPE>=lowPE and thisPE<highPE) {
            std::ifstream file;
            file.open(filename.c_str(), std::ios::binary | std::ios::in);

std::cout << "Process " << thisPE << " reading from " << beginLocalRange << " to " << endLocalRange << ", in total, localN= " << localN << " nodes/lines" << std::endl;

            ia.resize( localN +1);
            ia[0]=0;

            //
            // read the vertices offsets
            //
            SCAI_REGION_START("FileIO.readGraphBinary.fileRead")

            const ULONG startPos = (headerSize+beginLocalRange)*(sizeof(ULONG));
            ULONG* vertexOffsets = new ULONG[localN+1];
            file.seekg(startPos);
            file.read( (char *)(vertexOffsets), (localN+1)*sizeof(ULONG) );

            //
            // read the edges
            //
            ULONG edgeStartPos = vertexOffsets[0];

            const ULONG numReads = vertexOffsets[localN]-vertexOffsets[0];
            const ULONG numEdges = numReads/sizeof(ULONG);
            ULONG* edges = new ULONG[numEdges];
            file.seekg( edgeStartPos );
            file.read( (char *)(edges), (numEdges)*sizeof(ULONG) );

            SCAI_REGION_END("FileIO.readGraphBinary.fileRead")

            //TODO: construct the matrix outside of the while loop
            // not sure if can be done since we need the vertexOffsets and edges arrays

            //
            // construct CSRSparseMatrix
            //

            IndexType pos = 0;

            bool hasEdgeWeights = false;
            std::vector<ULONG> neighbors;

            for( IndexType i=0; i<localN; i++) {
                SCAI_REGION("FileIO.readGraphBinary.buildCSRmatrix")
                ULONG nodeDegree = (vertexOffsets[i+1]-vertexOffsets[i])/sizeof(ULONG);
                SCAI_ASSERT_GT_ERROR( nodeDegree, 0, "Node with degree zero not allowed, for node " << i*(thisPE+1) );
                neighbors.resize(nodeDegree);

                for(ULONG j=0; j<nodeDegree; j++, pos++) {
                    SCAI_ASSERT_LE_ERROR(pos, numEdges, "Number of local non-zero values is greater than the total number of edges read.");

                    ULONG neighbor = edges[pos];
                    if (neighbor >= globalN || neighbor < 0) {
                        throw std::runtime_error(std::string(__FILE__) +", "+std::to_string(__LINE__) + ": Found illegal neighbor " + std::to_string(neighbor) + " in line " + std::to_string(i+beginLocalRange));
                    }

                    neighbors[j] = neighbor;
                }

                //set Ia array
                ia[i+1] = ia[i] + neighbors.size();
                //copy neighbors to Ja array
                std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(ja));
                if (hasEdgeWeights) {
                    assert(ja.size() == values.size());
                }
            }

            // if no edge weight values vector is just 1s
            if (!hasEdgeWeights) {
                assert(values.size() == 0);
                values.resize(ja.size(), 1);//unweighted edges
            }
            assert(ja.size() == ia[localN]);

            delete[] vertexOffsets;
            delete[] edges;
            file.close();
        }

        lowPE  += window_size;
        highPE += window_size;
        comm->synchronize();
    }

    //
    //assign matrix
    //

    scai::lama::CSRStorage<ValueType> myStorage(localN, globalN,
            HArray<IndexType>(ia.size(), ia.data()),
            HArray<IndexType>(ja.size(), ja.data()),
            HArray<ValueType>(values.size(), values.data()));

    // block distribution for rows and no distribution for columns
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

    // myStorage is exactly my local part corresponding to the block distribution
    // ThomasBrandes: is localN realy dist->getLocalSize()
    return scai::lama::CSRSparseMatrix<ValueType>( dist, std::move( myStorage ) );
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraphMatrixMarket(const std::string filename, const scai::dmemo::CommunicatorPtr comm) {
    SCAI_REGION( "FileIO.readGraphMatrixMarket" );
    std::ifstream file(filename);

    scai::common::Settings::putEnvironment( "SCAI_IO_TYPE_DATA", "_Pattern" );

    if(file.fail())
        throw std::runtime_error("Could not open file "+ filename + ".");

    //skip the first lines that have comments starting with '%'
    std::string line;
    std::getline(file, line);

    while( line[0]== '%') {
        std::getline(file, line);
    }
    std::stringstream ss;
    ss.str( line );

    IndexType numRows;
    IndexType numColumns;
    IndexType numValues;

    ss >> numRows>> numColumns >> numValues;

    SCAI_ASSERT( numRows==numColumns, "Number of rows should be equal to number of columns");

    scai::lama::CSRSparseMatrix<ValueType> graph;

    const scai::dmemo::DistributionPtr rowDist(new scai::dmemo::BlockDistribution(numRows, comm));

    graph.readFromFile( filename, rowDist );

    //unsetenv( "SCAI_IO_TYPE_DATA" );
    return graph;
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readEdgeList(const std::string filename, const scai::dmemo::CommunicatorPtr comm, const bool binary) {
    SCAI_REGION( "FileIO.readEdgeList" );

    typedef unsigned long long int ULLI;

    const auto flags = binary ? std::ios::in | std::ios::binary : std::ios::in;
    const IndexType headerSize = 2;
    std::ifstream file(filename, flags);

    if(file.fail())
        throw std::runtime_error("Could not open file "+ filename + ".");

    ULLI globalM, globalN;

    if (binary) {
        std::vector<ULLI> header(headerSize);

        bool success=false;

        if( comm->getRank()==0 ) {
            std::cout <<  "Reading binary edge list ...";
        }

        if(file) {
            success = true;
            file.read((char*)(&header[0]), headerSize*sizeof(ULLI));
        }


        if (not comm->all(success)) {
            throw std::runtime_error("Error while opening the file " + filename);
        }

        globalN = header[0];
        globalM = header[1];
    } else {
        //skip the first lines that have comments starting with '%'
        std::string line;
        std::getline(file, line);

        // skip comments, maybe not needed
        while( line[0]== '%') {
            std::getline(file, line);
        }
        std::stringstream ss;
        std::string item;
        ss.str( line );

        std::getline(ss, item, ' ');
        globalN = std::stoll(item);
        std::getline(ss, item, ' ');
        globalM = std::stoll(item);
    }
    
    if (comm->getRank() == 0) {
        std::cout << " expecting " << globalN << " nodes and " << globalM << " edges." << std::endl;
    }
    if( globalN<=0 or globalM<0 ) {
        throw std::runtime_error("Negative input, maybe int value is not big enough: globalN= " + std::to_string(globalN) + " , globalM= " + std::to_string(globalM));
    }

    const IndexType rank = comm->getRank();
    const IndexType size = comm->getSize();
    const ULLI avgEdgesPerPE = globalM/size;
    assert(avgEdgesPerPE >= 0);
    assert(avgEdgesPerPE <= globalM);

    const ULLI beginLocalRange = rank*avgEdgesPerPE;//TODO: possibly adapt with block distribution
    const ULLI endLocalRange = (rank == size-1) ? globalM : (rank+1)*avgEdgesPerPE;

    //seek own part of file
    if (binary) {
        const ULLI startPos = (headerSize+2*beginLocalRange)*(sizeof(ULLI));
        file.seekg(startPos);
    } else {
        // scroll to own part of file
        for (ULLI i = 0; i < beginLocalRange; i++) {
            std::string line;
            std::getline(file, line);
        }
    }

    std::vector< std::pair<IndexType, IndexType>> edgeList;
    std::vector<ULLI> binaryEdges(2*(endLocalRange-beginLocalRange));

    //read in edges
    if (binary) {
        file.read( (char *)(binaryEdges.data()), (2*(endLocalRange-beginLocalRange))*sizeof(ULLI) );
    } else {
        for (ULLI i = 0; i < endLocalRange - beginLocalRange; i++) {
            std::string line;
            std::getline(file, line);
            std::stringstream ss( line );

            IndexType v1, v2;
            ss >> v1;
            ss >> v2;
            binaryEdges[2*i] = v1;
            binaryEdges[2*i+1] = v2;
        }
    }

    IndexType maxEncounteredNode = 0;
    IndexType maxFirstNode = 0;

    //convert to edge lists
    for (ULLI i = 0; i < endLocalRange - beginLocalRange; i++) {
        IndexType v1 = binaryEdges[2*i];
        IndexType v2 = binaryEdges[2*i+1];
        maxEncounteredNode = std::max(maxEncounteredNode, v1);
        maxEncounteredNode = std::max(maxEncounteredNode, v2);
        maxFirstNode = std::max(maxFirstNode, v1);

        if (v1 >= globalN) {
            throw std::runtime_error("Process " + std::to_string(rank) + ": Illegal node id " + std::to_string(v1) + " in edge list for " + std::to_string(globalN) + " nodes.");
        }

        if (v2 >= globalN) {
            throw std::runtime_error("Process " + std::to_string(rank) + ": Illegal node id " + std::to_string(v2) + " in edge list for " + std::to_string(globalN) + " nodes.");
        }

        edgeList.push_back( std::make_pair( v1, v2) );
    }

    //std::cout << "Process " << comm->getRank() << ": maxFirstNode " << maxFirstNode << std::endl;

    maxEncounteredNode = comm->max(maxEncounteredNode);
    if (maxEncounteredNode < globalN/2 && comm->getRank() == 0) {
        std::cout << "Warning: More than half of all nodes are isolated!" << std::endl;
        std::cout << "Max encountered node: " << maxEncounteredNode << std::endl;
    }
    scai::lama::CSRSparseMatrix<ValueType> graph = GraphUtils<IndexType, ValueType>::edgeList2CSR( edgeList, comm );

    scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, globalN) );
    scai::dmemo::DistributionPtr noDist( new scai::dmemo::NoDistribution(globalN));
    graph.redistribute(rowDistPtr, noDist);

    return graph;
}
//-------------------------------------------------------------------------------------------------

//TODO: handle case where number of files != numPEs

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readEdgeListDistributed(const std::string prefix, const scai::dmemo::CommunicatorPtr comm) {
    SCAI_REGION( "FileIO.readEdgeListDistributed" );

    const IndexType thisPE = comm->getRank();
    PRINT0("About to read a distributed edge list");

    std::string thisFileName = prefix+std::to_string(thisPE);
    std::ifstream file( thisFileName );

    // open thisFile and store edges in the edge list
    std::vector< std::pair<IndexType, IndexType>> edgeList;

    if (file.fail()) {
        PRINT0("Read from multiple files, one file per PE");
        throw std::runtime_error("Reading graph from " + thisFileName + " failed for PE" + std::to_string(thisPE) );
    } else {
        if( comm->getRank()==0 ) {
            std::cout<< "Reading from file "<< prefix << std::endl;
        }
    }

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        std::stringstream ss( line );

        //comments
        if(ss.str()[0]=='#' or ss.str()[0]=='%'){
            continue;
        }

        IndexType v1, v2;
        ss >> v1;
        ss >> v2;

        edgeList.push_back( std::make_pair( v1, v2) );
    }

    return  GraphUtils<IndexType, ValueType>::edgeList2CSR( edgeList, comm );

}
//-------------------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType> > FileIO<IndexType, ValueType>::readCoordsOcean(const std::string filename, const IndexType dimension, const scai::dmemo::CommunicatorPtr comm) {
    SCAI_REGION( "FileIO.readCoords" );
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("Could not open file "+ filename + ".");

    std::string line;
    bool read = !std::getline(file, line).fail();
    if (!read) {
        throw std::runtime_error("Could not read first line of " + filename + ".");
    }

    std::stringstream ss( line );
    std::string item;
    bool readLine = !std::getline(ss, item, ' ').fail();
    if (!readLine or item.size() == 0) {
        throw std::runtime_error("Unexpected end of first line.");
    }

    //now read in file size
    const IndexType globalN = std::stoi(item);
    if (!(globalN >= 0)) {
        throw std::runtime_error(std::to_string(globalN) + " is not a valid node count.");
    }

    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll forward to begin of local range
    for (IndexType i = 0; i < beginLocalRange; i++) {
        std::getline(file, line);
    }

    //create result vector
    std::vector<HArray<ValueType> > coords(dimension);
    for (IndexType dim = 0; dim < dimension; dim++) {
        coords[dim] = HArray<ValueType>(localN, ValueType(0));
    }

    //read local range
    for (IndexType i = 0; i < localN; i++) {
        bool read = !std::getline(file, line).fail();
        if (!read) {
            throw std::runtime_error("Unexpected end of coordinate file. Was the number of nodes correct?");
        }
        std::stringstream ss( line );
        std::string item;

        //first column contains index
        bool readIndex = !std::getline(ss, item, ' ').fail();
        if (!readIndex or item.size() == 0) {
            throw std::runtime_error("Could not read first element of line " + std::to_string(i+1));
        }
        IndexType nodeIndex = std::stoi(item);

        if (!nodeIndex == i+1) {
            throw std::runtime_error("Found index " + std::to_string(nodeIndex) + " in line " + std::to_string(i+1));
        }

        //remaining columns contain coordinates
        IndexType dim = 0;
        while (dim < dimension) {
            bool read = !std::getline(ss, item, ' ').fail();
            if (!read or item.size() == 0) {
                throw std::runtime_error("Unexpected end of line. Was the number of dimensions correct?");
            }
            ValueType coord = std::stod(item);
            coords[dim][i] = coord;
            dim++;
        }
        if (dim < dimension) {
            throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimension) + " expected in line '" + line + "'");
        }
    }

    if (endLocalRange == globalN) {
        bool eof = std::getline(file, line).eof();
        if (!eof) {
            throw std::runtime_error(std::to_string(globalN) + " coordinates read, but file continues.");
        }
    }

    std::vector<DenseVector<ValueType> > result(dimension);

    for (IndexType i = 0; i < dimension; i++) {
        result[i] = DenseVector<ValueType>(dist, coords[i] );
    }

    return result;
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoordsTEEC ( std::string filename, IndexType numberOfCoords, IndexType dimension, std::vector<DenseVector<ValueType>>& nodeWeights, const scai::dmemo::CommunicatorPtr comm) {
    SCAI_REGION( "FileIO.readCoordsTEEC" );

    std::vector<DenseVector<ValueType> > tempResult = FileIO<IndexType, ValueType>::readCoords(filename, numberOfCoords, dimension+1, comm, Format::METIS);

    nodeWeights.resize(1);
    nodeWeights[0] = tempResult[dimension];//last column is node weights
    tempResult.resize(dimension);//omit last column from coordinates
    return tempResult;
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads these coordinates and returns a vector of DenseVectors, one for each dimension
 */
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoords( const std::string filename, const IndexType numberOfPoints, const IndexType dimension, const scai::dmemo::CommunicatorPtr comm, Format format) {
    SCAI_REGION( "FileIO.readCoords" );

    IndexType globalN= numberOfPoints;
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

    if (format == Format::ADCIRC) {
        PRINT0("Reading coordinates in ADCIRC format");
        return readCoordsOcean(filename, dimension, comm);
    }
    else if( format== Format::MATRIXMARKET) {
        PRINT0("Reading coordinates in MATRIXMARKET format");
        return readCoordsMatrixMarket( filename, comm );
    } else if( format==Format::BINARY) {
        PRINT0("Reading coordinates in BINARY format");
        return  readCoordsBinary( filename, numberOfPoints, dimension, comm);
    }

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    IndexType numCommentLines = 0;

    std::string line;
    std::getline(file, line);
    while( line[0]== '%') {
        std::getline(file, line);
        ++numCommentLines;
    }

    // rewind stream to beggining of file
    file.clear();
    file.seekg(0);

    SCAI_ASSERT_LE_ERROR( beginLocalRange+localN+numCommentLines, globalN+numCommentLines, "Wrong total number or rows to read");

    //scroll forward to begin of local range
    for (IndexType i = 0; i < beginLocalRange+numCommentLines; i++) {
        std::getline(file, line);
    }

    //create result vector
    std::vector<std::vector<ValueType> > coords(dimension);
    for (IndexType dim = 0; dim < dimension; dim++) {
        coords[dim].resize(localN);
        SCAI_ASSERT_EQUAL_ERROR(coords[dim].size(), localN);
    }

    //read local range
    for (IndexType i = 0; i < localN; i++) {
        bool read = !std::getline(file, line).fail();
        if (!read) {
            PRINT(*comm << ": "<<  beginLocalRange+numCommentLines+i );
            throw std::runtime_error("Unexpected end of coordinate file. Was the number of nodes correct?");
        }
        std::stringstream ss( line );
        std::string item;

        IndexType dim = 0;
        while (dim < dimension) {
            bool read;
            do {//skip multiple whitespace
                read = !std::getline(ss, item, ' ').fail();
            } while (item.size() == 0);

            if (!read) {
                throw std::runtime_error("Unexpected end of line " + line +". Was the number of dimensions correct?");
            }
            // WARNING: in supermuc (with the gcc/5) the std::stod returns the int part !!
            ValueType coord = std::stod(item);
            coords[dim][i] = coord;
            dim++;
        }
        if (dim < dimension) {
            throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimension) + " expected in line '" + line + "'");
        }
    }

    if (endLocalRange == globalN) {
        bool eof = std::getline(file, line).eof();
        if (!eof) {
            throw std::runtime_error(std::to_string(numberOfPoints) + " coordinates read, but file continues.");
        }
    }

    std::vector<DenseVector<ValueType> > result(dimension);

    for (IndexType dim = 0; dim < dimension; dim++) {
        result[dim] = DenseVector<ValueType>(dist, HArray<ValueType>(localN, coords[dim].data()) );
    }

    return result;
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads these coordinates and returns a vector of DenseVectors, one for each dimension
 */
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoordsBinary( std::string filename, const IndexType numberOfPoints, const IndexType dimension, const scai::dmemo::CommunicatorPtr comm) {
    SCAI_REGION( "FileIO.readCoordsBinary" );

    typedef unsigned long int UINT; // maybe IndexType is not big enough for file position

    const IndexType globalN= numberOfPoints;
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    PRINT0("Reading binary coordinates...");

    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();

    //
    // set local range
    //

    //we assume a block distribution
    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, thisPE, numPEs);
    const IndexType localN = endLocalRange - beginLocalRange;

    //WARNING: for the binary format files, in 2D cases the 3rd coordinate is 0 but we must always read
    //         3 coordinates from the file and just not copy the 3rd
    IndexType maxDimension = 3;

    if( dimension>maxDimension ) {
        PRINT0("Function readCoordsBinary can only read coordinates up to 3 dimensions.\nAborting...");
        throw std::runtime_error("Number of dimensions not supported.");
    }

    const UINT beginLocalCoords = beginLocalRange*maxDimension;
    //const UINT endLocalCoords = endLocalRange*maxDimension;
    const UINT localTotalNumOfCoords = localN*maxDimension;

    SCAI_ASSERT_EQ_ERROR( globalN, comm->sum(localN), "Mismatch in total number of coordinates" );
    SCAI_ASSERT_EQ_ERROR( globalN, comm->sum(localTotalNumOfCoords)/maxDimension, "Mismatch in total number of coordinates" );

    // set like in KaHiP/parallel/prallel_src/app/configuration.h in configuration::standard
    //const IndexType binary_io_window_size = 64;

    const IndexType window_size = numPEs;
    IndexType lowPE =0;
    IndexType highPE = window_size;

    //create local part result vector
    std::vector<HArray<ValueType> > coords(dimension);
    for (IndexType dim = 0; dim < dimension; dim++) {
        coords[dim] = HArray<ValueType>(localN, ValueType(0));
    }

    while( lowPE<numPEs ) {
        if( thisPE>=lowPE and thisPE<highPE ) {
            std::ifstream file;
            file.open(filename.c_str(), std::ios::binary | std::ios::in);

            //std::cout << "Process " << thisPE << " reading from " << beginLocalCoords << " to " << endLocalCoords << ", in total, localNumCoords= " << localTotalNumOfCoords << " coordinates and " << (localTotalNumOfCoords)*sizeof(ValueType) << " bytes." << std::endl;

            SCAI_REGION_START("FileIO.readCoordsBinary.fileRead" );
            const UINT startPos = beginLocalCoords*sizeof(double);
            double* localPartOfCoords = new double[localTotalNumOfCoords];
            file.seekg(startPos);
            file.read( (char *)(localPartOfCoords), (localTotalNumOfCoords)*sizeof(double) );
            SCAI_REGION_END("FileIO.readCoordsBinary.fileRead" );

            for(IndexType i=0; i<localN; i++) {
                for(IndexType dim=0; dim<dimension; dim++) { //this is expensive, since a writeAccess is created for each loop iteration
                    coords[dim][i] = localPartOfCoords[i*maxDimension+dim]; //always maxDimension coords per point
                }
            }

            if( thisPE==numPEs-1) {
                //SCAI_ASSERT_EQ_ERROR( file.tellg(), globalN*maxDimension*sizeof(ValueType) , "While reading coordinates in binary: Position in file " << filename << " is not correct for process " << thisPE );
                SCAI_ASSERT_EQ_ERROR( file.tellg()/(maxDimension*sizeof(double)), globalN, "While reading coordinates in binary: Position in file " << filename << " is not correct for process " << thisPE );
            } else {
                SCAI_ASSERT_EQ_ERROR( file.tellg()/( maxDimension*sizeof(double)*(thisPE+1) ), localN, "While reading coordinates in binary: Position in file " << filename << " is not correct for process " << thisPE );
            }

            delete[] localPartOfCoords;
            file.close();
        }
        lowPE  += window_size;
        highPE += window_size;
        comm->synchronize();
    }

    //
    // set the return vector
    //

    std::vector<DenseVector<ValueType> > result(dimension);

    //again, we assume a block distribution
    const scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(globalN, comm));

    for (IndexType i=0; i<  dimension; i++) {
        result[i] = DenseVector<ValueType>( blockDist, coords[i] );
    }

    return result;
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoordsMatrixMarket ( const std::string filename, const scai::dmemo::CommunicatorPtr comm) {
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    //skip the first lines that have comments starting with '%'
    std::string line;
    std::getline(file, line);

    while( line[0]== '%') {
        std::getline(file, line);
    }
    std::stringstream ss;
    ss.str( line );

    IndexType numPoints ;
    IndexType dimensions ;

    ss >> numPoints >> dimensions;

    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(numPoints, comm));

    PRINT0( "numPoints= "<< numPoints << " , " << dimensions);

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, numPoints, comm->getRank(), comm->getSize());

    // the local ranges for the MatrixMarket format
    const IndexType beginLocalRangeMM = beginLocalRange*dimensions;
    const IndexType endLocalRangeMM = endLocalRange*dimensions;
    const IndexType localN = endLocalRange - beginLocalRange;
    const IndexType localNMM = endLocalRangeMM - beginLocalRangeMM;

    //PRINT( *comm << ": localN= "<< localNMM << ", numPoints= "<< numPoints << ", beginLocalRange= "<< beginLocalRangeMM << ", endLocalRange= " << endLocalRangeMM );

    //scroll forward to begin of local range
    for (IndexType i = 0; i < beginLocalRangeMM; i++) {
        std::getline(file, line);
    }

    //create result vector
    std::vector<HArray<ValueType> > coords(dimensions);
    for (IndexType dim = 0; dim < dimensions; dim++) {
        coords[dim] = HArray<ValueType>(localN, ValueType(0));
    }

    //read local range
    for (IndexType i = 0; i < localNMM; i++) {
        bool read = !std::getline(file, line).fail();

        if (!read and i!=localNMM-1 ) {
            throw std::runtime_error("In FileIO.cpp, line " + std::to_string(__LINE__) +"Unexpected end of coordinate file. Was the number of nodes correct?");
        }

        std::stringstream ss;
        ss.str( line );

        ValueType c;
        ss >> c;
        coords[i%dimensions][int(i/dimensions)] = c;
//PRINT( *comm << ": " << i%dimensions << " , " << i/dimensions << " :: " << coords[i%dimensions][int(i/dimensions)]  );
    }

    if (endLocalRange == numPoints) {
        bool eof = std::getline(file, line).eof();
        if (!eof) {
            throw std::runtime_error(std::to_string(numPoints) + " coordinates read, but file continues.");
        }
    }

    std::vector<DenseVector<ValueType> > result(dimensions);

    for (IndexType i = 0; i < dimensions; i++) {
        result[i] = DenseVector<ValueType>(dist, coords[i] );
    }

    return result;
}
//-------------------------------------------------------------------------------------------------

/* Read graph and coordinates from a OFF file. Coordinates are (usually) in 3D.
 */

template<typename IndexType, typename ValueType>
void  FileIO<IndexType, ValueType>::readOFFTriangularCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const std::string filename ) {

    std::ifstream file(filename);
    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    std::string line;
    std::getline(file, line);

    if( line!="OFF" ) {
        PRINT("The OFF file" << filename << " should start with the word OFF in the first line. Aborting.");
        throw std::runtime_error("Reading from an OFF file");
    }

    std::getline(file, line);
    std::stringstream ss;
    ss.str( line );

    IndexType N, numFaces, numEdges;

    ss >> N >> numFaces >> numEdges;

    //
    // first, read the N 3D coordinates
    //

    IndexType dimension = 3;

    std::vector<HArray<ValueType> > coordsLA(dimension);
    for (IndexType dim = 0; dim < dimension; dim++) {
        coordsLA[dim] = HArray<ValueType>(N, ValueType(0));
    }

    for(IndexType i=0; i<N; i++) {
        bool read = !std::getline(file, line).fail();
        if (!read) {
            //PRINT();
            throw std::runtime_error("Unexpected end of coordinates part in OFF file. Was the number of nodes correct?");
        }
        std::stringstream ss( line );
        std::string item;

        IndexType dim = 0;
        while (dim < dimension) {
            bool read;
            do {//skip multiple whitespace
                read = !std::getline(ss, item, ' ').fail();
            } while (item.size() == 0);

            if (!read) {
                throw std::runtime_error("Unexpected end of line " + line +". Was the number of dimensions correct?");
            }
            // WARNING: in supermuc (with the gcc/5) the std::stod returns the int part !!
            ValueType coord = std::stod(item);
            coordsLA[dim][i] = coord;
            dim++;
        }
        if (dim < dimension) {
            throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimension) + " expected in line '" + line + "'");
        }
    }

    coords.resize( dimension );

    for (IndexType i = 0; i < dimension; i++) {
        coords[i] = DenseVector<ValueType>( coordsLA[i] );
    }

    //
    // now, read the faces. Create the graph as an adjacency list to prevent N^2 space and convert to CSR later
    // each line contains the info for one face:
    // example: 4 20 7 4 12
    // first is the number of vertices for that face, 4 here, and then the vertices that define the face (20, 7, 4, 12)
    //

    std::vector<std::set<IndexType>> adjList( N );
    IndexType edgeCnt =0;

    for(IndexType i=0; i<numFaces; i++) {
        bool read = !std::getline(file, line).fail();
        if (!read) {
            //PRINT();
            throw std::runtime_error("Unexpected end of faces part in OFF file. Was the number of nodes correct?");
        }
        std::stringstream ss( line );

        IndexType numVertices;  // number of vertices of this face
        ss >> numVertices;
        SCAI_ASSERT_EQ_ERROR( numVertices, 3, "Found face with more than 3 vertices; this is not a triangular mesh. Aborting");

        std::vector<IndexType> face( numVertices );

        for(IndexType f=0; f<numVertices; f++) {
            ss >> face[f];
        }

        for(IndexType v1=0; v1<numVertices; v1++) {
            SCAI_ASSERT_LE_ERROR( v1, N, "Found vertex with too big index.");
            for(IndexType v2=v1+1; v2<numVertices; v2++) {
                adjList[face[v1]].insert(face[v2]);
                adjList[face[v2]].insert(face[v1]);
                ++edgeCnt;
            }
        }
    }
    if( numEdges!=0 ) {
        SCAI_ASSERT_EQ_ERROR( numEdges, edgeCnt/2, "Possibly wrong number of edges");
    }

    //
    // convert adjacency list to CSR matrix
    //

    graph = GraphUtils<IndexType, ValueType>::getCSRmatrixFromAdjList_NoEgdeWeights( adjList );
    SCAI_ASSERT_EQ_ERROR( graph.getNumColumns(), N, "Wrong number of columns");
    SCAI_ASSERT_EQ_ERROR( graph.getNumRows(), N, "Wrong number of rows");
    if( numEdges!=0 ) {
        SCAI_ASSERT_EQ_ERROR( graph.getNumValues()/2, numEdges, "Wrong number of edges");
    }
}
//-------------------------------------------------------------------------------------------------

/* Read graph and coordinates from a dom.geo file of the ALYA tool. Coordinates are (usually) in 3D.
 * The mesh is composed out of elements (eg. hexagons, tetrahedra etc) and each elements is composed out of nodes.
 */

template<typename IndexType, typename ValueType>
void  FileIO<IndexType, ValueType>::readAlyaCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const IndexType N, const IndexType dimensions,  const std::string filename ) {

    std::ifstream file(filename);
    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    std::string line;

    // skip first part of file until we find the line with keyword "ELEMENTS"

    while( std::getline(file, line) ) {
        size_t pos = line.find("ELEMENTS");
        if(pos!=std::string::npos) { 		//found
            std::cout<< "FOUND start of ELEMENTS" << std::endl;
            break;
        }
    }

    // We are in the "ELEMENTS" part, each line starts with a number, the ID of this element.
    // Next, there are some numbers, each corresponding to one node (not elements) ID. We will construct
    // the graph using these nodes as graph nodes.

    {
        std::vector<std::set<IndexType>> adjList( N );

        IndexType edgeCnt =0;
        int numElems = 0;

        while( std::getline(file, line) ) {
            size_t pos = line.find("END_ELEMENTS");
            if(pos!=std::string::npos) { 		//found
                std::cout<< "FOUND end of elements" << std::endl;
                break;
            }

            std::vector<IndexType> face;

            std::stringstream ss( line );
            IndexType currElem = 0;
            ss >> currElem;

            IndexType v;
            while( ss >> v) {
                face.push_back(v);
                assert(v>0);
            }

            for(IndexType v1=0; v1<face.size()-1; v1++) {
                SCAI_ASSERT_LE_ERROR( face[v1], N, "Found vertex with too big index.");
                auto ret = adjList[face[v1]-1].insert(face[v1+1]-1);		//TODO: check if correct: abstract 1 to start from 0
                adjList[face[v1+1]-1].insert(face[v1]-1);
                //std::cout << face[v1] << "-" << face[v1+1] << "    ";
                if(ret.second==true) {
                    ++edgeCnt;
                }
            }
            auto ret = adjList[face[0]-1].insert(face.back()-1);
            adjList[face.back()-1].insert(face[0]-1);

            if(ret.second==true) {
                ++edgeCnt;
            }

            numElems++;
            //if( numElems>9800300) std::cout<<"Current element=" <<  currElem << std::endl;
        }
        std::cout<< "Counted " << numElems << " elements and " << edgeCnt << " edges" << std::endl;

        // convert adjacency list to CSR matrix
        //

        graph = GraphUtils<IndexType, ValueType>::getCSRmatrixFromAdjList_NoEgdeWeights( adjList );
    }


    //
    // get the coordinates
    //

    while( std::getline(file, line) ) {
        size_t pos = line.find("COORDINATES");
        if(pos!=std::string::npos) { 		//found
            std::cout<< "FOUND start of COORDINATES" << std::endl;
            break;
        }
    }

    std::vector<scai::hmemo::HArray<ValueType> > coordsLA(dimensions);
    for (IndexType dim = 0; dim < dimensions; dim++) {
        coordsLA[dim] = scai::hmemo::HArray<ValueType>(N, 0.0);
    }

    for(IndexType i=0; i<N; i++) {
        bool read = !std::getline(file, line).fail();
        if (!read) {
            throw std::runtime_error("Unexpected end of coordinates part in OFF file. Was the number of nodes correct?");
        }
        std::stringstream ss( line );

        IndexType currElem = 0;
        ss >> currElem;

        ValueType coord;
        IndexType dim = 0;
        while( ss >> coord) {
            coordsLA[dim][i] = coord;
            dim++;
        }

        if (dim < dimensions) {
            throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimensions) + " expected in line '" + line + "'");
        }
    }
    std::getline(file, line);
    size_t pos = line.find("END_COORDINATES");
    if(pos==std::string::npos) { 		// NOT found
        std::cout<< "Did not find end of coordinates but: " << line << std::endl;
        throw std::runtime_error("Wrong number of points and coordinates?");
    }


    coords.resize( dimensions );

    for (IndexType i = 0; i < dimensions; i++) {
        coords[i] = DenseVector<ValueType>( coordsLA[i] );
    }

}//readAlyaCentral



template<typename IndexType, typename ValueType>
DenseVector<IndexType> FileIO<IndexType, ValueType>::readPartition(const std::string filename, const IndexType globalN) {
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    //skip the first lines that have comments starting with '%'
    std::string line;
    bool read = !std::getline(file, line).fail();

    while( line[0]== '%') {
        read = !std::getline(file, line).fail();
    }
    std::stringstream ss;
    ss.str( line );

    //get local range
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));
    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll to begin of local range.
    for (IndexType i = 0; i < beginLocalRange; i++) {
        read = !std::getline(file, line).fail();
    }

    std::vector<IndexType> localPart;

    for (IndexType i = 0; i < localN; i++) {
        if (!read) {
            throw std::runtime_error("In FileIO.cpp, line " + std::to_string(__LINE__) +": Unexpected end of file " + filename + ". Was the number of nodes correct?");
        }
        localPart.push_back(std::stoi(line));
        read = !std::getline(file, line).fail();
    }

    scai::hmemo::HArray<IndexType> hLocal(localPart.size(), localPart.data());
    DenseVector<IndexType> result(dist, hLocal);

    return result;
}

template<typename IndexType, typename ValueType>
std::pair<std::vector<ValueType>, std::vector<ValueType>> FileIO<IndexType, ValueType>::getBoundingCoords(std::vector<ValueType> centralCoords, IndexType level) {
    const IndexType dimension = centralCoords.size();
    const ValueType offset = 0.5 * (IndexType(1) << IndexType(level));
    std::vector<ValueType> minCoords(dimension);
    std::vector<ValueType> maxCoords(dimension);

    for (IndexType i = 0; i < dimension; i++) {
        minCoords[i] = centralCoords[i] - offset;
        maxCoords[i] = centralCoords[i] + offset;
    }
    return {minCoords, maxCoords};
}

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readQuadTree( std::string filename, std::vector<DenseVector<ValueType>> &coords ) {
    SCAI_REGION( "FileIO.readQuadTree" );

    const IndexType dimension = 3;
    const IndexType valuesPerLine = 1+2*dimension + 2*dimension*dimension;

    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("Reading file "+ filename+ " failed.");

    std::map<std::vector<ValueType>, std::shared_ptr<SpatialCell<ValueType>>> nodeMap;
    std::map<std::vector<ValueType>, std::set<std::vector<ValueType>>> pendingEdges;
    std::map<std::vector<ValueType>, std::set<std::vector<ValueType>>> confirmedEdges;
    std::set<std::shared_ptr<SpatialCell<ValueType>> > roots;

    IndexType duplicateNeighbors = 0;

    std::string line;
    while (!std::getline(file, line).fail()) {
        std::vector<ValueType> values;
        std::stringstream ss( line );
        std::string item;

        std::string comparison("timestep");
        std::string prefix(line.begin(), line.begin()+comparison.size());
        if (prefix == comparison) {
            std::cout << "Caught other timestep. Skip remainder of file." << std::endl;
            break;
        }

        IndexType i = 0;

        while (!std::getline(ss, item, ' ').fail()) {
            if (item.size() == 0) {
                continue;
            }

            try {
                values.push_back(std::stod(item));
            } catch (...) {
                std::cout << item << " could not be resolved as number." << std::endl;
                throw;
            }

            i++;
        }

        if (i == 0) {
            //empty line
            continue;
        } else if (i != valuesPerLine) {
            throw std::runtime_error("Expected "+std::to_string(valuesPerLine)+" values, but got "+std::to_string(i)+".");
        }

        //process quadtree node
        const std::vector<ValueType> ownCoords = {values[0], values[1], values[2]};
        const ValueType level = values[3];
        const std::vector<ValueType> parentCoords = {values[4], values[5], values[6]};

        assert(ownCoords != parentCoords);

        assert(*std::min_element(ownCoords.begin(), ownCoords.end()) >= 0);

        std::vector<ValueType> minCoords, maxCoords;
        std::tie(minCoords, maxCoords) = getBoundingCoords(ownCoords, level);

        //create own cell and add to node map
        std::shared_ptr<QuadNodeCartesianEuclid<ValueType>> quadNodePointer(new QuadNodeCartesianEuclid<ValueType>( Point<ValueType>(minCoords), Point<ValueType>(maxCoords)) );
        assert(nodeMap.count(ownCoords) == 0);
        nodeMap[ownCoords] = quadNodePointer;
        assert(confirmedEdges.count(ownCoords) == 0);
        confirmedEdges[ownCoords] = {};

        //check for pending edges
        if (pendingEdges.count(ownCoords) > 0) {
            std::set<std::vector<ValueType>> thisNodesPendingEdges = pendingEdges.at(ownCoords);
            for (std::vector<ValueType> otherCoords : thisNodesPendingEdges) {
                confirmedEdges[ownCoords].insert(otherCoords);
                confirmedEdges[otherCoords].insert(ownCoords);
            }
            pendingEdges.erase(ownCoords);
        }

        //check for parent pointer
        if (parentCoords[0] != -1 && nodeMap.count(parentCoords) == 0) {
            std::tie(minCoords, maxCoords) = getBoundingCoords(parentCoords, level+1);
            std::shared_ptr<QuadNodeCartesianEuclid<ValueType>> parentPointer(new QuadNodeCartesianEuclid<ValueType>(minCoords, maxCoords));
            nodeMap[parentCoords] = parentPointer;
            assert(confirmedEdges.count(parentCoords) == 0);
            confirmedEdges[parentCoords] = {};
            roots.insert(parentPointer);

            //check for pending edges of parent
            if (pendingEdges.count(parentCoords) > 0) {
                //std::cout << "Found pending edges for parent " << parentCoords[0] << ", " << parentCoords[1] << ", " << parentCoords[2] << std::endl;
                std::set<std::vector<ValueType>> thisNodesPendingEdges = pendingEdges.at(parentCoords);
                for (std::vector<ValueType> otherCoords : thisNodesPendingEdges) {
                    confirmedEdges[parentCoords].insert(otherCoords);
                    confirmedEdges[otherCoords].insert(parentCoords);
                }
                pendingEdges.erase(parentCoords);
            }
        }

        if (parentCoords[0] != -1) {
            roots.erase(quadNodePointer);
        }

        assert(nodeMap.count(parentCoords));//Why does this assert work? Why can't it happen that the parentCoords are -1?
        nodeMap[parentCoords]->addChild(quadNodePointer);
        assert(nodeMap[parentCoords]->height() > 1);

        //check own edges, possibly add to pending
        for (IndexType i = 0; i < 2*dimension; i++) {
            const IndexType beginIndex = 2*dimension+1+i*dimension;
            const IndexType endIndex = beginIndex+dimension;
            assert(endIndex <= values.size());
            if (i == 2*dimension -1) {
                assert(endIndex == values.size());
            }
            const std::vector<ValueType> possibleNeighborCoords(values.begin()+beginIndex, values.begin()+endIndex);
            assert(possibleNeighborCoords.size() == dimension);

            if (possibleNeighborCoords[0] == -1) {
                assert(possibleNeighborCoords[1] == -1);
                assert(possibleNeighborCoords[2] == -1);
                continue;
            } else {
                assert(possibleNeighborCoords[1] != -1);
                assert(possibleNeighborCoords[2] != -1);
            }

            if (nodeMap.count(possibleNeighborCoords)) {
                // this is actually not necessary, since if the other node was before this one,
                // the edges were added to the pending list and processed above - except if it was an implicitly referenced parent node.
                confirmedEdges[ownCoords].insert(possibleNeighborCoords);
                confirmedEdges[possibleNeighborCoords].insert(ownCoords);
            } else {
                //create empty set if not yet done
                if (pendingEdges.count(possibleNeighborCoords) == 0) {
                    pendingEdges[possibleNeighborCoords] = {};
                }
                //target doesn't exist yet, can't have confirmed edges
                assert(confirmedEdges.count(possibleNeighborCoords) == 0);

                //if edge is already there, it was duplicate
                if (pendingEdges[possibleNeighborCoords].count(ownCoords)) {
                    duplicateNeighbors++;
                }

                //finally, add pending edge
                pendingEdges[possibleNeighborCoords].insert(ownCoords);
            }
        }
    }

    file.close();

    //only used form printing messaged
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if(comm->getRank()==0){
        std::cout << "Read file, found or created " << nodeMap.size() << " nodes and pending edges for " << pendingEdges.size() << " ghost nodes." << std::endl;
    }
    if (duplicateNeighbors > 0) {
        std::cout << "Found " << duplicateNeighbors << " duplicate neighbors." << std::endl;
    }

    assert(confirmedEdges.size() == nodeMap.size());

    for (auto pendingSets : pendingEdges) {
        assert(nodeMap.count(pendingSets.first) == 0);
    }

    IndexType nodesInForest = 0;
    for (auto root : roots) {
        nodesInForest += root->countNodes();
    }

    if(comm->getRank()==0){
        std::cout << "Found " << roots.size() << " roots with " << nodesInForest << " nodes hanging from them." << std::endl;
    }
    assert(nodesInForest == nodeMap.size());

    //check whether all nodes have either no or the full amount of children
    for (std::pair<std::vector<ValueType>, std::shared_ptr<SpatialCell<ValueType>>> elems : nodeMap) {
        bool consistent = elems.second->isConsistent();
        if (!consistent) {
            std::vector<ValueType> coords = elems.first;
            //throw std::runtime_error("Node at " + std::to_string(coords[0]) + ", " + std::to_string(coords[1]) + ", " + std::to_string(coords[2]) + " inconsistent.");
        }
        //assert(elems.second->isConsistent());
        assert(pendingEdges.count(elems.first) == 0);//list of pending edges was erased when node was handled, new edges should not be added to pending list
    }

    IndexType i = 0;
    IndexType totalEdges = 0;
    IndexType numLeaves = 0;
    IndexType leafEdges = 0;
    std::vector<std::set<std::shared_ptr<SpatialCell<ValueType>> > > result(nodeMap.size());
    for (std::pair<std::vector<ValueType>, std::set<std::vector<ValueType> > > edgeSet : confirmedEdges) {
        result[i] = {};
        for (std::vector<ValueType> neighbor : edgeSet.second) {
            assert(nodeMap.count(neighbor) > 0);
            result[i].insert(nodeMap[neighbor]);
            totalEdges++;
        }
        assert(result[i].size() == edgeSet.second.size());

        if (nodeMap[edgeSet.first]->height() == 1) {
            numLeaves++;
            leafEdges += result[i].size();
            if (result[i].size() == 0) {
                //only parent nodes are allowed to have no edges.
                throw std::runtime_error("Node at " + std::to_string(edgeSet.first[0]) + ", " + std::to_string(edgeSet.first[1]) + ", " + std::to_string(edgeSet.first[2])
                                         + " is isolated leaf node.");
            }
        }

        i++;
    }
    assert(nodeMap.size() == i++);
    if(comm->getRank()==0){
        std::cout << "Read " << totalEdges << " confirmed edges, among them " << leafEdges << " edges between " << numLeaves << " leaves." << std::endl;
    }
    
    /*
     * now convert into CSRSparseMatrix
     */

    IndexType offset = 0;
    for (auto root : roots) {
        offset = root->indexSubtree(offset);
    }

    std::vector<std::shared_ptr<const SpatialCell<ValueType>> > rootVector(roots.begin(), roots.end());

    coords.clear();
    coords.resize(dimension);

    std::vector<std::vector<ValueType> > vCoords(dimension);
    std::vector< std::set<std::shared_ptr<const SpatialCell<ValueType>>>> graphNgbrsCells(nodesInForest);

    for (auto outgoing : confirmedEdges) {
        std::set<std::shared_ptr<const SpatialCell<ValueType>>> edgeSet;
        for (std::vector<ValueType> edgeTarget : outgoing.second) {
            edgeSet.insert(nodeMap[edgeTarget]);
        }
        graphNgbrsCells[nodeMap[outgoing.first]->getID()] = edgeSet;
    }

    scai::lama::CSRSparseMatrix<ValueType> matrix = SpatialTree<ValueType>::template getGraphFromForest<IndexType>( graphNgbrsCells, rootVector, vCoords);

    for (IndexType d = 0; d < dimension; d++) {
        assert(vCoords[d].size() == numLeaves);
        HArray<ValueType> localValues(vCoords[d].size(), vCoords[d].data());
        coords[d] = DenseVector<ValueType>(std::move(localValues));
    }
    return matrix;
}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
std::pair<IndexType, IndexType> FileIO<IndexType, ValueType>::getMatrixMarketCoordsInfos(const std::string filename) {

    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    //skip the first lines that have comments starting with '%'
    std::string line;
    std::getline(file, line);

    while( line[0]== '%') {
        std::getline(file, line);
    }
    std::stringstream ss;
    ss.str( line );

    IndexType numPoints ;
    IndexType dimensions ;

    ss >> numPoints >> dimensions;

    return std::make_pair( numPoints, dimensions);
}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > FileIO<IndexType, ValueType>::readBlockSizes(const std::string filename, const IndexType numBlocks, const IndexType numWeights) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    std::vector<std::vector<ValueType> > blockSizes(numWeights, std::vector<ValueType>(numBlocks,0));

    if( comm->getRank()==0 ) {
        std::ifstream file(filename);
        if(file.fail())
            throw std::runtime_error("File "+ filename+ " failed.");

        //read first line, has the number of blocks
        std::string line;
        std::getline(file, line);
        std::stringstream ss;
        ss.str( line );

        IndexType fileNumBlocks;
        ss >> fileNumBlocks;
        SCAI_ASSERT_EQ_ERROR( numBlocks, fileNumBlocks, "Number of blocks mismatch, given "<< numBlocks << " but the file has "<< fileNumBlocks );

        for(int i=0; i<numBlocks; i++) {
            bool read = !std::getline(file, line).fail();

            if (!read and i!=numBlocks-1 ) {
                throw std::runtime_error("In FileIO.cpp, line " + std::to_string(__LINE__) +": Unexpected end of block sizes file " + filename + ". Was the number of blocks correct?");
            }
            std::stringstream ss;
            ss.str( line );

            for (IndexType j = 0; j < numWeights; j++) {
                ValueType bSize;
                ss >> bSize;
                //blockSizes.push_back(bSize);
                blockSizes[j][i]= bSize;
            }
        }
        SCAI_ASSERT( blockSizes[0].size()==numBlocks, "Wrong number of blocks: "  <<blockSizes[0].size() << " for file " << filename);
        file.close();

        bool eof = std::getline(file, line).eof();
        if (!eof) {
            throw std::runtime_error(std::to_string(numBlocks) + " blocks read, but file continues.");
        }
    }

    for (IndexType i = 0; i < numWeights; i++) {
        comm->bcast( blockSizes[0].data(), numBlocks, 0);
    }

    return blockSizes;
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
CommTree<IndexType,ValueType> FileIO<IndexType, ValueType>::readPETree( const std::string& filename ) {

    if( not fileExists(filename) ) {
        std::cout<<"Erros, file " << filename << " to read processor tree does not exists.\nAborting..." << std::endl;
        throw std::runtime_error("File "+ filename+ " failed.");
    } else {
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        if( comm->getRank()==0 ) {
            std::cout<< "Reading from file "<< filename << std::endl;
        }
    }

    std::ifstream file(filename);

    //skip the first lines that have comments starting with '%' or '#'
    std::string line;
    std::getline(file, line);

    while( line[0]== '%' or line[0]== '#' ) {
        std::getline(file, line);
    }
    std::stringstream ss;
    ss.str( line );

    IndexType numPEs;
    IndexType numWeights;

    //first line after comments should have the number of PEs
    //and number of weights
    ss >> numPEs;
    ss >> numWeights;
    {
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        if( comm->getRank()==0 ) {
            std::cout<< "\t... with "<< numPEs << " and " << numWeights << " weights each"<< std::endl;
        }
    }

    //next entries should have a bit for every node weight indicating if it is  proportional

    std::vector<bool> isProp( numWeights );
    for( unsigned int w=0; w<numWeights; w++ ) {
        std::string b;
        ss >> b;
        isProp[w] = bool( std::stoi(b) );
    }

    //read by line to create the leaves

    std::vector<cNode<IndexType,ValueType>> leaves(numPEs);

    for(int i=0; i<numPEs; i++) {
        bool read = !std::getline(file, line).fail();

        if (!read and i!=numPEs-1 ) {
            throw std::runtime_error("In FileIO.cpp, line " + std::to_string(__LINE__) +": Unexpected end of processor tree file " + filename + ". Was the number of PEs correct?");
        }

        //TODO: this (with all the stings and streams) is probably a stupid and not efficient way; fix
        std::stringstream ss;
        ss.str( line );

        std::vector<unsigned int> hierarchy;
        std::vector<ValueType> weights( numWeights );

        std::string tok;
        ss >> tok;

        while( tok!="#") {
            hierarchy.push_back( std::stoi(tok) );
            ss >> tok;
        }

        // found '#', create the weights

        for( unsigned int w=0; w<numWeights; w++) {
            ss >> tok;
            weights[w] = ValueType(std::stod(tok));
        }

        cNode<IndexType,ValueType> node(hierarchy, weights );
        leaves[i]= node;
    }

    return CommTree<IndexType,ValueType>( leaves, isProp );
}


// taken from https://stackoverflow.com/questions/4316442/stdofstream-check-if-file-exists-before-writing
/** Check if a file exists
@param[in] filename - the name of the file to check
@return    true if the file exists, else false
*/
template<typename IndexType, typename ValueType>
bool FileIO<IndexType, ValueType>::fileExists(const std::string& filename) {
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}
//

// functions to trim a string of whitespaces
// taken from 
//https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

// trim from start (in place)
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}


//---------------------------------------------------------------------------


template class FileIO<IndexType, double>;
template class FileIO<IndexType, float>;
//template class FileIO<IndexType, IndexType>;

} /* namespace ITI */
