/*
 * IO.cpp
 *
 *  Created on: 15.02.2017
 *      Author: moritzl
 */

#include "FileIO.h"

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/Scalar.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/common/Math.hpp>
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
#include <iterator>

namespace ITI {

//-------------------------------------------------------------------------------------------------
/*Given the adjacency matrix it writes it in the file "filename" using the METIS format. In the
 * METIS format the first line has two numbers, first is the number on vertices and the second
 * is the number of edges. Then, row i has numbers e1, e2, e3, ... notating the edges:
 * (i, e1), (i, e2), (i, e3), ....
 *
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeGraphToFile (const CSRSparseMatrix<ValueType> &adjM, const std::string filename){
    SCAI_REGION( "IO.writeInFileMetisFormat" )
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    //PRINT(*comm << " In writeInFileMetisFormat");

    std::ofstream f;
    std::string oldFile = filename + "OLD";
    f.open(oldFile);
    IndexType cols= adjM.getNumColumns() , rows= adjM.getNumRows();
    IndexType i, j;

    SCAI_REGION_START( "IO.writeInFileMetisFormat.newVersion" )
    // new version
    std::ofstream fNew;
    std::string newFile = filename;// + "NEW";
    fNew.open(newFile);

    //assert( true == adjM.checkSymmetry() ); // this can be expensive
    assert(((int) adjM.getNumValues())%2==0); // even number of edges
    assert(cols==rows);

    // first line is number of nodes and edges
    fNew << cols <<" "<< adjM.getNumValues()/2 << std::endl;
    //std::cout << cols <<" "<< adjM.getNumValues()/2 << std::endl;

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    //const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

    for(IndexType i=0; i< ia.size(); i++){        // for all local nodes
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){             // for all the edges of a node
            SCAI_REGION("IO.writeInFileMetisFormat.newVersion.writeInFile");
            fNew << ja[j]+1 << " ";
    	}
    	fNew << std::endl;
    }
    fNew.close();
    SCAI_REGION_END( "IO.writeInFileMetisFormat.newVersion" )
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeGraphToDistributedFiles (const CSRSparseMatrix<ValueType> &adjM, const std::string filename){
    SCAI_REGION("IO.writeInFileMetisFormat_dist")

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    std::string fileTo = filename + std::to_string(comm->getRank());
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

    for(IndexType i=0; i< ia.size()-1; i++){                  // for all local nodes
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){             // for all the edges of a node
            f << ja[j]+1 << " ";
    	}
    	f << std::endl;
    }
    f.close();
}
//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and their dimension, writes them in file "filename".
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeCoordsToFile (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const std::string filename){
    SCAI_REGION( "IO.writeInFileCoords" )

    std::ofstream f(filename);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    IndexType i, j;
    IndexType dimension= coords.size();

    assert(coords.size() == dimension );
    assert(coords[0].size() == numPoints);
    for(i=0; i<numPoints; i++){
        for(j=0; j<dimension; j++)
            f<< coords[j].getValue(i).Scalar::getValue<ValueType>() << " ";
        f<< std::endl;
    }

}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and transforms
 * it to the adjacency matrix as a CSRSparseMatrix.
 */

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraphFromFile(const std::string filename) {
	SCAI_REGION("IO.readFromFile2AdjMatrix");

	std::ifstream file(filename);

	if (file.fail()) {
		throw std::runtime_error("Reading graph from " + filename + " failed.");
	}

	IndexType globalN, globalM;

	file >> globalN >> globalM;

	const ValueType avgDegree = ValueType(2*globalM) / globalN;

	//get distribution and local range
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( globalN ));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll to begin of local range. Neighbors of node i are in line i+1
    std::string line;
    std::getline(file, line);
    for (IndexType i = 0; i < beginLocalRange; i++) {
    	std::getline(file, line);
    }

    std::vector<IndexType> ia(localN+1, 0);
    std::vector<IndexType> ja;

    //we don't know exactly how many edges we are going to have, but in a regular mesh the average degree times the local nodes is a good estimate.
    ja.reserve(localN*avgDegree*1.1);

    //now read in local edges
    for (IndexType i = 0; i < localN; i++) {
    	bool read = std::getline(file, line);
    	assert(read);//if we have read past the end of the file, the node count was incorrect
        std::stringstream ss( line );
        std::string item;
        std::vector<IndexType> neighbors;

        while (std::getline(ss, item, ' ')) {
        	IndexType neighbor = std::stoi(item)-1;//-1 because of METIS format
        	if (neighbor >= globalN || neighbor < 0) {
        		throw std::runtime_error("Found illegal neighbor " + std::to_string(neighbor) + " in line " + std::to_string(i+beginLocalRange));
        	}
        	//std::cout << "Converted " << item << " to " << neighbor << std::endl;
        	neighbors.push_back(neighbor);
        }

        //set Ia array
        ia[i+1] = ia[i] + neighbors.size();
        //copy neighbors to Ja array
        std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(ja));
    }

    //TODO: maybe check that file is not longer than expected

    file.close();

    scai::utilskernel::LArray<ValueType> values(ja.size(), 1);//unweighted edges
    assert(ja.size() == ia[localN]);
    assert(comm->sum(localN) == globalN);

    if (comm->sum(ja.size()) != 2*globalM) {
    	throw std::runtime_error("Expected " + std::to_string(2*globalM) + " edges, got " + std::to_string(comm->sum(ja.size())));
    }

    //assign matrix
    scai::lama::CSRStorage<ValueType> myStorage(localN, globalN, ja.size(), scai::utilskernel::LArray<IndexType>(ia.size(), ia.data()),
    		scai::utilskernel::LArray<IndexType>(ja.size(), ja.data()), values);

    return scai::lama::CSRSparseMatrix<ValueType>(myStorage, dist, noDist);
}


//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads these coordinates and returns a vector of DenseVectors, one for each dimension
 */
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoordsFromFile( std::string filename, IndexType numberOfPoints, IndexType dimension){
    SCAI_REGION( "IO.fromFile2Coords" )
    IndexType globalN= numberOfPoints;
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll forward to begin of local range
    std::string line;
    for (IndexType i = 0; i < beginLocalRange; i++) {
    	std::getline(file, line);
    }

    //create result vector
    std::vector<scai::utilskernel::LArray<ValueType> > coords(dimension);
    for (IndexType dim = 0; dim < dimension; dim++) {
    	coords[dim] = scai::utilskernel::LArray<ValueType>(localN, 0);
    }

    //read local range
    for (IndexType i = 0; i < localN; i++) {
		bool read = std::getline(file, line);
		assert(read);//if we have read past the end of the file, the node count was incorrect
		std::stringstream ss( line );
		std::string item;

		IndexType dim = 0;
		while (std::getline(ss, item, ' ') && dim < dimension) {
			ValueType coord = std::stod(item);
			coords[dim][i] = coord;
			dim++;
		}
		if (dim < dimension) {
			throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimension) + " expected in line '" + line + "'");
		}
    }

    std::vector<DenseVector<ValueType> > result(dimension);

    for (IndexType i = 0; i < dimension; i++) {
    	result[i] = DenseVector<ValueType>(coords[i], dist);
    }

    return result;
}

template void FileIO<int, double>::writeGraphToFile (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeGraphToDistributedFiles (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeCoordsToFile (const std::vector<DenseVector<double>> &coords, int numPoints, const std::string filename);
template CSRSparseMatrix<double> FileIO<int, double>::readGraphFromFile(const std::string filename);
template std::vector<DenseVector<double>>  FileIO<int, double>::readCoordsFromFile( std::string filename, int numberOfCoords, int dimension);


} /* namespace ITI */
