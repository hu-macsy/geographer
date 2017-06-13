/*
 * Author Charilaos "Harry" Tzovas
 *
 *
*/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>

#include <scai/dmemo/BlockDistribution.hpp>

#include "FileIO.h"
#include "ParcoRepart.h"

#include <parmetis.h>


typedef double ValueType;
typedef int IndexType;


int main(int argc, char** argv) {
    
    std::string file = std::string(argv[1]);
    std::ifstream f(file);
    IndexType dimensions= 2;
    if(f.fail()) 
        throw std::runtime_error("File "+ file+ " failed.");
    IndexType N, edges;
    f >> N >> edges; 
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    CSRSparseMatrix<ValueType> graph = ITI::FileIO<IndexType, ValueType>::readGraph( file );
    
    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    
    SCAI_ASSERT_EQUAL( graph.getNumColumns(),  graph.getNumRows() , "matrix not square");
    SCAI_ASSERT_EQUAL( edges, (graph.getNumValues())/2 , "numEdges and numValues do not agree"); 
    std::vector<DenseVector<ValueType>> coords = ITI::FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size() , coords[1].getLocalValues().size(), "coordinates not of same size" );
    
    //get the vtx array
    
    IndexType size = comm->getSize()+1;
    scai::hmemo::HArray<IndexType> sendVtx(size, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvVtx(size);
    
    IndexType lb, ub;
    scai::dmemo::BlockDistribution blockDist(N, comm);
    blockDist.getLocalRange(lb, ub, N, comm->getRank(), comm->getSize() );
    PRINT(*comm<< ": "<< lb << " _ "<< ub);
    

    for(IndexType round=0; round<comm->getSize(); round++){
        SCAI_REGION("ParcoRepart.getBlockGraph.shiftArray");
        {   // write your part 
            scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendVtx );
            sendPartWrite[0]=0;
            sendPartWrite[comm->getRank()+1]=ub;
        }
        comm->shiftArray(recvVtx , sendVtx, 1);
        sendVtx.swap(recvVtx);
    } 
    
    scai::hmemo::ReadAccess<IndexType> recvPartRead( recvVtx );
    // vtxDist is and array of size numPEs and is replicated in every processor
    idx_t vtxDist[ comm->getSize()+1 ]; 
    vtxDist[0]= 0;
    for(int i=0; i<recvPartRead.size(); i++){
        vtxDist[i+1]= recvPartRead[i+1];
    }
    
/*
    for(IndexType i=0; i<recvPartRead.size(); i++){
        PRINT(*comm<< " , " << i <<": " << vtxDist[i]);
    }
*/
  
    recvPartRead.release();
    
    // setting xadj=ia and adjncy=ja values, these are the local values of every processor
    scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();
    scai::utilskernel::LArray<IndexType>& ia = localMatrix.getIA();
    scai::utilskernel::LArray<IndexType>& ja = localMatrix.getJA();
    
    idx_t xadj[ia.size()], adjncy[ja.size()];
    for(int i=0; i<ia.size(); i++){
        xadj[i]= ia[i];
        SCAI_ASSERT( xadj[i] >=0, "negative value for i= "<< i << " , val= "<< xadj[i]);
    }
    for(int i=0; i<ja.size(); i++){
        adjncy[i]= ja[i];
        SCAI_ASSERT( adjncy[i] >=0, "negative value for i= "<< i << " , val= "<< adjncy[i]);
    }

    //vwgt , adjwgt store the weigths of edges and vertices. Here we have
    // no weight so are both NULL.
    idx_t* vwgt= NULL;
    idx_t* adjwgt= NULL;
    
    // wgtflag is for the weight and can take 4 values. Here =0.
    idx_t wgtflag= 0;
    
    // numflag: 0 for C-style (start from 0), 1 for Fortrant-style (start from 1)
    idx_t numflag= 0;
    
    // ndims: the number of dimensions
    idx_t ndims= 2;

    // the xyz array for coordinates of size dim*localN contains the local coords
    // convert the vector<DenseVector> to idx_t*
    IndexType localN= dist->getLocalSize();
    real_t xyzLocal[2*localN];

    // TODO: only for 2D now
    scai::utilskernel::LArray<ValueType>& localPartOfCoords0 = coords[0].getLocalValues();
    scai::utilskernel::LArray<ValueType>& localPartOfCoords1 = coords[1].getLocalValues();
    
    for(unsigned int i=0; i<localN; i++){
        xyzLocal[2*i]= real_t(localPartOfCoords0[i]);
        xyzLocal[2*i+1]= real_t(localPartOfCoords1[i]);
        //PRINT(*comm <<": "<< xyzLocal[i] << ", "<< xyzLocal[i+1]);
    }
  
    // ncon: the numbers of weigths each vertex has. Here 1;
    idx_t ncon = 1;
    
    // nparts: the number of parts to partition (=k). Here, nparts=p
    IndexType k = comm->getSize();
    idx_t nparts= k;
  
    // tpwgts: array of size ncons*nparts, that is used to specify the fraction of 
    // vertex weight that should be distributed to each sub-domain for each balance
    // constraint. Here we want equal sizes, so every value is 1/nparts.
    real_t tpwgts[ k];
    real_t total = 0;
    for(int i=0; i<sizeof(tpwgts)/sizeof(real_t) ; i++){
	tpwgts[i] = real_t(1)/k;
	//PRINT(i <<": "<< tpwgts[i]);
	total += tpwgts[i];
    }

    // ubvec: array of size ncon to specify imbalance for every vertex weigth.
    // 1 is perfect balance and nparts perfect imbalance. Here 1 for now
    real_t ubvec= 1.05;
    
    // options: array of integers for passing arguments.
    // Here, options[0]=0 for the default values.
    idx_t options[0];
    options[0]= 0;
    
    // OUTPUT parameters
    // edgecut: the size of cut
    idx_t edgecut;
    
    // partition array of size localN, contains the block every vertex belongs
    //idx_t* part = (idx_t*) malloc(sizeof(idx_t)*localN);
    idx_t partKway[localN];
    
    // comm: the MPI comunicator
    MPI_Comm metisComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &metisComm);
     
    //PRINT(*comm<< ": xadj.size()= "<< sizeof(xadj) << "  adjncy.size=" <<sizeof(adjncy) ); 
    
    if(comm->getRank()==0){
	    PRINT("dims=" << ndims << ", nparts= " << nparts<<", ubvec= "<< ubvec << ", options="<< *options << ", ncon= "<< ncon );
    }
    
    //PRINT(*comm << ": "<< sizeof(xyzLocal)/sizeof(real_t) << " ## "<< sizeof(part)/sizeof(idx_t) );

	//
    // get the partiotions with parMetis
	//

    std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
    int metisRet;

    // parmetis with Kway
    metisRet = ParMETIS_V3_PartGeomKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ndims, xyzLocal, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );
    
    DenseVector<IndexType> partitionKway(dist);
    for(unsigned int i=0; i<localN; i++){
        partitionKway.getLocalValues()[i] = partKway[i];
    }
    ValueType cutKway = ITI::ParcoRepart<IndexType, ValueType>::computeCut(graph, partitionKway, true); 
    ValueType imbalanceKway = ITI::ParcoRepart<IndexType, ValueType>::computeImbalance( partitionKway, comm->getSize() );
    assert(sizeof(xyzLocal)/sizeof(real_t) == 2*sizeof(partKway)/sizeof(idx_t) );
  
    // check correct trasformation to DenseVector
    for(int i=0; i<localN; i++){
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        assert( partKway[i]== partitionKway.getLocalValues()[i]);
    }
    std::chrono::duration<double> partitionKwayTime =  std::chrono::system_clock::now() - beforePartTime;

	//
	// only using coordinates
	//
    // this is very fast but the cuts are very bad, so after k=60 (or 90) it gives way worst cuts than the Hilbert curve 

   /*
    std::chrono::time_point<std::chrono::system_clock> beforePartTime2 =  std::chrono::system_clock::now();

    idx_t partGeom[localN];

    metisRet = ParMETIS_V3_PartGeom(vtxDist, &ndims, xyzLocal, partGeom, &metisComm);

    DenseVector<IndexType> partitionGeom(dist);
    for(unsigned int i=0; i<localN; i++){
        //localWrite[i]= part[i];
        partitionGeom.getLocalValues()[i] = partGeom[i];
    }   
    ValueType cutGeom = ITI::ParcoRepart<IndexType, ValueType>::computeCut(graph, partitionGeom, true); 
    ValueType imbalanceGeom = ITI::ParcoRepart<IndexType, ValueType>::computeImbalance( partitionGeom, comm->getSize() );
    assert(sizeof(xyzLocal)/sizeof(real_t) == 2*sizeof(partGeom)/sizeof(idx_t) );
 
    std::chrono::duration<double> partitionGeomTime =  std::chrono::system_clock::now() - beforePartTime2;
     
    // check correct trasformation to DenseVector
    for(int i=0; i<localN; i++){
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        assert( partGeom[i]== partitionGeom.getLocalValues()[i]);
    }
    */
    

    double partKwayTime= comm->max(partitionKwayTime.count() );
    //double partGeomTime= comm->max(partitionGeomTime.count() );
    
    if(comm->getRank()==0){
        std::cout<< std::endl << "ParMetisKway cut= "<< cutKway <<" and imbalance= " << imbalanceKway<<", time for partition: "<< partKwayTime << std::endl;
        //td::cout<< std::endl << "ParMetisGeom cut= "<< cutGeom <<" and imbalance= " << imbalanceGeom<<", time for partition: "<< partGeomTime << std::endl;

    }
    return 0;
}

