/*
 * Wrappers.cpp
 *
 *  Created on: 02.02.2018
 *      Author: tzovas
 */


#include "Wrappers.h"

/* wrapper<metis> call
 * wrapper<zoltan> call( ... )
 * 
 * */
namespace ITI {

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::metisWrapper (
	const CSRSparseMatrix<ValueType> &graph,
	const std::vector<DenseVector<ValueType>> &coords, 
	const DenseVector<ValueType> &nodeWeights,
	int parMetisGeom,
	struct Settings &settings,
	struct Metrics &metrics){
		
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const IndexType N = graph.getNumRows();
	
	PRINT0("\t\tStarting the metis wrapper");
	 
	if( comm->getRank()==0 ){
        std::cout << "\033[1;31m";
        std::cout << "IndexType size: " << sizeof(IndexType) << " , ValueType size: "<< sizeof(ValueType) << std::endl;
        if( int(sizeof(IndexType)) != int(sizeof(idx_t)) ){
            std::cout<< "WARNING: IndexType size= " << sizeof(IndexType) << " and idx_t size=" << sizeof(idx_t) << "  do not agree, this may cause problems (even if this print looks OK)." << std::endl;
        }
        if( sizeof(ValueType)!=sizeof(real_t) ){
            std::cout<< "WARNING: IndexType size= " << sizeof(IndexType) << " and idx_t size=" << sizeof(idx_t) << "  do not agree, this may cause problems " << std::endl;
        }
        std::cout<<"\033[0m";
    }
    //-----------------------------------------------------
    //
    // convert to parMetis data types
    //
    
    //get the vtx array
    
    IndexType size = comm->getSize();

	scai::hmemo::HArray<IndexType> sendVtx(size+1, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvVtx(size+1);
    
    IndexType lb, ub;
    scai::dmemo::BlockDistribution blockDist(N, comm);
    blockDist.getLocalRange(lb, ub, N, comm->getRank(), comm->getSize() );
    //PRINT(*comm<< ": "<< lb << " _ "<< ub);
    

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

    // vtxDist is an array of size numPEs and is replicated in every processor
    idx_t vtxDist[ size+1 ];
    vtxDist[0]= 0;
 
    for(int i=0; i<recvPartRead.size()-1; i++){
        vtxDist[i+1]= recvPartRead[i+1];
    }
    /*
    for(IndexType i=0; i<recvPartRead.size(); i++){
        PRINT(*comm<< " , " << i <<": " << vtxDist[i]);
    }
    */
    recvPartRead.release();

    //
    // setting xadj=ia and adjncy=ja values, these are the local values of every processor
    //
    
    const scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();
    
    scai::hmemo::ReadAccess<IndexType> ia( localMatrix.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localMatrix.getJA() );
    IndexType iaSize= ia.size();

    idx_t* xadj = new idx_t[ iaSize ];
    idx_t* adjncy = new idx_t[ ja.size() ];
    
    for(int i=0; i<iaSize ; i++){
        xadj[i]= ia[i];
        SCAI_ASSERT( xadj[i] >=0, "negative value for i= "<< i << " , val= "<< xadj[i]);
    }

    for(int i=0; i<ja.size(); i++){
        adjncy[i]= ja[i];
        SCAI_ASSERT( adjncy[i] >=0, "negative value for i= "<< i << " , val= "<< adjncy[i]);
        SCAI_ASSERT( adjncy[i] <N , "too large value for i= "<< i << " , val= "<< adjncy[i]);
    }
    ia.release();
    ja.release();


    //vwgt , adjwgt store the weigths of edges and vertices. Here we have
    // no weight so are both NULL.
    idx_t* vwgt= NULL;
    idx_t* adjwgt= NULL;
    
    // wgtflag is for the weight and can take 4 values. Here =0.
    idx_t wgtflag= 0;
    
    // numflag: 0 for C-style (start from 0), 1 for Fortrant-style (start from 1)
    idx_t numflag= 0;
    
    // ndims: the number of dimensions
    idx_t ndims = settings.dimensions;

    // the xyz array for coordinates of size dim*localN contains the local coords
    // convert the vector<DenseVector> to idx_t*
    IndexType localN= dist->getLocalSize();
    real_t *xyzLocal;
	
    if( parMetisGeom!=0 or settings.writeDebugCoordinates ){
        xyzLocal = new real_t[ ndims*localN ];
        
        std::vector<scai::utilskernel::LArray<ValueType>> localPartOfCoords( ndims );
        for(int d=0; d<ndims; d++){
            localPartOfCoords[d] = coords[d].getLocalValues();
        }
        for(unsigned int i=0; i<localN; i++){
            SCAI_ASSERT_LE_ERROR( ndims*(i+1), ndims*localN, "Too large index, localN= " << localN );
            for(int d=0; d<ndims; d++){
                xyzLocal[ndims*i+d] = real_t(localPartOfCoords[d][i]);
            }
        }
        
    }
    // ncon: the numbers of weigths each vertex has. Here 1;
    idx_t ncon = 1;
    
    // nparts: the number of parts to partition (=k)
    idx_t nparts= settings.numBlocks;
  
    // tpwgts: array of size ncons*nparts, that is used to specify the fraction of 
    // vertex weight that should be distributed to each sub-domain for each balance
    // constraint. Here we want equal sizes, so every value is 1/nparts.
    real_t tpwgts[ nparts ];
    real_t total = 0;
    for(int i=0; i<sizeof(tpwgts)/sizeof(real_t) ; i++){
		tpwgts[i] = real_t(1)/nparts;
		//PRINT(*comm << ": " << i <<": "<< tpwgts[i]);
		total += tpwgts[i];
    }

    // ubvec: array of size ncon to specify imbalance for every vertex weigth.
    // 1 is perfect balance and nparts perfect imbalance. Here 1 for now
    real_t ubvec= settings.epsilon + 1;
    
    // options: array of integers for passing arguments.
    // Here, options[0]=0 for the default values.
    idx_t options[1]= {0};
    
    //
    // OUTPUT parameters
    //
    // edgecut: the size of cut
    idx_t edgecut;
    
    // partition array of size localN, contains the block every vertex belongs
    idx_t *partKway = new idx_t[ localN ];
    
    // comm: the MPI comunicator
    MPI_Comm metisComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &metisComm);
     
    //PRINT(*comm<< ": xadj.size()= "<< sizeof(xadj) << "  adjncy.size=" <<sizeof(adjncy) ); 
    //PRINT(*comm << ": "<< sizeof(xyzLocal)/sizeof(real_t) << " ## "<< sizeof(partKway)/sizeof(idx_t) << " , localN= "<< localN);
    
    if(comm->getRank()==0){
	    PRINT("dims=" << ndims << ", nparts= " << nparts<<", ubvec= "<< ubvec << ", options="<< *options << ", ncon= "<< ncon );
    }
		 
    //
    // get the partitions with parMetis
    //
    
    double sumKwayTime = 0.0;
    int repeatTimes = settings.repeatTimes;
    
    int metisRet;
    
    //
    // parmetis partition
    //
    int r;
    for( r=0; r<repeatTimes; r++){
        
        std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
		
		if( parMetisGeom==0){
			metisRet = ParMETIS_V3_PartKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );
		}else if( parMetisGeom==1 ){
            metisRet = ParMETIS_V3_PartGeomKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ndims, xyzLocal, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );  
        }else{
			metisRet = ParMETIS_V3_PartGeom( vtxDist, &ndims, xyzLocal, partKway, &metisComm ); 
		}
		
		std::chrono::duration<double> partitionKwayTime = std::chrono::system_clock::now() - beforePartTime;
        double partKwayTime= comm->max(partitionKwayTime.count() );
        sumKwayTime += partKwayTime;
        
        if( comm->getRank()==0 ){
            std::cout<< "Running time for run number " << r << " is " << partKwayTime << std::endl;
        }
        
        if( sumKwayTime>500){
			std::cout<< "Stopping runs because of excessive running total running time: " << sumKwayTime << std::endl;
            break;
        }
    }

	if( r!=repeatTimes){		// in case it has to break before all the runs are completed
		repeatTimes = r+1;
	}
	if(comm->getRank()==0 ){
        std::cout<<"Number of runs: " << repeatTimes << std::endl;	
    }
    
    double avgKwayTime = sumKwayTime/repeatTimes;
	metrics.timeFinalPartition = avgKwayTime;
	
    //
    // free arrays
    //
    delete[] xadj;
    delete[] adjncy;
    if( parMetisGeom or settings.writeDebugCoordinates ){
        delete[] xyzLocal;
    }
    
    //
    // convert partition to a DenseVector
    //
    DenseVector<IndexType> partitionKway(dist);
    for(unsigned int i=0; i<localN; i++){
        partitionKway.getLocalValues()[i] = partKway[i];
    }
    
    // check correct transformation to DenseVector
    for(int i=0; i<localN; i++){
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        assert( partKway[i]== partitionKway.getLocalValues()[i]);
    }
    
    delete[] partKway;
    
	return partitionKway;
		
	}

	
	 template class Wrappers<IndexType, ValueType>;
	
}//namespace