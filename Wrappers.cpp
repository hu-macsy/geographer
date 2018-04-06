/*
 * Wrappers.cpp
 *
 *  Created on: 02.02.2018
 *      Author: tzovas
 */


#include "Wrappers.h"


IndexType HARD_TIME_LIMIT= 600; 	// hard limit in seconds to stop execution if exceeded

namespace ITI {

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::partition(
	const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, 
	const scai::lama::DenseVector<ValueType> &nodeWeights, 
	bool nodeWeightsFlag,
	Tool tool,
	struct Settings &settings,
	struct Metrics &metrics	){

	switch( tool){
		case Tool::parMetisGraph:
			return metisPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, 0, settings, metrics);
		
		case Tool::parMetisGeom:
			return metisPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, 1, settings, metrics);
			
		case Tool::parMetisSFC:
			return metisPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, 2, settings, metrics);
			
		case Tool::zoltanRIB:
			return zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rib", settings, metrics);
		
		case Tool::zoltanRCB:
			return zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rcb", settings, metrics);
		
		case Tool::zoltanMJ:
			return zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "multijagged", settings, metrics);
			
		case Tool::zoltanSFC:
			return zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "hsfc", settings, metrics);
			
		default:
			throw std::runtime_error("Wrong tool given to partition.\nAborting...");
			return scai::lama::DenseVector<IndexType>(graph.getLocalNumRows(), -1 );
	}
}
//-----------------------------------------------------------------------------------------	

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::repartition (
	const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, 
	const scai::lama::DenseVector<ValueType> &nodeWeights, 
	bool nodeWeightsFlag,
	Tool tool,
	struct Settings &settings,
	struct Metrics &metrics){
	
	switch( tool){
		// for repartition, metis uses the same function
		case Tool::parMetisGraph:		
		case Tool::parMetisGeom:			
		case Tool::parMetisSFC:
			return metisRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, settings, metrics);
			
		case Tool::zoltanRIB:
			return zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rib", settings, metrics);
		
		case Tool::zoltanRCB:
			return zoltanRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rcb", settings, metrics);
		
		case Tool::zoltanMJ:
			return zoltanRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "multijagged", settings, metrics);
			
		case Tool::zoltanSFC:
			return zoltanRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "hsfc", settings, metrics);
			
		default:
			throw std::runtime_error("Wrong tool given to repartition.\nAborting...");
			return scai::lama::DenseVector<IndexType>(graph.getLocalNumRows(), -1 );
	}
}
//-----------------------------------------------------------------------------------------	


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::metisPartition (
	const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
	const scai::lama::DenseVector<ValueType> &nodeWeights, 
	bool nodeWeightsFlag,
	int parMetisGeom,
	struct Settings &settings,
	struct Metrics &metrics){
		
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const IndexType N = graph.getNumRows();
	const IndexType localN= dist->getLocalSize();
	
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
    
    double sumKwayTime = 0.0;
    int repeatTimes = settings.repeatTimes;
    
    idx_t *partKway;
	
    //
    // parmetis partition
    //
	if( parMetisGeom==0 and comm->getRank()==0 ) std::cout<< "About to call ParMETIS_V3_PartKway" << std::endl;
	if( parMetisGeom==1 and comm->getRank()==0 ) std::cout<< "About to call ParMETIS_V3_PartGeom" << std::endl;
	if( parMetisGeom==2 and comm->getRank()==0 ) std::cout<< "About to call ParMETIS_V3_PartSfc" << std::endl;
	
    int r;
    for( r=0; r<repeatTimes; r++){
    
		//get the vtx array
		
		IndexType size = comm->getSize();
		
		//TODO: generalize for any distribution or throw appropriate message
		/*
		//this, obviously, only applies for a block distribution
		IndexType lb, ub;
		scai::dmemo::BlockDistribution blockDist(N, comm);
		blockDist.getLocalRange(lb, ub, N, comm->getRank(), comm->getSize() );
		PRINT(*comm<< ": "<< lb << " _ "<< ub);
		*/
		
		// get local range of indices
				
		IndexType lb2=N+1, ub2=-1;
		{
			scai::hmemo::HArray<IndexType> myGlobalIndexes;
			dist->getOwnedIndexes( myGlobalIndexes );
			scai::hmemo::ReadAccess<IndexType> rIndices( myGlobalIndexes );
			SCAI_ASSERT_EQ_ERROR( localN, myGlobalIndexes.size(), "Local size mismatch" );
			
			for( int i=0; i<localN; i++){
				if( rIndices[i]<lb2 ) lb2=rIndices[i];
				if( rIndices[i]>ub2 ) ub2=rIndices[i];
			}
			++ub2;	// we need max+1
		}
		//PRINT(*comm<< ": "<< lb2 << " - "<< ub2);

		scai::hmemo::HArray<IndexType> sendVtx(size+1, static_cast<ValueType>( 0 ));
		scai::hmemo::HArray<IndexType> recvVtx(size+1);
		
		//TODO: use a sumArray instead of shiftArray
		for(IndexType round=0; round<comm->getSize(); round++){
			SCAI_REGION("ParcoRepart.getBlockGraph.shiftArray");
			{   // write your part 
				scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendVtx );
				sendPartWrite[0]=0;
				sendPartWrite[comm->getRank()+1]=ub2;
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


		// wgtflag is for the weight and can take 4 values. Here =0.
		idx_t wgtflag= 0;
		
		// vwgt , adjwgt store the weigths of vertices and edges.
		idx_t* vwgt= NULL;
		
		// if node weights are given
		if( nodeWeightsFlag ){
			scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
			SCAI_ASSERT_EQ_ERROR( localN, localWeights.size(), "Local weights size mismatch. Are node weights distributed correctly?");
			vwgt = new idx_t[localN];
			
			for(unsigned int i=0; i<localN; i++){
				vwgt[i] = idx_t (localWeights[i]);
			}
			
			wgtflag = 2;	//weights only in vertices
		}
		
		// edges weights not supported	
		idx_t* adjwgt= NULL;
		
		// numflag: 0 for C-style (start from 0), 1 for Fortrant-style (start from 1)
		idx_t numflag= 0;
		
		// ndims: the number of dimensions
		idx_t ndims = settings.dimensions;

		// the xyz array for coordinates of size dim*localN contains the local coords
		// convert the vector<DenseVector> to idx_t*
		real_t *xyzLocal;
		
		if( parMetisGeom==1 or parMetisGeom==2 or settings.writeDebugCoordinates ){
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
		partKway = new idx_t[ localN ];
		
		// comm: the MPI comunicator
		MPI_Comm metisComm;
		MPI_Comm_dup(MPI_COMM_WORLD, &metisComm);
		int metisRet;
		
		//PRINT(*comm<< ": xadj.size()= "<< sizeof(xadj) << "  adjncy.size=" <<sizeof(adjncy) ); 
		//PRINT(*comm << ": "<< sizeof(xyzLocal)/sizeof(real_t) << " ## "<< sizeof(partKway)/sizeof(idx_t) << " , localN= "<< localN);
		
		//if(comm->getRank()==0){
		//	PRINT("dims=" << ndims << ", nparts= " << nparts<<", ubvec= "<< ubvec << ", options="<< *options << ", wgtflag= "<< wgtflag );
		//}
			
		//
		// get the partitions with parMetis
		//
			
        std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
		
		if( parMetisGeom==0){
			metisRet = ParMETIS_V3_PartKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );
		}else if( parMetisGeom==1 ){
            metisRet = ParMETIS_V3_PartGeomKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ndims, xyzLocal, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );  
        }else if( parMetisGeom==2 ){
			metisRet = ParMETIS_V3_PartGeom( vtxDist, &ndims, xyzLocal, partKway, &metisComm ); 
		}else { // parMetisGeom==3 
			//repartition
			
			//TODO: check if vsize is correct 
			idx_t* vsize = new idx_t[localN];
			for(unsigned int i=0; i<localN; i++){
				vsize[i] = 1;
			}
			
			/*
			//TODO-CHECK: does repartition requires edge weights?
			IndexType localM = graph.getLocalNumValues();
			adjwgt =  new idx_t[localM];
			for(unsigned int i=0; i<localM; i++){
				adjwgt[i] = 1;
			}
			*/
			real_t itr = 1000;	//TODO: check other values too
			
			metisRet = ParMETIS_V3_AdaptiveRepart( vtxDist, xadj, adjncy, vwgt, vsize, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, &ubvec, &itr, options, &edgecut, partKway, &metisComm );
			
			delete[] vsize;
		}
		PRINT0("\n\t\tedge cut returned by parMetis: " << edgecut <<"\n");
		
		std::chrono::duration<double> partitionKwayTime = std::chrono::system_clock::now() - beforePartTime;
        double partKwayTime= comm->max(partitionKwayTime.count() );
        sumKwayTime += partKwayTime;
        
        if( comm->getRank()==0 ){
            std::cout<< "Running time for run number " << r << " is " << partKwayTime << std::endl;
        }
        
		//
		// free arrays
		//
		delete[] xadj;
		delete[] adjncy;
		if( parMetisGeom==1 or parMetisGeom==2 or settings.writeDebugCoordinates ){
			delete[] xyzLocal;
		}
		if( nodeWeightsFlag ){
			delete[] vwgt;
		}
        if( sumKwayTime>HARD_TIME_LIMIT){
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
    
	metrics.timeFinalPartition = sumKwayTime/(ValueType)repeatTimes;
	
     
    //
    // convert partition to a DenseVector
    //
    scai::lama::DenseVector<IndexType> partitionKway(dist);
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
//-----------------------------------------------------------------------------------------
	
//
//TODO: parMetis assumes that vertices are stored in a consecutive manner. This is not true for a
//		general distribution. Must reindex vertices for parMetis
//
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::metisRepartition (
	const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
	const scai::lama::DenseVector<ValueType> &nodeWeights, 
	bool nodeWeightsFlag,
	struct Settings &settings,
	struct Metrics &metrics){
	
	// copy graph and reindex
	scai::lama::CSRSparseMatrix<ValueType> copyGraph = graph;
	GraphUtils::reindex<IndexType, ValueType>(copyGraph);
	
	/*
	{// check that inidces are consecutive, TODO: maybe not needed, remove?
		
		const scai::dmemo::DistributionPtr dist( copyGraph.getRowDistributionPtr() );
		//scai::hmemo::HArray<IndexType> myGlobalIndexes;
		//dist.getOwnedIndexes( myGlobalIndexes );
		const IndexType globalN = graph.getNumRows();
		const IndexType localN= dist->getLocalSize();
		const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	
		std::vector<IndexType> myGlobalIndexes(localN);
		for(IndexType i=0; i<localN; i++){
			myGlobalIndexes[i] = dist->local2global( i );
		}
		
		std::sort( myGlobalIndexes.begin(), myGlobalIndexes.end() );
		SCAI_ASSERT_GE_ERROR( myGlobalIndexes[0], 0, "Invalid index");
		SCAI_ASSERT_LE_ERROR( myGlobalIndexes.back(), globalN, "Invalid index");
		
		for(IndexType i=1; i<localN; i++){
			SCAI_ASSERT_EQ_ERROR( myGlobalIndexes[i], myGlobalIndexes[i-1]+1, *comm << ": Invalid index for local index " << i);
		}
		
		//PRINT(*comm << ": min global ind= " <<  myGlobalIndexes.front() << " , max global ind= " << myGlobalIndexes.back() );
	}
	*/
	
	//trying Moritz version that also redistributes coordinates
	const scai::dmemo::DistributionPtr dist( copyGraph.getRowDistributionPtr() );
	SCAI_ASSERT_NE_ERROR(dist->getBlockDistributionSize(), nIndex, "Reindexed distribution should be a block distribution.");
	SCAI_ASSERT_EQ_ERROR(graph.getNumRows(), copyGraph.getNumRows(), "Graph sizes must be equal.");
	
	std::vector<scai::lama::DenseVector<ValueType>> copyCoords = coords;
	scai::lama::DenseVector<ValueType> copyNodeWeights = nodeWeights;
	
	// TODO: use constuctor to redistribute or a Redistributor
	for (IndexType d = 0; d < settings.dimensions; d++) {
	    copyCoords[d].redistribute(dist);
	}
	
	if (nodeWeights.size() > 0) {
	    copyNodeWeights.redistribute(dist);
	}
	
	int parMetisVersion = 3; // flag for repartition
	scai::lama::DenseVector<IndexType> partition = Wrappers<IndexType, ValueType>::metisPartition( copyGraph, copyCoords, copyNodeWeights, nodeWeightsFlag, parMetisVersion, settings, metrics);
	
	//because of the reindexing, we must redistribute the partition
	partition.redistribute( graph.getRowDistributionPtr() );
	
	return partition;
	//return Wrappers<IndexType, ValueType>::metisPartition( copyGraph, coords, nodeWeights, nodeWeightsFlag, parMetisVersion, settings, metrics);
}
	
	
//---------------------------------------------------------
//						zoltan
//---------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::zoltanPartition (
	const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
	const scai::lama::DenseVector<ValueType> &nodeWeights, 
	bool nodeWeightsFlag,
	std::string algo,
	struct Settings &settings,
	struct Metrics &metrics){
		
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	PRINT0("\t\tStarting the zoltan wrapper for partition with "<< algo);
	
	bool repart = false;
	
	return Wrappers<IndexType, ValueType>::zoltanCore(graph, coords, nodeWeights, nodeWeightsFlag, algo, repart, settings, metrics);
}
//---------------------------------------------------------------------------------------	
		
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::zoltanRepartition (
	const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
	const scai::lama::DenseVector<ValueType> &nodeWeights, 
	bool nodeWeightsFlag,
	std::string algo,
	struct Settings &settings,
	struct Metrics &metrics){
		
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	PRINT0("\t\tStarting the zoltan wrapper for repartition with " << algo);
	
	bool repart = true;
	
	return Wrappers<IndexType, ValueType>::zoltanCore(graph, coords, nodeWeights, nodeWeightsFlag, algo, repart, settings, metrics);	
	}
//---------------------------------------------------------------------------------------	

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::zoltanCore (
	const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
	const scai::lama::DenseVector<ValueType> &nodeWeights, 
	bool nodeWeightsFlag,
	std::string algo,
	bool repart,
	struct Settings &settings,
	struct Metrics &metrics){
		
	typedef Zoltan2::BasicUserTypes<ValueType, IndexType, IndexType> myTypes;
	typedef Zoltan2::BasicVectorAdapter<myTypes> inputAdapter_t;
	//typedef Zoltan2::EvaluatePartition<inputAdapter_t> quality_t;
	//typedef inputAdapter_t::part_t part_t;
	
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const IndexType thisPE = comm->getRank();
	//const IndexType numPEs = comm->getSize();
	//const IndexType N = graph.getNumRows();
	const IndexType numBlocks = settings.numBlocks;
	 
	IndexType dimensions = settings.dimensions;
	IndexType localN= dist->getLocalSize();
	
	//TODO: point directly to the localCoords data and save time and space for zoltanCoords
	// the local part of coordinates for zoltan
	ValueType *zoltanCoords = new ValueType [dimensions * localN];

	std::vector<scai::utilskernel::LArray<ValueType>> localPartOfCoords( dimensions );
	for(unsigned int d=0; d<dimensions; d++){
		localPartOfCoords[d] = coords[d].getLocalValues();
	}
	IndexType coordInd = 0;
	for(int d=0; d<dimensions; d++){
		//SCAI_ASSERT_LE_ERROR( dimensions*(i+1), dimensions*localN, "Too large index, localN= " << localN );
		for(IndexType i=0; i<localN; i++){
			SCAI_ASSERT_LT_ERROR( coordInd, localN*dimensions, "Too large coordinate index");
			zoltanCoords[coordInd++] = localPartOfCoords[d][i];
		}
	}
	
	std::vector<const ValueType *>coordVec( dimensions );
	std::vector<int> coordStrides(dimensions);
	
	coordVec[0] = zoltanCoords; 	// coordVec[d] = localCoords[d].data(); or something
	coordStrides[0] = 1;

	for( int d=1; d<dimensions; d++){
		coordVec[d] = coordVec[d-1] + localN;
		coordStrides[d] = 1;
	}	
		
	///////////////////////////////////////////////////////////////////////
	// Create parameters
	
	ValueType tolerance = 1+settings.epsilon;
	
	if (thisPE == 0)
		std::cout << "Imbalance tolerance is " << tolerance << std::endl;
	
	Teuchos::ParameterList params("test params");
	//params.set("debug_level", "basic_status");
	params.set("debug_level", "no_status");
	params.set("debug_procs", "0");
	params.set("error_check_level", "debug_mode_assertions");
	
	params.set("algorithm", algo);
	params.set("imbalance_tolerance", tolerance );
	params.set("num_global_parts", (int)numBlocks );
	
	params.set("compute_metrics", false);
	
	// chose if partition or repartition
	if( repart ){
		params.set("partitioning_approach", "repartition");
	}else{
		params.set("partitioning_approach", "partition");
	}

	//TODO:	params.set("partitioning_objective", "minimize_cut_edge_count");
	//		or something else, check at 
	//		https://trilinos.org/docs/r12.12/packages/zoltan2/doc/html/z2_parameters.html
	
	// Create global ids for the coordinates.
	IndexType *globalIds = new IndexType [localN];
	IndexType offset = thisPE * localN;

	//TODO: can also be taken from the distribution?
	for (size_t i=0; i < localN; i++)
		globalIds[i] = offset++;

	//set node weights
	std::vector<ValueType> localUnitWeight(localN);
	
	if( nodeWeightsFlag ){
		scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
		for(unsigned int i=0; i<localN; i++){
			localUnitWeight[i] = localWeights[i];
		}
	}else{
		localUnitWeight.assign( localN, 1.0);
	}
	std::vector<const ValueType *>weightVec(1);
	weightVec[0] = localUnitWeight.data();

	std::vector<int> weightStrides(1);
	weightStrides[0] = 1;
	
	//create the problem and solve it
	inputAdapter_t *ia= new inputAdapter_t(localN, globalIds, coordVec, 
                                         coordStrides, weightVec, weightStrides);

	Zoltan2::PartitioningProblem<inputAdapter_t> *problem =
           new Zoltan2::PartitioningProblem<inputAdapter_t>(ia, &params);	
		   
	if( comm->getRank()==0 )
		std::cout<< "About to call zoltan, algo " << algo << std::endl;
	
	int repeatTimes = settings.repeatTimes;
	double sumPartTime = 0.0;
	int r=0;
	
	for( r=0; r<repeatTimes; r++){
		std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
		problem->solve();
  	
		std::chrono::duration<double> partitionTmpTime = std::chrono::system_clock::now() - beforePartTime;
		double partitionTime= comm->max(partitionTmpTime.count() );
		sumPartTime += partitionTime;
		if( comm->getRank()==0 ){
            std::cout<< "Running time for run number " << r << " is " << partitionTime << std::endl;
        }
		if( sumPartTime>HARD_TIME_LIMIT){
			std::cout<< "Stopping runs because of excessive running total running time: " << sumPartTime << std::endl;
            break;
        }
	}
	
	if( r!=repeatTimes){		// in case it has to break before all the runs are completed
		repeatTimes = r+1;
	}
	if(comm->getRank()==0 ){
        std::cout<<"Number of runs: " << repeatTimes << std::endl;	
    }
    
	metrics.timeFinalPartition = sumPartTime/(ValueType)repeatTimes;
	
	//
	// convert partition to a DenseVector
    //
	scai::lama::DenseVector<IndexType> partitionZoltan(dist);

	//std::vector<IndexType> localBlockSize( numBlocks, 0 );
	
	const Zoltan2::PartitioningSolution<inputAdapter_t> &solution = problem->getSolution();
	const int *partAssignments = solution.getPartListView();
	for(unsigned int i=0; i<localN; i++){
		IndexType thisBlock = partAssignments[i];
		SCAI_ASSERT_LT_ERROR( thisBlock, numBlocks, "found wrong vertex id");
		SCAI_ASSERT_GE_ERROR( thisBlock, 0, "found negetive vertex id");
		partitionZoltan.getLocalValues()[i] = thisBlock;
		//localBlockSize[thisBlock]++;
	}
	for(int i=0; i<localN; i++){
		//PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
		SCAI_ASSERT_EQ_ERROR( partitionZoltan.getLocalValues()[i], partAssignments[i], "Wrong conversion to DenseVector");
	}
	
	delete[] globalIds;
	delete[] zoltanCoords;
	
	return partitionZoltan;
		
	}

//---------------------------------------------------------------------------------------	
		
	template class Wrappers<IndexType, ValueType>;
	
}//namespace