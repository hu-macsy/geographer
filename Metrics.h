#pragma once

#include <numeric>
#include <math.h>
#include <scai/lama.hpp>
#include <chrono>
#include <algorithm>

//#include <scai/lama.hpp>
#include "GraphUtils.h"

struct Metrics{
    
    // timing results
    //
    std::vector<ValueType>  timeMigrationAlgo;
	std::vector<ValueType>  timeConstructRedistributor;
    std::vector<ValueType>  timeFirstDistribution;
    std::vector<ValueType>  timeKmeans;
    std::vector<ValueType>  timeSecondDistribution;
    std::vector<ValueType>  timePreliminary;
    
   	ValueType inputTime = -1;
	ValueType timeFinalPartition = -1;
	ValueType reportTime = -1 ;
	ValueType timeTotal = -1;
	ValueType timeSpMV = -1;
	ValueType timeComm = -1;
    
    //metrics, each for every time we repeat the algo
    //
    ValueType preliminaryCut = 0;
    ValueType preliminaryImbalance = 0;
    
    ValueType finalCut = 0;
    ValueType finalImbalance = 0;
    IndexType maxBlockGraphDegree= 0;
    IndexType totalBlockGraphEdges= 0;
    IndexType maxCommVolume= 0;
    IndexType totalCommVolume= 0;
    IndexType maxBoundaryNodes= 0;
    IndexType totalBoundaryNodes= 0;
    ValueType maxBorderNodesPercent= 0;
    ValueType avgBorderNodesPercent= 0;

	IndexType maxBlockDiameter = 0;
	IndexType harmMeanDiam = 0;
	IndexType numDisconBlocks = 0;

	// used for redistribution
	IndexType maxRedistVol = 0;
	IndexType totRedistVol = 0;
	
	// various other needed info
	IndexType numBlocks = -1;
	
	//constructor
	//
	
	Metrics( IndexType k = 1) {
		initialize(k);
	}
	
	void initialize(IndexType k ){
		timeMigrationAlgo.resize(k);
		timeConstructRedistributor.resize(k);
		timeFirstDistribution.resize(k);
		timeKmeans.resize(k);
		timeSecondDistribution.resize(k);
		timePreliminary.resize(k);
	}
	
	//print metrics
	//
	void print( std::ostream& out){
		
		// for these time we have one measurement per PE and must make a max
		//ValueType maxTimeMigrationAlgo = *std::max_element( timeMigrationAlgo.begin(), timeMigrationAlgo.end() );
		//ValueType maxTimeFirstDistribution = *std::max_element( timeFirstDistribution.begin(), timeFirstDistribution.end() );
		ValueType maxTimeKmeans = *std::max_element( timeKmeans.begin(), timeKmeans.end() );
		//ValueType maxTimeSecondDistribution = *std::max_element( timeSecondDistribution.begin(), timeSecondDistribution.end() );
		ValueType maxTimePreliminary = *std::max_element( timePreliminary.begin(), timePreliminary.end() );
			
		ValueType timeLocalRef = timeFinalPartition - maxTimePreliminary;
		
		std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
		std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
		out << "date and time: " << std::ctime(&timeNow) << std::endl;
		
		out << "numBlocks= " << numBlocks << std::endl;

		//TODO: this is quite ugly. Refactor as dictionary with key-value-pairs, much more extensible.		
		if( maxBlockGraphDegree==-1 ){
			out << " ### WARNING: setting dummy value -1 for expensive (and not used) metrics max and total blockGraphDegree ###" << std::endl;
		}else if (maxBlockGraphDegree==0 ){
			out << " ### WARNING: possibly not all metrics calculated ###" << std::endl;
		}
		out << "gather" << std::endl;
				
		out << "timeKmeans timeGeom timeGraph timeTotal prelCut finalCut imbalance maxCommVol totCommVol maxDiameter harmMeanDiam numDisBlocks timeSpMV timeComm" << std::endl;

		auto oldprecision = out.precision();
		out << std::setprecision(4) << std::fixed;

		//times
		out<< maxTimeKmeans << " , ";
		out<< maxTimePreliminary << " , ";
		out<< timeLocalRef << " , ";
		out<< timeFinalPartition << " , ";
		
		//solution quality
		out<< preliminaryCut << " , ";
		out<< finalCut << " , ";
		out<< finalImbalance << " , ";
		//out<< maxBoundaryNodes << " , ";
		//out<< totalBoundaryNodes << " , ";
		out<< maxCommVolume << " , ";
		out<< totalCommVolume << " , ";
		out<< maxBlockDiameter << " , ";
		out<< harmMeanDiam<< " , ";
		out<< numDisconBlocks<< " , ";
		out<< std::setprecision(8) << std::fixed;
		out<< timeSpMV << " , ";
		out<< timeComm << std::endl;
		
		out.precision(oldprecision);
	}
//---------------------------------------------------------------------------
	
	void getAllMetrics(scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings ){
		
		getEasyMetrics( graph, partition, nodeWeights, settings );
		
		int numIter = 100;
		getRedistRequiredMetrics( graph, partition, settings, numIter );
		
	}
//---------------------------------------------------------------------------

	void getRedistMetrics( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings ){
		
		getAllMetrics( graph, partition, nodeWeights, settings);
		
		scai::dmemo::DistributionPtr newDist = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partition.getDistribution(), partition.getLocalValues() ) );	
		scai::dmemo::DistributionPtr oldDist = graph.getRowDistributionPtr();
		
		std::tie( maxRedistVol, totRedistVol ) = getRedistributionVol( newDist, oldDist);
		
	}
//---------------------------------------------------------------------------
	void getEasyMetrics( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings ){
		
		finalCut = ITI::GraphUtils::computeCut(graph, partition, true);
		finalImbalance = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( partition, settings.numBlocks, nodeWeights );
		
		//TODO: getting the block graph probably fails for p>5000, removed this metric since we do not use it so much
		//std::tie(maxBlockGraphDegree, totalBlockGraphEdges) = ITI::GraphUtils::computeBlockGraphComm<IndexType, ValueType>( graph, partition, settings.numBlocks );
		
		//set to dummy value -1
		maxBlockGraphDegree = -1;
		totalBlockGraphEdges = -1;

		// communication volume

		// 3 vector each of size numBlocks
		std::vector<IndexType> commVolume;
		std::vector<IndexType> numBorderNodesPerBlock;  
		std::vector<IndexType> numInnerNodesPerBlock;
		
		// TODO: can re returned in an auto, check if it is faster
		// it is a bit uglier but saves time
		std::tie( commVolume, numBorderNodesPerBlock, numInnerNodesPerBlock ) = \
				ITI::GraphUtils::computeCommBndInner( graph, partition, settings.numBlocks );
		
		maxCommVolume = *std::max_element( commVolume.begin(), commVolume.end() );
		totalCommVolume = std::accumulate( commVolume.begin(), commVolume.end(), 0 );
		

		
		//TODO: are num of boundary nodes needed ????         
		maxBoundaryNodes = *std::max_element( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end() );
		totalBoundaryNodes = std::accumulate( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end(), 0 );
		
		std::vector<ValueType> percentBorderNodesPerBlock( settings.numBlocks, 0);
		SCAI_ASSERT_EQ_ERROR( settings.numBlocks, numBorderNodesPerBlock.size(), "Vector size mismatch.");
		SCAI_ASSERT_EQ_ERROR( settings.numBlocks, numInnerNodesPerBlock.size(), "Vector size mismatch.");
		
		for(IndexType i=0; i<settings.numBlocks; i++){
			percentBorderNodesPerBlock[i] = (ValueType (numBorderNodesPerBlock[i]))/(numBorderNodesPerBlock[i]+numInnerNodesPerBlock[i]);
		}
		
		maxBorderNodesPercent = *std::max_element( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end() );
		avgBorderNodesPercent = std::accumulate( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end(), 0.0 )/(ValueType(settings.numBlocks));
		
		//get diameter
		std::tie( maxBlockDiameter, harmMeanDiam, numDisconBlocks ) = getDiameter(graph, partition, settings);
		
	}
//---------------------------------------------------------------------------


	std::tuple<IndexType,IndexType,IndexType> getDiameter( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, struct Settings settings ){
		
		std::chrono::time_point<std::chrono::high_resolution_clock> diameterStart = std::chrono::high_resolution_clock::now();
		IndexType maxBlockDiameter = 0;
		IndexType avgBlockDiameter = 0;
		IndexType numDisconBlocks = 0;
		ValueType harmMeanDiam = 0;
		
		scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
		const IndexType localN = dist->getLocalSize();
		const IndexType numPEs = comm->getSize();
		
		if (settings.numBlocks == numPEs && settings.computeDiameter) {
			//maybe possible to compute diameter
			bool allLocalNodesInSameBlock;
			{
				scai::hmemo::ReadAccess<IndexType> rPart(partition.getLocalValues());
				auto result = std::minmax_element(rPart.get(), rPart.get()+localN);
				allLocalNodesInSameBlock = ((*result.first) == (*result.second));
			}
			if (comm->all(allLocalNodesInSameBlock)) {
				IndexType maxRounds = settings.maxDiameterRounds;
				if (maxRounds < 0) {
					maxRounds = localN;
				}
				IndexType localDiameter = ITI::GraphUtils::getLocalBlockDiameter<IndexType, ValueType>(graph, localN/2, 0, 0, maxRounds);
				
				ValueType sumInverseDiam = comm->sum( 1.0/localDiameter );
				harmMeanDiam = comm->getSize()/sumInverseDiam;
				
				// count the number of disconnected blocks
				IndexType isDisconnected = 0;
								
				if( localDiameter== std::numeric_limits<IndexType>::max()){
					isDisconnected=1;
					//set to 0 so it does not affect the max and avg diameter
					localDiameter = 0;	
				}
				
				numDisconBlocks = comm->sum(isDisconnected);
				PRINT0("number of disconnected blocks: " << numDisconBlocks);
				
				//PRINT(*comm << ": "<< localDiameter);
				maxBlockDiameter = comm->max(localDiameter);
				
				// in case all blocks are disconnected
				if( numPEs-numDisconBlocks==0 ){
					avgBlockDiameter = 0;
				}else{
					avgBlockDiameter = comm->sum(localDiameter) / (numPEs-numDisconBlocks);
				}
			}else{
				PRINT0("\tWARNING: Not computing diameter, not all vertices are in same block everywhere");
			}
		}
		std::chrono::duration<ValueType,std::ratio<1>> diameterTime = std::chrono::high_resolution_clock::now() - diameterStart; 
		ValueType time = comm->max( diameterTime.count() );
		PRINT0("time to get the diameter: " <<  time );
	
		return std::make_tuple( maxBlockDiameter, harmMeanDiam, numDisconBlocks);
	}
//---------------------------------------------------------------------------
	
	
	void getRedistRequiredMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, struct Settings settings, const IndexType repeatTimes ){
	
		scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		const IndexType N = graph.getNumRows();
	
		//get the distribution from the partition
		scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partition.getDistribution(), partition.getLocalValues() ) );
		
		std::chrono::time_point<std::chrono::system_clock> beforeRedistribution = std::chrono::system_clock::now();

		// redistribute graph according to partition distribution		
		// distribute only rows for the diameter calculation
		
		scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( graph.getNumRows() ));
		scai::lama::CSRSparseMatrix<ValueType> copyGraph( graph, distFromPartition, noDistPtr);
			
		
		std::chrono::duration<ValueType> redistributionTime =  std::chrono::system_clock::now() - beforeRedistribution;
		
		ValueType time = 0;
		time = comm->max( redistributionTime.count() );
		PRINT0("time to redistribute: " << time);		
		
		const IndexType localN = distFromPartition->getLocalSize();
		SCAI_ASSERT_EQ_ERROR( localN, copyGraph.getLocalNumRows(), "Distribution mismatch")
		
		const IndexType maxLocalN = comm->max(localN);
		const IndexType minLocalN = comm->min(localN);
		const ValueType optSize = ValueType(N)/comm->getSize();
		
		ValueType imbalance = ValueType( maxLocalN - optSize)/optSize;
		PRINT0("minLocalN= "<< minLocalN <<", maxLocalN= " << maxLocalN << ", imbalance= " << imbalance);
						
		// diameter
		if( maxBlockDiameter==0 or harmMeanDiam==0){
			scai::lama::DenseVector<IndexType> copyPartition( partition, distFromPartition );	
			std::tie( maxBlockDiameter, harmMeanDiam, numDisconBlocks ) = getDiameter(copyGraph, copyPartition, settings);
		}
		
		// redistribute for SpMV and commTime
		copyGraph.redistribute( distFromPartition, distFromPartition );	
		
		// SpMV 
		{
			PRINT0("starting SpMV...");
			// vector for multiplication
			scai::lama::DenseVector<ValueType> x ( copyGraph.getColDistributionPtr(), 1.0 );
			scai::lama::DenseVector<ValueType> y ( copyGraph.getRowDistributionPtr(), 0.0 );
			copyGraph.setCommunicationKind( scai::lama::Matrix::SyncKind::ASYNCHRONOUS );
			comm->synchronize();
			
			// perfom the actual multiplication
			std::chrono::time_point<std::chrono::system_clock> beforeSpMVTime = std::chrono::system_clock::now();
			for(IndexType r=0; r<repeatTimes; r++){
				y = copyGraph *x +y;
			}
			comm->synchronize();
			std::chrono::duration<ValueType> SpMVTime = std::chrono::system_clock::now() - beforeSpMVTime;
			//PRINT(" SpMV time for PE "<< comm->getRank() << " = " << SpMVTime.count() );
			
			time = comm->max(SpMVTime.count());
			timeSpMV = time/repeatTimes;
			
			ValueType minTime = comm->min( SpMVTime.count() );
			PRINT0("max time for " << repeatTimes <<" SpMVs: " << time << " , min time " << minTime);
		}
		
		//TODO: maybe extract this time from the actual SpMV above
		// comm time in SpMV
		{
			PRINT0("starting comm shcedule...");
			const scai::dmemo::Halo& matrixHalo = copyGraph.getHalo();
			const scai::dmemo::CommunicationPlan& sendPlan  = matrixHalo.getProvidesPlan();
			const scai::dmemo::CommunicationPlan& recvPlan  = matrixHalo.getRequiredPlan();
			
			//PRINT(*comm << ": " << 	sendPlan.size() << " ++ " << sendPlan << " ___ " << recvPlan);
			scai::hmemo::HArray<ValueType> sendData( sendPlan.totalQuantity(), 1.0 );
			scai::hmemo::HArray<ValueType> recvData;
			
			comm->synchronize();
			std::chrono::time_point<std::chrono::system_clock> beforeCommTime = std::chrono::system_clock::now();
			for ( IndexType i = 0; i < repeatTimes; ++i ){
				comm->exchangeByPlan( recvData, recvPlan, sendData, sendPlan );
			}
			//comm->synchronize();
			std::chrono::duration<ValueType> commTime = std::chrono::system_clock::now() - beforeCommTime;
			
			//PRINT(*comm << ": "<< sendPlan );		
			time = comm->max(commTime.count());
			timeComm = time/repeatTimes;
			
			ValueType minTime = comm->min( commTime.count() );
			PRINT0("max time for " << repeatTimes <<" communications: " << time << " , min time " << minTime);
		}
		
		//redistibute back to initial distributions
		//graph.redistribute( initRowDistPtr, initColDistPtr );
		
	}

	/* Calculate the volume, aka the data that will be exchanged when redistributing from oldDist to newDist.
	 */
	std::pair<IndexType,IndexType> getRedistributionVol( const scai::dmemo::DistributionPtr newDist , const scai::dmemo::DistributionPtr oldDist){
		
		scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		
		//get the distribution from the partition
		scai::dmemo::Redistributor prepareRedist( newDist, oldDist );	
				
		// redistribution load
		scai::hmemo::HArray<IndexType> sourceIndices = prepareRedist.getHaloSourceIndexes();
		scai::hmemo::HArray<IndexType> targetIndices = prepareRedist.getHaloTargetIndexes();
		IndexType thisSourceSize = prepareRedist.getHaloSourceSize();
		IndexType thisTargetSize = prepareRedist.getHaloTargetSize();

		IndexType globTargetSize = comm->sum( thisTargetSize);
		IndexType globSourceSize = comm->sum( thisSourceSize);
		SCAI_ASSERT_EQ_ERROR( globSourceSize, globTargetSize, "Mismatch in total migartion volume.");
		//PRINT0("total migration volume= " << globSourceSize);
		
		IndexType maxTargetSize = comm->max( thisTargetSize);
		IndexType maxSourceSize = comm->max( thisSourceSize);
		//PRINT0("maxTargetSize= " << maxTargetSize);
		//PRINT0("maxSourceSize= " << maxSourceSize);
		
		return std::make_pair( std::max(maxTargetSize,maxSourceSize), globSourceSize);
	}
	
	
	//TODO: deprecated version, remove
	ValueType getCommScheduleTime( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, const IndexType repeatTimes){
			
		scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		const IndexType N = graph.getNumRows();
		
		PRINT0("starting comm shcedule...");
		
		// the original row and  column distributions
		const scai::dmemo::DistributionPtr initRowDistPtr = graph.getRowDistributionPtr();
		const scai::dmemo::DistributionPtr initColDistPtr = graph.getColDistributionPtr();
		
		//get the distribution from the partition
		scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partition.getDistribution(), partition.getLocalValues() ) );
		
		std::chrono::time_point<std::chrono::system_clock> beforeRedistribution = std::chrono::system_clock::now();
		// redistribute graph according to partition distribution
		//graph.redistribute( distFromPartition, initColDistPtr);
		graph.redistribute( distFromPartition, distFromPartition);
		std::chrono::duration<ValueType> redistributionTime =  std::chrono::system_clock::now() - beforeRedistribution;
		
		ValueType time = 0;
		time = comm->max( redistributionTime.count() );
		PRINT0("time to redistribute: " << time);
	
		const IndexType localN = distFromPartition->getLocalSize();
		SCAI_ASSERT_EQ_ERROR( localN, graph.getLocalNumRows(), "Distribution mismatch")
		
		const IndexType maxLocalN = comm->max(localN);
		const IndexType minLocalN = comm->min(localN);
		const ValueType optSize = ValueType(N)/comm->getSize();
		
		ValueType imbalance = ValueType( maxLocalN - optSize)/optSize;
		PRINT0("minLocalN= "<< minLocalN <<", maxLocalN= " << maxLocalN << ", imbalance= " << imbalance);
		
		const scai::dmemo::Halo& matrixHalo = graph.getHalo();
		const scai::dmemo::CommunicationPlan& sendPlan  = matrixHalo.getProvidesPlan();
		const scai::dmemo::CommunicationPlan& recvPlan  = matrixHalo.getRequiredPlan();
		
		scai::hmemo::HArray<ValueType> sendData( sendPlan.totalQuantity(), 1.0 );
		scai::hmemo::HArray<ValueType> recvData;
		
		comm->synchronize();
		std::chrono::time_point<std::chrono::system_clock> beforeCommTime = std::chrono::system_clock::now();
		for ( IndexType i = 0; i < repeatTimes; ++i ){
			comm->exchangeByPlan( recvData, recvPlan, sendData, sendPlan );
		}
		//comm->synchronize();
		std::chrono::duration<ValueType> commTime = std::chrono::system_clock::now() - beforeCommTime;
		
		//PRINT(*comm << ": "<< sendPlan );		
		time = comm->max(commTime.count());
		
		ValueType minTime = comm->min( commTime.count() );
		PRINT0("max time for " << repeatTimes <<" communications: " << time << " , min time " << minTime);
	
		//redistibute back to initial distributions
		graph.redistribute( initRowDistPtr, initColDistPtr );
		
		return time;
	}

};


//------------------------------------------------------------------------------------------------------------

inline void printMetricsShort(struct Metrics metrics, std::ostream& out){
	
	std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
	std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
	out << "date and time: " << std::ctime(&timeNow); //<< std::endl;
	out << "numBlocks= " << metrics.numBlocks << std::endl;
	out << "gather" << std::endl;
	out << "timeTotal finalCut imbalance maxBnd totBnd maxCommVol totCommVol maxDiameter harmMeanDiam numDisBlocks timeSpMV timeComm" << std::endl;
	out << metrics.timeFinalPartition<< " " \
		<< metrics.finalCut << " "\
		<< metrics.finalImbalance << " "\
		<< metrics.maxBoundaryNodes << " "\
		<< metrics.totalBoundaryNodes << " "\
		<< metrics.maxCommVolume << " "\
		<< metrics.totalCommVolume << " "\
		<< metrics.maxBlockDiameter << " " \
		<< metrics.harmMeanDiam << " " \
		<< metrics.numDisconBlocks << " ";
	out << std::setprecision(8) << std::fixed;
	out	<< metrics.timeSpMV << " "\
		<< metrics.timeComm \
		<< std::endl; 
}

//------------------------------------------------------------------------------------------------------------

inline void printRedistMetricsShort(struct Metrics metrics, std::ostream& out){
	
	std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
	std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
	out << "date and time: " << std::ctime(&timeNow); //<< std::endl;
	out << "numBlocks= " << metrics.numBlocks << std::endl;
	out << "gather" << std::endl;
	out << "timeTotal finalCut imbalance maxCommVol totCommVol maxRedistVol totRedistVol maxDiameter harmMeanDiam numDisBlocks timeSpMV timeComm" << std::endl;
	out << metrics.timeFinalPartition<< " " \
		<< metrics.finalCut << " "\
		<< metrics.finalImbalance << " "\
		<< metrics.maxCommVolume << " "\
		<< metrics.totalCommVolume << " " \
		<< metrics.maxRedistVol << " "\
		<< metrics.totRedistVol << " " \
		<< metrics.maxBlockDiameter << " " \
		<< metrics.harmMeanDiam << " " \
		<< metrics.numDisconBlocks << " ";
	out << std::setprecision(8) << std::fixed;
	out	<< metrics.timeSpMV << " "\
		<< metrics.timeComm \
		<< std::endl; 
}

//-------------------------------------------------------------------------------------------------------------


inline void printVectorMetrics( std::vector<struct Metrics>& metricsVec, std::ostream& out){
	
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	
	IndexType numRuns = metricsVec.size();
	
	if( comm->getRank()==0 ){
		std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
		std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
		out << "date and time: " << std::ctime(&timeNow) << std::endl;
		out << "numBlocks= " << metricsVec[0].numBlocks << std::endl;
		out << "# times, input, migrAlgo, 1distr, kmeans, 2redis, prelim, localRef, total,    prel cut, finalcut, imbalance,    maxBnd, totalBnd,    maxCommVol, totalCommVol,   max diameter , avg diameter, numDisBlocks   timeSpMV timeComm" << std::endl;
	}

	ValueType sumMigrAlgo = 0;
	ValueType sumFirstDistr = 0;
	ValueType sumKmeans = 0;
	ValueType sumSecondDistr = 0;
	ValueType sumPrelimanry = 0; 
	ValueType sumLocalRef = 0; 
	ValueType sumFinalTime = 0;
	
	IndexType sumPreliminaryCut = 0;
	IndexType sumFinalCut = 0;
	ValueType sumImbalace = 0;
	IndexType sumMaxBnd = 0;
	IndexType sumTotBnd = 0;
	IndexType sumMaxCommVol = 0;
	IndexType sumTotCommVol = 0;
	
	ValueType sumMaxDiameter = 0;
	ValueType sumharmMeanDiam = 0;
	ValueType sumDisconBlocks = 0;
	
	ValueType sumTimeSpMV = 0;
	ValueType sumTimeComm = 0;
	
	for(IndexType run=0; run<numRuns; run++){
		Metrics thisMetric = metricsVec[ run ];
		
		SCAI_ASSERT_EQ_ERROR(thisMetric.timeMigrationAlgo.size(), comm->getSize(), "Wrong vector size" );
		
		// for these time we have one measurement per PE and must make a max
		ValueType maxTimeMigrationAlgo = *std::max_element( thisMetric.timeMigrationAlgo.begin(), thisMetric.timeMigrationAlgo.end() );
		ValueType maxTimeFirstDistribution = *std::max_element( thisMetric.timeFirstDistribution.begin(), thisMetric.timeFirstDistribution.end() );
		ValueType maxTimeKmeans = *std::max_element( thisMetric.timeKmeans.begin(), thisMetric.timeKmeans.end() );
		ValueType maxTimeSecondDistribution = *std::max_element( thisMetric.timeSecondDistribution.begin(), thisMetric.timeSecondDistribution.end() );
		ValueType maxTimePreliminary = *std::max_element( thisMetric.timePreliminary.begin(), thisMetric.timePreliminary.end() );
		
		// these times are global, no need to max
		ValueType timeFinal = thisMetric.timeFinalPartition;
		ValueType timeLocalRef = timeFinal - maxTimePreliminary;
		
		if( comm->getRank()==0 ){
			out << std::setprecision(4) << std::fixed;
			out<< run << " ,       "<< thisMetric.inputTime << ",  " << maxTimeMigrationAlgo << ",  " << maxTimeFirstDistribution << ",  " << maxTimeKmeans << ",  " << maxTimeSecondDistribution << ",  " << maxTimePreliminary << ",  " << timeLocalRef << ",  "<< timeFinal << " , \t "\
			<< thisMetric.preliminaryCut << ",  "<< thisMetric.finalCut << ",  " << thisMetric.finalImbalance << ",    "  \
			// << thisMetric.maxBlockGraphDegree << ",  " << thisMetric.totalBlockGraphEdges << " ," 
			<< thisMetric.maxBoundaryNodes << ", " << thisMetric.totalBoundaryNodes << ",    " \
			<< thisMetric.maxCommVolume << ",  " << thisMetric.totalCommVolume << ",    " \
			<< thisMetric.maxBlockDiameter << ",  " << thisMetric.harmMeanDiam<< ", " \
			<< thisMetric.numDisconBlocks <<", ";
			out << std::setprecision(8) << std::fixed;
			out <<  thisMetric.timeSpMV << " , " \
			<<thisMetric.timeComm  << std::endl;
		}
		
		sumMigrAlgo += maxTimeMigrationAlgo;
		sumFirstDistr += maxTimeFirstDistribution;
		sumKmeans += maxTimeKmeans;
		sumSecondDistr += maxTimeSecondDistribution;
		sumPrelimanry += maxTimePreliminary;
		sumLocalRef += timeLocalRef;
		sumFinalTime += timeFinal;
		
		sumPreliminaryCut += thisMetric.preliminaryCut;
		sumFinalCut += thisMetric.finalCut;
		sumImbalace += thisMetric.finalImbalance;
		sumMaxBnd += thisMetric.maxBoundaryNodes  ;
		sumTotBnd += thisMetric.totalBoundaryNodes ;
		sumMaxCommVol +=  thisMetric.maxCommVolume;
		sumTotCommVol += thisMetric.totalCommVolume;
		sumMaxDiameter += thisMetric.maxBlockDiameter;
		sumharmMeanDiam += thisMetric.harmMeanDiam;
		sumDisconBlocks += thisMetric.numDisconBlocks;
		
		sumTimeSpMV += thisMetric.timeSpMV;
		sumTimeComm += thisMetric.timeComm;
	}
	
	if( comm->getRank()==0 ){
		out << std::setprecision(4) << std::fixed;
		out << "average,  "\
			<< ValueType (metricsVec[0].inputTime)<< ",  "\
			<< ValueType(sumMigrAlgo)/numRuns<< ",  " \
			<< ValueType(sumFirstDistr)/numRuns<< ",  " \
			<< ValueType(sumKmeans)/numRuns<< ",  " \
			<< ValueType(sumSecondDistr)/numRuns<< ",  " \
			<< ValueType(sumPrelimanry)/numRuns<< ",  " \
			<< ValueType(sumLocalRef)/numRuns<< ",  " \
			<< ValueType(sumFinalTime)/numRuns<< ", \t " \
			<< ValueType(sumPreliminaryCut)/numRuns<< ",  " \
			<< ValueType(sumFinalCut)/numRuns<< ",  " \
			<< ValueType(sumImbalace)/numRuns<< ",    " \
			<< ValueType(sumMaxBnd)/numRuns<< ",  " \
			<< ValueType(sumTotBnd)/numRuns<< ",    " \
			<< ValueType(sumMaxCommVol)/numRuns<< ", " \
			<< ValueType(sumTotCommVol)/numRuns<< ",    "\
			<< ValueType(sumMaxDiameter)/numRuns<< ", " \
			<< ValueType(sumharmMeanDiam)/numRuns << ", " \
			<< ValueType(sumDisconBlocks)/numRuns << ", " ;
			out << std::setprecision(8) << std::fixed;
			out << ValueType(sumTimeSpMV)/numRuns << ", " \
			<< ValueType(sumTimeComm)/numRuns \
			<< std::endl;
			
		out << std::setprecision(4) << std::fixed;
		out << "gather" << std::endl;
		out << "timeKmeans timeGeom timeGraph timeTotal prelCut finalCut imbalance maxBnd totBnd maxCommVol totCommVol maxDiameter harmMeanDiam numDisBlocks timeSpMV timeComm" << std::endl;
		out << ValueType(sumKmeans)/numRuns<< " " \
			<< ValueType(sumPrelimanry)/numRuns<< " " \
			<< ValueType(sumLocalRef)/numRuns<< " " \
			<< ValueType(sumFinalTime)/numRuns<< " " \
			<< ValueType(sumPreliminaryCut)/numRuns<< " " \
			<< ValueType(sumFinalCut)/numRuns<< " " \
			<< ValueType(sumImbalace)/numRuns<< " " \
			<< ValueType(sumMaxBnd)/numRuns<< " " \
			<< ValueType(sumTotBnd)/numRuns<< " " \
			<< ValueType(sumMaxCommVol)/numRuns<< " " \
			<< ValueType(sumTotCommVol)/numRuns<< " "\
			<< ValueType(sumMaxDiameter)/numRuns<< " " \
			<< ValueType(sumharmMeanDiam)/numRuns << " " \
			<< ValueType(sumDisconBlocks)/numRuns << " " ;
			out << std::setprecision(8) << std::fixed;
			out << ValueType(sumTimeSpMV)/numRuns << " " \
			<< ValueType(sumTimeComm)/numRuns \
			<< std::endl;        
	}
	
}


//-------------------------------------------------------------------------------------------------------------


inline void printVectorMetricsShort( std::vector<struct Metrics>& metricsVec, std::ostream& out){
	
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	
	IndexType numRuns = metricsVec.size();
	
	if( comm->getRank()==0 ){
		std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
		std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
		out << "date and time: " << std::ctime(&timeNow) << std::endl;
		out << "numBlocks= " << metricsVec[0].numBlocks << std::endl;
		out << "timeTotal finalcut imbalance maxBnd totalBnd maxCommVol totalCommVol maxDiameter harmMeanDiam numDisBlocks timeSpMV timeComm" << std::endl;
	}

	ValueType sumFinalTime = 0;
	
	IndexType sumFinalCut = 0;
	ValueType sumImbalace = 0;
	IndexType sumMaxBnd = 0;
	IndexType sumTotBnd = 0;
	IndexType sumMaxCommVol = 0;
	IndexType sumTotCommVol = 0;
	ValueType sumMaxDiameter = 0;
	ValueType sumharmMeanDiam = 0;
	ValueType sumDisconBlocks = 0;
	ValueType sumTimeSpMV = 0;
	ValueType sumTimeComm = 0;
	
	for(IndexType run=0; run<numRuns; run++){
		Metrics thisMetric = metricsVec[ run ];
		
		SCAI_ASSERT_EQ_ERROR(thisMetric.timeMigrationAlgo.size(), comm->getSize(), "Wrong vector size" );
		
		// these times are global, no need to max
		ValueType timeFinal = thisMetric.timeFinalPartition;
		
		if( comm->getRank()==0 ){
			out << std::setprecision(4) << std::fixed;
			//out<< run <<  maxTimeKmeans << ",  " << maxTimeSecondDistribution << ",  " << maxTimePreliminary << ",  " << timeLocalRef << ",  ";
			out << timeFinal << "  ";
			//<< thisMetric.preliminaryCut << ",  "
			out << thisMetric.finalCut << "  " << thisMetric.finalImbalance << "  "  \
			<< thisMetric.maxBoundaryNodes << " " << thisMetric.totalBoundaryNodes << "  " \
			<< thisMetric.maxCommVolume << "  " << thisMetric.totalCommVolume << " " \
			<< thisMetric.maxBlockDiameter << " " << thisMetric.harmMeanDiam << " "\
			<< thisMetric.numDisconBlocks << " ";
			out << std::setprecision(8) << std::fixed;
			out << thisMetric.timeSpMV << " " \
			<< thisMetric.timeComm \
			<< std::endl;
		}
		
		sumFinalTime += timeFinal;
		
		sumFinalCut += thisMetric.finalCut;
		sumImbalace += thisMetric.finalImbalance;
		sumMaxBnd += thisMetric.maxBoundaryNodes  ;
		sumTotBnd += thisMetric.totalBoundaryNodes ;
		sumMaxCommVol +=  thisMetric.maxCommVolume;
		sumTotCommVol += thisMetric.totalCommVolume;
		sumMaxDiameter += thisMetric.maxBlockDiameter;
		sumharmMeanDiam += thisMetric.harmMeanDiam;
		sumDisconBlocks += thisMetric.numDisconBlocks;
		sumTimeSpMV += thisMetric.timeSpMV;
		sumTimeComm += thisMetric.timeComm;
	}
	
	if( comm->getRank()==0 ){
		out << "gather" << std::endl;
		out << "timeTotal finalcut imbalance maxBnd totalBnd maxCommVol totalCommVol maxDiameter harmMeanDiam numDisBlocks timeSpMV timeComm " << std::endl;
		
		out << std::setprecision(4) << std::fixed;
		out << ValueType(sumFinalTime)/numRuns<< " " \
			<< ValueType(sumFinalCut)/numRuns<< " " \
			<< ValueType(sumImbalace)/numRuns<< " " \
			<< ValueType(sumMaxBnd)/numRuns<< " " \
			<< ValueType(sumTotBnd)/numRuns<< " " \
			<< ValueType(sumMaxCommVol)/numRuns<< " " \
			<< ValueType(sumTotCommVol)/numRuns<< " " \
			<< ValueType(sumMaxDiameter)/numRuns<< " " \
			<< ValueType(sumharmMeanDiam)/numRuns  <<" "\
			<< ValueType(sumDisconBlocks)/numRuns  <<" ";
			out << std::setprecision(8) << std::fixed;
			out << ValueType(sumTimeSpMV)/numRuns << " "\
			<< ValueType(sumTimeComm)/numRuns \
			<< std::endl;
	}
	
}

