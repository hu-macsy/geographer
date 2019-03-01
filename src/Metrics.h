#pragma once

#include <numeric>
#include <math.h>
#include <scai/lama.hpp>
#include <scai/dmemo/RedistributePlan.hpp>
#include <chrono>
#include <algorithm>

//#include <scai/lama.hpp>
#include "GraphUtils.h"

struct Metrics{
    
    // timing results
    //
    std::vector<ValueType> timeMigrationAlgo;
	std::vector<ValueType> timeConstructRedistributor;
    std::vector<ValueType> timeFirstDistribution;
    std::vector<ValueType> timeKmeans;
    std::vector<ValueType> timeSecondDistribution;
    std::vector<ValueType> timePreliminary;

    //vector specific for kmeans
    // tuple has 3 values: (delta, maxTime, imbalance)
    std::vector<std::tuple<ValueType, ValueType, ValueType>> kmeansProfiling;
    std::vector<IndexType> numBalanceIter;
    
    //std::map< std::pair<int,int>, int, ValueType > localRefDetails;

    //std::pair<int,ValueType> ;
    std::vector< std::vector<std::pair<ValueType,ValueType>> > localRefDetails;

   	ValueType inputTime = 0;
	ValueType timeFinalPartition = 0;
	ValueType reportTime = 0 ;
	ValueType timeTotal = 0;
	ValueType timeSpMV = 0;
	ValueType timeComm = 0;
    
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
	
	//constructors
	//
	
	Metrics( Settings settings) {
		initialize( settings.numBlocks );
		localRefDetails.resize( settings.multiLevelRounds+1 );
		for( int i=0; i<settings.multiLevelRounds+1; i++){
			//WARNING: problem if refinement rounds are more than 50. Very, very difficult to happen
			localRefDetails[i].resize( 50, std::make_pair(-1,-1) );	
		}
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
	void print( std::ostream& out);

	void getAllMetrics(const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings );
	
	void getRedistMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings );

	void getRedistRequiredMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, struct Settings settings, const IndexType repeatTimes );

	void getEasyMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings );

	std::tuple<IndexType,IndexType,IndexType> getDiameter( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, struct Settings settings );

	/* Calculate the volume, aka the data that will be exchanged when redistributing from 	oldDist to newDist.
	 */
	std::pair<IndexType,IndexType> getRedistributionVol( const scai::dmemo::DistributionPtr newDist , const scai::dmemo::DistributionPtr oldDist);

	ValueType getCommScheduleTime( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, const IndexType repeatTimes);

	void getMappingMetrics(
		const scai::lama::CSRSparseMatrix<ValueType> blockGraph, 
		const scai::lama::CSRSparseMatrix<ValueType> PEGraph, 
		const std::vector<IndexType> mapping);

}; //struct Metrics


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
	//umLocalRefDetails[i][j].first is the time, for local refinement round i,j
	//umLocalRefDetails[i][j].second is the gain (edges) in cut, for local refinement round i,j
	std::vector< std::vector<std::pair<ValueType,ValueType>> > sumLocalRefDetails(50);
	int counter[50][50] = {0};
	for(int i=0; i<50; i++){
		sumLocalRefDetails[i].resize(50, std::make_pair(0.0,0.0) );
		std::fill( counter[i], counter[i]+50, 0);
	}
	
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
		
		SCAI_ASSERT_EQ_ERROR(thisMetric.timeMigrationAlgo.size(), (unsigned int)comm->getSize(), "Wrong vector size" );
		
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


		
		//std::vector< std::tuple<int, int, int, double>> lala;

		//sumLocalRefDetails.resize(50);
		//out<< "localRefinement details for run " << run << std::endl;
		for( unsigned int i=0; i<thisMetric.localRefDetails.size(); i++){
			for( unsigned int j=0; j<thisMetric.localRefDetails[i].size(); j++){
				if( thisMetric.localRefDetails[i][j].first != -1){
					sumLocalRefDetails[i][j].first += thisMetric.localRefDetails[i][j].first;
					sumLocalRefDetails[i][j].second += thisMetric.localRefDetails[i][j].second;
					counter[i][j]++;
					//PRINT0( i << " _ " << j << " = " << counter[i][j]);
				}
			}
		}

	}
	
	if( comm->getRank()==0 ){
				
		out << std::setprecision(4) << std::fixed;
		out << "average,  "\
			<< ValueType(metricsVec[0].inputTime)<< ",  "\
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
		
		out<< "localRefinement detail" << std::endl;
		for( unsigned int i=0; i<sumLocalRefDetails.size(); i++){
			if( sumLocalRefDetails[i][0].first != 0){
				out << "MLRound " << i << std::endl;
			}
			for( unsigned int j=0; j<sumLocalRefDetails[i].size(); j++){
				if( sumLocalRefDetails[i][j].first != 0 or sumLocalRefDetails[i][j].first != 0){
//PRINT0( sumLocalRefDetails[i][j].first << "  + + + " << sumLocalRefDetails[i][j].second  );
					SCAI_ASSERT_NE_ERROR(counter[i][j], 0 , "wrong counter value for i,j= " << i << ", "<< j);
					out << "\t refine round " << j <<", gain: " << \
						sumLocalRefDetails[i][j].first/counter[i][j] << ", time: "<< \
						sumLocalRefDetails[i][j].second/counter[i][j] << std::endl;
				}
			}
		}

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
	//ValueType sumFMStep = 0;
	
	for(IndexType run=0; run<numRuns; run++){
		Metrics thisMetric = metricsVec[ run ];
		
		SCAI_ASSERT_EQ_ERROR(thisMetric.timeMigrationAlgo.size(), (unsigned int) comm->getSize(), "Wrong vector size" );
		
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
		//sumFMStep += thisMetric.timeDistFMStep;
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

