#pragma once

#include <numeric>
#include <math.h>
#include <scai/lama.hpp>
#include <scai/dmemo/RedistributePlan.hpp>
#include <chrono>
#include <algorithm>

#include "GraphUtils.h"

struct Metrics{
    
    // timing results
    //
    
    //vector specific for kmeans
    // tuple has 3 values: (delta, maxTime, imbalance)
    std::vector<std::tuple<ValueType, ValueType, ValueType>> kmeansProfiling;
    std::vector<IndexType> numBalanceIter;
    
    std::vector< std::vector<std::pair<ValueType,ValueType>> > localRefDetails;

    //MM, metrics map
	std::map<std::string,ValueType> MM = {
		{"timeMigrationAlgo",-1.0} , {"timeFirstDistribution",-1.0}, {"timeTotal",-1.0} , {"timeSpMV",-1.0}, {"timeComm",-1.0} , {"reportTime",-1.0},
		{"inputTime",-1.0} , {"timeFinalPartition",-1.0}, {"timeKmeans",-1.0} , {"timeSecondDistribution",-1.0}, {"timePreliminary",-1.0} ,
		{"preliminaryCut",-1.0} , {"preliminaryImbalance",-1.0}, {"finalCut",-1.0} , {"finalImbalance",-1.0}, {"maxBlockGraphDegree",-1.0},
		{"totalBlockGraphEdges",-1.0}, {"maxCommVolume",-1.0} , {"totalCommVolume",-1.0}, {"maxBoundaryNodes",-1.0} , {"totalBoundaryNodes",-1.0},
		{"maxBorderNodesPercent",-1.0} , {"avgBorderNodesPercent",-1.0},
		{"maxBlockDiameter",-1.0}, {"harmMeanDiam",-1.0}, {"numDisconBlocks",-1.0},
		{"maxRedistVol",-1.0}, {"totRedistVol",-1.0},	 //redistribution metrics
		{"maxCongestion",-1.0}, {"maxDilation",-1.0}, {"avgDilation",0.0}		//mapping metrics
	};
	
	//constructors
	//
	
	Metrics( Settings settings) {
		//initialize( settings.numBlocks );
		localRefDetails.resize( settings.multiLevelRounds+1 );
		for( int i=0; i<settings.multiLevelRounds+1; i++){
			//WARNING: problem if refinement rounds are more than 50. Very, very difficult to happen
			localRefDetails[i].resize( 50, std::make_pair(-1,-1) );	
		}
	}

/*
	void initialize(IndexType k ){
		timeMigrationAlgo.resize(k);
		timeConstructRedistributor.resize(k);
		timeFirstDistribution.resize(k);
		timeKmeans.resize(k);
		timeSecondDistribution.resize(k);
		timePreliminary.resize(k);
	}
*/

	void getAllMetrics(const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings );
	
	void getRedistMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings );

	void getRedistRequiredMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, struct Settings settings, const IndexType repeatTimes );

	void getEasyMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings );

	std::tuple<IndexType,IndexType,IndexType> getDiameter( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, struct Settings settings );

	/* Calculate the volume, aka the data that will be exchanged when redistributing from 	oldDist to newDist.
	 */
	std::pair<IndexType,IndexType> getRedistributionVol( const scai::dmemo::DistributionPtr newDist , const scai::dmemo::DistributionPtr oldDist);

	ValueType getCommScheduleTime( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, const IndexType repeatTimes);

	/** 
	@param[in] blockGraph The block (or communication) graph.
	@param[in] PEGraph The processor graph (or physical network).
	@param[in] mapping A mapping from blocks to PEs.
	**/
	void getMappingMetrics(
		const scai::lama::CSRSparseMatrix<ValueType> blockGraph, 
		const scai::lama::CSRSparseMatrix<ValueType> PEGraph, 
		const std::vector<IndexType> mapping);

	/** 
	Internally, the identity mapping is assumed.
	@param[in] appGraph The application graph
	@param[in] partition A partition of the application graph. The number of blocks
	should be equal to the number of nodes of the PE graph.
	@param[in] PEGraph The processor graph (or physical network).

	**/
	void getMappingMetrics(
		const scai::lama::CSRSparseMatrix<ValueType> appGraph,
		const scai::lama::DenseVector<IndexType> partition,
		const scai::lama::CSRSparseMatrix<ValueType> PEGraph );

	
	//print metrics
	//
	void print( std::ostream& out);

	//void printMetricsShort( std::ostream& out );

}; //struct Metrics


//------------------------------------------------------------------------------------------------------------
/*
inline void Metrics::printMetricsShort(std::ostream& out){
	
	std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
	std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
	out << "date and time: " << std::ctime(&timeNow); //<< std::endl;

	
	for( auto mapIt= metricsMap.begin(); mapIt!=metricsMap.end(); mapIt++ ){
		out<< mapIt->first <<": " << mapIt->second << std::endl;
	}

}
*/
//------------------------------------------------------------------------------------------------------------

/*
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
*/
//-------------------------------------------------------------------------------------------------------------


inline void printVectorMetrics( std::vector<struct Metrics>& metricsVec, std::ostream& out, Settings settings){
	
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	
	IndexType numRuns = metricsVec.size();

	Metrics aggregateMetrics( settings );
	
	for(IndexType run=0; run<numRuns; run++){
		Metrics thisMetric = metricsVec[ run ];
		
		//SCAI_ASSERT_EQ_ERROR(thisMetric.timeMigrationAlgo.size(), (unsigned int)comm->getSize(), "Wrong vector size" );
		
		// for these time we have one measurement per PE and must make a max
		aggregateMetrics.MM["timeMigrationAlgo"] += comm->max( thisMetric.MM["timeMigrationAlgo"] );
		aggregateMetrics.MM["timeMigrationAlgo"] /= numRuns;
		aggregateMetrics.MM["timeFirstDistribution"] +=  comm->max( thisMetric.MM["timeFirstDistribution"] );
		aggregateMetrics.MM["timeFirstDistribution"] /= numRuns;
		aggregateMetrics.MM["timeKmeans"] += comm->max( thisMetric.MM["timeKmeans"] );
		aggregateMetrics.MM["timeKmeans"] /= numRuns;
		aggregateMetrics.MM["timeSecondDistribution"] += comm->max( thisMetric.MM["timeSecondDistribution"] );
		aggregateMetrics.MM["timeSecondDistribution"] /= numRuns;
		aggregateMetrics.MM["timePreliminary"] += comm->max( thisMetric.MM["timePreliminary"] );
		aggregateMetrics.MM["timePreliminary"] /= numRuns;
		
		// these times are global, no need to max
		aggregateMetrics.MM["timeFinalPartition"] += thisMetric.MM["timeFinalPartition"];
		aggregateMetrics.MM["timeFinalPartition"] /= numRuns;

		aggregateMetrics.MM["timeSpMV"] += thisMetric.MM["timeSpMV"];
		aggregateMetrics.MM["timeSpMV"] /= numRuns;
		aggregateMetrics.MM["timeComm"] += thisMetric.MM["timeComm"];
		aggregateMetrics.MM["timeComm"] /= numRuns;

	/*
		ValueType timeLocalRef = aggregateMetrics.MM["timeFinalPartition"] - aggregateMetrics.MM["maxTimePreliminary"];
		sumLocalRef += timeLocalRef;
		sumLocalRef /= numRuns;
	*/
		//TODO: is this too much? maybe add a shorter print?
		thisMetric.print( out );

	}
	
	if( comm->getRank()==0 ){
				
		out << std::setprecision(4) << std::fixed;
		aggregateMetrics.print( out );

/*		
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
*/

	}
	
}


//-------------------------------------------------------------------------------------------------------------

/*
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
*/
