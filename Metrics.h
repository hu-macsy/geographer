#pragma once

#include <numeric>
#include <math.h>
#include <scai/lama.hpp>
#include "GraphUtils.h"

struct Metrics{
    
    // timing results
    //
    std::vector<ValueType>  timeMigrationAlgo;
    std::vector<ValueType>  timeFirstDistribution;
    std::vector<ValueType>  timeKmeans;
    std::vector<ValueType>  timeSecondDistribution;
    std::vector<ValueType>  timePreliminary;
    
    ValueType inputTime = -1;
    ValueType timeFinalPartition = -1;
    ValueType reportTime = -1 ;
    ValueType timeTotal = -1;
    
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

    
    //constructor
    //
    Metrics(){}
    
    Metrics( IndexType k) {
        timeMigrationAlgo.resize(k);
        timeFirstDistribution.resize(k);
        timeKmeans.resize(k);
        timeSecondDistribution.resize(k);
        timePreliminary.resize(k);
    }
    
    void initialize(IndexType k ){
        timeMigrationAlgo.resize(k);
        timeFirstDistribution.resize(k);
        timeKmeans.resize(k);
        timeSecondDistribution.resize(k);
        timePreliminary.resize(k);
    }
    
    //print metrics
    //
    void print( std::ostream& out){
        
        // for these time we have one measurement per PE and must make a max
        ValueType maxTimeMigrationAlgo = *std::max_element( timeMigrationAlgo.begin(), timeMigrationAlgo.end() );
        ValueType maxTimeFirstDistribution = *std::max_element( timeFirstDistribution.begin(), timeFirstDistribution.end() );
        ValueType maxTimeKmeans = *std::max_element( timeKmeans.begin(), timeKmeans.end() );
        ValueType maxTimeSecondDistribution = *std::max_element( timeSecondDistribution.begin(), timeSecondDistribution.end() );
        ValueType maxTimePreliminary = *std::max_element( timePreliminary.begin(), timePreliminary.end() );
            
        ValueType timeLocalRef = timeFinalPartition - maxTimePreliminary;
        
        if( maxBlockGraphDegree==-1 ){
            out << " ### WARNING: setting dummy value -1 for expensive (and not used) metrics max and total blockGraphDegree ###" << std::endl;
        }else if (maxBlockGraphDegree==0 ){
            out << " ### WARNING: possibly not all metrics calculated ###" << std::endl;
        }
        //out << "times: input, migrAlgo , 1redistr , k-means , 2redistr , prelim, localRef, total  , metrics:  prel cut, cut, imbalance,    maxBnd, totalBnd,    maxCommVol, totalCommVol,    BorNodes max, avg  " << std::endl;
        out << "gather" << std::endl;
        out << "timeKmeans timeGeom timeGraph timeTotal prelCut finalCut imbalance maxBnd totBnd maxCommVol totCommVol maxBndPercnt avgBndPercnt" << std::endl;
		
        out << std::setprecision(3) << std::fixed;
        out<<  "         "<< inputTime << ",  " << maxTimeMigrationAlgo << ",  " << maxTimeFirstDistribution << ",  " << maxTimeKmeans << ",  " << maxTimeSecondDistribution << ",  " << maxTimePreliminary << ",  " << timeLocalRef << " ,  "<< timeFinalPartition << " ,  \t "\
        << preliminaryCut << ",  "<< finalCut << ",  " << finalImbalance << ",    "  \
        << maxBoundaryNodes << ",  " << totalBoundaryNodes << ",    "  \
        << maxCommVolume << ",  " << totalCommVolume << ",    ";
        out << std::setprecision(6) << std::fixed;
        out << maxBorderNodesPercent << ",  " << avgBorderNodesPercent \
        << std::endl;
    }
    
    void getMetrics( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings ){
        
        finalCut = ITI::GraphUtils::computeCut(graph, partition, true);
        finalImbalance = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( partition, settings.numBlocks, nodeWeights );
        
        //TODO: getting the block graph probably fails for p>5000, removed this metric since we do not use it so much
        //std::tie(maxBlockGraphDegree, totalBlockGraphEdges) = ITI::GraphUtils::computeBlockGraphComm<IndexType, ValueType>( graph, partition, settings.numBlocks );
        
        //set to dummy value -1
        maxBlockGraphDegree = -1;
        totalBlockGraphEdges = -1;

        // communication volume
        std::vector<IndexType> commVolume = ITI::GraphUtils::computeCommVolume( graph, partition );
        
        maxCommVolume = *std::max_element( commVolume.begin(), commVolume.end() );
        totalCommVolume = std::accumulate( commVolume.begin(), commVolume.end(), 0 );
        
        // 2 vectors of size k
        std::vector<IndexType> numBorderNodesPerBlock;  
        std::vector<IndexType> numInnerNodesPerBlock;
        
        std::tie( numBorderNodesPerBlock, numInnerNodesPerBlock ) = ITI::GraphUtils::getNumBorderInnerNodes( graph, partition, settings);
        
        //TODO: are num of boundary nodes needed ????         
        maxBoundaryNodes = *std::max_element( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end() );
        totalBoundaryNodes = std::accumulate( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end(), 0 );
        
        std::vector<ValueType> percentBorderNodesPerBlock( settings.numBlocks, 0);
		SCAI_ASSERT_EQ_ERROR( settings.numBlocks, numBorderNodesPerBlock.size(), "Vector size mismatch.");
		SCAI_ASSERT_EQ_ERROR( settings.numBlocks, numInnerNodesPerBlock.size(), "Vector size mismatch.");
		
        for(IndexType i=0; i<settings.numBlocks; i++){
            percentBorderNodesPerBlock[i] = (ValueType (numBorderNodesPerBlock[i]))/(numBorderNodesPerBlock[i]+numInnerNodesPerBlock[i]);
			if( std::isnan(percentBorderNodesPerBlock[i]) ){
					PRINT("\n\t\t\t WARNING: found NaN value for percentBnd for block " << i <<", probably is has no vertices.\n\n");
			}
			PRINT( percentBorderNodesPerBlock[i] );
        }
        
        maxBorderNodesPercent = *std::max_element( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end() );
        avgBorderNodesPercent = std::accumulate( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end(), 0.0 )/(ValueType(settings.numBlocks));
        
    }
};

//------------------------------------------------------------------------------------------------------------

inline void printMetricsShort(struct Metrics metrics, std::ostream& out){
	out << "gather" << std::endl;
	out << "timeTotal finalCut imbalance maxBnd totBnd maxCommVol totCommVol maxBndPercnt avgBndPercnt" << std::endl;
    out << metrics.timeFinalPartition<< " " \
		<< metrics.finalCut << " "\
		<< metrics.finalImbalance << " "\
		<< metrics.maxBoundaryNodes << " "\
		<< metrics.totalBoundaryNodes << " "\
		<< metrics.maxCommVolume << " "\
		<< metrics.totalCommVolume << " ";
	out << std::setprecision(6) << std::fixed;
	out <<  metrics.maxBorderNodesPercent << " " \
		<<  metrics.avgBorderNodesPercent \
		<< std::endl; 
}

//-------------------------------------------------------------------------------------------------------------


inline void printVectorMetrics( std::vector<struct Metrics>& metricsVec, std::ostream& out){
    
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    IndexType numRuns = metricsVec.size();
    
    if( comm->getRank()==0 ){
        out << "# times, input, migrAlgo, 1distr, kmeans, 2redis, prelim, localRef, total,    prel cut, finalcut, imbalance,    maxBnd, totalBnd,    maxCommVol, totalCommVol,    BorNodes max, avg  " << std::endl;
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
    IndexType maxBoundaryNodes = 0;
    IndexType totalBoundaryNodes = 0;
    ValueType sumMaxBorderNodesPerc = 0;
    ValueType sumAvgBorderNodesPerc = 0;

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
            out << std::setprecision(2) << std::fixed;
            out<< run << " ,       "<< thisMetric.inputTime << ",  " << maxTimeMigrationAlgo << ",  " << maxTimeFirstDistribution << ",  " << maxTimeKmeans << ",  " << maxTimeSecondDistribution << ",  " << maxTimePreliminary << ",  " << timeLocalRef << ",  "<< timeFinal << " , \t "\
            << thisMetric.preliminaryCut << ",  "<< thisMetric.finalCut << ",  " << thisMetric.finalImbalance << ",    "  \
            // << thisMetric.maxBlockGraphDegree << ",  " << thisMetric.totalBlockGraphEdges << " ," 
            << thisMetric.maxBoundaryNodes << ", " << thisMetric.totalBoundaryNodes << ",    " \
            << thisMetric.maxCommVolume << ",  " << thisMetric.totalCommVolume << ",    ";
            out << std::setprecision(6) << std::fixed;
            out << thisMetric.maxBorderNodesPercent << ",  " << thisMetric.avgBorderNodesPercent \
            << std::endl;
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
        sumMaxBorderNodesPerc += thisMetric.maxBorderNodesPercent;
        sumAvgBorderNodesPerc += thisMetric.avgBorderNodesPercent;
    }
    
    if( comm->getRank()==0 ){
        out << std::setprecision(2) << std::fixed;
        out << "average,  "\
            <<  ValueType (metricsVec[0].inputTime)<< ",  "\
            <<  ValueType(sumMigrAlgo)/numRuns<< ",  " \
            <<  ValueType(sumFirstDistr)/numRuns<< ",  " \
            <<  ValueType(sumKmeans)/numRuns<< ",  " \
            <<  ValueType(sumSecondDistr)/numRuns<< ",  " \
            <<  ValueType(sumPrelimanry)/numRuns<< ",  " \
            <<  ValueType(sumLocalRef)/numRuns<< ",  " \
            <<  ValueType(sumFinalTime)/numRuns<< ", \t " \
            <<  ValueType(sumPreliminaryCut)/numRuns<< ",  " \
            <<  ValueType(sumFinalCut)/numRuns<< ",  " \
            <<  ValueType(sumImbalace)/numRuns<< ",    " \
            <<  ValueType(sumMaxBnd)/numRuns<< ",  " \
            <<  ValueType(sumTotBnd)/numRuns<< ",    " \
            <<  ValueType(sumMaxCommVol)/numRuns<< ", " \
            <<  ValueType(sumTotCommVol)/numRuns<< ",    ";
            out << std::setprecision(6) << std::fixed;
            out <<  ValueType(sumMaxBorderNodesPerc)/numRuns<< ", " \
            <<  ValueType(sumAvgBorderNodesPerc)/numRuns  \
            << std::endl;
            
        out << std::setprecision(2) << std::fixed;
        out << "gather" << std::endl;
        out << "timeKmeans timeGeom timeGraph timeTotal prelCut finalCut imbalance maxBnd totBnd maxCommVol totCommVol maxBndPercnt avgBndPercnt" << std::endl;
        out <<  ValueType(sumKmeans)/numRuns<< " " \
            <<  ValueType(sumPrelimanry)/numRuns<< " " \
            <<  ValueType(sumLocalRef)/numRuns<< " " \
            <<  ValueType(sumFinalTime)/numRuns<< " " \
            <<  ValueType(sumPreliminaryCut)/numRuns<< " " \
            <<  ValueType(sumFinalCut)/numRuns<< " " \
            <<  ValueType(sumImbalace)/numRuns<< " " \
            <<  ValueType(sumMaxBnd)/numRuns<< " " \
            <<  ValueType(sumTotBnd)/numRuns<< " " \
            <<  ValueType(sumMaxCommVol)/numRuns<< " " \
            <<  ValueType(sumTotCommVol)/numRuns<< " ";
            out << std::setprecision(6) << std::fixed;
            out <<  ValueType(sumMaxBorderNodesPerc)/numRuns<< " " \
            <<  ValueType(sumAvgBorderNodesPerc)/numRuns  \
            << std::endl;        
    }
    
}


//-------------------------------------------------------------------------------------------------------------


inline void printVectorMetricsShort( std::vector<struct Metrics>& metricsVec, std::ostream& out){
    
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    IndexType numRuns = metricsVec.size();
    
    if( comm->getRank()==0 ){
        out << "timeTotal finalcut imbalance maxBnd totalBnd maxCommVol totalCommVol maxBndPercnt avgBndPercnt " << std::endl;
    }

    //ValueType sumKmeans = 0;
    //ValueType sumPrelimanry = 0; 
    //ValueType sumLocalRef = 0; 
    ValueType sumFinalTime = 0;
    
    IndexType sumFinalCut = 0;
    ValueType sumImbalace = 0;
    IndexType sumMaxBnd = 0;
    IndexType sumTotBnd = 0;
    IndexType sumMaxCommVol = 0;
    IndexType sumTotCommVol = 0;
    IndexType maxBoundaryNodes = 0;
    IndexType totalBoundaryNodes = 0;
    ValueType sumMaxBorderNodesPerc = 0;
    ValueType sumAvgBorderNodesPerc = 0;

    for(IndexType run=0; run<numRuns; run++){
        Metrics thisMetric = metricsVec[ run ];
        
        SCAI_ASSERT_EQ_ERROR(thisMetric.timeMigrationAlgo.size(), comm->getSize(), "Wrong vector size" );
        
        // for these time we have one measurement per PE and must make a max
        //ValueType maxTimeKmeans = *std::max_element( thisMetric.timeKmeans.begin(), thisMetric.timeKmeans.end() );
        //ValueType maxTimeSecondDistribution = *std::max_element( thisMetric.timeSecondDistribution.begin(), thisMetric.timeSecondDistribution.end() );
        //ValueType maxTimePreliminary = *std::max_element( thisMetric.timePreliminary.begin(), thisMetric.timePreliminary.end() );
        
        // these times are global, no need to max
        //ValueType timeLocalRef = timeFinal - maxTimePreliminary;
		ValueType timeFinal = thisMetric.timeFinalPartition;
        
        if( comm->getRank()==0 ){
            out << std::setprecision(2) << std::fixed;
            //out<< run <<  maxTimeKmeans << ",  " << maxTimeSecondDistribution << ",  " << maxTimePreliminary << ",  " << timeLocalRef << ",  ";
            out << timeFinal << "  ";
            //<< thisMetric.preliminaryCut << ",  "
			out << thisMetric.finalCut << "  " << thisMetric.finalImbalance << "  "  \
            << thisMetric.maxBoundaryNodes << " " << thisMetric.totalBoundaryNodes << "  " \
            << thisMetric.maxCommVolume << "  " << thisMetric.totalCommVolume << " ";
            out << std::setprecision(6) << std::fixed;
            out << thisMetric.maxBorderNodesPercent << " " << thisMetric.avgBorderNodesPercent \
            << std::endl;
        }
        
        //sumKmeans += maxTimeKmeans;
        //sumPrelimanry += maxTimePreliminary;
        //sumLocalRef += timeLocalRef;
        sumFinalTime += timeFinal;
        
        sumFinalCut += thisMetric.finalCut;
        sumImbalace += thisMetric.finalImbalance;
        sumMaxBnd += thisMetric.maxBoundaryNodes  ;
        sumTotBnd += thisMetric.totalBoundaryNodes ;
        sumMaxCommVol +=  thisMetric.maxCommVolume;
        sumTotCommVol += thisMetric.totalCommVolume;
        sumMaxBorderNodesPerc += thisMetric.maxBorderNodesPercent;
        sumAvgBorderNodesPerc += thisMetric.avgBorderNodesPercent;
    }
    
    if( comm->getRank()==0 ){
        out << "gather" << std::endl;
        out << "timeKmeans timeGeom timeGraph timeTotal prelCut finalCut imbalance maxBnd totBnd maxCommVol totCommVol maxBndPercnt avgBndPercnt" << std::endl;
		
        out << std::setprecision(2) << std::fixed;
        out << "average,  "\
            //<<  ValueType(sumKmeans)/numRuns<< "  " \
            //<<  ValueType(sumLocalRef)/numRuns<< ",  "  
            <<  ValueType(sumFinalTime)/numRuns<< " " \
            //<<  ValueType(sumPreliminaryCut)/numRuns<< ",  " 
            <<  ValueType(sumFinalCut)/numRuns<< " " \
            <<  ValueType(sumImbalace)/numRuns<< " " \
            <<  ValueType(sumMaxBnd)/numRuns<< " " \
            <<  ValueType(sumTotBnd)/numRuns<< " " \
            <<  ValueType(sumMaxCommVol)/numRuns<< " " \
            <<  ValueType(sumTotCommVol)/numRuns<< " ";
            out << std::setprecision(6) << std::fixed;
            out <<  ValueType(sumMaxBorderNodesPerc)/numRuns<< " " \
            <<  ValueType(sumAvgBorderNodesPerc)/numRuns  \
            << std::endl;
		
		
        
		/*
        out <<  ValueType(sumKmeans)/numRuns<< " " \
            <<  ValueType(sumPrelimanry)/numRuns<< " " \
            <<  ValueType(sumLocalRef)/numRuns<< " " \
            <<  ValueType(sumFinalTime)/numRuns<< " " \
            <<  ValueType(sumPreliminaryCut)/numRuns<< " " \
            <<  ValueType(sumFinalCut)/numRuns<< " " \
            <<  ValueType(sumImbalace)/numRuns<< " " \
            <<  ValueType(sumMaxBnd)/numRuns<< " " \
            <<  ValueType(sumTotBnd)/numRuns<< " " \
            <<  ValueType(sumMaxCommVol)/numRuns<< " " \
            <<  ValueType(sumTotCommVol)/numRuns<< " ";
            out << std::setprecision(6) << std::fixed;
            out <<  ValueType(sumMaxBorderNodesPerc)/numRuns<< " " \
            <<  ValueType(sumAvgBorderNodesPerc)/numRuns  \
            << std::endl;        
			*/
    }
    
}
