#pragma once

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
    
    ValueType inputTime;
    ValueType timeFinalPartition;
    ValueType reportTime;
    ValueType timeTotal;
    
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
        out << "# times: input, migrAlgo , 1redistr , k-means , 2redistr , prelim, localRef, total  , metrics:  prel cut, cut, imbalance  ,  BlGr maxDeg, edges  ,  CommVol max, total  ,  BorNodes max, avg  " << std::endl;
        
        out << std::setprecision(3) << std::fixed;
        out<<  "           "<< inputTime << ",  " << maxTimeMigrationAlgo << ",  " << maxTimeFirstDistribution << ",  " << maxTimeKmeans << ",  " << maxTimeSecondDistribution << ",  " << maxTimePreliminary << ",  " << timeLocalRef << " ,  "<< timeFinalPartition << " ,   "\
        << preliminaryCut << ",  "<< finalCut << ",  " << finalImbalance << " , "  \
        << maxBlockGraphDegree << ",  " << totalBlockGraphEdges << " , "  \
        << maxCommVolume << ",  " << totalCommVolume << " , ";
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

        // 2 vectors of size k
        std::vector<IndexType> numBorderNodesPerBlock;  
        std::vector<IndexType> numInnerNodesPerBlock;
        
        std::tie( numBorderNodesPerBlock, numInnerNodesPerBlock ) = ITI::GraphUtils::getNumBorderInnerNodes( graph, partition);
        
        maxCommVolume = *std::max_element( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end() );
        totalCommVolume = std::accumulate( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end(), 0 );
                
        std::vector<ValueType> percentBorderNodesPerBlock( settings.numBlocks, 0);
    
        for(IndexType i=0; i<settings.numBlocks; i++){
            percentBorderNodesPerBlock[i] = (ValueType (numBorderNodesPerBlock[i]))/(numBorderNodesPerBlock[i]+numInnerNodesPerBlock[i]);
        }
        
        maxBorderNodesPercent = *std::max_element( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end() );
        avgBorderNodesPercent = std::accumulate( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end(), 0.0 )/(ValueType(settings.numBlocks));
        
    }
};
