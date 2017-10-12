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
    //Metrics(){}
    
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
        
//        if( comm->getRank()==0 ){
            out << "# times: migrAlgo , 1redistr , k-means , 2redistr , total  , metrics:  cut, imbalance  ,  BlGr maxDeg, edges  ,  CommVol max, total  ,  BorNodes max, avg  " << std::endl;        
        
            out << std::setprecision(2) << std::fixed;
            out<<  "           "<< inputTime << ",  " << maxTimeMigrationAlgo << ",  " << maxTimeFirstDistribution << ",  " << maxTimeKmeans << ",  " << maxTimeSecondDistribution << ",  " << maxTimePreliminary << ",  " << timeLocalRef << " ,  "<< timeFinalPartition << " ,  \t   "\
            << preliminaryCut << ",  "<< finalCut << ",  " << finalImbalance << " , \t "  \
            << maxBlockGraphDegree << ",  " << totalBlockGraphEdges << " ,\t "  \
            << maxCommVolume << ",  " << totalCommVolume << " , \t ";
            out << std::setprecision(6) << std::fixed;
            out << maxBorderNodesPercent << ",  " << avgBorderNodesPercent \
            << std::endl;
//        }
        
    }
    
    void getMetrics( scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> partition, scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings ){
        
        finalCut = ITI::GraphUtils::computeCut(graph, partition, true);
        finalImbalance = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( partition, settings.numBlocks, nodeWeights );
        
        std::tie(maxBlockGraphDegree, totalBlockGraphEdges) = ITI::GraphUtils::computeBlockGraphComm<IndexType, ValueType>( graph, partition, settings.numBlocks );
        
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
