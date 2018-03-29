#pragma once

#include <algorithm>

#include <scai/lama.hpp>
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
    IndexType maxBoundaryNodes= 0;
    IndexType totalBoundaryNodes= 0;
    ValueType maxBorderNodesPercent= 0;
    ValueType avgBorderNodesPercent= 0;

    IndexType maxBlockDiameter = 0;
    IndexType avgBlockDiameter = 0;

    
    //constructor
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
        ValueType maxTimeMigrationAlgo = *std::max_element( timeMigrationAlgo.begin(), timeMigrationAlgo.end() );
        ValueType maxTimeConstructRedist = *std::max_element( timeConstructRedistributor.begin(), timeConstructRedistributor.end() );
        ValueType maxTimeFirstDistribution = *std::max_element( timeFirstDistribution.begin(), timeFirstDistribution.end() );
        ValueType maxTimeKmeans = *std::max_element( timeKmeans.begin(), timeKmeans.end() );
        ValueType maxTimeSecondDistribution = *std::max_element( timeSecondDistribution.begin(), timeSecondDistribution.end() );
        ValueType maxTimePreliminary = *std::max_element( timePreliminary.begin(), timePreliminary.end() );
            
        ValueType timeLocalRef = timeFinalPartition - maxTimePreliminary;
        
        //TODO: this is quite ugly. Refactor as dictionary with key-value-pairs, much more extensible.
        if( maxBlockGraphDegree==-1 ){
            out << " ### WARNING: setting dummy value -1 for expensive (and not used) metrics max and total blockGraphDegree ###" << std::endl;
        }else if (maxBlockGraphDegree==0 ){
            out << " ### WARNING: possibly not all metrics calculated ###" << std::endl;
        }
        out << "# times: input, migrAlgo , constRedist, 1redistr , k-means , 2redistr , prelim, localRef, total";
        out << ", metrics:  prel cut, cut, imbalance, maxCommVol, totalCommVol, maxDiameter, avgDiameter" << std::endl;

        auto oldprecision = out.precision();
        
        out << std::setprecision(3) << std::fixed;

        //times
        out << inputTime    << ", ";
        out << maxTimeMigrationAlgo << ", ";
        out << maxTimeConstructRedist << ", ";
        out << maxTimeFirstDistribution << ", ";
        out << maxTimeKmeans    << ", ";
        out << maxTimeSecondDistribution    << ", ";
        out << maxTimePreliminary   << ", ";
        out << timeLocalRef << ", ";
        out << timeFinalPartition   << ", ";

        //solution quality
        out << preliminaryCut   << ", ";
        out << finalCut << ", ";
        out << finalImbalance   << ", ";
        //out << maxBoundaryNodes << ", ";
        //out << totalBoundaryNodes   << ", ";
        out << maxCommVolume    << ", ";
        out << totalCommVolume  << ", ";
        out << maxBlockDiameter << ", ";
        out << avgBlockDiameter << ", ";

        out << std::setprecision(6) << std::fixed;
        //out << maxBorderNodesPercent << ", ";
        //out << avgBorderNodesPercent << ", ";
        out << std::endl;

        out.precision(oldprecision);
        
        
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
        
        std::tie( numBorderNodesPerBlock, numInnerNodesPerBlock ) = ITI::GraphUtils::getNumBorderInnerNodes( graph, partition);
        
        //TODO: are num of boundary nodes needed ????         
        maxBoundaryNodes = *std::max_element( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end() );
        totalBoundaryNodes = std::accumulate( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end(), 0 );
        
        std::vector<ValueType> percentBorderNodesPerBlock( settings.numBlocks, 0);
    
        for(IndexType i=0; i<settings.numBlocks; i++){
            percentBorderNodesPerBlock[i] = (ValueType (numBorderNodesPerBlock[i]))/(numBorderNodesPerBlock[i]+numInnerNodesPerBlock[i]);
        }
        
        maxBorderNodesPercent = *std::max_element( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end() );
        avgBorderNodesPercent = std::accumulate( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end(), 0.0 )/(ValueType(settings.numBlocks));
        
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
        const IndexType localN = dist->getLocalSize();

        if (settings.numBlocks == comm->getSize() && settings.computeDiameter) {
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
                maxBlockDiameter = comm->max(localDiameter);
                avgBlockDiameter = comm->sum(localDiameter) / comm->getSize();
            }

        }


    }
};
