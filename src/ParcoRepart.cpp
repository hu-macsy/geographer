/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include "ParcoRepart.h"

#include <assert.h>
#include <cmath>
#include <climits>
#include <queue>
#include <string>
#include <unordered_set>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <set>
#include <iostream>
#include <iomanip>

#include <scai/tracing.hpp>

#include "PrioQueue.h"
#include "HilbertCurve.h"
#include "MultiLevel.h"
#include "SpectralPartition.h"
#include "KMeans.h"
#include "AuxiliaryFunctions.h"
#include "MultiSection.h"
#include "GraphUtils.h"
#include "Mapping.h"



namespace ITI {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(
    CSRSparseMatrix<ValueType> &input,
    std::vector<DenseVector<ValueType>> &coordinates,
    Settings settings,
    struct Metrics& metrics)
{
    std::vector<DenseVector<ValueType> > uniformWeights(1);
    uniformWeights[0] = fill<DenseVector<ValueType>>(input.getRowDistributionPtr(), 1);
    return partitionGraph(input, coordinates, uniformWeights, settings, metrics);
}


template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(
    CSRSparseMatrix<ValueType> &input,
    std::vector<DenseVector<ValueType>> &coordinates,
    struct Settings settings) {

    struct Metrics metrics(settings);
    assert(settings.storeInfo == false); // Cannot return timing information. Better throw an error than silently drop it.

    return partitionGraph(input, coordinates, settings, metrics);
}

// overloaded version without a graph
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(
    std::vector<DenseVector<ValueType>> &coordinates,
    std::vector<DenseVector<ValueType>> &nodeWeights,
    struct Settings settings,
    struct Metrics& metrics) {

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if( settings.initialPartition!=ITI::Tool::geoSFC ) {
        if( comm->getRank()==0) {
            std::cout<< "Called ParcoRepart::partitionGraph without the graph as input argument but the tool to partition is "\
                     << settings.initialPartition << " and it require the graph. Call again by also providing the graph" << std::endl;
        }
        throw std::runtime_error("Graph not given but required.");
    }

    if( not settings.noRefinement ) {
        if( comm->getRank()==0) {
            std::cout << "The refinement flag is on but no graph is provided. Call again by also providing the graph" << std::endl;
        }
        throw std::runtime_error("Graph not given but required.");
    }

    const scai::dmemo::DistributionPtr dist = coordinates[0].getDistributionPtr();
    const IndexType N = dist->getGlobalSize();
    const scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));

    //generate dummy matrix
    scai::lama::CSRSparseMatrix<ValueType> graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDistPointer);

    return partitionGraph( graph, coordinates, nodeWeights, settings, metrics );
}


// overloaded version with metrics
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(
    CSRSparseMatrix<ValueType> &input,
    std::vector<DenseVector<ValueType>> &coordinates,
    std::vector<DenseVector<ValueType>> &nodeWeights,
    Settings settings,
    struct Metrics& metrics) {

    DenseVector<IndexType> previous;
    assert(!settings.repartition);

    CommTree<IndexType,ValueType> commTree;
    commTree.createFlatHomogeneous( settings.numBlocks );
    commTree.adaptWeights( nodeWeights );

    return partitionGraph(input, coordinates, nodeWeights, previous, commTree, settings, metrics);

}

//TODO: maybe we do not need that. But how to decide since we do not have
// the settings.blockSizes anymore?
//overloaded with blocksizes
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input,
        std::vector<DenseVector<ValueType>> &coordinates,
        std::vector<DenseVector<ValueType>> &nodeWeights,
        std::vector<std::vector<ValueType>> &blockSizes,
        Settings settings,
        struct Metrics& metrics) {

    DenseVector<IndexType> previous;
    assert(!settings.repartition);

    CommTree<IndexType,ValueType> commTree;
    commTree.createFlatHeterogeneous( blockSizes );

    return partitionGraph(input, coordinates, nodeWeights, previous, commTree, settings, metrics);

}

/* wrapper for input in metis-like format

*   vtxDist, size=numPEs,  is a replicated array, it is the prefix sum of the number of nodes per PE
        eg: [0, 15, 25, 50], PE0 has 15 vertices, PE1 10 and PE2 25
*   xadj, size=localN+1, (= IA array of the CSR sparse matrix format), is the prefix sum of the degrees
        of the local nodes, ie, how many non-zero values the row has.
*   adjncy, size=localM (number of local edges = the JA array), contains numbers >0 and <N, each
        number is the global id of the neighboring vertex
*/
template<typename IndexType, typename ValueType>
std::vector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(
    IndexType *vtxDist, IndexType *xadj, IndexType *adjncy, IndexType localM,
    IndexType *vwgt, IndexType dimensions, ValueType *xyz,
    Settings  settings, Metrics &metrics ) {

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();

    //SCAI_ASSERT_EQ_ERROR(numPEs+1, sizeof(vtxDist)/sizeof(IndexType), "wrong size for array vtxDist" );
    // ^^ this is wrong,  sizeof(vtxDist)=size of a pointer. How to check if size is correct?
    const IndexType N = vtxDist[numPEs];

    // how to check if array has the correct size?
    const IndexType localN = vtxDist[thisPE+1]-vtxDist[thisPE];
    SCAI_ASSERT_GT_ERROR( localN, 0, "Wrong value for localN for PE " << thisPE << ". Probably wrong vtxDist array");
    SCAI_ASSERT_EQ_ERROR( N, comm->sum(localN), "Global number of vertices mismatch");

    PRINT0("N= " << N);

    // contains the size of each part
    std::vector<IndexType> partSize( numPEs );
    for( int i=0; i<numPEs; i++) {
        partSize[i] = vtxDist[i+1]-vtxDist[i];
    }

    // pointer to the general block distribution created using the vtxDist array
    const auto genBlockDistPtr = genBlockDistributionBySizes( partSize, comm );

    //-----------------------------------------------------
    //
    // convert to scai data types
    //

    //
    // graph
    //

    scai::hmemo::HArray<IndexType> localIA(localN+1, xadj);
    scai::hmemo::HArray<IndexType> localJA(localM, adjncy);
    scai::hmemo::HArray<ValueType> localValues(localM, 1.0);      //TODO: weight 1.0=> no edge weights, change/generalize

    scai::lama::CSRStorage<ValueType> graphLocalStorage( localN, N, localIA, localJA, localValues);
    scai::lama::CSRSparseMatrix<ValueType> graph (genBlockDistPtr, graphLocalStorage);

    SCAI_ASSERT_EQ_ERROR( graph.getLocalNumRows(), localN, "Local size mismatch");
    SCAI_ASSERT_EQ_ERROR( genBlockDistPtr->getLocalSize(), localN, "Local size mismatch");

    //
    // coordinates
    //

    std::vector<std::vector<ValueType>> localCoords(dimensions);

    for (IndexType dim = 0; dim < dimensions; dim++) {
        localCoords[dim].resize(localN);
        for( int i=0; i<localN; i++) {
            localCoords[dim][i] = xyz[dimensions*i+dim];
        }
    }

    std::vector<scai::lama::DenseVector<ValueType>> coordinates(dimensions);
    for (IndexType dim = 0; dim < dimensions; dim++) {
        coordinates[dim] = scai::lama::DenseVector<ValueType>(genBlockDistPtr, scai::hmemo::HArray<ValueType>(localN, localCoords[dim].data()) );
    }

    //
    // node weights
    //
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(1, scai::lama::DenseVector<ValueType>(genBlockDistPtr, scai::hmemo::HArray<ValueType>(localN, *vwgt)));

    scai::lama::DenseVector<IndexType> localPartitionDV = partitionGraph( graph, coordinates, nodeWeights, settings, metrics);

    //WARNING: must check that is correct
    localPartitionDV.redistribute( graph.getRowDistributionPtr() );

    //copy the local values to a std::vector and return
    scai::hmemo::ReadAccess<IndexType> localPartRead ( localPartitionDV.getLocalValues() );
    std::vector<IndexType> localPartition( localPartRead.get(), localPartRead.get()+ localN );
    return localPartition;
}

//-------------------------------------------------------------------------------------------------

//the core implementation
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(
    CSRSparseMatrix<ValueType> &input,
    std::vector<DenseVector<ValueType>> &coordinates,
    std::vector<DenseVector<ValueType>> &nodeWeights,
    DenseVector<IndexType>& previous,
    CommTree<IndexType,ValueType> commTree,
    Settings settings,
    struct Metrics& metrics)
{
    IndexType k = settings.numBlocks;
    ValueType epsilon = settings.epsilon;
    const IndexType dimensions = coordinates.size();

    SCAI_REGION( "ParcoRepart.partitionGraph" )

    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

    SCAI_REGION_START("ParcoRepart.partitionGraph.inputCheck")
    /*
    * check input arguments for sanity
    */
    IndexType n = input.getNumRows();
    for( int d=0; d<dimensions; d++) {
        if (n != coordinates[d].size()) {
            throw std::runtime_error("Matrix has " + std::to_string(n) + " rows, but " + std::to_string(coordinates[0].size())
                                     + " coordinates are given.");
        }
    }

    if (n != input.getNumColumns()) {
        throw std::runtime_error("Matrix must be quadratic.");
    }

    if (!input.isConsistent()) {
        throw std::runtime_error("Input matrix inconsistent");
    }

    if (k > n) {
        throw std::runtime_error("Creating " + std::to_string(k) + " blocks from " + std::to_string(n) + " elements is impossible.");
    }

    if (epsilon < 0) {
        throw std::runtime_error("Epsilon " + std::to_string(epsilon) + " is invalid.");
    }

    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
    const IndexType rank = comm->getRank();

    if (!coordDist->isEqual( *inputDist) ) {
        throw std::runtime_error( "Distributions should be equal.");
    }

    bool nodesUnweighted = nodeWeights.size() == 1 && (nodeWeights[0].max() == nodeWeights[0].min());

    SCAI_REGION_END("ParcoRepart.partitionGraph.inputCheck")
    {
        SCAI_REGION("ParcoRepart.synchronize")
        comm->synchronize();
    }

    if (settings.repartition && nodeWeights.size() > 1) {
        throw std::logic_error("Repartitioning not implemented for multiple node weights.");
    }

    if (settings.initialPartition == ITI::Tool::geoMS && nodeWeights.size() > 1) {
        throw std::logic_error("MultiSection not implemented for multiple weights.");
    }

    // get an initial partition
    DenseVector<IndexType> result;

    for (IndexType i = 0; i < nodeWeights.size(); i++) {
        assert(nodeWeights[i].getDistribution().isEqual(*inputDist));
    }

    //-------------------------
    //
    // timing info

    std::chrono::duration<double> kMeansTime = std::chrono::duration<double>(0.0);
    std::chrono::duration<double> migrationCalculation = std::chrono::duration<double> (0.0);
    std::chrono::duration<double> migrationTime= std::chrono::duration<double>(0.0);
    std::chrono::duration<double> secondRedistributionTime = std::chrono::duration<double>(0.0) ;
    std::chrono::duration<double> partitionTime= std::chrono::duration<double>(0.0);

    std::chrono::time_point<std::chrono::system_clock> beforeInitPart =  std::chrono::system_clock::now();

    if( settings.initialPartition==ITI::Tool::geoSFC) {
        PRINT0("Initial partition with SFCs");
        result= HilbertCurve<IndexType, ValueType>::computePartition(coordinates, settings);
        std::chrono::duration<double> sfcTime = std::chrono::system_clock::now() - beforeInitPart;
        if ( settings.verbose ) {
            ValueType totSFCTime = ValueType(comm->max(sfcTime.count()) );
            if(comm->getRank() == 0)
                std::cout << "SFC Time:" << totSFCTime << std::endl;
        }
    }
    else if (settings.initialPartition == ITI::Tool::geoKmeans or settings.initialPartition == ITI::Tool::geoHierKM \
             or  settings.initialPartition == ITI::Tool::geoHierRepart) {
        if (comm->getRank() == 0) {
            std::cout << "Initial partition with K-Means" << std::endl;
        }

        //prepare coordinates for k-means
        std::vector<DenseVector<ValueType> > coordinateCopy = coordinates;
        std::vector<DenseVector<ValueType> > nodeWeightCopy = nodeWeights;
        if (comm->getSize() > 1 && (settings.dimensions == 2 || settings.dimensions == 3)) {
            SCAI_REGION("ParcoRepart.partitionGraph.initialPartition.prepareForKMeans")

            if (!settings.repartition || comm->getSize() != settings.numBlocks) {

                if (settings.initialMigration != ITI::Tool::geoSFC) {
                    throw std::logic_error("KMeans depends on pre-sorting with space filling curves.");
                }

                HilbertCurve<IndexType,ValueType>::redistribute(coordinateCopy, nodeWeightCopy, settings, metrics);
            }
        }

        std::vector<ValueType> weightSum(nodeWeights.size());
        for (int i = 0; i < nodeWeights.size(); i++) {
            weightSum[i] = nodeWeights[i].sum();
        }

        // vector of size k, each element represents the size of one block
        std::vector<std::vector<ValueType>> blockSizes = commTree.getBalanceVectors(1);
        SCAI_ASSERT_EQ_ERROR( blockSizes.size(), nodeWeights.size(), "Wrong number of weights");
        SCAI_ASSERT_EQ_ERROR( blockSizes[0].size(), settings.numBlocks, "Wrong size of weights" );
        if( blockSizes.empty() ) {
            blockSizes = std::vector<std::vector<ValueType> >(nodeWeights.size());
            for (int i = 0; i < nodeWeights.size(); i++) {
                blockSizes[i].assign( settings.numBlocks, std::ceil(weightSum[i]/settings.numBlocks) );
            }
        }

        std::chrono::time_point<std::chrono::system_clock> beforeKMeans =  std::chrono::system_clock::now();

        if (settings.repartition) {
            result = ITI::KMeans::computeRepartition<IndexType, ValueType>(coordinateCopy, nodeWeightCopy, blockSizes, previous, settings);
        } else if (settings.initialPartition == ITI::Tool::geoKmeans) {
            result = ITI::KMeans::computePartition<IndexType, ValueType>(coordinateCopy, nodeWeightCopy, blockSizes, settings, metrics);
        } else if (settings.initialPartition == ITI::Tool::geoHierKM or settings.initialPartition == ITI::Tool::geoHierRepart) {
            const IndexType numWeights = nodeWeightCopy.size();
            SCAI_ASSERT_GT_ERROR( settings.hierLevels.size(), 0, "Must provide the tree level sizes in order to call hierarchical KMeans" );
            //TODO: adapt also for when tree is given in a file

            CommTree<IndexType,ValueType> commTree( settings.hierLevels, numWeights );
            const IndexType numLeaves = commTree.getNumLeaves();
            PRINT0("About to call hierarchical kmeans with " << settings.hierLevels.size() << " levels and " << numLeaves << " leaves." );
            SCAI_ASSERT_EQ_ERROR( numLeaves, settings.numBlocks, "The number of leaves and blocks should agree" );

            commTree.adaptWeights( nodeWeightCopy );

            if (settings.initialPartition == ITI::Tool::geoHierKM) {
                result = ITI::KMeans::computeHierarchicalPartition<IndexType, ValueType>( coordinateCopy, nodeWeightCopy, commTree, settings, metrics);
            }
            if (settings.initialPartition == ITI::Tool::geoHierRepart) {
                //settings.debugMode = true;
                result = ITI::KMeans::computeHierPlusRepart<IndexType, ValueType>( coordinateCopy, nodeWeightCopy, commTree, settings, metrics);
            }
            SCAI_ASSERT_EQ_ERROR( nodeWeightCopy[0].getDistributionPtr()->getLocalSize(),\
                                  result.getDistributionPtr()->getLocalSize(), "Partition distribution mismatch(?)");
        }

        kMeansTime = std::chrono::system_clock::now() - beforeKMeans;
        metrics.MM["timeKmeans"] = kMeansTime.count();
        //timeForKmeans = ValueType ( comm->max(kMeansTime.count() ));
        assert(scai::utilskernel::HArrayUtils::min(result.getLocalValues()) >= 0);
        SCAI_ASSERT_LT_ERROR(scai::utilskernel::HArrayUtils::max(result.getLocalValues()),k, "");

        if (settings.verbose) {
            ValueType totKMeansTime = ValueType( comm->max(kMeansTime.count()) );
            if(comm->getRank() == 0)
                std::cout << "K-Means, Time:" << totKMeansTime << std::endl;
        }

        SCAI_ASSERT_EQ_ERROR( result.max(), settings.numBlocks -1, "Wrong index in partition" );
        //assert(result.max() == settings.numBlocks -1);
        assert(result.min() == 0);

    } else if (settings.initialPartition == ITI::Tool::geoMS) {// multisection
        PRINT0("Initial partition with multisection");
        if( not settings.cutsPerDim.empty() ) { // specific cuts per dimension are provided
            IndexType k = std::accumulate( settings.cutsPerDim.begin(), settings.cutsPerDim.end(), 1, std::multiplies<IndexType>() );
            if( settings.numBlocks!=k ) {
                PRINT0("Input argument numBlocks= " << settings.numBlocks << " but the cutsPerDim provided will partition into "\
                       << k << " blocks. These values should agree.\nAborting...");
                throw std::runtime_error("Input argument values do not agree");
            }
        }

        DenseVector<ValueType> convertedWeights(nodeWeights[0]);
        result = ITI::MultiSection<IndexType, ValueType>::computePartition(input, coordinates, convertedWeights, settings);
        std::chrono::duration<double> msTime = std::chrono::system_clock::now() - beforeInitPart;

        if ( settings.verbose ) {
            ValueType totMsTime = ValueType ( comm->max(msTime.count()) );
            if(comm->getRank() == 0)
                std::cout << "MS Time:" << totMsTime << std::endl;
        }
    } else if (settings.initialPartition == ITI::Tool::none) {
        //no need to explicitly check for repartitioning mode or not.
        assert(comm->getSize() == settings.numBlocks);
        result = DenseVector<IndexType>(input.getRowDistributionPtr(), comm->getRank());
    }
    else {
        throw std::runtime_error("Initial Partitioning mode unsupported.");
    }

    //-----------------------------------------------------------
    //
    // At this point we have the initial, geometric partition.
    //

    // if noRefinement then these are the times, if we do refinement they will be overwritten
    partitionTime =  std::chrono::system_clock::now() - beforeInitPart;
    metrics.MM["timePreliminary"] = partitionTime.count();

    if (comm->getSize() == k) {
        //WARNING: the result  is not redistributed. must redistribute afterwards
        if( !settings.noRefinement ) {

            //uncomment to store the first, geometric partition into a file that then can be visualized using matlab and GPI's code
            //std::string filename = "geomPart.mtx";
            //result.writeToFile( filename );

            SCAI_REGION("ParcoRepart.partitionGraph.initialRedistribution");
            if (nodeWeights.size() > 1) {
                throw std::logic_error("Local refinement not yet implemented for multiple weights.");
            }
            /*
             * redistribute to prepare for local refinement
             */
            bool useRedistributor = true;
            aux<IndexType, ValueType>::redistributeFromPartition( result, input, coordinates, nodeWeights[0], settings, useRedistributor);

            partitionTime =  std::chrono::system_clock::now() - beforeInitPart;
            ValueType cut = GraphUtils<IndexType,ValueType>::computeCut( input, result, true);
            ValueType imbalance = GraphUtils<IndexType, ValueType>::computeImbalance(result, k, nodeWeights[0]);


            //-----------------------------------------------------------
            //
            // output: in std and file
            //

            //now, every PE store its own times. These will be maxed afterwards, before printing in Metrics
            ValueType timeToCalcInitMigration = migrationCalculation.count();
            ValueType timeForFirstRedistribution = migrationTime.count();
            ValueType timeForKmeans = kMeansTime.count();
            ValueType timeForSecondRedistr = secondRedistributionTime.count();
            ValueType timeForInitPart = partitionTime.count();

            metrics.MM["timeSecondDistribution"] = timeForSecondRedistr;
            metrics.MM["timePreliminary"] = timeForInitPart;
            metrics.MM["preliminaryCut"] = cut;
            metrics.MM["preliminaryImbalance"] = imbalance;

            if (settings.verbose ) {
                //TODO: do this comm->max here? otherwise it is just PE 0 times
                timeToCalcInitMigration = comm->max(migrationCalculation.count()) ;
                timeForFirstRedistribution = comm->max( migrationTime.count() );
                timeForKmeans = comm->max( kMeansTime.count() );
                timeForSecondRedistr = comm->max( secondRedistributionTime.count() );
                timeForInitPart = comm->max( partitionTime.count() );
                if(comm->getRank() == 0 ) {
                    std::cout<< std::endl << "\033[1;32mTiming: migration algo: "<< timeToCalcInitMigration << ", 1st redistr: " << timeForFirstRedistribution << ", only k-means: " << timeForKmeans <<", only 2nd redistr: "<< timeForSecondRedistr <<", total:" << timeForInitPart << std::endl;
                    std::cout << "# of cut edges:" << cut << ", imbalance:" << imbalance<< " \033[0m" <<std::endl << std::endl;
                }
            }

            SCAI_REGION_START("ParcoRepart.partitionGraph.multiLevelStep")
            scai::dmemo::HaloExchangePlan halo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(input);
            ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(input, result, nodeWeights[0], coordinates, halo, settings, metrics);
            SCAI_REGION_END("ParcoRepart.partitionGraph.multiLevelStep")
        }
    } else {
        result.redistribute(inputDist);
        if (comm->getRank() == 0 && !settings.noRefinement) {
            std::cout << "Local refinement only implemented for one block per process. Called with " << comm->getSize() << " processes and " << k << " blocks." << std::endl;
        }
    }

    std::chrono::duration<double> elapTime = std::chrono::system_clock::now() - startTime;
    metrics.MM["timeTotal"] = elapTime.count();

    //possible mapping at the end
    if( settings.mappingRenumbering ) {
        PRINT0("Applying renumbering of blocks based on the SFC index of their centers.");
        std::chrono::time_point<std::chrono::system_clock> startRnb = std::chrono::system_clock::now();

        if( not result.getDistribution().isEqual(coordinates[0].getDistribution()) ) {
            PRINT0("WARNING:\nCoordinates and partition do not have the same distribution.\nRedistributing coordinates to match distribution");
            for( int d=0; d<dimensions; d++) {
                coordinates[d].redistribute( result.getDistributionPtr() );
            }
        }
        Mapping<IndexType,ValueType>::applySfcRenumber( coordinates, nodeWeights, result, settings );

        std::chrono::duration<double> elapTime = std::chrono::system_clock::now() - startRnb;
        PRINT0("renumbering time " << elapTime.count() );
    }

    return result;
} //partitionGraph
//---------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::pixelPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings) {
    SCAI_REGION( "ParcoRepart.pixelPartition" )

    SCAI_REGION_START("ParcoRepart.pixelPartition.initialise")
    std::chrono::time_point<std::chrono::steady_clock> start, round;
    start = std::chrono::steady_clock::now();

    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();

    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    const IndexType localN = coordDist->getLocalSize();
    const IndexType globalN = coordDist->getGlobalSize();

    if (k != comm->getSize() && comm->getRank() == 0) {
        throw std::logic_error("Pixel partition only implemented for same number of blocks and processes.");
    }

    std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    DenseVector<IndexType> result(coordDist, 0);

    //TODO: probably minimum is not needed
    //TODO: if we know maximum from the input we could save that although is not too costly

    /*
     * get minimum / maximum of local coordinates
     */
    for (IndexType dim = 0; dim < dimensions; dim++) {
        //get local parts of coordinates
        scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[dim].getLocalValues() );
        for (IndexType i = 0; i < localN; i++) {
            ValueType coord = localPartOfCoords[i];
            if (coord < minCoords[dim]) minCoords[dim] = coord;
            if (coord > maxCoords[dim]) maxCoords[dim] = coord;
        }
    }

    /*
     * communicate to get global min / max
     */
    for (IndexType dim = 0; dim < dimensions; dim++) {
        minCoords[dim] = comm->min(minCoords[dim]);
        maxCoords[dim] = comm->max(maxCoords[dim]);
    }

    // measure density with rounding
    // have to handle 2D and 3D cases seperately
    const IndexType sideLen = settings.pixeledSideLen;
    const IndexType cubeSize = std::pow(sideLen, dimensions);

    //TODO: generalize this to arbitrary dimensions, do not handle 2D and 3D differently
    //TODO: by a  for(int d=0; d<dimension; d++){ ... }
    // a 2D or 3D arrays as a one dimensional vector
    // [i][j] is in position: i*sideLen + j
    // [i][j][k] is in: i*sideLen*sideLen + j*sideLen + k

    //std::vector<IndexType> density( cubeSize ,0);
    scai::hmemo::HArray<IndexType> density( cubeSize, IndexType(0) );
    scai::hmemo::WriteAccess<IndexType> wDensity(density);

    SCAI_REGION_END("ParcoRepart.pixelPartition.initialise")

    if(dimensions==2) {
        SCAI_REGION( "ParcoRepart.pixelPartition.localDensity" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );

        IndexType scaledX, scaledY;
        //the +1 is needed
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;

        for(IndexType i=0; i<localN; i++) {
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            IndexType pixelInd = scaledX*sideLen + scaledY;
            SCAI_ASSERT( pixelInd < wDensity.size(), "Index too big: "<< std::to_string(pixelInd) );
            ++wDensity[pixelInd];
        }
    } else if(dimensions==3) {
        SCAI_REGION( "ParcoRepart.pixelPartition.localDensity" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );

        IndexType scaledX, scaledY, scaledZ;

        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;
        IndexType maxZ = maxCoords[2]+1;

        for(IndexType i=0; i<localN; i++) {
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            scaledZ = coordAccess2[i]/maxZ * sideLen;
            IndexType pixelInd = scaledX*sideLen*sideLen + scaledY*sideLen + scaledZ;

            SCAI_ASSERT( pixelInd < wDensity.size(), "Index too big: "<< std::to_string(pixelInd) );
            ++wDensity[pixelInd];
        }
    } else {
        throw std::runtime_error("Available only for 2D and 3D. Data given have dimension:" + std::to_string(dimensions) );
    }
    wDensity.release();

    // sum density from all PEs
    {
        SCAI_REGION( "ParcoRepart.pixelPartition.sumDensity" )
        comm->sumArray( density );
    }

    //TODO: is that needed? we just can overwrite density array.
    // use the summed density as a Dense vector
    scai::lama::DenseVector<IndexType> sumDensity( density );

    //
    //using the summed density get an initial pixeled partition

    std::vector<IndexType> pixeledPartition( density.size(), -1);

    IndexType pointsLeft= globalN;
    IndexType pixelsLeft= cubeSize;
    IndexType maxBlockSize = globalN/k * 1.02; // allowing some imbalance
    PRINT0("max allowed block size: " << maxBlockSize );
    IndexType thisBlockSize;

    //for all the blocks
    for(IndexType block=0; block<k; block++) {
        SCAI_REGION( "ParcoRepart.pixelPartition.localPixelGrowing")

        ValueType averagePointsPerPixel = ValueType(pointsLeft)/pixelsLeft;
        // a factor to force the block to spread more
        ValueType spreadFactor;
        // make a block spread towards the borders (and corners) of our input space
        ValueType geomSpread;
        // to measure the distance from the first, center pixel
        ValueType pixelDistance;

        // start from the densest pixel
        //IndexType maxDensityPixel = std::distance( sumDensity.begin(), std::max_element(sumDensity.begin(), sumDensity.end()) );

        //TODO: sumDensity is local/not distributed. No need for that, just to avoid getValue.
        scai::hmemo::WriteAccess<IndexType> localSumDens( sumDensity.getLocalValues() );

        //TODO: bad way to do that. linear time for every block. maybe sort or use a priority queue
        IndexType maxDensityPixel=-1;
        IndexType maxDensity=-1;
        for(IndexType ii=0; ii<sumDensity.size(); ii++) {
            if(localSumDens[ii]>maxDensity) {
                maxDensityPixel = ii;
                maxDensity= localSumDens[ii];
            }
        }

        if(maxDensityPixel<0) {
            PRINT0("Max density pixel id = -1. Should not happen(?) or pixels are finished. For block "<< block<< " and k= " << k);
            break;
        }

        SCAI_ASSERT(maxDensityPixel < sumDensity.size(), "Too big index: " + std::to_string(maxDensityPixel));
        SCAI_ASSERT(maxDensityPixel >= 0, "Negative index: " + std::to_string(maxDensityPixel));
        spreadFactor = averagePointsPerPixel/localSumDens[ maxDensityPixel ];

        //TODO: change to more appropriate data type
        // insert all the neighbouring pixels
        std::vector<std::pair<IndexType, ValueType>> border;
        std::vector<IndexType> neighbours = ParcoRepart<IndexType, ValueType>::neighbourPixels( maxDensityPixel, sideLen, dimensions);

        // insert in border if not already picked
        for(IndexType j=0; j<neighbours.size(); j++) {
            // make sure this neighbour does not belong to another block
            if(localSumDens[ neighbours[j]] != -1 ) {
                std::pair<IndexType, ValueType> toInsert;
                toInsert.first = neighbours[j];
                SCAI_ASSERT(neighbours[j] < sumDensity.size(), "Too big index: " + std::to_string(neighbours[j]));
                SCAI_ASSERT(neighbours[j] >= 0, "Negative index: " + std::to_string(neighbours[j]));
                geomSpread = 1 + 1/std::log2(sideLen)*( std::abs(sideLen/2 - neighbours[j]/sideLen)/(0.8*sideLen/2) + std::abs(sideLen/2 - neighbours[j]%sideLen)/(0.8*sideLen/2) );
                //PRINT0( geomSpread );
                // value to pick a border node
                pixelDistance = aux<IndexType, ValueType>::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);
                toInsert.second = (1/pixelDistance)* geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[maxDensityPixel], 0.5) );
                border.push_back(toInsert);
            }
        }
        thisBlockSize = localSumDens[maxDensityPixel];

        pixeledPartition[maxDensityPixel] = block;

        // set this pixel to -1 so it is not picked again
        localSumDens[maxDensityPixel] = -1;


        while(border.size() !=0 ) {     // there are still pixels to check

            //TODO: different data type to avoid that
            // sort border by the value in increasing order
            std::sort( border.begin(), border.end(),
            [](const std::pair<IndexType, ValueType> &left, const std::pair<IndexType, ValueType> &right) {
                return left.second < right.second;
            });

            std::pair<IndexType, ValueType> bestPixel;
            IndexType bestIndex=-1;
            do {
                bestPixel = border.back();
                border.pop_back();
                bestIndex = bestPixel.first;

            } while( localSumDens[ bestIndex] +thisBlockSize > maxBlockSize and border.size()>0); // this pixel is too big

            // picked last pixel in border but is too big
            if(localSumDens[ bestIndex] +thisBlockSize > maxBlockSize ) {
                break;
            }
            SCAI_ASSERT(localSumDens[ bestIndex ] != -1, "Wrong pixel choice.");

            // this pixel now belongs in this block
            SCAI_ASSERT(bestIndex < sumDensity.size(), "Wrong pixel index: " + std::to_string(bestIndex));
            pixeledPartition[ bestIndex ] = block;
            thisBlockSize += localSumDens[ bestIndex ];
            --pixelsLeft;
            pointsLeft -= localSumDens[ bestIndex ];

            //averagePointsPerPixel = ValueType(pointsLeft)/pixelsLeft;
            //spreadFactor = localSumDens[ bestIndex ]/averagePointsPerPixel;
            //spreadFactor = (k-block)*averagePointsPerPixel/localSumDens[ bestIndex ];
            spreadFactor = averagePointsPerPixel/localSumDens[ bestIndex ];

            //get the neighbours of the new pixel
            std::vector<IndexType> neighbours = ParcoRepart<IndexType, ValueType>::neighbourPixels( bestIndex, sideLen, dimensions);

            //insert neighbour in border or update value if already there
            for(IndexType j=0; j<neighbours.size(); j++) {

                SCAI_ASSERT(neighbours[j] < sumDensity.size(), "Too big index: " + std::to_string(neighbours[j]));
                SCAI_ASSERT(neighbours[j] >= 0, "Negative index: " + std::to_string(neighbours[j]));

                //geomSpread = 1 + 1.0/detailLvl*( std::abs(sideLen/2.0 - neighbours[j]/sideLen)/(0.8*sideLen/2.0) + std::abs(sideLen/2.0 - neighbours[j]%sideLen)/(0.8*sideLen/2.0) );
                IndexType ngbrX = neighbours[j]/sideLen;
                IndexType ngbrY = neighbours[j]%sideLen;

                geomSpread= 1+ (std::pow(ngbrX-sideLen/2, 2) + std::pow(ngbrY-sideLen/2, 2))*(2/std::pow(sideLen,2));
                //geomSpread = geomSpread * geomSpread;// std::pow(geomSpread, 0.5);
                //
                geomSpread = 1;
                //

                if (localSumDens[ neighbours[j]] == -1) { // this pixel is already picked by a block (maybe this)
                    continue;
                } else {
                    bool inBorder = false;

                    for (IndexType l=0; l<border.size(); l++) {
                        if( border[l].first == neighbours[j]) { // its already in border, update value
                            //border[l].second = 1.3*border[l].second + geomSpread * (spreadFactor*(std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5) );
                            pixelDistance = aux<IndexType, ValueType>::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);
                            border[l].second += geomSpread*  (1/(pixelDistance*pixelDistance))* ( spreadFactor *std::pow(localSumDens[neighbours[j]], 0.5) + std::pow(localSumDens[bestIndex], 0.5) );
                            inBorder= true;
                        }
                    }
                    if (!inBorder) {
                        std::pair<IndexType, ValueType> toInsert;
                        toInsert.first = neighbours[j];
                        //toInsert.second = geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5));
                        pixelDistance = aux<IndexType, ValueType>::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);
                        //toInsert.second = (1/(pixelDistance*pixelDistance))* geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5));
                        toInsert.second = geomSpread*  (1/(pixelDistance*pixelDistance))* ( spreadFactor *(std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5) );
                        //toInsert.second = geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5))/(std::pow( std::abs( localSumDens[bestIndex] - localSumDens[neighbours[j]]),0.5));
                        border.push_back(toInsert);
                    }
                }
            }

            localSumDens[ bestIndex ] = -1;
        }
        //PRINT0("##### final blockSize for block "<< block << ": "<< thisBlockSize);
    } // for(IndexType block=0; block<k; block++)

    // assign all orphan pixels to last block
    for(unsigned long int pp=0; pp<pixeledPartition.size(); pp++) {
        scai::hmemo::ReadAccess<IndexType> localSumDens( sumDensity.getLocalValues() );
        if(pixeledPartition[pp] == -1) {
            pixeledPartition[pp] = k-1;
            thisBlockSize += localSumDens[pp];
        }
    }
    //PRINT0("##### final blockSize for block "<< k-1 << ": "<< thisBlockSize);

    /*
     * here all pixels should have a partition
    */

    // set your local part of the partition/result
    scai::hmemo::WriteOnlyAccess<IndexType> wLocalPart ( result.getLocalValues() );

    if(dimensions==2) {
        SCAI_REGION( "ParcoRepart.pixelPartition.setLocalPartition" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );

        IndexType scaledX, scaledY;
        //the +1 is needed
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;

        for(IndexType i=0; i<localN; i++) {
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            IndexType densInd = scaledX*sideLen + scaledY;
            //PRINT(densInd << " # " << coordAccess0[i] << " _ " << coordAccess1[i] );
            SCAI_ASSERT( densInd < density.size(), "Index too big: "<< std::to_string(densInd) );

            wLocalPart[i] = pixeledPartition[densInd];
            SCAI_ASSERT(wLocalPart[i] < k, " Wrong block number: " + std::to_string(wLocalPart[i] ) );
        }
    } else if(dimensions==3) {
        SCAI_REGION( "ParcoRepart.pixelPartition.setLocalPartition" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );

        IndexType scaledX, scaledY, scaledZ;

        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;
        IndexType maxZ = maxCoords[2]+1;

        for(IndexType i=0; i<localN; i++) {
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            scaledZ = coordAccess2[i]/maxZ * sideLen;
            IndexType densInd = scaledX*sideLen*sideLen + scaledY*sideLen + scaledZ;

            SCAI_ASSERT( densInd < density.size(), "Index too big: "<< std::to_string(densInd) );
            wLocalPart[i] = pixeledPartition[densInd];
            SCAI_ASSERT(wLocalPart[i] < k, " Wrong block number: " + std::to_string(wLocalPart[i] ) );
        }
    } else {
        throw std::runtime_error("Available only for 2D and 3D. Data given have dimension:" + std::to_string(dimensions) );
    }
    wLocalPart.release();

    return result;
}

//-----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<DenseVector<IndexType>> ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( CSRSparseMatrix<ValueType> &adjM, Settings settings) {
    IndexType N= adjM.getNumRows();
    SCAI_REGION("ParcoRepart.getCommunicationPairs_local");
    // coloring.size()=3: coloring(i,j,c) means that edge with endpoints i and j is colored with color c.
    // and coloring[i].size()= number of edges in input graph

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    assert(adjM.getNumColumns() == adjM.getNumRows() );
    IndexType colors;
    std::vector<std::vector<IndexType>> coloring;
    {
        std::chrono::time_point<std::chrono::system_clock> beforeColoring =  std::chrono::system_clock::now();
        if (!adjM.getRowDistributionPtr()->isReplicated()) {
            //PRINT0("***WARNING: In getCommunicationPairs_local: given graph is not replicated; will replicate now");
            const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));
            adjM.redistribute(noDist, noDist);
            //throw std::runtime_error("Input matrix must be replicated.");
        }
        //here graph is replicated. PE0 will write it in a file

        coloring = GraphUtils<IndexType, ValueType>::mecGraphColoring( adjM, colors); // our implementation

        std::chrono::duration<double> coloringTime = std::chrono::system_clock::now() - beforeColoring;
        ValueType maxTime = comm->max( coloringTime.count() );
        ValueType minTime = comm->min( coloringTime.count() );
        if (settings.verbose) PRINT0("coloring done in time " << minTime << " -- " << maxTime << ", using " << colors << " colors" );
    }
    std::vector<DenseVector<IndexType>> retG(colors);

    if (adjM.getNumRows()==2) {
        assert(colors<=1);
        assert(coloring[0].size()<=1);
    }

    for(IndexType i=0; i<colors; i++) {
        retG[i].allocate(N);
        // TODO: although not distributed maybe try to avoid setValue, change to std::vector ?
        // initialize so retG[i][j]= j instead of -1
        for( IndexType j=0; j<N; j++) {
            retG[i].setValue( j, j );
        }
    }

    // for all the edges:
    // coloring[0][i] = the first block , coloring[1][i] = the second block,
    // coloring[2][i]= the color/round in which the two blocks shall communicate
    for(IndexType i=0; i<coloring[0].size(); i++) {
        IndexType color = coloring[2][i]; // the color/round of this edge
        //assert(color<colors);
        SCAI_ASSERT_LT_ERROR( color, colors, "Wrong number of colors?");
        IndexType firstBlock = coloring[0][i];
        IndexType secondBlock = coloring[1][i];
        retG[color].setValue( firstBlock, secondBlock);
        retG[color].setValue( secondBlock, firstBlock );
    }

    return retG;
}
//---------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
std::vector<IndexType> ParcoRepart<IndexType, ValueType>::neighbourPixels(const IndexType thisPixel, const IndexType sideLen, const IndexType dimension) {
    SCAI_REGION("ParcoRepart.neighbourPixels");

    SCAI_ASSERT(thisPixel>=0, "Negative pixel value: " << std::to_string(thisPixel));
    SCAI_ASSERT(sideLen> 0, "Negative or zero side length: " << std::to_string(sideLen));
    SCAI_ASSERT(sideLen> 0, "Negative or zero dimension: " << std::to_string(dimension));

    IndexType totalSize = std::pow(sideLen,dimension);
    SCAI_ASSERT( thisPixel < totalSize, "Wrong side length or dimension, sideLen=" + std::to_string(sideLen)+ " and dimension= " + std::to_string(dimension) );

    std::vector<IndexType> result;

    //calculate the index of the neighbouring pixels
    for(IndexType i=0; i<dimension; i++) {
        for( int j : {
                    -1, 1
                    } ) {
            // possible neighbour
            IndexType ngbrIndex = thisPixel + j*std::pow(sideLen,i );
            // index is within bounds
            if( ngbrIndex < 0 or ngbrIndex >=totalSize) {
                continue;
            }
            if(dimension==2) {
                IndexType xCoord = thisPixel/sideLen;
                IndexType yCoord = thisPixel%sideLen;
                if( ngbrIndex/sideLen == xCoord or ngbrIndex%sideLen == yCoord) {
                    result.push_back(ngbrIndex);
                }
            } else if(dimension==3) {
                IndexType planeSize= sideLen*sideLen;
                IndexType xCoord = thisPixel/planeSize;
                IndexType yCoord = (thisPixel%planeSize) /  sideLen;
                IndexType zCoord = (thisPixel%planeSize) % sideLen;
                IndexType ngbrX = ngbrIndex/planeSize;
                IndexType ngbrY = (ngbrIndex%planeSize)/sideLen;
                IndexType ngbrZ = (ngbrIndex%planeSize)%sideLen;
                if( ngbrX == xCoord and  ngbrY == yCoord ) {
                    result.push_back(ngbrIndex);
                } else if(ngbrX == xCoord and  ngbrZ == zCoord) {
                    result.push_back(ngbrIndex);
                } else if(ngbrY == yCoord and  ngbrZ == zCoord) {
                    result.push_back(ngbrIndex);
                }
            } else {
                throw std::runtime_error("Implemented only for 2D and 3D. Dimension given: " + std::to_string(dimension) );
            }
        }
    }
    return result;
}
//---------------------------------------------------------------------------------------

//to force instantiation
template class ParcoRepart<IndexType, ValueType>;

} //namespace ITI
