/*
 * GraphUtils.cpp
 *
 *  Created on: 29.06.2017
 *      Author: moritzl
 */

#include <assert.h>
#include <queue>
#include <unordered_set>
#include <chrono>

#include <scai/dmemo/mpi/MPICommunicator.hpp>
#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>
#include <scai/dmemo/HaloExchangePlan.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>
#include <scai/lama/matrix/DenseMatrix.hpp>
#include <scai/lama/matrix/DIASparseMatrix.hpp>
#include <scai/logging.hpp>
#include <scai/tracing.hpp>

#include <JanusSort.hpp>

#include "GraphUtils.h"

SCAI_LOG_DEF_LOGGER( logger, "GraphUtilsLogger" );

using std::vector;
using std::queue;

namespace ITI {

//namespace GraphUtils {

using scai::hmemo::ReadAccess;
using scai::dmemo::Distribution;
using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::lama::CSRStorage;


template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr GraphUtils<IndexType,ValueType>::genBlockRedist(scai::lama::CSRSparseMatrix<ValueType> &graph) {
    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();

    //get the global IDs of all the local indices
    scai::dmemo::DistributionPtr blockDist = scai::dmemo::genBlockDistributionBySize(globalN, localN, comm);

    graph.redistribute( blockDist, graph.getColDistributionPtr() );

    return blockDist;
}

//TODO: deprecated version, fix and use or remove
template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr GraphUtils<IndexType,ValueType>::reindex(scai::lama::CSRSparseMatrix<ValueType> &graph) {
    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();

    //get the global IDs of all the local indices
    scai::dmemo::DistributionPtr blockDist = scai::dmemo::genBlockDistributionBySize(globalN, localN, comm);

    DenseVector<IndexType> result(blockDist,0); 
    blockDist->getOwnedIndexes(result.getLocalValues());

    //for(int i=0; i<localN; i++){
    //   PRINT( comm->getRank() << ": i=" << i << ", glob i= " <<  result.getLocalValues()[i] );
    //}
    SCAI_ASSERT_EQUAL_ERROR(result.sum(), globalN*(globalN-1)/2);

    scai::dmemo::HaloExchangePlan partHalo = buildNeighborHalo(graph);
    scai::hmemo::HArray<IndexType> haloData;
    partHalo.updateHalo( haloData, result.getLocalValues(), *comm );

    CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    scai::hmemo::HArray<IndexType> newJA(localStorage.getJA());
    {
        scai::hmemo::ReadAccess<IndexType> rHalo(haloData);
        scai::hmemo::ReadAccess<IndexType> rResult(result.getLocalValues());
        scai::hmemo::WriteAccess<IndexType> ja( newJA );//TODO: no longer allowed, change
        for (IndexType i = 0; i < ja.size(); i++) {
            IndexType oldNeighborID = ja[i];
            IndexType localNeighbor = inputDist->global2Local(oldNeighborID);
            //this neighboring vertex is also local in this PE
            if (localNeighbor != scai::invalidIndex) {
                ja[i] = rResult[localNeighbor];
                assert(blockDist->isLocal(ja[i]));
            } else {
                IndexType haloIndex = partHalo.global2Halo(oldNeighborID);
                assert(haloIndex != scai::invalidIndex);
                ja[i] = rHalo[haloIndex];
                assert(!blockDist->isLocal(ja[i]));
            }
        }
    }

    CSRStorage<ValueType> newStorage(localN, globalN, localStorage.getIA(), newJA, localStorage.getValues());
    graph = CSRSparseMatrix<ValueType>(blockDist, std::move(newStorage));

    graph.redistribute( blockDist, graph.getColDistributionPtr() );

    return blockDist;
}

template<typename IndexType, typename ValueType>
std::vector<IndexType> GraphUtils<IndexType,ValueType>::localBFS(const scai::lama::CSRSparseMatrix<ValueType> &graph, const IndexType u)
{
    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const IndexType localN = inputDist->getLocalSize();
    assert(u < localN);
    assert(u >= 0);

    std::vector<IndexType> result(localN, std::numeric_limits<IndexType>::max());
    std::queue<IndexType> queue;
    std::queue<IndexType> alternateQueue;
    std::vector<bool> visited(localN, false);

    queue.push(u);
    result[u] = 0;
    visited[u] = true;

    const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> localIa(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> localJa(localStorage.getJA());

    IndexType round = 0;
    while (!queue.empty()) {
        while (!queue.empty()) {
            const IndexType v = queue.front();
            queue.pop();
            assert(v < localN);
            assert(v >= 0);

            const IndexType beginCols = localIa[v];
            const IndexType endCols = localIa[v+1];
            for (IndexType j = beginCols; j < endCols; j++) {
                IndexType globalNeighbor = localJa[j];
                IndexType localNeighbor = inputDist->global2Local(globalNeighbor);
                if (localNeighbor != scai::invalidIndex && !visited[localNeighbor])  {
                    assert(localNeighbor < localN);
                    assert(localNeighbor >= 0);
                    assert(localNeighbor != u);

                    alternateQueue.push(localNeighbor);
                    result[localNeighbor] = round+1;
                    visited[localNeighbor] = true;
                }
            }
        }
        round++;
        std::swap(queue, alternateQueue);
        //if alternateQueue was empty, queue is now empty and outer loop will abort
    }
    assert(result[u] == 0);

    return result;
}
//------------------------------------------------------------------------------------

//very similar to localBFS

template<typename IndexType, typename ValueType>
std::vector<ValueType> GraphUtils<IndexType,ValueType>::localDijkstra(const scai::lama::CSRSparseMatrix<ValueType> &graph, const IndexType u, std::vector<IndexType>& predecessor )
{
    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const IndexType localN = graph.getNumRows();
    SCAI_ASSERT_LT_ERROR(u, localN, "Index too large");
    SCAI_ASSERT_EQ_ERROR( localN, graph.getNumColumns(), "Matrix not square");
    assert(u >= 0);

    //std::vector<IndexType> result(localN, std::numeric_limits<IndexType>::max());
    //std::queue<IndexType> queue;
    // first is the distance to the node, second is the node ID
    typedef std::pair<ValueType, IndexType> iPair;
    std::priority_queue<iPair, std::vector<iPair>, std::greater<iPair> > queue;
    std::priority_queue<iPair, std::vector<iPair>, std::greater<iPair> > alternateQueue;

    // fill predessor vector with wrong value
    predecessor.resize( localN, -1);

    std::vector<bool> visited(localN, false);
    std::vector<ValueType> dist(localN, std::numeric_limits<ValueType>::max() );

    queue.push( std::make_pair( 0, u ) );
    visited[u] = true;
    dist[u] = 0;

    const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> localIa(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> localJa(localStorage.getJA());
    const scai::hmemo::ReadAccess<ValueType> localValues(localStorage.getValues());

    IndexType round = 0;
    while (!queue.empty()) {
        while (!queue.empty()) {
            const IndexType v = queue.top().second; // node ID of the nearest node
            queue.pop();
            assert(v < localN);
            assert(v >= 0);

            // check all neighbors of node v
            const IndexType beginCols = localIa[v];
            const IndexType endCols = localIa[v+1];
            for (IndexType j = beginCols; j < endCols; j++) {
                IndexType globalNeighbor = localJa[j];
                IndexType localNeighbor = inputDist->global2Local(globalNeighbor);

                //neighbor is local and not visited
                if (localNeighbor != scai::invalidIndex	and !visited[localNeighbor] ) {
                    assert(localNeighbor < localN);
                    assert(localNeighbor >= 0);
                    assert(localNeighbor != u);

                    //weight of edge (v, localNeighbor)
                    ValueType edgeWeight = localValues[j];

                    if( dist[localNeighbor] > dist[v] + edgeWeight ) {
                        dist[localNeighbor] = dist[v] + edgeWeight;
                        alternateQueue.push( std::make_pair( dist[localNeighbor],localNeighbor) );
                        //queue.push( std::make_pair( dist[localNeighbor],localNeighbor) );
                        predecessor[localNeighbor] = v;
                        visited[v] = true;
                    }
                }
            }
        }
        round++;
        std::swap(queue, alternateQueue);
        //if alternateQueue was empty, queue is now empty and outer loop will abort
    }
    assert(dist[u] == 0);

    return dist;
}

template<typename IndexType, typename ValueType>
IndexType GraphUtils<IndexType,ValueType>::getLocalBlockDiameter(const CSRSparseMatrix<ValueType> &graph, const IndexType u, IndexType lowerBound, const IndexType k, IndexType maxRounds)
{
    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

    const IndexType localN = inputDist->getLocalSize();
    if (comm->getRank() == 0) {
        std::cout << "Starting Diameter calculation..." << std::endl;
    }
    assert(u < localN);
    assert(u >= 0);
    std::vector<IndexType> ecc(localN);
    std::vector<IndexType> distances = localBFS(graph, u);
    assert(distances[u] == 0);
    ecc[u] = *std::max_element(distances.begin(), distances.end());

    if (localN > 1) {
        SCAI_ASSERT_GT_ERROR( ecc[u], 0, *comm << ": Wrong eccentricity value");
    }

    if (ecc[u] > localN) {
        SCAI_ASSERT_EQ_ERROR(ecc[u], std::numeric_limits<IndexType>::max(), "invalid ecc value");
        return ecc[u];
    }
    IndexType i = ecc[u];
    lowerBound = std::max(ecc[u], lowerBound);
    IndexType upperBound = 2*ecc[u];
    if (maxRounds == -1) {
        maxRounds = localN;
    }

    while (upperBound - lowerBound > k && ecc[u] - i < maxRounds) {
        assert(i > 0);
        // get max eccentricity in fringe i
        IndexType B_i = 0;
        for (IndexType j = 0; j < localN; j++) {
            if (distances[j] == i) {
                assert(j != u);
                std::vector<IndexType> jDistances = localBFS(graph, j);
                ecc[j] = *std::max_element(jDistances.begin(), jDistances.end());
                B_i = std::max(B_i, ecc[j]);
            }
        }

        lowerBound = std::max(lowerBound, B_i);
        if (lowerBound > 2*(i-1)) {
            return lowerBound;
        }   else {
            upperBound = 2*(i-1);
        }
        //std::cout << "proc " << comm->getRank() << ", i: " << i << ", lb:" << lowerBound << ", ub:" << upperBound << std::endl;
        i -= 1;
    }
    return lowerBound;
}

template<typename IndexType, typename ValueType>
ValueType GraphUtils<IndexType,ValueType>::computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const bool weighted) {
    SCAI_REGION( "ParcoRepart.computeCut" )
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();

    scai::dmemo::CommunicatorPtr comm = partDist->getCommunicatorPtr();
    if( comm->getRank()==0 ) {
        std::cout<<"Computing the cut...";
        std::cout.flush();
    }

    [[maybe_unused]] const IndexType n = inputDist->getGlobalSize();
    const IndexType localN = inputDist->getLocalSize();

    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

    if (partDist->getLocalSize() != localN) {
        PRINT(comm->getRank() << ": Local mismatch for matrix and partition");
        throw std::runtime_error("partition has " + std::to_string(partDist->getLocalSize()) + " local values, but matrix has " + std::to_string(localN));
    }

    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    scai::hmemo::HArray<IndexType> localData = part.getLocalValues();
    scai::hmemo::ReadAccess<IndexType> partAccess(localData);

    scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());
    scai::dmemo::HaloExchangePlan partHalo = buildNeighborHalo(input);
    scai::hmemo::HArray<IndexType> haloData;
    partHalo.updateHalo( haloData, localData, partDist->getCommunicator() );

    ValueType result = 0;
    for (IndexType i = 0; i < localN; i++) {
        const IndexType beginCols = ia[i];
        const IndexType endCols = ia[i+1];
        assert(ja.size() >= endCols);

        const IndexType globalI = inputDist->local2Global(i);
        //assert(partDist->isLocal(globalI));
        SCAI_ASSERT_ERROR(partDist->isLocal(globalI), "non-local index, globalI= " << globalI << " for PE " << comm->getRank() );

        IndexType thisBlock = partAccess[i];

        for (IndexType j = beginCols; j < endCols; j++) {
            IndexType neighbor = ja[j];
            assert(neighbor >= 0);
            assert(neighbor < n);

            IndexType neighborBlock;
            if (partDist->isLocal(neighbor)) {
                neighborBlock = partAccess[partDist->global2Local(neighbor)];
            } else {
                neighborBlock = haloData[partHalo.global2Halo(neighbor)];
            }

            if (neighborBlock != thisBlock) {
                if (weighted) {
                    result += values[j];
                } else {
                    result++;
                }
            }
        }
    }

    if (!inputDist->isReplicated()) {
        //sum values over all processes
        result = inputDist->getCommunicatorPtr()->sum(result);
    }

    std::chrono::duration<double> endTime = std::chrono::steady_clock::now() - startTime;
    double totalTime= comm->max(endTime.count() );

    if( comm->getRank()==0 ) {
        std::cout<< result/2 << ", done in " << totalTime << " seconds " << std::endl;
    }

    return result / 2; //counted each edge from both sides
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType GraphUtils<IndexType,ValueType>::computeImbalance(
    const DenseVector<IndexType> &part,
    IndexType k,
    const DenseVector<ValueType> &nodeWeights,
    const std::vector<ValueType> &optBlockSizes) {
    SCAI_REGION( "ParcoRepart.computeImbalance" )

    const IndexType globalN = part.getDistributionPtr()->getGlobalSize();
    const IndexType localN = part.getDistributionPtr()->getLocalSize();
    const IndexType weightsSize = nodeWeights.getDistributionPtr()->getGlobalSize();

    const bool weighted = (weightsSize != 0);

    scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
    SCAI_ASSERT_EQ_ERROR(weighted, comm->any(weighted), "inconsistent input!");

    ValueType minWeight, maxWeight;
    if (weighted) {
        assert(weightsSize == globalN);
        SCAI_ASSERT_EQ_ERROR(nodeWeights.getDistributionPtr()->getLocalSize(), localN, "in PE " << comm->getRank() );
        minWeight = nodeWeights.min();
        maxWeight = nodeWeights.max();
    } else {
        minWeight = 1;
        maxWeight = 1;
    }
    if (maxWeight <= 0) {
        throw std::runtime_error("Node weight vector given, but all weights non-positive.");
    }
    if (minWeight < 0) {
        throw std::runtime_error("Negative node weights not supported.");
    }

    std::vector<ValueType> globalSubsetSizes = getBlocksWeights( part, k, nodeWeights);
    assert( globalSubsetSizes.size()==k );

    const ValueType maxBlockSize = *std::max_element(globalSubsetSizes.begin(), globalSubsetSizes.end());

    //if an optBlockSizes vector is give, then we do not have homogeneous blocks weights
    const bool homogeneous = (optBlockSizes.size()==0);

    //to be returned
    ValueType imbalance;

    if (weighted) {
        if( homogeneous) {
            //get global weight sum
            //weightSum = comm->sum(weightSum);
            const ValueType weightSum = std::accumulate( globalSubsetSizes.begin(), globalSubsetSizes.end(), 0.0 );
            ValueType optSize = weightSum / k + (maxWeight - minWeight);
            assert(maxBlockSize >= optSize);

            imbalance = (ValueType(maxBlockSize - optSize)/ optSize);
        } else {
            //optBlockSizes is the optimum weight/size for every block
            SCAI_ASSERT_EQ_ERROR( k, optBlockSizes.size(), "Number of blocks do not agree with the size of the vector of the block sizes");
            //TODO: vector not really needed, only the max value
            std::vector<ValueType> imbalances( k );
            for( IndexType i=0; i<k; i++) {
                imbalances[i] = (ValueType( globalSubsetSizes[i]- optBlockSizes[i]))/optBlockSizes[i];
            }
            imbalance = *std::max_element(imbalances.begin(), imbalances.end() );
        }
//TODO: can we a have heterogeneous network but no node weights?
    } else {
        ValueType optSize = ValueType(globalN) / k;
        assert(maxBlockSize >= optSize);

        imbalance = (ValueType(maxBlockSize - optSize)/ optSize);
    }

    return imbalance;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType>  GraphUtils<IndexType,ValueType>::getBlocksWeights(
    const scai::lama::DenseVector<IndexType> &part,
    const IndexType numBlocks,
    const scai::lama::DenseVector<ValueType> &nodeWeights
){
    
    const IndexType globalN = part.getDistributionPtr()->getGlobalSize();
    const IndexType localN = part.getDistributionPtr()->getLocalSize();
    const IndexType weightsSize = nodeWeights.getDistributionPtr()->getGlobalSize();

    const bool weighted = (weightsSize != 0);

    scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
    SCAI_ASSERT_EQ_ERROR(weighted, comm->any(weighted), "inconsistent input!");

    //TODO: is this needed here? remove or wrap around settings.debugMode?

    const IndexType minK = part.min();
    const IndexType maxK = part.max();
    if (minK < 0) {
        throw std::runtime_error("Block id " + std::to_string(minK) + " found in partition with supposedly " + std::to_string(numBlocks) + " blocks.");
    }
    if (maxK >= numBlocks) {
        throw std::runtime_error("Block id " + std::to_string(maxK) + " found in partition with supposedly " + std::to_string(numBlocks) + " blocks.");
    }

    std::vector<ValueType> subsetSizes(numBlocks, 0.0);
    scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());
    scai::hmemo::ReadAccess<ValueType> localWeight(nodeWeights.getLocalValues());
    assert(localPart.size() == localN);

    //calculate weight of each block and global weight sum
    ValueType weightSum = 0.0;
    for (IndexType i = 0; i < localN; i++) {
        IndexType partID = localPart[i];
        ValueType weight = weighted ? localWeight[i] : 1;
        subsetSizes[partID] += weight;
        weightSum += weight;
    }

    std::vector<ValueType> globalSubsetSizes(numBlocks);
    const bool isReplicated = part.getDistribution().isReplicated();
    SCAI_ASSERT_EQ_ERROR(isReplicated, comm->any(isReplicated), "inconsistent distribution!");

    if (isReplicated) {
        SCAI_ASSERT_EQUAL_ERROR(localN, globalN);
    }

    if (!isReplicated) {
        //sum block sizes over all processes
        comm->sumImpl( globalSubsetSizes.data(), subsetSizes.data(), numBlocks, scai::common::TypeTraits<ValueType>::stype);
    } else {
        globalSubsetSizes = subsetSizes;
    }

ValueType globWsum = std::accumulate( globalSubsetSizes.begin(), globalSubsetSizes.end(), 0.0 );
SCAI_ASSERT_EQ_ERROR( globWsum, comm->sum(weightSum), " global sum mismatch" );

    return globalSubsetSizes;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::dmemo::HaloExchangePlan GraphUtils<IndexType,ValueType>::buildNeighborHalo(const CSRSparseMatrix<ValueType>& input) {

    SCAI_REGION( "ParcoRepart.buildPartHalo" )

    std::vector<IndexType> requiredHaloIndices = nonLocalNeighbors(input);

    scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );

    return haloExchangePlan( input.getRowDistribution(), arrRequiredIndexes );
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> GraphUtils<IndexType, ValueType>::nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
    SCAI_REGION( "ParcoRepart.nonLocalNeighbors" )
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    [[maybe_unused]] const IndexType n = inputDist->getGlobalSize();
    const IndexType localN = inputDist->getLocalSize();

    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    //WARNING: this can affect how other functions behave. For example, getPEGraphTest
    //	fails if we use a set
    //TODO: template or add an option so it can be called both ways?
    //std::set<IndexType> neighborSet;	//does not allows duplicates, we count vertices
    std::multiset<IndexType> neighborSet; //since this allows duplicates,  we count edges

    for (IndexType i = 0; i < localN; i++) {
        const IndexType beginCols = ia[i];
        const IndexType endCols = ia[i+1];

        for (IndexType j = beginCols; j < endCols; j++) {
            IndexType neighbor = ja[j];
            assert(neighbor >= 0);
            assert(neighbor < n);

            if (!inputDist->isLocal(neighbor)) {
                neighborSet.insert(neighbor);
            }
        }
    }
    return std::vector<IndexType>(neighborSet.begin(), neighborSet.end()) ;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
inline bool GraphUtils<IndexType, ValueType>::hasNonLocalNeighbors(const CSRSparseMatrix<ValueType> &input, IndexType globalID) {
    SCAI_REGION( "ParcoRepart.hasNonLocalNeighbors" )

    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    const IndexType localID = inputDist->global2Local(globalID);
    assert(localID != scai::invalidIndex);

    const IndexType beginCols = ia[localID];
    const IndexType endCols = ia[localID+1];

    for (IndexType j = beginCols; j < endCols; j++) {
        if (!inputDist->isLocal(ja[j])) {
            return true;
        }
    }
    return false;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> GraphUtils<IndexType, ValueType>::getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input, const std::set<IndexType>& candidates) {
    SCAI_REGION( "ParcoRepart.getNodesWithNonLocalNeighbors_cache" );
    std::vector<IndexType> result;
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    for (IndexType globalI : candidates) {
        const IndexType localI = inputDist->global2Local(globalI);
        if (localI == scai::invalidIndex) {
            continue;
        }
        const IndexType beginCols = ia[localI];
        const IndexType endCols = ia[localI+1];

        //over all edges
        for (IndexType j = beginCols; j < endCols; j++) {
            if (inputDist->isLocal(ja[j]) == 0) {
                result.push_back(globalI);
                break;
            }
        }
    }

    //nodes should have been sorted to begin with, so a subset of them will be sorted as well
    std::sort(result.begin(), result.end());
    return result;
}


template<typename IndexType, typename ValueType>
std::vector<IndexType> GraphUtils<IndexType, ValueType>::getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
    SCAI_REGION( "ParcoRepart.getNodesWithNonLocalNeighbors" )
    std::vector<IndexType> result;

    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    if (inputDist->isReplicated()) {
        //everything is local
        return result;
    }

    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const IndexType localN = inputDist->getLocalSize();

    scai::hmemo::HArray<IndexType> ownIndices;
    inputDist->getOwnedIndexes(ownIndices);
    scai::hmemo::ReadAccess<IndexType> rIndices(ownIndices);

    //iterate over all nodes
    for (IndexType localI = 0; localI < localN; localI++) {
        const IndexType beginCols = ia[localI];
        const IndexType endCols = ia[localI+1];

        //over all edges
        for (IndexType j = beginCols; j < endCols; j++) {
            if (!inputDist->isLocal(ja[j])) {
                IndexType globalI = rIndices[localI];
                result.push_back(globalI);
                break;
            }
        }
    }

    //nodes should have been sorted to begin with, so a subset of them will be sorted as well
    assert(std::is_sorted(result.begin(), result.end()));
    return result;
}
//---------------------------------------------------------------------------------------

/* The results returned is already distributed
 */
template<typename IndexType, typename ValueType>
DenseVector<IndexType> GraphUtils<IndexType, ValueType>::getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {

    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();
    DenseVector<IndexType> border(dist,0);
    scai::hmemo::HArray<IndexType>& localBorder= border.getLocalValues();

    //const IndexType globalN = dist->getGlobalSize();
    [[maybe_unused]] IndexType max = part.max();

    if( !dist->isEqual( part.getDistribution() ) ) {
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

    auto partHalo = buildNeighborHalo(adjM);
    auto haloData = partHalo.updateHaloF( localPart, dist->getCommunicator() );

    auto rHaloData = scai::hmemo::hostReadAccess( haloData );

    for(IndexType i=0; i<localN; i++) {   // for all local nodes
        IndexType thisBlock = localPart[i];
        for(IndexType j=ia[i]; j<ia[i+1]; j++) {                  // for all the edges of a node
            IndexType neighbor = ja[j];
            IndexType neighborBlock;
            if (dist->isLocal(neighbor)) {
                neighborBlock = partAccess[dist->global2Local(neighbor)];
            } else {
                neighborBlock = rHaloData[partHalo.global2Halo(neighbor)];
            }
            assert( neighborBlock < max +1 );
            if (thisBlock != neighborBlock) {
                localBorder[i] = 1;
                break;
            }
        }
    }

    assert(border.getDistributionPtr()->getLocalSize() == localN);
    return border;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>,std::vector<IndexType>> GraphUtils<IndexType, ValueType>::getNumBorderInnerNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, const struct Settings settings) {

    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

    if( comm->getRank()==0 ) {
        std::cout<<"Computing the border and inner nodes..." << std::endl;
    }
    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();

    //const IndexType globalN = dist->getGlobalSize();
    IndexType max = part.max();

    if(max!=settings.numBlocks-1) {
        PRINT("\n\t\tWARNING: the max block id is " << max << " but it should be " << settings.numBlocks-1);
        max = settings.numBlocks-1;
    }

    // the number of border nodes per block
    std::vector<IndexType> borderNodesPerBlock( max+1, 0 );
    // the number of inner nodes
    std::vector<IndexType> innerNodesPerBlock( max+1, 0 );


    if( !dist->isEqual( part.getDistribution() ) ) {
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

    auto partHalo = buildNeighborHalo(adjM);
    auto haloData = partHalo.updateHaloF( localPart, dist->getCommunicator() );
    auto rHaloData = scai::hmemo::hostReadAccess( haloData );

    for(IndexType i=0; i<localN; i++) {   // for all local nodes
        IndexType thisBlock = localPart[i];
        SCAI_ASSERT_LE_ERROR( thisBlock, max, "Wrong block id." );
        bool isBorderNode = false;

        for(IndexType j=ia[i]; j<ia[i+1]; j++) {     // for all the edges of a node
            IndexType neighbor = ja[j];
            IndexType neighborBlock;
            if (dist->isLocal(neighbor)) {
                neighborBlock = partAccess[dist->global2Local(neighbor)];
            } else {
                neighborBlock = rHaloData[partHalo.global2Halo(neighbor)];
            }
            SCAI_ASSERT_LE_ERROR( neighborBlock, max, "Wrong block id." );
            if (thisBlock != neighborBlock) {
                borderNodesPerBlock[thisBlock]++;   //increase number of border nodes found
                isBorderNode = true;
                break;
            }
        }
        //if all neighbors are in the same block then this is an inner node
        if( !isBorderNode ) {
            innerNodesPerBlock[thisBlock]++;
        }
    }

    comm->sumImpl( borderNodesPerBlock.data(), borderNodesPerBlock.data(), max+1, scai::common::TypeTraits<IndexType>::stype);

    comm->sumImpl( innerNodesPerBlock.data(), innerNodesPerBlock.data(), max+1, scai::common::TypeTraits<IndexType>::stype);

    std::chrono::duration<double> endTime = std::chrono::steady_clock::now() - startTime;
    double totalTime= comm->max(endTime.count() );
    if( comm->getRank()==0 ) {
        std::cout<<"\t\t\t time to get number of border and inner nodes : " << totalTime <<  std::endl;
    }

    return std::make_pair( borderNodesPerBlock, innerNodesPerBlock );
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> GraphUtils<IndexType, ValueType>::computeCommVolume( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, Settings settings) {
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    const IndexType numBlocks = settings.numBlocks;

    if( comm->getRank()==0 && settings.verbose) {
        std::cout<<"Computing the communication volume ...";
        std::cout.flush();
    }
    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();

    // the communication volume per block for this PE
    std::vector<IndexType> commVolumePerBlock( numBlocks, 0 );


    if( !dist->isEqual( part.getDistribution() ) ) {
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

    auto partHalo = buildNeighborHalo(adjM);
    auto haloData = partHalo.updateHaloF( localPart, dist->getCommunicator() );
    auto rHaloData = scai::hmemo::hostReadAccess( haloData );

    for(IndexType i=0; i<localN; i++) {   // for all local nodes
        IndexType thisBlock = localPart[i];
        SCAI_ASSERT_LE_ERROR( thisBlock, numBlocks, "Wrong block id." );
        std::set<IndexType> allNeighborBlocks;

        for(IndexType j=ia[i]; j<ia[i+1]; j++) {      // for all the edges of a node
            IndexType neighbor = ja[j];
            IndexType neighborBlock;
            if (dist->isLocal(neighbor)) {
                neighborBlock = partAccess[dist->global2Local(neighbor)];
            } else {
                neighborBlock = rHaloData[partHalo.global2Halo(neighbor)];
            }
            SCAI_ASSERT_LE_ERROR( neighborBlock, numBlocks, "Wrong block id." );

            // found a neighbor that belongs to a different block
            if (thisBlock != neighborBlock) {

                typename std::set<IndexType>::iterator it = allNeighborBlocks.find( neighborBlock );

                if( it==allNeighborBlocks.end() ) {  // this block has not been encountered before
                    allNeighborBlocks.insert( neighborBlock );
                    commVolumePerBlock[thisBlock]++;   //increase volume
                } else {
                    // if neighnor belongs to a different block but we have already found another neighbor
                    // from that block, then do not increase volume
                }
            }
        }
    }

    // sum local volume
    comm->sumImpl( commVolumePerBlock.data(), commVolumePerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype);

    std::chrono::duration<double> endTime = std::chrono::steady_clock::now() - startTime;
    double totalTime= comm->max(endTime.count() );
    if( comm->getRank()==0 && settings.verbose) {
        std::cout<<" done in " << totalTime <<  std::endl;
    }
    return commVolumePerBlock;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::tuple<std::vector<IndexType>, std::vector<IndexType>, std::vector<IndexType>> GraphUtils<IndexType, ValueType>::computeCommBndInner(
            const CSRSparseMatrix<ValueType> &adjM,
            const DenseVector<IndexType> &part,
            Settings settings) {

    const IndexType numBlocks = settings.numBlocks;
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

    if( comm->getRank()==0 && settings.verbose) {
        std::cout<<"Computing the communication volume, number of border and inner nodes ...";
        std::cout.flush();
    }
    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();


    // the communication volume per block for this PE
    std::vector<IndexType> commVolumePerBlock( numBlocks, 0 );
    // the number of border nodes per block
    std::vector<IndexType> borderNodesPerBlock( numBlocks, 0 );
    // the number of inner nodes
    std::vector<IndexType> innerNodesPerBlock( numBlocks, 0 );

    if( !dist->isEqual( part.getDistribution() ) ) {
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

    auto partHalo = buildNeighborHalo(adjM);
    auto haloData = partHalo.updateHaloF( localPart, dist->getCommunicator() );
    auto rHaloData = scai::hmemo::hostReadAccess( haloData );

    for(IndexType i=0; i<localN; i++) {   // for all local nodes
        IndexType thisBlock = localPart[i];
        SCAI_ASSERT_LT_ERROR( thisBlock, numBlocks, "Wrong block id." );
        bool isBorderNode = false;
        std::set<IndexType> allNeighborBlocks;

        for(IndexType j=ia[i]; j<ia[i+1]; j++) {       // for all the edges of a node
            IndexType neighbor = ja[j];
            IndexType neighborBlock;
            if (dist->isLocal(neighbor)) {
                neighborBlock = partAccess[dist->global2Local(neighbor)];
            } else {
                neighborBlock = rHaloData[partHalo.global2Halo(neighbor)];
            }
            SCAI_ASSERT_LT_ERROR( neighborBlock, numBlocks, "Wrong block id." );

            // found a neighbor that belongs to a different block
            if (thisBlock != neighborBlock) {
                if( not isBorderNode) {
                    borderNodesPerBlock[thisBlock]++;   //increase number of border nodes found
                    isBorderNode = true;
                }

                typename std::set<IndexType>::iterator it = allNeighborBlocks.find( neighborBlock );

                if( it==allNeighborBlocks.end() ) {  // this block has not been encountered before
                    allNeighborBlocks.insert( neighborBlock );
                    commVolumePerBlock[thisBlock]++;   //increase volume
                } else {
                    // if neighnor belongs to a different block but we have already found another neighbor
                    // from that block, then do not increase volume
                }
            }
        }
        //if all neighbors are in the same block then this is an inner node
        if( !isBorderNode ) {
            innerNodesPerBlock[thisBlock]++;
        }
    }

    // sum local volume
    comm->sumImpl( commVolumePerBlock.data(), commVolumePerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype);
    // sum border nodes
    comm->sumImpl( borderNodesPerBlock.data(), borderNodesPerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype);
    // sum inner nodes
    comm->sumImpl( innerNodesPerBlock.data(), innerNodesPerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype);

    std::chrono::duration<double> endTime = std::chrono::steady_clock::now() - startTime;
    double totalTime= comm->max(endTime.count() );
    if( comm->getRank()==0 && settings.verbose) {
        std::cout<<" done in " << totalTime <<  std::endl;
    }
    return std::make_tuple( std::move(commVolumePerBlock), std::move(borderNodesPerBlock), std::move(innerNodesPerBlock) );
}

//---------------------------------------------------------------------------------------

/* Get the maximum degree of a graph.
 * */
template<typename IndexType, typename ValueType>
IndexType GraphUtils<IndexType, ValueType>::getGraphMaxDegree( const scai::lama::CSRSparseMatrix<ValueType>& adjM) {

    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = distPtr->getCommunicatorPtr();
    const IndexType globalN = distPtr->getGlobalSize();

    {
        scai::dmemo::DistributionPtr noDist (new scai::dmemo::NoDistribution( globalN ));
        SCAI_ASSERT( adjM.getColDistributionPtr()->isEqual(*noDist), "Adjacency matrix should have no column distribution." );
    }

    const scai::lama::CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());

    // local maximum degree
    IndexType maxDegree = ia[1]-ia[0];

    for(int i=1; i<ia.size(); i++) {
        IndexType thisDegree = ia[i]-ia[i-1];
        if( thisDegree>maxDegree) {
            maxDegree = thisDegree;
        }
    }
    //return global maximum
    return comm->max( maxDegree );
}
//------------------------------------------------------------------------------

/* Compute maximum communication= max degree of the block graph, and total communication= sum of all edges
 */
template<typename IndexType, typename ValueType>
std::pair<IndexType,IndexType> GraphUtils<IndexType, ValueType>::computeBlockGraphComm( const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType> &part) {

    scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();

    if( comm->getRank()==0 ) {
        std::cout<<"Computing the block graph communication..." << std::endl;
    }
    IndexType k = part.max()-1;
    //TODO: getting the block graph probably fails for p>5000,
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = getBlockGraph( adjM, part, k);

    IndexType maxComm = getGraphMaxDegree( blockGraph );
    IndexType totalComm = blockGraph.getNumValues()/2;

    return std::make_pair(maxComm, totalComm);
}
//-----------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType>  GraphUtils<IndexType, ValueType>::getBlockGraph( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k) {
    SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges");

    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();    
    const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
    //TODO/check: should dist==partDist??
    const IndexType localN = partDist->getLocalSize();
    SCAI_ASSERT( partDist->isEqual(*dist), "Graph and partition distributions must agree" );

    if( !dist->isEqual( part.getDistribution() ) ) {
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    //access to graph and partition
    const scai::lama::CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

    //assert( max(ja)<globalN);

    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();
    const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

    //get halo for non-local values
    scai::dmemo::HaloExchangePlan partHalo = buildNeighborHalo( adjM );
    scai::hmemo::HArray<IndexType> haloData;
    partHalo.updateHalo( haloData, localPart, partDist->getCommunicator() );

    // TODO: memory costly for big k
    IndexType size= k*k;
    //localBlockGraphEdges[i] = w, is the weight for edge (i/k,i%k)
    scai::hmemo::HArray<ValueType> localBlockGraphEdges( size,  static_cast<ValueType>(0.0) );

    //go over local data
    for (IndexType i = 0; i < localN; i++) {
        const IndexType beginCols = ia[i];
        const IndexType endCols = ia[i+1];
        assert(ja.size() >= endCols);

        const IndexType globalI = dist->local2Global(i);
        //assert(partDist->isLocal(globalI));
        SCAI_ASSERT_ERROR(partDist->isLocal(globalI), "non-local index, globalI= " << globalI << " for PE " << comm->getRank() );

        IndexType thisBlock = partAccess[i];

        for (IndexType j = beginCols; j < endCols; j++) {
            IndexType neighbor = ja[j];
            IndexType neighborBlock;
            if (partDist->isLocal(neighbor)) {
                neighborBlock = partAccess[partDist->global2Local(neighbor)];
            } else {
                neighborBlock = haloData[partHalo.global2Halo(neighbor)];
            }

            if (neighborBlock != thisBlock) {
                IndexType index = thisBlock*k + neighborBlock;
                //there is no += operator for HArray
                localBlockGraphEdges[index] = localBlockGraphEdges[index] + values[j];
            }
        }
    }//for

    comm->sumArray( localBlockGraphEdges );
    //now, localBlockGraphEdges contains the global summed values; it should be the same in every PE

    //count number of edges
    IndexType numEdges=0;
    {
        scai::hmemo::ReadAccess<ValueType> globalEdges( localBlockGraphEdges );
        for(IndexType i=0; i<globalEdges.size(); i++) {
            if( globalEdges[i]>0 )
                ++numEdges;
        }
    }

    //convert the k*k HArray to a [k x k] CSRSparseMatrix
    scai::lama::CSRStorage<ValueType> localMatrix;
    localMatrix.allocate( k,k );

    scai::hmemo::HArray<IndexType> csrIA;
    scai::hmemo::HArray<IndexType> csrJA;
    scai::hmemo::HArray<ValueType> csrValues( numEdges, 0.0 );
    {
        scai::hmemo::WriteOnlyAccess<IndexType> ia( csrIA, k +1 );
        scai::hmemo::WriteOnlyAccess<IndexType> ja( csrJA, numEdges );
        scai::hmemo::WriteOnlyAccess<ValueType> values( csrValues );
        scai::hmemo::ReadAccess<ValueType> globalEdges( localBlockGraphEdges );
        ia[0]= 0;

        IndexType rowCounter = 0; // count rows
        IndexType nnzCounter = 0; // count non-zero elements

        for(IndexType i=0; i<k; i++) {
            IndexType rowNums=0;
            // traverse the part of the HArray that represents a row and find how many elements are in this row
            for(IndexType j=0; j<k; j++) {
                if( globalEdges[i*k+j] >0  ) {
                    ++rowNums;
                }
            }
            ia[rowCounter+1] = ia[rowCounter] + rowNums;

            for(IndexType j=0; j<k; j++) {
                if( globalEdges[i*k +j] >0) {  // there exist edge (i,j)
                    ja[nnzCounter] = j;
                    //values[nnzCounter] = 1;
                    values[nnzCounter] = globalEdges[i*k +j];
                    //PRINT0("edge ["<< i <<", "<< j << "]= " << values[nnzCounter] );
                    ++nnzCounter;
                }
            }
            ++rowCounter;
        }
    }
    SCAI_REGION_START("ParcoRepart.getBlockGraph.swapAndAssign");
    scai::lama::CSRSparseMatrix<ValueType> matrix;
    localMatrix.swap( csrIA, csrJA, csrValues );
    matrix.assign(localMatrix);
    SCAI_REGION_END("ParcoRepart.getBlockGraph.swapAndAssign");
    return matrix;
}
//-----------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType>  GraphUtils<IndexType, ValueType>::getBlockGraph_dist( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k) {

    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
    //TODO/check: should dist==partDist??
    const IndexType localN = partDist->getLocalSize();
    SCAI_ASSERT( partDist->isEqual(*dist), "Graph and partition distributions must agree" );

    if( !dist->isEqual( part.getDistribution() ) ) {
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    //use a map to store edge weights
    typedef std::pair<IndexType,IndexType> edge;
    //TODO: check: this is supposedto be initialized to 0?
    std::map<edge,ValueType> edgeMap;

    // read local values and create the local edge list of the block graph
    {
        //access to graph and partition
        const scai::lama::CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
        const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
        const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
        const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

        const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();
        const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

        //get halo for non-local values
        scai::dmemo::HaloExchangePlan partHalo = buildNeighborHalo( adjM );
        scai::hmemo::HArray<IndexType> haloData;
        partHalo.updateHalo( haloData, localPart, partDist->getCommunicator() );

        //go over local data
        for (IndexType i = 0; i < localN; i++) {
            const IndexType beginCols = ia[i];
            const IndexType endCols = ia[i+1];
            assert(ja.size() >= endCols);

            const IndexType globalI = dist->local2Global(i);
            //assert(partDist->isLocal(globalI));
            SCAI_ASSERT_ERROR(partDist->isLocal(globalI), "non-local index, globalI= " << globalI << " for PE " << comm->getRank() );

            IndexType thisBlock = partAccess[i];

            for (IndexType j = beginCols; j < endCols; j++) {
                IndexType neighbor = ja[j];
                IndexType neighborBlock;
                if (partDist->isLocal(neighbor)) {
                    neighborBlock = partAccess[partDist->global2Local(neighbor)];
                } else {
                    neighborBlock = haloData[partHalo.global2Halo(neighbor)];
                }

                //found an edge between two blocks
                if (neighborBlock != thisBlock) {
                    edge e1(thisBlock, neighborBlock);
                    edgeMap[e1] += values[j];
                }
            }
        }//for
    }

    const IndexType mySize = edgeMap.size()*3;  // [...,u,v,weight,..]

    //convert map to a vector
    std::vector<ValueType> edgeVec( mySize );

    IndexType ind=0;
    for( auto edgeIt=edgeMap.begin(); edgeIt!=edgeMap.end(); edgeIt ++ ) {
        edgeVec[ind] = edgeIt->first.first;		//first vertex
        edgeVec[ind+1] = edgeIt->first.second;	//second
        edgeVec[ind+2] = edgeIt->second;		//edge weight
        ind += 3;
    }

    const IndexType rootPE = 0; // set PE 0 as root

    //TODO: the array size can get very big. Another way would be to use a custom
    //communication plan where PE i sends to i+1 ... using logk rounds (similar to sum)
    //and in every summation step duplicate edges are eliminated.
    IndexType sumSize = comm->sum( mySize );
    IndexType arraySize=1;
    if( comm->getRank()==rootPE ) {
        arraySize = sumSize;
        //PRINT(*comm <<": array size= " << arraySize );
    }

    //every PE has different size to send, this is needed for the gather
    std::vector<IndexType> allSizes( comm->getSize(), 0 );
    allSizes[comm->getRank()] = mySize;
    comm->sumImpl( allSizes.data(), allSizes.data(), comm->getSize(),  scai::common::TypeTraits<IndexType>::stype );

    std::vector<ValueType> allEdges(arraySize, -1); //set a dummy value of -1
    comm->gatherV( allEdges.data(), mySize, rootPE, edgeVec.data(), allSizes.data() );

    /*
    allEdges is a concatenation of edge lists. Every PE stores its local edgelist.
    After gathering, the root PE traverses the gathered	edges lists and constructs
    the block graph.
     **/

    IndexType numEdges = 0; //only root will change it

    std::vector<IndexType> ia(k+1,0);
    std::vector<IndexType> ja;
    std::vector<ValueType> values;

    if( comm->getRank()==rootPE ) { //only root constructs the graph

        //sort map lexicographically by vertex ids
        struct lexEdgeSort {
            //bool operator()(const std::pair<edge,ValueType> u, const std::pair<edge,ValueType> v){
            bool operator() (const edge u, const edge v) const {
                //sort edges based on the first node of the edge or the second if the first is the same
                if (u.first == v.first)
                    return u.second < v.second;
                else
                    return u.first < v.first;
            }
        };

        std::map<edge,ValueType,lexEdgeSort> allEdgeMap;

        for(IndexType i=0; i<arraySize; i+=3 ) {
            edge e( allEdges[i], allEdges[i+1] );
            ValueType w = allEdges[i+2];
            allEdgeMap[e] += w;
            //PRINT0("inserting edge ("<< allEdges[i] << ", " << allEdges[i+1] << "), weight " << allEdgeMap[e] );
        }
        allEdges.clear();//not needed anymore

        numEdges = allEdgeMap.size();

        //create CSR storage

        ia[0] = 0;

        auto edgeIt = allEdgeMap.begin();
        for(IndexType v=0; v<k; v++ ) { //for every vertex of the block graph
            SCAI_ASSERT_EQ_ERROR( edgeIt->first.first, v, "Missing node id " << v );
            SCAI_ASSERT( edgeIt!=allEdgeMap.end(), "Edge list iterator out of bounds" ); //not very helpful assertion

            IndexType degree = 0;
            while( v==edgeIt->first.first ) {
                IndexType u = edgeIt->first.second;
                ValueType w = edgeIt->second;	//edge weight
                ja.push_back( u );
                values.push_back( w );
                degree++;
                edgeIt++; //move iterator to next edge
            }
            ia[v+1] = ia[v]+degree;
        }
        SCAI_ASSERT_EQ_ERROR( ja.size(), numEdges, "Wrong CSR ja size" );
        SCAI_ASSERT_EQ_ERROR( values.size(), numEdges, "Wrong CSR values size" );
        SCAI_ASSERT( edgeIt==allEdgeMap.end(), "Edge list iterator did not reach the end. Some edges were not considered" );

    }//if rootPE

    //only in root it has a non-zero value
    numEdges = comm->max( numEdges );

    if( comm->getRank()!=rootPE ) {
        ja.resize( numEdges, 0);
        values.resize( numEdges, 0 );
    }

    //broadcast CSR data from root to all other PEs
    comm->bcast( ia.data(), k+1, rootPE );
    comm->bcast( ja.data(), numEdges, rootPE );
    comm->bcast( values.data(), numEdges, rootPE );

    scai::lama::CSRStorage<ValueType> storage ( k, k,
            scai::hmemo::HArray<IndexType>(ia.size(), ia.data()),
            scai::hmemo::HArray<IndexType>(ja.size(), ja.data()),
            scai::hmemo::HArray<ValueType>(values.size(), values.data()));

    scai::lama::CSRSparseMatrix<ValueType> blockG( storage );

    return blockG;
}
//-----------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> GraphUtils<IndexType, ValueType>::getPEGraph( const CSRSparseMatrix<ValueType> &adjM) {
    SCAI_REGION("ParcoRepart.getPEGraph");
    
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    const IndexType numPEs = comm->getSize();

    const std::vector<IndexType> nonLocalIndices = GraphUtils<IndexType, ValueType>::nonLocalNeighbors(adjM);

    SCAI_REGION_START("ParcoRepart.getPEGraph.getOwners");
    scai::hmemo::HArray<IndexType> indexTransport(nonLocalIndices.size(), nonLocalIndices.data());
    // find the PEs that own every non-local index
    scai::hmemo::HArray<IndexType> owners(nonLocalIndices.size(), -1);
    dist->computeOwners( owners, indexTransport);
    SCAI_REGION_END("ParcoRepart.getPEGraph.getOwners");

    scai::hmemo::ReadAccess<IndexType> rOwners(owners);
    std::vector<IndexType> neighborPEs(rOwners.get(), rOwners.get()+rOwners.size());
    rOwners.release();

    SCAI_ASSERT_EQ_ERROR( neighborPEs.size(), nonLocalIndices.size(), "Vector size mismatch");

    //first count how many edges are between PEs and then remove duplicates

    //we use map because, for example, a PE can have only 2 neighbors but with high ranks
    std::map<IndexType, ValueType> edgeWeights;
    //initialize map
    for( int i=0; i<neighborPEs.size(); i++) {
        edgeWeights[ neighborPEs[i] ] = 0;
    }
    for( int i=0; i<neighborPEs.size(); i++) {
        edgeWeights[ neighborPEs[i] ]++;
    }

    std::sort(neighborPEs.begin(), neighborPEs.end());

    //remove duplicates
    neighborPEs.erase(std::unique(neighborPEs.begin(), neighborPEs.end()), neighborPEs.end());
    const IndexType numNeighbors = neighborPEs.size();
    SCAI_ASSERT_EQ_ERROR(edgeWeights.size(), numNeighbors, "Num neighbors mismatch");

    // create the PE adjacency matrix to be returned
    scai::dmemo::DistributionPtr distPEs ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, numPEs) );
    assert(distPEs->getLocalSize() == 1);
    scai::dmemo::DistributionPtr noDistPEs (new scai::dmemo::NoDistribution( numPEs ));

    SCAI_REGION_START("ParcoRepart.getPEGraph.buildMatrix");
    scai::hmemo::HArray<IndexType> ia( { 0, numNeighbors } );
    scai::hmemo::HArray<IndexType> ja(numNeighbors, neighborPEs.data());
    scai::hmemo::HArray<ValueType> values(numNeighbors, 1);

    int ii=0;
    for( auto edgeW = edgeWeights.begin(); edgeW!=edgeWeights.end(); edgeW++ ) {
        values[ii] = edgeW->second;
        //PRINT(*comm << ": values[" << ii << "]= " << values[ii] );
        ii++;
    }

    scai::lama::CSRStorage<ValueType> myStorage(1, numPEs, std::move(ia), std::move(ja), std::move(values));
    SCAI_REGION_END("ParcoRepart.getPEGraph.buildMatrix");

    scai::lama::CSRSparseMatrix<ValueType> PEgraph(distPEs, std::move(myStorage));

    return PEgraph;
}
//-----------------------------------------------------------------------------------
//TODO: add edge weights

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> GraphUtils<IndexType, ValueType>::getCSRmatrixFromAdjList_NoEgdeWeights( const std::vector<std::set<IndexType>>& adjList) {

    IndexType N = adjList.size();

    // the CSRSparseMatrix vectors
    std::vector<IndexType> ia(N+1);
    ia[0] = 0;
    std::vector<IndexType> ja;

    for(IndexType i=0; i<N; i++) {
        std::set<IndexType> neighbors = adjList[i]; // the neighbors of this vertex
        for( typename std::set<IndexType>::iterator it=neighbors.begin(); it!=neighbors.end(); it++) {
            ja.push_back( *it );
        }
        ia[i+1] = ia[i]+neighbors.size();
    }

    std::vector<ValueType> values(ja.size(), 1);

    scai::lama::CSRStorage<ValueType> myStorage( N, N,
            scai::hmemo::HArray<IndexType>(ia.size(), ia.data()),
            scai::hmemo::HArray<IndexType>(ja.size(), ja.data()),
            scai::hmemo::HArray<ValueType>(values.size(), values.data())
                                               );

    return scai::lama::CSRSparseMatrix<ValueType>(std::move(myStorage));
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> GraphUtils<IndexType, ValueType>::edgeList2CSR(
    std::vector< std::pair<IndexType, IndexType>> &edgeList,
    const scai::dmemo::CommunicatorPtr comm,
    const bool duplicateEdges,
    const bool removeSelfLoops) {

    const IndexType thisPE = comm->getRank();
    IndexType localM = edgeList.size();

    //Cannot use this, getMPIDatatype() does not work. Look into RBC/SortingDataType.hpp
    //typedef std::pair<int,int> int_pair;

    int typesize;
    MPI_Type_size(MPI_2INT, &typesize);
    //SCAI_ASSERT_EQ_ERROR(typesize, sizeof(int_pair), "Wrong size"); //not valid for int_double, presumably due to padding

    //-------------------------------------------------------------------
    //
    // add edges to the local_pairs vector for sorting
    //

    //TODO: not filling with dummy values, each localPairs can have different sizes
    std::vector<int_pair> localPairs(localM);
    if( duplicateEdges ){
        localPairs.reserve( 2*localM );
        localPairs.resize( 2*localM );
        PRINT0("will duplicate edges");
    }

    // TODO: duplicate and reverse all edges before sorting to ensure matrix will be symmetric?
    // TODO: any better way to avoid edge duplication?

    IndexType maxLocalVertex=0;
    IndexType minLocalVertex=std::numeric_limits<IndexType>::max();

    //get min and max first;
    for(IndexType i=0; i<localM; i++) {
        IndexType v1 = edgeList[i].first;
        IndexType v2 = edgeList[i].second;
        IndexType minV = std::min(v1,v2);
        IndexType maxV = std::max(v1,v2);

        if( minV<minLocalVertex ) {
            minLocalVertex = minV;
        }
        if( maxV>maxLocalVertex ) {
            maxLocalVertex = maxV;
        }
    }
    //PRINT(thisPE << ": vertices range from "<< minLocalVertex << " to " << maxLocalVertex);

    const IndexType globalMinIndex = comm->min(minLocalVertex);
    globalMinIndex==0 ? maxLocalVertex : maxLocalVertex-- ;

    //if vertices are numbered starting from 1, subtract 1 by every vertex index
    const IndexType oneORzero = globalMinIndex==0 ? 0: 1;
    IndexType selfLoopsCnt = 0;
    
    for(IndexType i=0; i<localM; i++) {
        IndexType v1 = edgeList[i].first - oneORzero;
        IndexType v2 = edgeList[i].second - oneORzero;
        if( removeSelfLoops && v1==v2 ){
            selfLoopsCnt++;
            continue;
        }
        localPairs[i].first = v1;
        localPairs[i].second = v2;

        //TODO?: insert also reversed edge to keep matrix symmetric
        if( duplicateEdges ){
            localPairs[localM+i].first = v2;
            localPairs[localM+i].second = v1;
        }
    }

    const IndexType N = comm->max( maxLocalVertex );
    const IndexType globalSelfLoops = comm->sum( selfLoopsCnt );
    //localM *=2 ;	// for the duplicated edges

    PRINT0("removed globally " << globalSelfLoops << " self loops");

    //
    // globally sort edges
    //
    std::chrono::time_point<std::chrono::steady_clock> beforeSort =  std::chrono::steady_clock::now();
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // as MPI communicator might have been splitted, take the one used by comm
    if ( comm->getType() == scai::dmemo::CommunicatorType::MPI ){
        const auto& mpiComm = static_cast<const scai::dmemo::MPICommunicator&>( *comm );
        mpi_comm = mpiComm.getMPIComm();
    }

    //sort globally
    JanusSort::sort(mpi_comm, localPairs, MPI_2INT);

    std::chrono::duration<double> sortTmpTime = std::chrono::steady_clock::now() - beforeSort;
    ValueType sortTime = comm->max( sortTmpTime.count() );
    PRINT0("time to sort edges: " << sortTime);

    //check for isolated nodes and wrong conversions
    IndexType prevVertex = localPairs[0].first;
    for (int_pair edge : localPairs) {
        //TODO: should we allow isolated vertices? (see also below)
        //not necessarily an error
        SCAI_ASSERT_LE_ERROR(edge.first, prevVertex + 1, "Gap in sorted node IDs before edge exchange in PE " << comm->getRank() );
        if( edge.first>prevVertex+1 /* and settings.verbose */ ){
        	std::cout<< "WARNING, node " << prevVertex << " has no edges, in PE " << comm->getRank() << std::endl;
        }
        prevVertex = edge.first;
    }


    //PRINT(thisPE << ": "<< localPairs.back().first << " - " << localPairs.back().second << " in total " <<  localPairs.size() );

    //-------------------------------------------------------------------
    //
    // communicate so each PE have all the edges of the last node
    // each PE just collect the edges of it last node and sends them to its +1 neighbor
    //

    // get vertex with max local id
    IndexType newMaxLocalVertex = localPairs.back().first;

    //TODO: communicate first to see if you need to send. now, just send to your +1 the your last vertex
    // store the edges you must send
    std::vector<IndexType> sendEdgeList;

    IndexType numEdgesToRemove = 0;
    for( std::vector<int_pair>::reverse_iterator edgeIt = localPairs.rbegin(); edgeIt->first==newMaxLocalVertex; ++edgeIt) {
        sendEdgeList.push_back( edgeIt->first);
        sendEdgeList.push_back( edgeIt->second);
        ++numEdgesToRemove;
    }

    if( thisPE!= comm->getSize()-1) {
        for( int i=0; i<numEdgesToRemove; i++ ) {
            localPairs.pop_back();
        }
    }

    //PRINT( thisPE << ": maxLocalVertex= " << newMaxLocalVertex << ", removed edges " << numEdgesToRemove );

    // make communication plan
    std::vector<IndexType> quantities(comm->getSize(), 0);

    if( thisPE==comm->getSize()-1 ) {	//the last PE will only receive
        // do nothing, quantities is 0 for all
    } else {
        quantities[thisPE+1] = sendEdgeList.size();		// will only send to your +1 neighbor
    }

    scai::dmemo::CommunicationPlan sendPlan( quantities.data(), comm->getSize() );

    PRINT0("allocated send plan");

    scai::dmemo::CommunicationPlan recvPlan = comm->transpose( sendPlan );

    IndexType recvEdgesSize = recvPlan.totalQuantity();
    SCAI_ASSERT_EQ_ERROR(recvEdgesSize % 2, 0, "List of received edges must have even length.");
    scai::hmemo::HArray<IndexType> recvEdges(recvEdgesSize, -1);		// the edges to be received
    //PRINT(thisPE <<": received  " << recvEdgesSize << " edges");

    PRINT0("allocated communication plans");

    {
        scai::hmemo::WriteOnlyAccess<IndexType> recvVals( recvEdges, recvEdgesSize );
        comm->exchangeByPlan( recvVals.get(), recvPlan, sendEdgeList.data(), sendPlan );
    }

    PRINT0("exchanged edges");

    // insert all the received edges to your local edges
    {
        scai::hmemo::ReadAccess<IndexType> rRecvEdges(recvEdges);
        std::vector<int_pair> recvEdgesV;
        recvEdgesV.reserve(recvEdgesSize);
        SCAI_ASSERT_EQ_ERROR(rRecvEdges.size(), recvEdgesSize, "mismatch");
        for( IndexType i=0; i<recvEdgesSize; i+=2) {
            SCAI_ASSERT_LT_ERROR(i+1, rRecvEdges.size(), "index mismatch");
            int_pair sp;
            sp.first = rRecvEdges[i];
            sp.second = rRecvEdges[i+1];
            recvEdgesV.push_back(sp);
            //PRINT( thisPE << ": received edge: "<< recvEdges[i] << " - " << recvEdges[i+1] );
        }
        std::sort( recvEdgesV.begin(), recvEdgesV.end() );

        localPairs.insert( localPairs.begin(), recvEdgesV.begin(), recvEdgesV.end() );
    }

    PRINT0("rebuild local edge list");

    //
    //remove duplicates
    //
    {
        const IndexType numEdges = localPairs.size();
        localPairs.erase( unique( localPairs.begin(), localPairs.end(), [](int_pair p1, int_pair p2) {
            return ( (p1.second==p2.second) and (p1.first==p2.first));
        }), localPairs.end() );
        const IndexType numRemoved = numEdges - localPairs.size();
        const IndexType totalNumRemoved = comm->sum(numRemoved);
        PRINT0("removed duplicates, total removed edges: " << totalNumRemoved);
    }

    SCAI_ASSERT_ERROR( std::is_sorted(localPairs.begin(), localPairs.end()), \
        "Disorder after insertion of received edges in PE " << thisPE );

    //remove self loops one more time; probably not needed
    if(removeSelfLoops) {
        const IndexType numEdges = localPairs.size();
        std::remove_if( localPairs.begin(), localPairs.end(), 
            [](int_pair p) {
                return p.first==p.second;
            }
        );
        const IndexType numRemoved = numEdges - localPairs.size();
        const IndexType totalNumRemoved = comm->sum(numRemoved);
        PRINT0("removed self loops, total removed edges: " << totalNumRemoved);
    }

    //
    // check that all is correct
    //
    newMaxLocalVertex = localPairs.back().first;
    IndexType newMinLocalVertex = localPairs[0].first;
    IndexType checkSum = newMaxLocalVertex - newMinLocalVertex  ;
    IndexType globCheckSum = comm->sum( checkSum )+ comm->getSize() -1;

    //TODO: should we allow isolated vertices?
    //this assertion triggers when the graph has isolated (with no edges) vertices
    SCAI_ASSERT_EQ_ERROR( globCheckSum, N, "Checksum mismatch, maybe some node id missing." );
    //if( globCheckSum!=N ){
    //	std::cout<<"WARNING, there is at least one node with no edges." << std::endl;
    //}

    //PRINT( *comm << ": from "<< newMinLocalVertex << " to " << newMaxLocalVertex );

    localM = localPairs.size();					// after sorting, exchange and removing duplicates

    IndexType localN = newMaxLocalVertex-newMinLocalVertex+1;
    IndexType globalN = comm->sum( localN );
    //IndexType globalM = comm->sum( localM );
    //PRINT(thisPE << ": N: localN, global= " << localN << ", " << globalN << ", \tM: local, global= " << localM  << ", " << globalM );

    //
    // create local indices and general distribution
    //
    scai::hmemo::HArray<IndexType> localIndices( localN, -1);
    IndexType index = 1;
    PRINT0("prepared data structure for local indices");

    {
        scai::hmemo::WriteAccess<IndexType> wLocalIndices(localIndices);
        IndexType oldVertex = localPairs[0].first;
        wLocalIndices[0] = oldVertex;

        // go through all local edges and add a local index if it is not already added
        for(IndexType i=1; i<localPairs.size(); i++) {
            IndexType newVertex = localPairs[i].first;
            if( newVertex!=wLocalIndices[index-1] ) {
                wLocalIndices[index++] = newVertex;
                SCAI_ASSERT_LE_ERROR( index, localN,"Too large index for localIndices array.");
            }
            // newVertex-oldVertex should be either 0 or 1, either are the same or differ by 1
            SCAI_ASSERT_LE_ERROR( newVertex-oldVertex, 1, "Vertex with id " << newVertex-1 <<" is missing. Error in edge list, vertex should be contunious");
            oldVertex = newVertex;
        }
        SCAI_ASSERT_NE_ERROR( wLocalIndices[localN-1], -1, "localIndices array not full");
    }

    PRINT0("assembled local indices");


    //-------------------------------------------------------------------
    //
    // turn the local edge list to a CSRSparseMatrix
    //

    // the CSRSparseMatrix vectors
    std::vector<IndexType> ia(localN+1);
    ia[0] = 0;
    index = 0;
    std::vector<IndexType> ja;

    for( IndexType e=0; e<localM; ) {
        IndexType v1 = localPairs[e].first;		//the vertices of this edge
        IndexType v1Degree = 0;
        // for all edges of v1
        for( std::vector<int_pair>::iterator edgeIt = localPairs.begin()+e; edgeIt->first==v1 and edgeIt!=localPairs.end(); ++edgeIt) {
            ja.push_back( edgeIt->second );	// the neighbor of v1
            //PRINT( thisPE << ": " << v1 << " -- " << 	edgeIt->second );
            ++v1Degree;
            ++e;
        }
        index++;
        //TODO: can remove the assertion if we do not initialise ia and use push_back
        SCAI_ASSERT_LE_ERROR( index, localN, thisPE << ": Wrong ia size and localN.");
        ia[index] = ia[index-1] + v1Degree;
    }
    SCAI_ASSERT_EQ_ERROR( ja.size(), localM, thisPE << ": Wrong ja size and localM.");
    std::vector<ValueType> values(ja.size(), 1);

    PRINT0("assembled CSR arrays");

    //assign/assemble the matrix
    scai::lama::CSRStorage<ValueType> myStorage ( localN, globalN,
            scai::hmemo::HArray<IndexType>(ia.size(), ia.data()),
            scai::hmemo::HArray<IndexType>(ja.size(), ja.data()),
            scai::hmemo::HArray<ValueType>(values.size(), values.data()));//no longer allowed. TODO: change

    const scai::dmemo::DistributionPtr blockDist = scai::dmemo::genBlockDistributionBySize(globalN, localN, comm);
    PRINT0("assembled CSR storage");

    return scai::lama::CSRSparseMatrix<ValueType>(blockDist, std::move(myStorage));

}//edgeList2CSR

//---------------------------------------------------------------------------------------
//WARNING,TODO: Assumes the graph is undirected
// given a non-distributed csr undirected matrix converts it to an edge list
// two first numbers are the vertex IDs and the third one is the edge weight
template<typename IndexType, typename ValueType>
std::vector<std::tuple<IndexType,IndexType,ValueType>> GraphUtils<IndexType, ValueType>::CSR2EdgeList_repl(const CSRSparseMatrix<ValueType> &graph, IndexType &maxDegree) {

    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    CSRSparseMatrix<ValueType> tmpGraph(graph);
    const IndexType N= graph.getNumRows();

    // TODO: maybe handle differently? with an error message?
    if (!tmpGraph.getRowDistributionPtr()->isReplicated()) {
        PRINT0("***WARNING: In CSR2EdgeList_repl: given graph is not replicated; will replicate now");
        const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N) );
        tmpGraph.redistribute(noDist, noDist);
        PRINT0("Graph replicated");
    }

    std::vector<std::tuple<IndexType,IndexType,ValueType>> edgeList = localCSR2GlobalEdgeList(graph,  maxDegree );

    return edgeList;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::tuple<IndexType,IndexType,ValueType>> GraphUtils<IndexType, ValueType>::localCSR2GlobalEdgeList(
    const CSRSparseMatrix<ValueType> &graph,
    IndexType &maxDegree){

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());
    
    const IndexType numLocalEdges = values.size();
    const IndexType localN = localStorage.getNumRows();

    //not needed assertion
    SCAI_ASSERT_EQ_ERROR( ja.size(), values.size(), "Size mismatch for csr sparse matrix" );
    SCAI_ASSERT_EQ_ERROR( ia.size(), localN+1, "Wrong ia size?" );

    std::vector<std::tuple<IndexType,IndexType,ValueType>> edgeList;//( numEdges/2 );
    IndexType edgeIndex = 0;

    //WARNING: we only need the upper, left part of the matrix values since
    //      matrix is symmetric
    for(IndexType i=0; i<localN; i++) {
        const IndexType v1 = dist->local2Global(i); //first vertex
        SCAI_ASSERT_LE_ERROR( i+1, ia.size(), "Wrong index for ia[i+1]" );
        IndexType thisDegree = ia[i+1]-ia[i];
        if(thisDegree>maxDegree) {
            maxDegree = thisDegree;
        }
        for (IndexType j = ia[i]; j < ia[i+1]; j++) {
            const IndexType v2 =  ja[j]; //second vertex
            //WARNING: here, we assume graph is directed
            // so we DO enter every edge twice as (u,v) and (v,u)
            //if ( v2<v1 ) {
            //    edgeIndex++;
            //    continue;
            //}

            SCAI_ASSERT_LE_ERROR( edgeIndex, numLocalEdges, "Wrong edge index");
            edgeList.push_back( std::make_tuple( v1, v2, values[edgeIndex]) );

            edgeIndex++;
        }
    }

    //SCAI_ASSERT_EQ_ERROR( edgeList.size()*2, numLocalEdges, "Wrong number of edges");// assertion when NOT adding all edges
    SCAI_ASSERT_EQ_ERROR( edgeList.size(), numLocalEdges, "Wrong number of edges");
    return edgeList;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> GraphUtils<IndexType, ValueType>::constructLaplacian(const CSRSparseMatrix<ValueType>& graph) {
    using scai::lama::CSRStorage;
    using scai::hmemo::HArray;
    using std::vector;

    const IndexType globalN = graph.getNumRows();
    const IndexType localN = graph.getLocalNumRows();

    if (graph.getNumColumns() != globalN) {
        throw std::runtime_error("Matrix must be square to be an adjacency matrix");
    }

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    vector<ValueType> targetDegree(localN,0);
    {
        const CSRStorage<ValueType>& storage = graph.getLocalStorage();
        const ReadAccess<IndexType> ia(storage.getIA());
        const ReadAccess<ValueType> values(storage.getValues());
        assert(ia.size() == localN+1);

        for (IndexType i = 0; i < localN; i++) {
            for (IndexType j = ia[i]; j < ia[i+1]; j++) {
                targetDegree[i] += values[j];
            }
        }
    }

    //the degree matrix
    CSRSparseMatrix<ValueType> degreeM;
    degreeM.setIdentity(dist);

    SCAI_ASSERT_ERROR( degreeM.isConsistent(), "identity matrix not consistent");
    scai::hmemo::HArray<ValueType> localDiagValues(targetDegree.size(), targetDegree.data() );
    scai::lama::DenseVector<ValueType> distDiagonal( dist, localDiagValues);

    degreeM.setDiagonal( distDiagonal );
    degreeM.redistribute( graph.getRowDistributionPtr(), graph.getColDistributionPtr() );

    SCAI_ASSERT_ERROR( degreeM.isConsistent(), "degree matrix not consistent");
    SCAI_ASSERT_EQ_ERROR( degreeM.getNumValues(), globalN, "matrix should be diagonal" );

    CSRSparseMatrix<ValueType> L = graph;
    
    //TODO: check, this might break the matrix's consistency
    //L = D-A
    L.matrixPlusMatrix( 1.0, degreeM, -1.0, graph );

    SCAI_ASSERT_EQ_ERROR( dist, L.getRowDistributionPtr(), "distribution mismatch" );
    SCAI_ASSERT_ERROR(L.isConsistent(), "laplacian matrix not consistent");

    return L;
}

//---------------------------------------------------------------------------------------
//TODO: diagonal elements wrongly found when row distribution and column distribution are some general distribution
//  while it works if both are a block distribution
template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> GraphUtils<IndexType, ValueType>::constructLaplacianPlusIdentity(const CSRSparseMatrix<ValueType>& graph) {
    using scai::lama::CSRStorage;
    using scai::hmemo::HArray;
    using std::vector;

    const IndexType globalN = graph.getNumRows();
    const IndexType localN = graph.getLocalNumRows();

    if (graph.getNumColumns() != globalN) {
        throw std::runtime_error("Matrix must be square to be an adjacency matrix");
    }
    //SCAI_ASSERT_EQ_ERROR( globalN, graph.getLocalNumColumns(), "Row must not have a distribution" );

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    const CSRStorage<ValueType>& storage = graph.getLocalStorage();
    const ReadAccess<IndexType> ia(storage.getIA());
    const ReadAccess<IndexType> ja(storage.getJA());
    const ReadAccess<ValueType> values(storage.getValues());
    assert(ia.size() == localN+1);
    assert(ja.size() == graph.getLocalNumValues() );

    std::vector<IndexType> newIA(ia.size());
    std::vector<IndexType> newJA(ja.size()+localN);
    std::vector<ValueType> newValues(ja.size()+localN);

    vector<ValueType> targetDegree(localN,0);
    for (IndexType i = 0; i < localN; i++) {
        const IndexType globalI = dist->local2Global(i);
        bool rightOfDiagonal = false;
        newIA[i] = ia[i] + i;//adding values at the diagonal
        newIA[i+1] = ia[i+1] + i + 1;

        for (IndexType j = ia[i]; j < ia[i+1]; j++) {
			const IndexType neighbor = ja[j];
            if (neighbor == globalI) {
                throw std::runtime_error("Forbidden self loop at " + std::to_string(globalI) + " with weight " + std::to_string(values[j]));
            }
            //if (ja[j] < globalI && rightOfDiagonal)  {
            //    throw std::runtime_error("Outgoing edges are not sorted.");
            //}
            if (neighbor > globalI) {
                if (!rightOfDiagonal) {
                    newJA[j + i] = globalI;
                }
                rightOfDiagonal = true;
            }

            const IndexType jaOffset = i + rightOfDiagonal;
            newJA[j+jaOffset] = neighbor;
            newValues[j+jaOffset] = -values[j];

            targetDegree[i] += values[j];
        }

		//if all neighbor vertices are before the diagonal
        if (!rightOfDiagonal) {
            newJA[ia[i+1] + i] = globalI;
        }

        //edge weights are summed, can now enter value at diagonal
        [[maybe_unused]] bool foundDiagonal = false;
        for (IndexType j = newIA[i]; j < newIA[i+1]; j++) {
            if (newJA[j] == globalI) {
                assert(!foundDiagonal);
                foundDiagonal = true;
                newValues[j] = targetDegree[i]*1.1;
            }
        }
        assert(foundDiagonal);
    }
    assert(newIA[localN] == newJA.size());

    const CSRStorage<ValueType> resultStorage(localN, globalN, newIA, newJA, newValues);

    CSRSparseMatrix<ValueType> result = graph;
    result.assignLocal( resultStorage, dist );
    SCAI_ASSERT_EQ_ERROR( newValues.size(), values.size()+localN, "values size wrong?");
    return result;
    //return CSRSparseMatrix<ValueType>(dist, resultStorage);
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> GraphUtils<IndexType, ValueType>::constructFJLTMatrix(ValueType epsilon, IndexType n, IndexType origDimension) {
    using scai::hmemo::HArray;
    using scai::lama::DenseMatrix;
    using scai::lama::DIASparseMatrix;
    using scai::lama::DIAStorage;
    using scai::hmemo::WriteAccess;
    using scai::lama::eval;


    const IndexType magicConstant = 0.1;
    const ValueType logn = std::log(n);
    const IndexType targetDimension = magicConstant * std::pow(epsilon, -2)*logn;

    if (origDimension <= targetDimension) {
        //better to just return the identity
        std::cout << "Target dimension " << targetDimension << " is higher than original dimension " << origDimension << ". Returning identity instead." << std::endl;
        DIASparseMatrix<ValueType> D(DIAStorage<ValueType>(origDimension, origDimension, HArray<IndexType>( { IndexType(0) } ), HArray<ValueType>(origDimension, IndexType(1) )));
        return CSRSparseMatrix<ValueType>(D);
    }

    const IndexType p = 2;
    ValueType q = std::min(ValueType((std::pow(epsilon, p-2)*std::pow(logn,p))/origDimension), ValueType(1.0));

    std::default_random_engine generator;
    std::normal_distribution<ValueType> gauss(0,q);
    std::uniform_real_distribution<ValueType> unit_interval(0.0,1.0);

    DenseMatrix<ValueType> P(targetDimension, origDimension);
    {
        WriteAccess<ValueType> wP(P.getLocalStorage().getData());
        for (IndexType i = 0; i < targetDimension*origDimension; i++) {
            if (unit_interval(generator) < q) {
                wP[i] = gauss(generator);
            } else {
                wP[i] = 0;
            }
        }
    }

    DenseMatrix<ValueType> H = constructHadamardMatrix(origDimension);

    HArray<ValueType> randomDiagonal(origDimension);
    {
        WriteAccess<ValueType> wDiagonal(randomDiagonal);
        //the following can definitely be optimized
        for (IndexType i = 0; i < origDimension; i++) {
            wDiagonal[i] = 1-2*(rand() ^ 1);
        }
    }
    //DIAStorage<ValueType> dstor(origDimension, origDimension, HArray<IndexType>({ IndexType(0) } ), randomDiagonal );
    //DIASparseMatrix<ValueType> D(std::move(dstor));
    DenseMatrix<ValueType> Ddense(origDimension, origDimension);
    Ddense.assignDiagonal(DenseVector<ValueType>(randomDiagonal));

    DenseMatrix<ValueType> PH = eval<DenseMatrix<ValueType>>(P*H);
    DenseMatrix<ValueType> denseTemp = eval<DenseMatrix<ValueType>>(PH*Ddense);
    return CSRSparseMatrix<ValueType>(denseTemp);
}

template<typename IndexType, typename ValueType>
scai::lama::DenseMatrix<ValueType> GraphUtils<IndexType, ValueType>::constructHadamardMatrix(IndexType d) {
    using scai::lama::DenseMatrix;
    using scai::hmemo::WriteAccess;
    const ValueType scalingFactor = 1/sqrt(d);
    DenseMatrix<ValueType> result(d,d);
    WriteAccess<ValueType> wResult(result.getLocalStorage().getData());
    for (IndexType i = 0; i < d; i++) {
        for (IndexType j = 0; j < d; j++) {
            IndexType dotProduct = (i-1) ^ (j-1);
            IndexType entry = 1-2*(dotProduct & 1);//(-1)^{dotProduct}
            wResult[i*d+j] = scalingFactor*entry;
        }
    }
    return result;
}

//-----------------------------------------------------------------------------------
// return[0][i] the first node, return[1][i] the second node, return[2][i] the color of the edge
// WARNING: the code from hasan thesis is faster, almost half the time.
//		For small graphs (~ <32K  edges) that is OK.
//TODO: invest possible optimizations

template<typename IndexType, typename ValueType>
std::vector< std::vector<IndexType>> GraphUtils<IndexType, ValueType>::mecGraphColoring( const CSRSparseMatrix<ValueType> &graph, IndexType &colors) {

    typedef std::tuple<IndexType,IndexType,ValueType> edge;
    typedef std::tuple<IndexType,IndexType> edgeNoWeight;
    const IndexType N = graph.getNumRows();
    colors = -1;

    std::chrono::time_point<std::chrono::steady_clock> start= std::chrono::steady_clock::now();
    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();

    if (!graph.getRowDistributionPtr()->isReplicated()) {
        PRINT0("***WARNING: In getCommunicationPairs_local: given graph is not replicated;\nAborting...");
        throw std::runtime_error("In mecGraphColoring, graph is not replicated");
    }

    // 1 - convert CSR to adjacency list

    IndexType maxDegree = 0;

    //edgeList[i] = a tuple (v1,v2,w) describing an edges. v1 and v2 are the two
    // edge IDs and w is the edge weight
    std::vector<edge> edgeList = CSR2EdgeList_repl(graph, maxDegree);

    //
    // 2 - sort adjacency list based on edge weights
    //
    std::sort( edgeList.begin(), edgeList.end(),
    [](edge v1, edge v2) {
        return std::get<2>(v1) > std::get<2>(v2);
    }
             );

    //
    // 3 - apply greedy algorithm for mec
    //

    // a map storing the color for each edge. Initialize coloring to 2*maxDegree
    std::map<edgeNoWeight, int> edgesColor;
    for(typename std::vector<edge>::iterator it=edgeList.begin(); it!=edgeList.end(); it++) {
        //edge thisEdge = *it;
        edgeNoWeight thisEdge = std::make_tuple( std::get<0>(*it), std::get<1>(*it) );
        edgesColor.insert( std::pair<edgeNoWeight,int>( thisEdge, 2*maxDegree));
    }

    // to be returned
    //retCol[0][i] the first node, retCol[1][i] the second node, retCol[2][i] the color of the edge
    std::vector< std::vector<IndexType>> retCol(3);
    start= std::chrono::steady_clock::now();
    // for all the edges
    for(typename std::vector<edge>::iterator edgeIt=edgeList.begin(); edgeIt!=edgeList.end(); edgeIt++) {
        //edge thisEdge = *edgeIt;
        IndexType v0 = std::get<0>(*edgeIt);
        IndexType v1 = std::get<1>(*edgeIt);

        // since the matrix is symmetric, we need only to check to top right half
        if( v0>v1 ) {
            continue;
        }

        //TODO: maybe too many assertion , consider removing them
        SCAI_ASSERT_LT_ERROR( v0, N, "Too large vertex ID");
        SCAI_ASSERT_LT_ERROR( v1, N, "Too large vertex ID");
        SCAI_ASSERT_GE_ERROR( v0, 0, "Negative vertex ID");
        SCAI_ASSERT_GE_ERROR( v1, 0, "Negative vertex ID");

        const CSRStorage<ValueType>& storage = graph.getLocalStorage();
        const scai::hmemo::ReadAccess<IndexType> ia(storage.getIA());
        const scai::hmemo::ReadAccess<IndexType> ja(storage.getJA());

        //TODO? maybe use a set for colors to automatically remove duplicates?
        std::vector<int> usedColors = {2*(int)maxDegree};

        // check the color of the rest of the edges for the two nodes of this edge,
        std::vector<IndexType> tmpEdge = {v0, v1};
        for(typename vector<IndexType>::iterator nodeIt=tmpEdge.begin(); nodeIt!=tmpEdge.end(); nodeIt++) {
            SCAI_ASSERT_LT_ERROR(*nodeIt, ia.size(), "Wrong node index");
            for(IndexType j=ia[*nodeIt]; j<ia[*nodeIt+1]; j++) { // for all the edges of a node
                IndexType neighbor = ja[j];
                // not check this edge, edgeIt=(v0, v1, ...)
                if(neighbor==v0 or neighbor==v1 ) {
                    continue;
                }
                // to find the edge (v0,v1), the smallest id must be first
                edgeNoWeight neighborEdge;
                if( neighbor<*nodeIt) {
                    neighborEdge = std::make_tuple( neighbor, *nodeIt );
                } else {
                    neighborEdge = std::make_tuple( *nodeIt, neighbor );
                }

                int color = edgesColor.find( neighborEdge )->second;
                usedColors.push_back( color );
            }
        }

        //find the minimum free color
        std::sort(usedColors.begin(), usedColors.end() );
        //remove duplicates
        usedColors.erase( std::unique(usedColors.begin(), usedColors.end()), usedColors.end() );

        int freeColor = -1;

        //TODO: worth to replace linear scan with an (adapted) binary search
        for(unsigned int i=0; i<usedColors.size(); i++) {
            if(usedColors[i]!=i) {
                freeColor = i;
                break;
            }
        }

        //update the colors variable
        if(freeColor>colors) {
            colors = freeColor;
        }
        SCAI_ASSERT_LT_ERROR( freeColor, 2*maxDegree, "Color too large" );

        edgesColor.find( std::make_tuple(v0, v1) )->second = freeColor;

        retCol[0].push_back( v0 );
        retCol[1].push_back( v1 );
        retCol[2].push_back( freeColor );
    }//for all edges in list

    //number of colors is the max color used +1
    colors++;

    return retCol;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType GraphUtils<IndexType, ValueType>::localSumOutgoingEdges(const CSRSparseMatrix<ValueType> &input, const bool weighted) {
    SCAI_REGION( "ParcoRepart.localSumOutgoingEdges" )
    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

    ValueType sumOutgoingEdgeWeights = 0;
    for (IndexType j = 0; j < ja.size(); j++) {
        if (!input.getRowDistributionPtr()->isLocal(ja[j])) sumOutgoingEdgeWeights += weighted ? values[j] : 1;
    }

    return sumOutgoingEdgeWeights;
}
//------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
bool GraphUtils<IndexType, ValueType>::hasSelfLoops(const CSRSparseMatrix<ValueType> &graph){
    
    const CSRStorage<ValueType>& storage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(storage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(storage.getJA());

    scai::hmemo::HArray<ValueType> diagonal;
    storage.getDiagonal( diagonal );
//for( int i)
 //   PRINT(comm->getRank() << ": " << x);
    const IndexType diagonalSum = scai::utilskernel::HArrayUtils::sum(diagonal);
    
    const scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const IndexType diagonalSumSum = comm->sum( diagonalSum );

    if( diagonalSumSum>0 ){
        return true;
    }
    return false;
}
//------------------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
std::vector<IndexType> GraphUtils<IndexType, ValueType>::indexReorderCantor(const IndexType maxIndex) {
    IndexType index = 0;
    std::vector<IndexType> ret(maxIndex, -1);
    std::vector<bool> chosen(maxIndex, false);
    //TODO: change vector of booleans?
    //bool chosen2[maxIndex]=1;

    IndexType denom;
    for( denom=1; denom<maxIndex; denom*=2) {
        for( IndexType numer=1; numer<denom; numer+=2) {
            IndexType val = maxIndex*((ValueType)numer/denom);
            //std::cout << numer <<"/" << denom << " = "<< val <<" <> ";
            ret[index++] = val;
            chosen[val]=true;
            //++index;
        }
    }
    //PRINT("Index= " << index <<", still "<< maxIndex-index << " to fill");
    for(IndexType i=0; i<maxIndex; i++) {
        if( chosen[i]==false ) {
            ret[index] = i;
            ++index;
            SCAI_ASSERT_LE_ERROR( index, maxIndex, "index too high");
        }
    }
    SCAI_ASSERT_EQ_ERROR( index, maxIndex, "index mismatch");

    return ret;
}
//-----------------------------------------------------------------------------------

template class GraphUtils<IndexType, double>;
template class GraphUtils<IndexType, float>;

} /* namespace ITI */
