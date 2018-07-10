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

#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>
#include <scai/dmemo/Halo.hpp>
#include <scai/dmemo/HaloBuilder.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>
#include <scai/lama/matrix/DenseMatrix.hpp>
#include <scai/lama/matrix/DIASparseMatrix.hpp>
#include <scai/lama/expression/all.hpp>

#include "GraphUtils.h"
#include "RBC/Sort/SQuick.hpp"


using std::vector;
using std::queue;

namespace ITI {

namespace GraphUtils {

using scai::hmemo::ReadAccess;
using scai::dmemo::Distribution;
using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::lama::CSRStorage;

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> reindex(scai::lama::CSRSparseMatrix<ValueType> &graph) {
    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    //const IndexType p = comm->getSize();

    scai::dmemo::DistributionPtr blockDist(new scai::dmemo::GenBlockDistribution(globalN, localN, comm));
    DenseVector<IndexType> result(blockDist,0);
    blockDist->getOwnedIndexes(result.getLocalValues());

    SCAI_ASSERT_EQUAL_ERROR(result.sum(), globalN*(globalN-1)/2);

    scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(graph);
    scai::hmemo::HArray<IndexType> haloData;
    comm->updateHalo( haloData, result.getLocalValues(), partHalo );

    CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    scai::hmemo::HArray<IndexType> newJA(localStorage.getJA());
    {
        scai::hmemo::ReadAccess<IndexType> rHalo(haloData);
        scai::hmemo::ReadAccess<IndexType> rResult(result.getLocalValues());
        scai::hmemo::WriteAccess<IndexType> ja( newJA );//TODO: no longer allowed, change
        for (IndexType i = 0; i < ja.size(); i++) {
            IndexType oldNeighborID = ja[i];
            IndexType localNeighbor = inputDist->global2local(oldNeighborID);
            if (localNeighbor != scai::invalidIndex) {
                ja[i] = rResult[localNeighbor];
                assert(blockDist->isLocal(ja[i]));
            } else {
                IndexType haloIndex = partHalo.global2halo(oldNeighborID);
                assert(haloIndex != scai::invalidIndex);
                ja[i] = rHalo[haloIndex];
                assert(!blockDist->isLocal(ja[i]));
            }
        }
    }

    CSRStorage<ValueType> newStorage(localN, globalN, localStorage.getIA(), newJA, localStorage.getValues());
    graph = CSRSparseMatrix<ValueType>(blockDist, std::move(newStorage));
    //graph.setDistributionPtr(blockDist);//TODO no longer allowed, change

    return result;
}

template<typename IndexType, typename ValueType>
std::vector<IndexType> localBFS(const scai::lama::CSRSparseMatrix<ValueType> &graph, const IndexType u)
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
                IndexType localNeighbor = inputDist->global2local(globalNeighbor);
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

template<typename IndexType, typename ValueType>
IndexType getLocalBlockDiameter(const CSRSparseMatrix<ValueType> &graph, const IndexType u, IndexType lowerBound, const IndexType k, IndexType maxRounds)
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
ValueType computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const bool weighted) {
	SCAI_REGION( "ParcoRepart.computeCut" )
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();

	scai::dmemo::CommunicatorPtr comm = partDist->getCommunicatorPtr();
	if( comm->getRank()==0 ){
        std::cout<<"Computing the cut...";
        std::cout.flush();
	}

	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();
	//const IndexType maxBlockID = part.max();

	std::chrono::time_point<std::chrono::system_clock> startTime =  std::chrono::system_clock::now();
    
	if (partDist->getLocalSize() != localN) {
		PRINT0("Local values mismatch for matrix and partition");
		throw std::runtime_error("partition has " + std::to_string(partDist->getLocalSize()) + " local values, but matrix has " + std::to_string(localN));
	}
	
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	scai::hmemo::HArray<IndexType> localData = part.getLocalValues();
	scai::hmemo::ReadAccess<IndexType> partAccess(localData);

	scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(input);
	scai::hmemo::HArray<IndexType> haloData;
	partDist->getCommunicatorPtr()->updateHalo( haloData, localData, partHalo );

	ValueType result = 0;
	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];
		assert(ja.size() >= endCols);

		const IndexType globalI = inputDist->local2global(i);
		//assert(partDist->isLocal(globalI));
		SCAI_ASSERT_ERROR(partDist->isLocal(globalI), "non-local index, globalI= " << globalI << " for PE " << comm->getRank() );
		
		IndexType thisBlock = partAccess[i];

		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			assert(neighbor >= 0);
			assert(neighbor < n);

			IndexType neighborBlock;
			if (partDist->isLocal(neighbor)) {
				neighborBlock = partAccess[partDist->global2local(neighbor)];
			} else {
				neighborBlock = haloData[partHalo.global2halo(neighbor)];
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

	std::chrono::duration<double> endTime = std::chrono::system_clock::now() - startTime;
	double totalTime= comm->max(endTime.count() );

    if( comm->getRank()==0 ){
        std::cout<<" done in " << totalTime << " seconds " << std::endl;
    }

    return result / 2; //counted each edge from both sides
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType computeImbalance(const DenseVector<IndexType> &part, IndexType k, const DenseVector<ValueType> &nodeWeights) {
	SCAI_REGION( "ParcoRepart.computeImbalance" )
	const IndexType globalN = part.getDistributionPtr()->getGlobalSize();
	const IndexType localN = part.getDistributionPtr()->getLocalSize();
	const IndexType weightsSize = nodeWeights.getDistributionPtr()->getGlobalSize();
	const bool weighted = (weightsSize != 0);
    scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
    
    /*
    if( comm->getRank()==0 ){
        std::cout<<"Computing the imbalance...";
        std::cout.flush();
    }
    */
    
    SCAI_ASSERT_EQ_ERROR(weighted, comm->any(weighted), "inconsistent input!");

	ValueType minWeight, maxWeight;
	if (weighted) {
		assert(weightsSize == globalN);
		assert(nodeWeights.getDistributionPtr()->getLocalSize() == localN);
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

	std::vector<ValueType> subsetSizes(k, 0);
	const IndexType minK = part.min();
	const IndexType maxK = part.max();

	if (minK < 0) {
		throw std::runtime_error("Block id " + std::to_string(minK) + " found in partition with supposedly " + std::to_string(k) + " blocks.");
	}

	if (maxK >= k) {
		throw std::runtime_error("Block id " + std::to_string(maxK) + " found in partition with supposedly " + std::to_string(k) + " blocks.");
	}

	scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());
	scai::hmemo::ReadAccess<ValueType> localWeight(nodeWeights.getLocalValues());
	assert(localPart.size() == localN);

	ValueType weightSum = 0.0;
	for (IndexType i = 0; i < localN; i++) {
		IndexType partID = localPart[i];
		ValueType weight = weighted ? localWeight[i] : 1;
		subsetSizes[partID] += weight;
		weightSum += weight;
	}
	//PRINT(*comm << ": " << ", local node weightSum= " << weightSum);
	
	ValueType optSize;
	
	if (weighted) {
		//get global weight sum
		weightSum = comm->sum(weightSum);
		optSize = std::ceil(weightSum / k + (maxWeight - minWeight));
        //optSize = std::ceil(ValueType(weightSum) / k );
	} else {
		optSize = std::ceil(ValueType(globalN) / k);
	}

    std::vector<ValueType> globalSubsetSizes(k);
    const bool isReplicated = part.getDistribution().isReplicated();
    SCAI_ASSERT_EQ_ERROR(isReplicated, comm->any(isReplicated), "inconsistent distribution!");

    if (isReplicated) {
        SCAI_ASSERT_EQUAL_ERROR(localN, globalN);
    }

	if (!isReplicated) {
            //sum block sizes over all processes
            comm->sumImpl( globalSubsetSizes.data() , subsetSizes.data(), k, scai::common::TypeTraits<ValueType>::stype);
	}else{
            globalSubsetSizes = subsetSizes;
	}

	ValueType maxBlockSize = *std::max_element(globalSubsetSizes.begin(), globalSubsetSizes.end());

	if (!weighted) {
		assert(maxBlockSize >= optSize);
	}

	/**
    if( comm->getRank()==0 ){
        std::cout<<" done" << std::endl;
    }
    */

	return (ValueType(maxBlockSize - optSize)/ optSize);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::dmemo::Halo buildNeighborHalo(const CSRSparseMatrix<ValueType>& input) {

	SCAI_REGION( "ParcoRepart.buildPartHalo" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	std::vector<IndexType> requiredHaloIndices = nonLocalNeighbors<IndexType, ValueType>(input);

	scai::dmemo::Halo halo;
	{
		scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
		scai::dmemo::HaloBuilder::build( *inputDist, arrRequiredIndexes, halo );
	}

	return halo;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
	SCAI_REGION( "ParcoRepart.nonLocalNeighbors" )
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

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
inline bool hasNonLocalNeighbors(const CSRSparseMatrix<ValueType> &input, IndexType globalID) {
	SCAI_REGION( "ParcoRepart.hasNonLocalNeighbors" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	const IndexType localID = inputDist->global2local(globalID);
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
std::vector<IndexType> getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input, const std::set<IndexType>& candidates) {
    SCAI_REGION( "ParcoRepart.getNodesWithNonLocalNeighbors_cache" );
    std::vector<IndexType> result;
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    for (IndexType globalI : candidates) {
        const IndexType localI = inputDist->global2local(globalI);
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
std::vector<IndexType> getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
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
DenseVector<IndexType> getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();
    DenseVector<IndexType> border(dist,0);
    scai::hmemo::HArray<IndexType>& localBorder= border.getLocalValues();

    //const IndexType globalN = dist->getGlobalSize();
    IndexType max = part.max();

    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

	scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(adjM);
	scai::hmemo::HArray<IndexType> haloData;
	dist->getCommunicatorPtr()->updateHalo( haloData, localPart, partHalo );

    for(IndexType i=0; i<localN; i++){    // for all local nodes
    	IndexType thisBlock = localPart[i];
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){                   // for all the edges of a node
    		IndexType neighbor = ja[j];
    		IndexType neighborBlock;
			if (dist->isLocal(neighbor)) {
				neighborBlock = partAccess[dist->global2local(neighbor)];
			} else {
				neighborBlock = haloData[partHalo.global2halo(neighbor)];
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
std::pair<std::vector<IndexType>,std::vector<IndexType>> getNumBorderInnerNodes	( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, const struct Settings settings) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    if( comm->getRank()==0 ){
        std::cout<<"Computing the border and inner nodes..." << std::endl;
    }
    std::chrono::time_point<std::chrono::system_clock> startTime =  std::chrono::system_clock::now();
	
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();

    //const IndexType globalN = dist->getGlobalSize();
    IndexType max = part.max();
	
	if(max!=settings.numBlocks-1){
		PRINT("\n\t\tWARNING: the max block id is " << max << " but it should be " << settings.numBlocks-1);
		max = settings.numBlocks-1;
	}
    
    // the number of border nodes per block
    std::vector<IndexType> borderNodesPerBlock( max+1, 0 );
    // the number of inner nodes
    std::vector<IndexType> innerNodesPerBlock( max+1, 0 );
    
    
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

	scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(adjM);
	scai::hmemo::HArray<IndexType> haloData;
	dist->getCommunicatorPtr()->updateHalo( haloData, localPart, partHalo );

    for(IndexType i=0; i<localN; i++){    // for all local nodes
    	IndexType thisBlock = localPart[i];
        SCAI_ASSERT_LE_ERROR( thisBlock , max , "Wrong block id." );
        bool isBorderNode = false;
        
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){      // for all the edges of a node
    		IndexType neighbor = ja[j];
    		IndexType neighborBlock;
			if (dist->isLocal(neighbor)) {
				neighborBlock = partAccess[dist->global2local(neighbor)];
			} else {
				neighborBlock = haloData[partHalo.global2halo(neighbor)];
			}
			SCAI_ASSERT_LE_ERROR( neighborBlock , max , "Wrong block id." );
			if (thisBlock != neighborBlock) {
                borderNodesPerBlock[thisBlock]++;   //increase number of border nodes found
                isBorderNode = true;
				break;
			}
    	}
    	//if all neighbors are in the same block then this is an inner node
        if( !isBorderNode ){
            innerNodesPerBlock[thisBlock]++; 
        }
    }

    comm->sumImpl( borderNodesPerBlock.data(), borderNodesPerBlock.data(), max+1, scai::common::TypeTraits<IndexType>::stype); 
    
    comm->sumImpl( innerNodesPerBlock.data(), innerNodesPerBlock.data(), max+1, scai::common::TypeTraits<IndexType>::stype); 
    
	std::chrono::duration<double> endTime = std::chrono::system_clock::now() - startTime;
	double totalTime= comm->max(endTime.count() );
	if( comm->getRank()==0 ){
        std::cout<<"\t\t\t time to get number of border and inner nodes : " << totalTime <<  std::endl;
    }
    
    return std::make_pair( borderNodesPerBlock, innerNodesPerBlock );
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> computeCommVolume( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, Settings settings) {
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType numBlocks = settings.numBlocks;
    
    if( comm->getRank()==0 && settings.verbose){
        std::cout<<"Computing the communication volume ...";
        std::cout.flush();
    }
    std::chrono::time_point<std::chrono::system_clock> startTime =  std::chrono::system_clock::now();
	
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();

    // the communication volume per block for this PE
    std::vector<IndexType> commVolumePerBlock( numBlocks, 0 );
    
    
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

	scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(adjM);
	scai::hmemo::HArray<IndexType> haloData;
	dist->getCommunicatorPtr()->updateHalo( haloData, localPart, partHalo );

    for(IndexType i=0; i<localN; i++){    // for all local nodes
    	IndexType thisBlock = localPart[i];
        SCAI_ASSERT_LE_ERROR( thisBlock , numBlocks , "Wrong block id." );
        std::set<IndexType> allNeighborBlocks;
        
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){       // for all the edges of a node
    		IndexType neighbor = ja[j];
    		IndexType neighborBlock;
			if (dist->isLocal(neighbor)) {
				neighborBlock = partAccess[dist->global2local(neighbor)];
			} else {
				neighborBlock = haloData[partHalo.global2halo(neighbor)];
			}
			SCAI_ASSERT_LE_ERROR( neighborBlock , numBlocks , "Wrong block id." );
            
            // found a neighbor that belongs to a different block
			if (thisBlock != neighborBlock) {
                
                typename std::set<IndexType>::iterator it = allNeighborBlocks.find( neighborBlock );
                
                if( it==allNeighborBlocks.end() ){   // this block has not been encountered before
                    allNeighborBlocks.insert( neighborBlock );
                    commVolumePerBlock[thisBlock]++;   //increase volume
                }else{
                    // if neighnor belongs to a different block but we have already found another neighbor 
                    // from that block, then do not increase volume
                }
			}
    	}
    }

    // sum local volume
    comm->sumImpl( commVolumePerBlock.data(), commVolumePerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype); 
	
	std::chrono::duration<double> endTime = std::chrono::system_clock::now() - startTime;
	double totalTime= comm->max(endTime.count() );
	if( comm->getRank()==0 && settings.verbose){
        std::cout<<" done in " << totalTime <<  std::endl;
    }
    return commVolumePerBlock;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::tuple<std::vector<IndexType>, std::vector<IndexType>, std::vector<IndexType>> computeCommBndInner( 
	const CSRSparseMatrix<ValueType> &adjM, 
	const DenseVector<IndexType> &part, 
	Settings settings) {

    const IndexType numBlocks = settings.numBlocks;
	
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    if( comm->getRank()==0 && settings.verbose){
        std::cout<<"Computing the communication volume, number of border and inner nodes ...";
        std::cout.flush();
    }
    std::chrono::time_point<std::chrono::system_clock> startTime =  std::chrono::system_clock::now();
	
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();


    // the communication volume per block for this PE
    std::vector<IndexType> commVolumePerBlock( numBlocks, 0 );
	// the number of border nodes per block
    std::vector<IndexType> borderNodesPerBlock( numBlocks, 0 );
    // the number of inner nodes
    std::vector<IndexType> innerNodesPerBlock( numBlocks, 0 );
    
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

	scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(adjM);
	scai::hmemo::HArray<IndexType> haloData;
	dist->getCommunicatorPtr()->updateHalo( haloData, localPart, partHalo );

    for(IndexType i=0; i<localN; i++){    // for all local nodes
    	IndexType thisBlock = localPart[i];
        SCAI_ASSERT_LT_ERROR( thisBlock , numBlocks , "Wrong block id." );
        bool isBorderNode = false;
        std::set<IndexType> allNeighborBlocks;
        
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){        // for all the edges of a node
    		IndexType neighbor = ja[j];
    		IndexType neighborBlock;
			if (dist->isLocal(neighbor)) {
				neighborBlock = partAccess[dist->global2local(neighbor)];
			} else {
				neighborBlock = haloData[partHalo.global2halo(neighbor)];
			}
			SCAI_ASSERT_LT_ERROR( neighborBlock , numBlocks , "Wrong block id." );
            
            // found a neighbor that belongs to a different block
			if (thisBlock != neighborBlock) {
				if( not isBorderNode){
					borderNodesPerBlock[thisBlock]++;   //increase number of border nodes found
					isBorderNode = true;
				}
				
                typename std::set<IndexType>::iterator it = allNeighborBlocks.find( neighborBlock );
                
                if( it==allNeighborBlocks.end() ){   // this block has not been encountered before
                    allNeighborBlocks.insert( neighborBlock );
                    commVolumePerBlock[thisBlock]++;   //increase volume
                }else{
                    // if neighnor belongs to a different block but we have already found another neighbor 
                    // from that block, then do not increase volume
                }
			}
    	}
		//if all neighbors are in the same block then this is an inner node
		if( !isBorderNode ){
			innerNodesPerBlock[thisBlock]++; 
		}
    }

    // sum local volume
    comm->sumImpl( commVolumePerBlock.data(), commVolumePerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype); 
	// sum border nodes
	comm->sumImpl( borderNodesPerBlock.data(), borderNodesPerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype); 
    // sum inner nodes
    comm->sumImpl( innerNodesPerBlock.data(), innerNodesPerBlock.data(), numBlocks, scai::common::TypeTraits<IndexType>::stype); 
	
	std::chrono::duration<double> endTime = std::chrono::system_clock::now() - startTime;
	double totalTime= comm->max(endTime.count() );
	if( comm->getRank()==0 && settings.verbose){
        std::cout<<" done in " << totalTime <<  std::endl;
    }
    return std::make_tuple( std::move(commVolumePerBlock), std::move(borderNodesPerBlock), std::move(innerNodesPerBlock) );
}

//---------------------------------------------------------------------------------------

/** Get the maximum degree of a graph.
 * */
template<typename IndexType, typename ValueType>
IndexType getGraphMaxDegree( const scai::lama::CSRSparseMatrix<ValueType>& adjM){

    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = distPtr->getCommunicatorPtr();
    const IndexType globalN = distPtr->getGlobalSize();
    
    {
        scai::dmemo::DistributionPtr noDist (new scai::dmemo::NoDistribution( globalN ));
        SCAI_ASSERT( adjM.getColDistributionPtr()->isEqual(*noDist) , "Adjacency matrix should have no column distribution." );
    }
    
    const scai::lama::CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    
    // local maximum degree 
    IndexType maxDegree = ia[1]-ia[0];
    
    for(int i=1; i<ia.size(); i++){
        IndexType thisDegree = ia[i]-ia[i-1];
        if( thisDegree>maxDegree){
            maxDegree = thisDegree;
        }
    }
    //return global maximum
    return comm->max( maxDegree );
}
//------------------------------------------------------------------------------

/** Compute maximum communication= max degree of the block graph, and total communication= sum of all edges
 */
template<typename IndexType, typename ValueType>
std::pair<IndexType,IndexType> computeBlockGraphComm( const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k){

    scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
    
    if( comm->getRank()==0 ){
        std::cout<<"Computing the block graph communication..." << std::endl;
    }
    //TODO: getting the block graph probably fails for p>5000, 
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = getBlockGraph( adjM, part, k);
    
    IndexType maxComm = getGraphMaxDegree<IndexType,ValueType>( blockGraph );
    IndexType totalComm = blockGraph.getNumValues()/2;
    
    return std::make_pair(maxComm, totalComm);
}

//------------------------------------------------------------------------------

/** Returns the edges of the block graph only for the local part. Eg. if blocks 1 and 2 are local
 * in this processor it finds the edge (1,2) ( and the edge (2,1)).
 * Also if the other endpoint is in another processor it finds this edge: block 1 is local, it
 * shares an edge with block 3 that is not local, this edge is found and returned.
 *
 * @param[in] adjM The adjacency matrix of the input graph.
 * @param[in] part The partition of the input graph.
 *
 * @return A 2 dimensional vector with the edges of the local parts of the block graph:
 * edge (u,v) is (ret[0][i], ret[1][i]) if block u and block v are connected.
 * edges[0]: first vetrex id, edges[1]: second vetrex id. edges[2]: the weight of the edge
 */
//return: there is an edge in the block graph between blocks ret[0][i]-ret[1][i]
template<typename IndexType, typename ValueType>
std::vector<std::vector<IndexType>> getLocalBlockGraphEdges( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part) {
    SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges");
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.initialise");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::hmemo::HArray<IndexType>& localPart= part.getLocalValues();
    //const IndexType N = adjM.getNumColumns();
    IndexType max = part.max();
   
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.initialise");
    
    
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.addLocalEdge_newVersion");
    
    scai::hmemo::HArray<IndexType> nonLocalIndices( dist->getLocalSize() ); 
    scai::hmemo::WriteAccess<IndexType> writeNLI(nonLocalIndices, dist->getLocalSize() );

    const scai::lama::CSRStorage<ValueType> localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());
    
    // we do not know the size of the non-local indices that is why we use an std::vector
    // with push_back, then convert that to a DenseVector in order to call DenseVector::gather
    // TODO: skip the std::vector to DenseVector conversion. maybe use HArray
    
    //std::vector< std::vector<IndexType> > edges(2);
    //edges[0]: first vetrex id, edges[1]: second vetrex id. edges[2]: the weight of the edge
    std::vector< std::vector<IndexType> > edges(3);	
    std::vector<IndexType> localInd, nonLocalInd;

    for(IndexType i=0; i<dist->getLocalSize(); i++){ 
        for(IndexType j=ia[i]; j<ia[i+1]; j++){ 
            if( dist->isLocal(ja[j]) ){ 
                IndexType u = localPart[i];         // partition(i)
                IndexType v = localPart[dist->global2local(ja[j])]; // partition(j), 0<j<N so take the local index of j
                assert( u < max +1);
                assert( v < max +1);
                if( u != v){    // the nodes belong to different blocks                  
                        bool add_edge = true;
                        for(IndexType k=0; k<edges[0].size(); k++){ //check that this edge is not already in
                            if( edges[0][k]==u && edges[1][k]==v ){
                            	// the edge (u,v) already exists, increase weight
                            	edges[2][k]++;
                                add_edge= false;
                                break;      
                            }
                        }
                        if( add_edge== true){       //if this edge does not exist, add it
                            edges[0].push_back(u);
                            edges[1].push_back(v);
                            edges[2].push_back(1);
                        }
                }
            } else{  // if(dist->isLocal(j)) 
                // there is an edge between i and j but index j is not local in the partition so we cannot get part[j].
                localInd.push_back(i);
                nonLocalInd.push_back(ja[j]);
            }
            
        }
    }
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.addLocalEdge_newVersion");
    
    // TODO: this seems to take quite a long !
    // take care of all the non-local indices found
    assert( localInd.size() == nonLocalInd.size() );
    scai::lama::DenseVector<IndexType> nonLocalDV( nonLocalInd.size(), 0 );
    scai::lama::DenseVector<IndexType> gatheredPart( nonLocalDV.size(),0 );
    
    //get a DenseVector from a vector
    for(IndexType i=0; i<nonLocalInd.size(); i++){
        nonLocalDV.setValue(i, nonLocalInd[i]);
    }
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.gatherNonLocal")
    //gather all non-local indexes
    gatheredPart.gather(part, nonLocalDV , scai::common::BinaryOp::COPY );
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.gatherNonLocal")
    
    assert( gatheredPart.size() == nonLocalInd.size() );
    assert( gatheredPart.size() == localInd.size() );
    
    for(IndexType i=0; i<gatheredPart.size(); i++){
        SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges.addNonLocalEdge");
        IndexType u = localPart[ localInd[i] ];         
        IndexType v = gatheredPart.getValue(i);
        assert( u < max +1);
        assert( v < max +1);
        if( u != v){    // the nodes belong to different blocks                  
            bool add_edge = true;
            for(IndexType k=0; k<edges[0].size(); k++){ //check that this edge is not already in
                if( edges[0][k]==u && edges[1][k]==v ){
                	// the edge (u,v) already exists, increase weight
                    add_edge= false;
                    edges[2][k]++;
                    break;      
                }
            }
            if( add_edge== true){       //if this edge does not exist, add it
                edges[0].push_back(u);
                edges[1].push_back(v);
                edges[2].push_back(1);
            }
        }
    }
    return edges;
}
//-----------------------------------------------------------------------------------

/** Builds the block graph of the given partition.
 * Creates an HArray that is passed around in numPEs (=comm->getSize()) rounds and every time
 * a processor writes in the array its part.
 *
 * Not distributed.
 *
 * @param[in] adjM The adjacency matric of the input graph.
 * @param[in] part The partition of the input garph.
 * @param[in] k Number of blocks.
 *
 * @return The "adjacency matrix" of the block graph. In this version is a 1-dimensional array
 * with size k*k and [i,j]= i*k+j.
 */
template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getBlockGraph( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k) {
    SCAI_REGION("ParcoRepart.getBlockGraph");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    
    // there are k blocks in the partition so the adjacency matrix for the block graph has dimensions [k x k]
    scai::dmemo::DistributionPtr distRowBlock ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, k) );  
    scai::dmemo::DistributionPtr distColBlock ( new scai::dmemo::NoDistribution( k ));
    
    // TODO: memory costly for big k
    IndexType size= k*k;
    // get, on each processor, the edges of the blocks that are local
    std::vector< std::vector<IndexType> > blockEdges = getLocalBlockGraphEdges( adjM, part);
    assert(blockEdges[0].size() == blockEdges[1].size());
    
    scai::hmemo::HArray<IndexType> sendPart(size, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvPart(size);
    
    for(IndexType round=0; round<comm->getSize(); round++){
        SCAI_REGION("ParcoRepart.getBlockGraph.shiftArray");
        {   // write your part 
            scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendPart );
            for(IndexType i=0; i<blockEdges[0].size(); i++){
                IndexType u = blockEdges[0][i];
                IndexType v = blockEdges[1][i];
                //sendPartWrite[ u*k + v ] = 1;
                sendPartWrite[ u*k + v ] = blockEdges[2][i];
            }
        }
        comm->shiftArray(recvPart , sendPart, 1);
        sendPart.swap(recvPart);
    } 
    
    // get numEdges
    IndexType numEdges=0;
    
    scai::hmemo::ReadAccess<IndexType> recvPartRead( recvPart );
    for(IndexType i=0; i<recvPartRead.size(); i++){
        if( recvPartRead[i]>0 )
            ++numEdges;
    }
    
    //convert the k*k HArray to a [k x k] CSRSparseMatrix
    scai::lama::CSRStorage<ValueType> localMatrix;
    localMatrix.allocate( k ,k );
    
    scai::hmemo::HArray<IndexType> csrIA;
    scai::hmemo::HArray<IndexType> csrJA;
    scai::hmemo::HArray<ValueType> csrValues( numEdges, 0.0 ); 
    {
        IndexType numNZ = numEdges;     // this equals the number of edges of the graph
        scai::hmemo::WriteOnlyAccess<IndexType> ia( csrIA, k +1 );
        scai::hmemo::WriteOnlyAccess<IndexType> ja( csrJA, numNZ );
        scai::hmemo::WriteOnlyAccess<ValueType> values( csrValues );   
        scai::hmemo::ReadAccess<IndexType> recvPartRead( recvPart );
        ia[0]= 0;
        
        IndexType rowCounter = 0; // count rows
        IndexType nnzCounter = 0; // count non-zero elements
        
        for(IndexType i=0; i<k; i++){
            IndexType rowNums=0;
            // traverse the part of the HArray that represents a row and find how many elements are in this row
            for(IndexType j=0; j<k; j++){
                if( recvPartRead[i*k+j] >0  ){
                    ++rowNums;
                }
            }
            ia[rowCounter+1] = ia[rowCounter] + rowNums;
           
            for(IndexType j=0; j<k; j++){
                if( recvPartRead[i*k +j] >0){   // there exist edge (i,j)
                    ja[nnzCounter] = j;
                    //values[nnzCounter] = 1;
                    values[nnzCounter] = recvPartRead[i*k +j];
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
//----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const scai::dmemo::Halo& halo) {
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	scai::dmemo::DistributionPtr distPEs ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, comm->getSize()) );
	assert(distPEs->getLocalSize() == 1);
	scai::dmemo::DistributionPtr noDistPEs (new scai::dmemo::NoDistribution( comm->getSize() ));

    const scai::dmemo::CommunicationPlan& plan = halo.getProvidesPlan();
    std::vector<IndexType> neighbors;
    std::vector<ValueType> edgeCount;
    for (IndexType i = 0; i < plan.size(); i++) {
    	if (plan[i].quantity > 0) {
    		neighbors.push_back(plan[i].partitionId);
    		edgeCount.push_back(plan[i].quantity);
    	}
    }
    const IndexType numNeighbors = neighbors.size();

    SCAI_REGION_START("ParcoRepart.getPEGraph.buildMatrix");
	scai::hmemo::HArray<IndexType> ia(2, 0, numNeighbors);
	scai::hmemo::HArray<IndexType> ja(numNeighbors, neighbors.data());
	scai::hmemo::HArray<ValueType> values(edgeCount.size(), edgeCount.data());
	scai::lama::CSRStorage<ValueType> myStorage(1, comm->getSize(), numNeighbors, ia, ja, values);
	SCAI_REGION_END("ParcoRepart.getPEGraph.buildMatrix");

    scai::lama::CSRSparseMatrix<ValueType> PEgraph(distPEs, noDistPEs);
    PEgraph.swapLocalStorage(myStorage);

    return PEgraph;
}
//-----------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const CSRSparseMatrix<ValueType> &adjM) {
    SCAI_REGION("ParcoRepart.getPEGraph");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr(); 
    const IndexType numPEs = comm->getSize();
    
    const std::vector<IndexType> nonLocalIndices = GraphUtils::nonLocalNeighbors<IndexType, ValueType>(adjM);
    
    SCAI_REGION_START("ParcoRepart.getPEGraph.getOwners");
    scai::hmemo::HArray<IndexType> indexTransport(nonLocalIndices.size(), nonLocalIndices.data());
    // find the PEs that own every non-local index
    scai::hmemo::HArray<IndexType> owners(nonLocalIndices.size() , -1);
    dist->computeOwners( owners, indexTransport);
    SCAI_REGION_END("ParcoRepart.getPEGraph.getOwners");
    
    scai::hmemo::ReadAccess<IndexType> rOwners(owners);
    std::vector<IndexType> neighborPEs(rOwners.get(), rOwners.get()+rOwners.size());
    rOwners.release();
    
    std::map<IndexType, unsigned int> edgeWeights;
    for( int i=0; i<neighborPEs.size(); i++){
	   	edgeWeights[ neighborPEs[i] ]++;
  	}
    
    /*
    {	
    	for(int i=0; i< neighborPEs.size(); i++)
    		PRINT(*comm << ": " << neighborPEs[i] );

	    int c=0;
    	for( auto edge = edgeWeights.begin(); edge!=edgeWeights.end(); edge++ ){
    		PRINT(comm->getRank() << "-" << edge->first << " , weight= " << edge->second);
    		//PRINT( *comm << ": " << edgeWeights[edge->first] );
    	}
    }
    */

    //TODO: maybe no need to sort and remove duplicates...
    
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
    for( auto edge = edgeWeights.begin(); edge!=edgeWeights.end(); edge++ ){
    	values[ii] = edge->second;
    	//PRINT(*comm << ": values[" << ii << "]= " << values[ii] );
    	ii++;
    }

    scai::lama::CSRStorage<ValueType> myStorage(1, numPEs, std::move(ia), std::move(ja), std::move(values));
    SCAI_REGION_END("ParcoRepart.getPEGraph.buildMatrix");
    
    scai::lama::CSRSparseMatrix<ValueType> PEgraph(distPEs, std::move(myStorage));

    return PEgraph;
}
//-----------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getCSRmatrixFromAdjList_NoEgdeWeights( const std::vector<std::set<IndexType>>& adjList) {
    
    IndexType N = adjList.size();

    // the CSRSparseMatrix vectors
    std::vector<IndexType> ia(N+1);
    ia[0] = 0;
    std::vector<IndexType> ja;
        
    for(IndexType i=0; i<N; i++){
        std::set<IndexType> neighbors = adjList[i]; // the neighbors of this vertex
        for( typename std::set<IndexType>::iterator it=neighbors.begin(); it!=neighbors.end(); it++){
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
scai::lama::CSRSparseMatrix<ValueType> edgeList2CSR( std::vector< std::pair<IndexType, IndexType>> &edgeList ){

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType thisPE = comm->getRank();
	IndexType localM = edgeList.size();
		
    int typesize;
	MPI_Type_size(SortingDatatype<int_pair>::getMPIDatatype(), &typesize);
	assert(typesize == sizeof(int_pair));
	
	//-------------------------------------------------------------------
	//
	// add edges to the local_pairs vector for sorting
	//
	
	//TODO: not filling with dummy values, each localPairs can have different sizes
	std::vector<int_pair> localPairs(localM*2);
	
	// duplicate and reverse all edges before sorting to ensure matrix will be symmetric
	// TODO: any better way to avoid edge duplication?
	
	IndexType maxLocalVertex=0;
	IndexType minLocalVertex=std::numeric_limits<IndexType>::max();
	
	for(IndexType i=0; i<localM; i++){
		IndexType v1 = edgeList[i].first;
		IndexType v2 = edgeList[i].second;
		localPairs[2*i].first = v1;
		localPairs[2*i].second = v2;
		
		//insert also reversed edge to keep matrix symmetric
		localPairs[2*i+1].first = v2;
		localPairs[2*i+1].second = v1;
		
		IndexType minV = std::min(v1,v2);
		IndexType maxV = std::max(v1,v2);
		
		if( minV<minLocalVertex ){
			minLocalVertex = minV;
		}
		if( maxV>maxLocalVertex ){
			maxLocalVertex = maxV;
		}
	}
	//PRINT(thisPE << ": vertices range from "<< minLocalVertex << " to " << maxLocalVertex);
	
	const IndexType N = comm->max( maxLocalVertex );
	localM *=2 ;	// for the duplicated edges
	
	//
	// globally sort edges
	//
    std::chrono::time_point<std::chrono::system_clock> beforeSort =  std::chrono::system_clock::now();
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
	SQuick::sort<int_pair>(mpi_comm, localPairs, -1);
	
	std::chrono::duration<double> sortTmpTime = std::chrono::system_clock::now() - beforeSort;
	ValueType sortTime = comm->max( sortTmpTime.count() );
	PRINT0("time to sort edges: " << sortTime);
	
	//check for isolated nodes and wrong conversions
	IndexType lastNode = localPairs[0].first;
	for (int_pair edge : localPairs) {
	    SCAI_ASSERT_LE_ERROR(edge.first, lastNode + 1, "Gap in sorted node IDs before edge exchange.");
	    lastNode = edge.first;
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
	for( std::vector<int_pair>::reverse_iterator edgeIt = localPairs.rbegin(); edgeIt->first==newMaxLocalVertex; ++edgeIt){
		sendEdgeList.push_back( edgeIt->first);
		sendEdgeList.push_back( edgeIt->second);
		++numEdgesToRemove;
	}
	
	if( thisPE!= comm->getSize()-1){
		for( int i=0; i<numEdgesToRemove; i++ ){
			localPairs.pop_back();
		}
	}

    // make communication plan
    std::vector<IndexType> quantities(comm->getSize(), 0);
		
	if( thisPE==comm->getSize()-1 ){	//the last PE will only receive
		// do nothing, quantities is 0 for all
	}else{
		quantities[thisPE+1] = sendEdgeList.size();		// will only send to your +1 neighbor
	}
	
	scai::dmemo::CommunicationPlan sendPlan( quantities.data(), comm->getSize() );
	
	PRINT0("allocated send plan");

	scai::dmemo::CommunicationPlan recvPlan;
	recvPlan.allocateTranspose( sendPlan, *comm );
	
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

	//const IndexType minLocalVertexBeforeInsertion = localPairs.front().first;

	// insert all the received edges to your local edges
	{
        scai::hmemo::ReadAccess<IndexType> rRecvEdges(recvEdges);
        SCAI_ASSERT_EQ_ERROR(rRecvEdges.size(), recvEdgesSize, "mismatch");
        for( IndexType i=0; i<recvEdgesSize; i+=2){
            SCAI_ASSERT_LT_ERROR(i+1, rRecvEdges.size(), "index mismatch");
            int_pair sp;
            sp.first = rRecvEdges[i];
            sp.second = rRecvEdges[i+1];
            localPairs.insert( localPairs.begin(), sp);//this is horribly expensive! Will move the entire list of local edges with each insertion!
            //PRINT( thisPE << ": recved edge: "<< recvEdges[i] << " - " << recvEdges[i+1] );
        }
	}

	PRINT0("rebuild local edge list");

	//IndexType numEdges = localPairs.size() ;
	
	SCAI_ASSERT_ERROR(std::is_sorted(localPairs.begin(), localPairs.end()), "Disorder after insertion of received edges." );

	//
	//remove duplicates
	//
	localPairs.erase(unique(localPairs.begin(), localPairs.end(), [](int_pair p1, int_pair p2) {
		return ( (p1.second==p2.second) and (p1.first==p2.first)); 	}), localPairs.end() );
	//PRINT( thisPE <<": removed " << numEdges - localPairs.size() << " duplicate edges" );

	PRINT0("removed duplicates");

	//
	// check that all is correct
	//
	newMaxLocalVertex = localPairs.back().first;
	IndexType newMinLocalVertex = localPairs[0].first;
	IndexType checkSum = newMaxLocalVertex - newMinLocalVertex;
	IndexType globCheckSum = comm->sum( checkSum ) + comm->getSize() -1;

	SCAI_ASSERT_EQ_ERROR( globCheckSum, N , "Checksum mismatch, maybe some node id missing." );
	
	//PRINT( *comm << ": from "<< newMinLocalVertex << " to " << newMaxLocalVertex );
	
	localM = localPairs.size();					// after sorting, exchange and removing duplicates
	
	IndexType localN = newMaxLocalVertex-newMinLocalVertex+1;	
	IndexType globalN = comm->sum( localN );	
	//IndexType globalM = comm->sum( localM );
	//PRINT(thisPE << ": N: localN, global= " << localN << ", " << globalN << ", \tM: local, global= " << localM  << ", " << globalM );

	//
	// create local indices and general distribution
	//
	scai::hmemo::HArray<IndexType> localIndices( localN , -1);
	IndexType index = 1;
	PRINT0("prepared data structure for local indices");
	
	{
		scai::hmemo::WriteAccess<IndexType> wLocalIndices(localIndices);
		IndexType oldVertex = localPairs[0].first;
		wLocalIndices[0] = oldVertex;
		
		// go through all local edges and add a local index if it is not already added
		for(IndexType i=1; i<localPairs.size(); i++){
			IndexType newVertex = localPairs[i].first;
			if( newVertex!=wLocalIndices[index-1] ){
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
	
	const scai::dmemo::DistributionPtr genDist(new scai::dmemo::GeneralDistribution(globalN, localIndices, comm));//this could be a GenBlockDistribution, right?
	
	//-------------------------------------------------------------------
	//
	// turn the local edge list to a CSRSparseMatrix
	//
	
	// the CSRSparseMatrix vectors
    std::vector<IndexType> ia(localN+1);
    ia[0] = 0;
	index = 0;
    std::vector<IndexType> ja;
	
	for( IndexType e=0; e<localM; ){
		IndexType v1 = localPairs[e].first;		//the vertices of this edge
		IndexType v1Degree = 0;
		// for all edges of v1
		for( std::vector<int_pair>::iterator edgeIt = localPairs.begin()+e; edgeIt->first==v1 and edgeIt!=localPairs.end(); ++edgeIt){
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
	
	const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

	PRINT0("assembled CSR storage");
	
	return scai::lama::CSRSparseMatrix<ValueType>(genDist, std::move(myStorage));
	
}

//--------------------------------------------------------------------------------------- 
// given a non-distributed csr matrix converts it to an edge list
// two first numbers are the vertex ID and the third one is the edge weight
template<typename IndexType, typename ValueType>
std::vector<std::tuple<IndexType,IndexType,IndexType>> CSR2EdgeList_local(const CSRSparseMatrix<ValueType> &graph, IndexType &maxDegree) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    CSRSparseMatrix<ValueType> tmpGraph(graph);
	const IndexType N= graph.getNumRows();

	// TODO: maybe handle differently? with an error message?
	if (!tmpGraph.getRowDistributionPtr()->isReplicated()) {
		PRINT0("***WARNING: In CSR2EdgeList_local: given graph is not replicated; will replicate now");
		const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N) );
		tmpGraph.redistribute(noDist, noDist);
		PRINT0("Graph replicated");
	}	

	const CSRStorage<ValueType>& localStorage = tmpGraph.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	//not needed assertion
	SCAI_ASSERT_EQ_ERROR( ja.size(), values.size(), "Size mismatch for csr sparse matrix" );
	SCAI_ASSERT_EQ_ERROR( ia.size(), N+1, "Wrong ia size?" );	

	const IndexType numEdges = values.size();
	std::vector<std::tuple<IndexType,IndexType,IndexType>> edgeList;//( numEdges/2 );
	IndexType edgeIndex = 0;
	//IndexType maxDegree = 0;

	//WARNING: we only need the upper, left part of the matrix values since
	//		matrix is symmetric
	for(IndexType i=0; i<N; i++){
		const IndexType v1 = i;	//first vertex
		SCAI_ASSERT_LE_ERROR( i+1, ia.size(), "Wrong index for ia[i+1]" );
		IndexType thisDegree = ia[i+1]-ia[i];
		if(thisDegree>maxDegree){
			maxDegree = thisDegree;
		}
    	for (IndexType j = ia[i]; j < ia[i+1]; j++) {
    		const IndexType v2 = ja[j]; //second vertex
    		// so we do not enter every edge twice, assuming graph is undirected
    		if ( v2<v1 ){
    			edgeIndex++;
    			continue;
    		}
    		SCAI_ASSERT_LE_ERROR( edgeIndex, numEdges, "Wrong edge index");
    		edgeList.push_back( std::make_tuple( v1, v2, values[edgeIndex]) );
    		edgeIndex++;
    	}
    }
    SCAI_ASSERT_EQ_ERROR( edgeList.size()*2, numEdges, "Wrong number of edges");
    return edgeList;
}

//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> constructLaplacian(CSRSparseMatrix<ValueType> graph) {
    using scai::lama::CSRStorage;
    using scai::hmemo::HArray;
    using std::vector;

    const IndexType globalN = graph.getNumRows();
    const IndexType localN = graph.getLocalNumRows();

    if (graph.getNumColumns() != globalN) {
        throw std::runtime_error("Matrix must be square to be an adjacency matrix");
    }

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

    const CSRStorage<ValueType>& storage = graph.getLocalStorage();
    const ReadAccess<IndexType> ia(storage.getIA());
    const ReadAccess<IndexType> ja(storage.getJA());
    const ReadAccess<ValueType> values(storage.getValues());
    assert(ia.size() == localN+1);

    vector<ValueType> targetDegree(localN,0);
    for (IndexType i = 0; i < localN; i++) {
        const IndexType globalI = dist->local2global(i);
        for (IndexType j = ia[i]; j < ia[i+1]; j++) {
            if (ja[j] == globalI) {
                throw std::runtime_error("Forbidden self loop at " + std::to_string(globalI) + " with weight " + std::to_string(values[j]));
            }
            targetDegree[i] += values[j];
        }
    }

    //in the diagonal matrix, each node has one loop
    scai::hmemo::HArray<IndexType> dIA(localN+1, IndexType(0));
    scai::utilskernel::HArrayUtils::setSequence(dIA, IndexType(0), IndexType(1), dIA.size());
    //... to itself
    scai::hmemo::HArray<IndexType> dJA(localN, IndexType(0));
    dist->getOwnedIndexes(dJA);
    // with the degree as value
    scai::hmemo::HArray<ValueType> dValues(localN, targetDegree.data());

    CSRStorage<ValueType> dStorage(localN, globalN, dIA, dJA, dValues );

    CSRSparseMatrix<ValueType> D(dist, std::move(dStorage));

    auto result = scai::lama::eval<CSRSparseMatrix<ValueType>>(D - graph);
    assert(result.getNumValues() == graph.getNumValues() + globalN);

    return result;
}

//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> constructFJLTMatrix(ValueType epsilon, IndexType n, IndexType origDimension) {
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
    ValueType q = std::min((std::pow(epsilon, p-2)*std::pow(logn,p))/origDimension, 1.0);

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

    DenseMatrix<ValueType> H = constructHadamardMatrix<IndexType,ValueType>(origDimension);

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
scai::lama::DenseMatrix<ValueType> constructHadamardMatrix(IndexType d) {
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
template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> mecGraphColoring( const CSRSparseMatrix<ValueType> &graph ) {

// 1 - convert CSR to adjacency list

// 2 - sort adjacency list based on edge weights

// 3 - apply greedy (quadratic) algorithm for mec


}
//-----------------------------------------------------------------------------------



template scai::lama::DenseVector<IndexType> reindex(CSRSparseMatrix<ValueType> &graph);
template std::vector<IndexType> localBFS(const CSRSparseMatrix<ValueType> &graph, IndexType u);
template IndexType getLocalBlockDiameter(const CSRSparseMatrix<ValueType> &graph, const IndexType u, IndexType lowerBound, const IndexType k, IndexType maxRounds);
template ValueType computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, bool weighted);
template ValueType computeImbalance(const DenseVector<IndexType> &part, IndexType k, const DenseVector<ValueType> &nodeWeights);
template scai::dmemo::Halo buildNeighborHalo<IndexType,ValueType>(const CSRSparseMatrix<ValueType> &input);
template bool hasNonLocalNeighbors(const CSRSparseMatrix<ValueType> &input, IndexType globalID);
template std::vector<IndexType> getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input);
template std::vector<IndexType> getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input, const std::set<IndexType>& candidates);
template std::vector<IndexType> nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input);
template DenseVector<IndexType> getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part);
template std::pair<std::vector<IndexType>,std::vector<IndexType>> getNumBorderInnerNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, Settings settings);
template std::tuple<std::vector<IndexType>, std::vector<IndexType>, std::vector<IndexType>> computeCommBndInner( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, Settings settings);
template std::vector<IndexType> computeCommVolume( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, Settings settings);
template std::vector<std::vector<IndexType>> getLocalBlockGraphEdges( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part);
template scai::lama::CSRSparseMatrix<ValueType> getBlockGraph( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k);
template IndexType getGraphMaxDegree( const scai::lama::CSRSparseMatrix<ValueType>& adjM);
template  std::pair<IndexType,IndexType> computeBlockGraphComm( const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k);
template scai::lama::CSRSparseMatrix<ValueType> getPEGraph<IndexType,ValueType>( const scai::lama::CSRSparseMatrix<ValueType> &adjM);
template scai::lama::CSRSparseMatrix<ValueType> getCSRmatrixFromAdjList_NoEgdeWeights( const std::vector<std::set<IndexType>> &adjList);
template scai::lama::CSRSparseMatrix<ValueType> edgeList2CSR( std::vector< std::pair<IndexType, IndexType>> &edgeList );
template std::vector<std::tuple<IndexType,IndexType,IndexType>> CSR2EdgeList_local(const CSRSparseMatrix<ValueType>& graph, IndexType& maxDegree);
template scai::lama::CSRSparseMatrix<ValueType> constructLaplacian<IndexType, ValueType>(scai::lama::CSRSparseMatrix<ValueType> graph);
template scai::lama::CSRSparseMatrix<ValueType> constructFJLTMatrix(ValueType epsilon, IndexType n, IndexType origDimension);
template scai::lama::DenseMatrix<ValueType> constructHadamardMatrix(IndexType d);


} /*namespace GraphUtils*/




} /* namespace ITI */
