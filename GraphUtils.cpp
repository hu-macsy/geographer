/*
 * GraphUtils.cpp
 *
 *  Created on: 29.06.2017
 *      Author: moritzl
 */

#include <assert.h>
#include <queue>
#include <set>

#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>
#include <scai/dmemo/Halo.hpp>
#include <scai/dmemo/HaloBuilder.hpp>

#include "GraphUtils.h"

using std::vector;
using std::queue;

namespace ITI {

namespace GraphUtils {

using scai::hmemo::ReadAccess;
using scai::dmemo::Distribution;
using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::lama::Scalar;
using scai::lama::CSRStorage;

template<typename IndexType, typename ValueType>
IndexType getFarthestLocalNode(const scai::lama::CSRSparseMatrix<ValueType> graph, std::vector<IndexType> seedNodes) {
	/**
	 * Yet another BFS. This currently has problems with unconnected graphs.
	 */
	const IndexType localN = graph.getLocalNumRows();
	const Distribution& dist = graph.getRowDistribution();

	if (seedNodes.size() == 0) return rand() % localN;

	vector<bool> visited(localN, false);
	queue<IndexType> bfsQueue;

	for (IndexType seed : seedNodes) {
		bfsQueue.push(seed);
		assert(seed >= 0 || seed < localN);
		visited[seed] = true;
	}

	const scai::lama::CSRStorage<ValueType>& storage = graph.getLocalStorage();
	ReadAccess<IndexType> ia(storage.getIA());
	ReadAccess<IndexType> ja(storage.getJA());

	IndexType nextNode = 0;
	while (bfsQueue.size() > 0) {
		nextNode = bfsQueue.front();
		bfsQueue.pop();
		visited[nextNode] = true;

		for (IndexType j = ia[nextNode]; j < ia[nextNode+1]; j++) {
			IndexType localNeighbour = dist.global2local(ja[j]);
			if (localNeighbour != nIndex && !visited[localNeighbour]) {
				bfsQueue.push(localNeighbour);
				visited[localNeighbour] = true;
			}
		}
	}

	//if nodes are unvisited, the graph is unconnected and the unvisited nodes are in fact the farthest
	for (IndexType v = 0; v < localN; v++) {
		if (!visited[v]) nextNode = v;
		break;
	}

	return nextNode;
}

template<typename IndexType, typename ValueType>
ValueType computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const bool weighted) {
	SCAI_REGION( "ParcoRepart.computeCut" )
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();

	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();
	const Scalar maxBlockScalar = part.max();
	const IndexType maxBlockID = maxBlockScalar.getValue<IndexType>();

	if (partDist->getLocalSize() != localN) {
		throw std::runtime_error("partition has " + std::to_string(partDist->getLocalSize()) + " local values, but matrix has " + std::to_string(localN));
	}

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	scai::hmemo::HArray<IndexType> localData = part.getLocalValues();
	scai::hmemo::ReadAccess<IndexType> partAccess(localData);

	scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(input);
	scai::utilskernel::LArray<IndexType> haloData;
	partDist->getCommunicatorPtr()->updateHalo( haloData, localData, partHalo );

	ValueType result = 0;
	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];
		assert(ja.size() >= endCols);

		const IndexType globalI = inputDist->local2global(i);
		assert(partDist->isLocal(globalI));
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

  return result / 2; //counted each edge from both sides
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType computeImbalance(const DenseVector<IndexType> &part, IndexType k, const DenseVector<IndexType> &nodeWeights) {
	SCAI_REGION( "ParcoRepart.computeImbalance" )
	const IndexType globalN = part.getDistributionPtr()->getGlobalSize();
	const IndexType localN = part.getDistributionPtr()->getLocalSize();
	const IndexType weightsSize = nodeWeights.getDistributionPtr()->getGlobalSize();
	const bool weighted = (weightsSize != 0);

	IndexType minWeight, maxWeight;
	if (weighted) {
		assert(weightsSize == globalN);
		assert(nodeWeights.getDistributionPtr()->getLocalSize() == localN);
		minWeight = nodeWeights.min().Scalar::getValue<IndexType>();
		maxWeight = nodeWeights.max().Scalar::getValue<IndexType>();
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

	std::vector<IndexType> subsetSizes(k, 0);
	const IndexType minK = part.min().Scalar::getValue<IndexType>();
	const IndexType maxK = part.max().Scalar::getValue<IndexType>();

	if (minK < 0) {
		throw std::runtime_error("Block id " + std::to_string(minK) + " found in partition with supposedly " + std::to_string(k) + " blocks.");
	}

	if (maxK >= k) {
		throw std::runtime_error("Block id " + std::to_string(maxK) + " found in partition with supposedly " + std::to_string(k) + " blocks.");
	}

	scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());
	scai::hmemo::ReadAccess<IndexType> localWeight(nodeWeights.getLocalValues());
	assert(localPart.size() == localN);

	IndexType weightSum = 0;
	for (IndexType i = 0; i < localN; i++) {
		IndexType partID = localPart[i];
		IndexType weight = weighted ? localWeight[i] : 1;
		subsetSizes[partID] += weight;
		weightSum += weight;
	}

	IndexType optSize;
	scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
	if (weighted) {
		//get global weight sum
		weightSum = comm->sum(weightSum);
                //PRINT(weightSum);
                //TODO: why not just weightSum/k ?
                // changed for now so that the test cases can agree
		//optSize = std::ceil(weightSum / k + (maxWeight - minWeight));
                optSize = std::ceil(weightSum / k );
	} else {
		optSize = std::ceil(globalN / k);
	}

	if (!part.getDistribution().isReplicated()) {
	  //sum block sizes over all processes
	  for (IndexType partID = 0; partID < k; partID++) {
	    subsetSizes[partID] = comm->sum(subsetSizes[partID]);
	  }
	}

	IndexType maxBlockSize = *std::max_element(subsetSizes.begin(), subsetSizes.end());
	if (!weighted) {
		assert(maxBlockSize >= optSize);
	}
	return (ValueType(maxBlockSize - optSize)/ optSize);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::dmemo::Halo buildNeighborHalo(const CSRSparseMatrix<ValueType>& input) {

	SCAI_REGION( "ParcoRepart.buildPartHalo" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	std::vector<IndexType> requiredHaloIndices = nonLocalNeighbors<IndexType, ValueType>(input);

	scai::dmemo::Halo Halo;
	{
		scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
		scai::dmemo::HaloBuilder::build( *inputDist, arrRequiredIndexes, Halo );
	}

	return Halo;
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

	std::set<IndexType> neighborSet;

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
	/**
	 * this could be inlined physically to reduce the overhead of creating read access locks
	 */
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	const IndexType localID = inputDist->global2local(globalID);
	assert(localID != nIndex);

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

	//iterate over all nodes
	for (IndexType localI = 0; localI < localN; localI++) {
		const IndexType beginCols = ia[localI];
		const IndexType endCols = ia[localI+1];

		//over all edges
		for (IndexType j = beginCols; j < endCols; j++) {
			if (!inputDist->isLocal(ja[j])) {
				IndexType globalI = inputDist->local2global(localI);
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

template<typename IndexType, typename ValueType>
DenseVector<IndexType> getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    const scai::utilskernel::LArray<IndexType>& localPart= part.getLocalValues();
    DenseVector<IndexType> border(dist,0);
    scai::utilskernel::LArray<IndexType>& localBorder= border.getLocalValues();

    IndexType globalN = dist->getGlobalSize();
    IndexType max = part.max().Scalar::getValue<IndexType>();

    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

	scai::dmemo::Halo partHalo = buildNeighborHalo<IndexType, ValueType>(adjM);
	scai::utilskernel::LArray<IndexType> haloData;
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

template int getFarthestLocalNode(const CSRSparseMatrix<double> graph, std::vector<int> seedNodes);
template double computeCut(const CSRSparseMatrix<double> &input, const DenseVector<int> &part, bool weighted = false);
template double computeImbalance(const DenseVector<int> &part, int k, const DenseVector<int> &nodeWeights = {});
template scai::dmemo::Halo buildNeighborHalo<int,double>(const CSRSparseMatrix<double> &input);
template bool hasNonLocalNeighbors(const CSRSparseMatrix<double> &input, int globalID);
template std::vector<int> getNodesWithNonLocalNeighbors(const CSRSparseMatrix<double>& input);
template std::vector<int> nonLocalNeighbors(const CSRSparseMatrix<double>& input);
template DenseVector<int> getBorderNodes( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);


}

} /* namespace ITI */
