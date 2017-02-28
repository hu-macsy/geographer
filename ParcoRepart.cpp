/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include <scai/dmemo/HaloBuilder.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>
#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>
#include <scai/tracing.hpp>

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

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "HilbertCurve.h"

#include "quadtree/QuadTreeCartesianEuclid.h"

namespace ITI {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings)
{
	IndexType k = settings.numBlocks;
	ValueType epsilon = settings.epsilon;
    
	SCAI_REGION( "ParcoRepart.partitionGraph" )

	std::chrono::time_point<std::chrono::steady_clock> start, afterSFC, round;
	start = std::chrono::steady_clock::now();

	SCAI_REGION_START("ParcoRepart.partitionGraph.inputCheck")
	/**
	* check input arguments for sanity
	*/
	IndexType n = input.getNumRows();
	if (n != coordinates[0].size()) {
		throw std::runtime_error("Matrix has " + std::to_string(n) + " rows, but " + std::to_string(coordinates[0].size())
		 + " coordinates are given.");
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

	const IndexType dimensions = coordinates.size();
        
	const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
	const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();

	const IndexType localN = inputDist->getLocalSize();
	const IndexType globalN = inputDist->getGlobalSize();

	if( !coordDist->isEqual( *inputDist) ){
		throw std::runtime_error( "Distributions should be equal.");
	}
	SCAI_REGION_END("ParcoRepart.partitionGraph.inputCheck")
	
	/**
	*	gather information for space-filling curves
	*/
	std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
	DenseVector<IndexType> result;

	{
		SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition" )

		/**
		 * get minimum / maximum of local coordinates
		 */
		for (IndexType dim = 0; dim < dimensions; dim++) {
			//get local parts of coordinates
			scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[dim].getLocalValues();
			for (IndexType i = 0; i < localN; i++) {
				ValueType coord = localPartOfCoords[i];
				if (coord < minCoords[dim]) minCoords[dim] = coord;
				if (coord > maxCoords[dim]) maxCoords[dim] = coord;
			}
		}

		/**
		 * communicate to get global min / max
		 */
		for (IndexType dim = 0; dim < dimensions; dim++) {
			minCoords[dim] = comm->min(minCoords[dim]);
			maxCoords[dim] = comm->max(maxCoords[dim]);
		}
	

		ValueType maxExtent = 0;
		for (IndexType dim = 0; dim < dimensions; dim++) {
			if (maxCoords[dim] - minCoords[dim] > maxExtent) {
				maxExtent = maxCoords[dim] - minCoords[dim];
			}
		}

		/**
		* Several possibilities exist for choosing the recursion depth.
		* Either by user choice, or by the maximum fitting into the datatype, or by the minimum distance between adjacent points.
		*/
		const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(n), double(21));
	
		/**
		*	create space filling curve indices.
		*/
		// trying the new version of getHilbertIndex
                
		scai::lama::DenseVector<ValueType> hilbertIndices(inputDist);
		// get local part of hilbert indices
		scai::utilskernel::LArray<ValueType>& hilbertIndicesLocal = hilbertIndices.getLocalValues();

		{
			SCAI_REGION("ParcoRepart.partitionGraph.initialPartition.spaceFillingCurve")
			// get read access to the local part of the coordinates
			// TODO: should be coordAccess[dimension] but I don't know how ... maybe HArray::acquireReadAccess? (harry)
			scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
			scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
			// this is faulty, if dimensions=2 coordAccess2 is equal to coordAccess1
			scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[dimensions-1].getLocalValues() );

			ValueType point[dimensions];
			for (IndexType i = 0; i < localN; i++) {
				coordAccess0.getValue(point[0], i);
				coordAccess1.getValue(point[1], i);
				// TODO change how I treat different dimensions
				if(dimensions == 3){
					coordAccess2.getValue(point[2], i);
				}
				ValueType globalHilbertIndex = HilbertCurve<IndexType, ValueType>::getHilbertIndex( point, dimensions, recursionDepth, minCoords, maxCoords);
				hilbertIndicesLocal[i] = globalHilbertIndex;
			}
		}
		
		/**
		* now sort the global indices by where they are on the space-filling curve.
		*/
		scai::lama::DenseVector<IndexType> permutation;
        {
			SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition.sorting" )
			hilbertIndices.sort(permutation, true);
        }

		/**
		* initial partitioning with sfc.
		*/
		if (!inputDist->isReplicated() && comm->getSize() == k) {
			SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition.redistribute" )

			scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(n, comm));
			permutation.redistribute(blockDist);
			scai::hmemo::WriteAccess<IndexType> wPermutation( permutation.getLocalValues() );
			std::sort(wPermutation.get(), wPermutation.get()+wPermutation.size());
			wPermutation.release();

			scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(n, permutation.getLocalValues(), comm));

			input.redistribute(newDistribution, input.getColDistributionPtr());
			result = DenseVector<IndexType>(newDistribution, comm->getRank());

			if (settings.useGeometricTieBreaking) {
				for (IndexType dim = 0; dim < dimensions; dim++) {
					coordinates[dim].redistribute(newDistribution);
				}
			}

		} else {
			scai::lama::DenseVector<IndexType> inversePermutation;
			DenseVector<IndexType> tmpPerm = permutation;
			tmpPerm.sort( inversePermutation, true);

			result.allocate(inputDist);

			for (IndexType i = 0; i < localN; i++) {
				result.getLocalValues()[i] = int( inversePermutation.getLocalValues()[i] *k/n);
			}
		}
	}

	IndexType numRefinementRounds = 0;

	if (comm->getSize() == 1 || comm->getSize() == k) {
		ValueType gain = settings.minGainForNextRound;
		ValueType cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;

		/**
		scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getPEGraph(input);

		std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph);

		std::vector<IndexType> nodesWithNonLocalNeighbors = getNodesWithNonLocalNeighbors(input);

		std::vector<double> distances;
		if (settings.useGeometricTieBreaking) {
			distances = distancesFromBlockCenter(coordinates);
		}
*/
		DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(input.getRowDistributionPtr(), 1);

		if (comm->getRank() == 0) {
			afterSFC = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsedSeconds = afterSFC-start;
			std::cout << "With SFC (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
		}

		multiLevelStep(input, result, uniformWeights, coordinates, settings);
		/*
		while (gain >= settings.minGainForNextRound) {
			if (inputDist->isReplicated()) {
				gain = replicatedMultiWayFM(input, result, k, epsilon);
			} else {
				std::vector<IndexType> gainPerRound = distributedFMStep(input, result, nodesWithNonLocalNeighbors, nonWeights, communicationScheme, coordinates, distances, settings);
				gain = 0;
				for (IndexType roundGain : gainPerRound) gain += roundGain;

				if (settings.skipNoGainColors) {
					IndexType i = 0;
					while (i < gainPerRound.size()) {
						if (gainPerRound[i] == 0) {
							//remove this color from future rounds. This is not terribly efficient, since it causes multiple copies, but all vectors are small and this method isn't called too often.
							communicationScheme.erase(communicationScheme.begin()+i);
							gainPerRound.erase(gainPerRound.begin()+i);
						} else {
							i++;
						}
					}
				}
			}
			ValueType oldCut = cut;
			cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;
			if (comm->getRank() == 0) {
				round = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsedSeconds = round-start;
				std::cout << "After " << numRefinementRounds + 1 << " refinement rounds, (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
			}
			if (cut != oldCut - gain) {
				throw std::logic_error("Old cut was " + std::to_string(oldCut) + ", new cut is " + std::to_string(cut) + " but gain is " + std::to_string(gain)+".");
			}

			numRefinementRounds++;
		}
		*/
	} else {
		std::cout << "Local refinement only implemented sequentially and for one block per process. Called with " << comm->getSize() << " processes and " << k << " blocks." << std::endl;
	}
	return result;
}

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::replicatedMultiWayFM(const CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted) {

	SCAI_REGION( "ParcoRepart.replicatedMultiWayFM" )
	const IndexType n = input.getNumRows();
	/**
	* check input and throw errors
	*/
	const Scalar minPartID = part.min();
	const Scalar maxPartID = part.max();
	if (minPartID.getValue<IndexType>() != 0) {
		throw std::runtime_error("Smallest block ID is " + std::to_string(minPartID.getValue<IndexType>()) + ", should be 0");
	}

	if (maxPartID.getValue<IndexType>() != k-1) {
		throw std::runtime_error("Highest block ID is " + std::to_string(maxPartID.getValue<IndexType>()) + ", should be " + std::to_string(k-1));
	}

	if (part.size() != n) {
		throw std::runtime_error("Partition has " + std::to_string(part.size()) + " entries, but matrix has " + std::to_string(n) + ".");
	}

	if (epsilon < 0) {
		throw std::runtime_error("Epsilon must be >= 0, not " + std::to_string(epsilon));
	}

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();

	if (!inputDist->isReplicated()) {
		throw std::runtime_error("Input matrix must be replicated, for now.");
	}

	if (!partDist->isReplicated()) {
		throw std::runtime_error("Input partition must be replicated, for now.");
	}

	if (k == 1) {
		//nothing to partition
		return 0;
	}

	/**
	* allocate data structures
	*/

	//const ValueType oldCut = computeCut(input, part, unweighted);

	scai::hmemo::ReadAccess<IndexType> partAccess(part.getLocalValues());

	const IndexType optSize = ceil(double(n) / k);
	const IndexType maxAllowablePartSize = optSize*(1+epsilon);

	std::vector<IndexType> bestTargetFragment(n);
	std::vector<PrioQueue<ValueType, IndexType>> queues(k, n);

	std::vector<IndexType> gains;
	std::vector<std::pair<IndexType, IndexType> > transfers;
	std::vector<IndexType> transferedVertices;
	std::vector<double> imbalances;

	std::vector<double> fragmentSizes(k);

	double maxFragmentSize = 0;

	for (IndexType i = 0; i < n; i++) {
		IndexType blockID = partAccess[i];
		assert(blockID >= 0);
		assert(blockID < k);
		fragmentSizes[blockID] += 1;

		if (fragmentSizes[blockID] < maxFragmentSize) {
			maxFragmentSize = fragmentSizes[blockID];
		}
	}

	std::vector<IndexType> degrees(n);
	std::vector<std::vector<ValueType> > edgeCuts(n);

	//Using ReadAccess here didn't give a performance benefit
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::utilskernel::LArray<IndexType>& ia = localStorage.getIA();
	const scai::utilskernel::LArray<IndexType>& ja = localStorage.getJA();
	const scai::utilskernel::LArray<IndexType>& values = localStorage.getValues();
	if (!unweighted && values.min() < 0) {
		throw std::runtime_error("Only positive edge weights are supported, " + std::to_string(values.min()) + " invalid.");
	}

	ValueType totalWeight = 0;

	for (IndexType v = 0; v < n; v++) {
		edgeCuts[v].resize(k, 0);

		const IndexType beginCols = ia[v];
		const IndexType endCols = ia[v+1];
		degrees[v] = endCols - beginCols;
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			IndexType localNeighbor = inputDist->global2local(neighbor);
			if (neighbor == v) continue;
			edgeCuts[v][partAccess[localNeighbor]] += unweighted ? 1 : values[j];
			totalWeight += unweighted ? 1 : values[j];
		}
	}

	//setting initial best target for each node
	for (IndexType v = 0; v < n; v++) {
		ValueType maxCut = -totalWeight;
		IndexType idAtMax = k;

		for (IndexType fragment = 0; fragment < k; fragment++) {
			if (unweighted) {
				assert(edgeCuts[v][fragment] <= degrees[v]);
			}
			assert(edgeCuts[v][fragment] >= 0);

			if (fragment != partAccess[v] && edgeCuts[v][fragment] > maxCut && fragmentSizes[fragment] <= maxAllowablePartSize) {
				idAtMax = fragment;
				maxCut = edgeCuts[v][fragment];
			}
		}

		assert(idAtMax < k);
		assert(maxCut >= 0);
		if (unweighted) assert(maxCut <= degrees[v]);
		bestTargetFragment[v] = idAtMax;
		assert(partAccess[v] < queues.size());
		if (fragmentSizes[partAccess[v]] > 1) {
			ValueType key = -(maxCut-edgeCuts[v][partAccess[v]]);
			assert(-key <= degrees[v]);
			queues[partAccess[v]].insert(key, v); //negative max gain
		}
	}

	ValueType gainsum = 0;
	bool allQueuesEmpty = false;

	std::vector<bool> moved(n, false);

	while (!allQueuesEmpty) {
	allQueuesEmpty = true;

	//choose largest partition with non-empty queue.
	IndexType largestMovablePart = k;
	IndexType largestSize = 0;

	for (IndexType partID = 0; partID < k; partID++) {
		if (queues[partID].size() > 0 && fragmentSizes[partID] > largestSize) {
			largestMovablePart = partID;
			largestSize = fragmentSizes[partID];
		}
	}

	if (largestSize > 1 && largestMovablePart != k) {
		//at least one queue is not empty
		allQueuesEmpty = false;
		IndexType partID = largestMovablePart;

		assert(partID < queues.size());
		assert(queues[partID].size() > 0);

		IndexType topVertex;
		ValueType topGain;
		std::tie(topGain, topVertex) = queues[partID].extractMin();
		topGain = -topGain;//invert, since the negative gain was used as priority.
		assert(topVertex < n);
		assert(topVertex >= 0);
		if (unweighted) {
			assert(topGain <= degrees[topVertex]);
		}
		assert(!moved[topVertex]);
		assert(partAccess[topVertex] == partID);

		//now get target partition.
		IndexType targetFragment = bestTargetFragment[topVertex];
		ValueType storedGain = edgeCuts[topVertex][targetFragment] - edgeCuts[topVertex][partID];
		
		assert(abs(storedGain - topGain) < 0.0001);
		assert(fragmentSizes[partID] > 1);


		//move node there
		part.setValue(topVertex, targetFragment);
		moved[topVertex] = true;

		//udpate size map
		fragmentSizes[partID] -= 1;
		fragmentSizes[targetFragment] += 1;

		//update history
		gainsum += topGain;
		gains.push_back(gainsum);
		transfers.emplace_back(partID, targetFragment);
		transferedVertices.push_back(topVertex);
		assert(transferedVertices.size() == transfers.size());
		assert(gains.size() == transfers.size());

		double imbalance = (*std::max_element(fragmentSizes.begin(), fragmentSizes.end()) - optSize) / optSize;
		imbalances.push_back(imbalance);

		//std::cout << "Moved node " << topVertex << " to block " << targetFragment << " for gain of " << topGain << ", bringing sum to " << gainsum 
		//<< " and imbalance to " << imbalance  << "." << std::endl;

		//I've tried to replace these direct accesses by ReadAccess, program was slower
		const IndexType beginCols = ia[topVertex];
		const IndexType endCols = ia[topVertex+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			const IndexType neighbour = ja[j];
			if (!moved[neighbour]) {
				//update edge cuts
				IndexType neighbourBlock = partAccess[neighbour];

				edgeCuts[neighbour][partID] -= unweighted ? 1 : values[j];
				assert(edgeCuts[neighbour][partID] >= 0);
				edgeCuts[neighbour][targetFragment] += unweighted ? 1 : values[j];
				assert(edgeCuts[neighbour][targetFragment] <= degrees[neighbour]);

				//find new fragment for neighbour
				ValueType maxCut = -totalWeight;
				IndexType idAtMax = k;

				for (IndexType fragment = 0; fragment < k; fragment++) {
					if (fragment != neighbourBlock && edgeCuts[neighbour][fragment] > maxCut  && fragmentSizes[fragment] <= maxAllowablePartSize) {
						idAtMax = fragment;
						maxCut = edgeCuts[neighbour][fragment];
					}
				}

				assert(maxCut >= 0);
				if (unweighted) assert(maxCut <= degrees[neighbour]);
				assert(idAtMax < k);
				bestTargetFragment[neighbour] = idAtMax;

				ValueType key = -(maxCut-edgeCuts[neighbour][neighbourBlock]);
				assert(-key == edgeCuts[neighbour][idAtMax] - edgeCuts[neighbour][neighbourBlock]);
				assert(-key <= degrees[neighbour]);

				//update prioqueue
				queues[neighbourBlock].remove(neighbour);
				queues[neighbourBlock].insert(key, neighbour);
				}
			}
		}
	}

	const IndexType testedNodes = gains.size();
	if (testedNodes == 0) return 0;
	assert(gains.size() == transfers.size());

    /**
	* now find best partition among those tested
	*/
	IndexType maxIndex = -1;
	ValueType maxGain = 0;

	for (IndexType i = 0; i < testedNodes; i++) {
		if (gains[i] > maxGain && imbalances[i] <= epsilon) {
			maxIndex = i;
			maxGain = gains[i];
		}
	}
	assert(testedNodes >= maxIndex);
	//assert(maxIndex >= 0);
	assert(testedNodes-1 < transfers.size());

	/**
	 * apply partition modifications in reverse until best is recovered
	 */
	for (int i = testedNodes-1; i > maxIndex; i--) {
		assert(transferedVertices[i] < n);
		assert(transferedVertices[i] >= 0);
		part.setValue(transferedVertices[i], transfers[i].first);
	}
	return maxGain;
}

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const bool weighted) {
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

	scai::dmemo::Halo partHalo = buildNeighborHalo(input);
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

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(const CSRSparseMatrix<ValueType> &input, const bool weighted) {
	SCAI_REGION( "ParcoRepart.localSumOutgoingEdges" )
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	IndexType sumOutgoingEdgeWeights = 0;
	for (IndexType j = 0; j < ja.size(); j++) {
		if (!input.getRowDistributionPtr()->isLocal(ja[j])) sumOutgoingEdgeWeights += weighted ? values[j] : 1;
	}

	return sumOutgoingEdgeWeights;
}

template<typename IndexType, typename ValueType>
IndexType ParcoRepart<IndexType, ValueType>::localBlockSize(const DenseVector<IndexType> &part, IndexType blockID) {
	SCAI_REGION( "ParcoRepart.localBlockSize" )
	IndexType result = 0;
	scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());

	for (IndexType i = 0; i < localPart.size(); i++) {
		if (localPart[i] == blockID) {
			result++;
		}
	}

	return result;
}

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::computeImbalance(const DenseVector<IndexType> &part, IndexType k, const DenseVector<IndexType> &nodeWeights) {
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
		throw std::runtime_error("Block id " + std::to_string(minK) + " found in partition with supposedly" + std::to_string(k) + " blocks.");
	}

	if (maxK >= k) {
		throw std::runtime_error("Block id " + std::to_string(maxK) + " found in partition with supposedly" + std::to_string(k) + " blocks.");
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
		optSize = std::ceil(weightSum / k + (maxWeight - minWeight));
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
	return ((maxBlockSize - optSize)/ optSize);
}

template<typename IndexType, typename ValueType>
std::vector<DenseVector<IndexType>> ParcoRepart<IndexType, ValueType>::computeCommunicationPairings(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part,
			const DenseVector<IndexType> &blocksToPEs) {
	/**
	 * for now, trivial communication pairing in 2^(ceil(log_2(p))) steps.
	 */
	SCAI_REGION( "ParcoRepart.computeCommunicationPairings" )

	const Scalar maxPartID = blocksToPEs.max();
	const IndexType p = maxPartID.getValue<IndexType>() + 1;
	const IndexType rounds = std::ceil(std::log(p) / std::log(2));
	const IndexType upperPowerP = 1 << rounds;
	assert(upperPowerP >= p);
	assert(upperPowerP < 2*p);
	const IndexType steps = upperPowerP-1;
	assert(steps >= p-1);

	std::vector<DenseVector<IndexType>> result;

	for (IndexType step = 1; step <= steps; step++) {
		IndexType commPerm[p];

		for (IndexType i = 0; i < p; i++) {
			IndexType partner = i ^ step;
			commPerm[i] = partner < p ? partner : i;
		}

		result.emplace_back(p, &commPerm[0]);
	}
	return result;
}

template<typename IndexType, typename ValueType>
std::vector<IndexType> ITI::ParcoRepart<IndexType, ValueType>::nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
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

template<typename IndexType, typename ValueType>
std::vector<ValueType> ITI::ParcoRepart<IndexType, ValueType>::distancesFromBlockCenter(const std::vector<DenseVector<ValueType>> &coordinates) {
	SCAI_REGION("ParcoRepart.distanceFromBlockCenter");

	const IndexType localN = coordinates[0].getDistributionPtr()->getLocalSize();
	const IndexType dimensions = coordinates.size();

	std::vector<ValueType> geometricCenter(dimensions);
	for (IndexType dim = 0; dim < dimensions; dim++) {
		const scai::utilskernel::LArray<ValueType>& localValues = coordinates[dim].getLocalValues();
		assert(localValues.size() == localN);
		geometricCenter[dim] = localValues.sum() / localN;
	}

	std::vector<ValueType> result(localN);
	for (IndexType i = 0; i < localN; i++) {
		ValueType distanceSquared = 0;
		for (IndexType dim = 0; dim < dimensions; dim++) {
			const ValueType diff = coordinates[dim].getLocalValues()[i] - geometricCenter[dim];
			distanceSquared += diff*diff;
		}
		result[i] = pow(distanceSquared, 0.5);
	}
	return result;
}

template<typename IndexType, typename ValueType>
template<typename T>
void ITI::ParcoRepart<IndexType, ValueType>::redistributeFromHalo(DenseVector<T>& input, scai::dmemo::DistributionPtr newDist, Halo& halo, scai::utilskernel::LArray<T>& haloData) {
	SCAI_REGION( "ParcoRepart.redistributeFromHalo.Vector" )

	using scai::utilskernel::LArray;

	scai::dmemo::DistributionPtr oldDist = input.getDistributionPtr();
	const IndexType newLocalN = newDist->getLocalSize();
	LArray<T> newLocalValues;

	{
		scai::hmemo::ReadAccess<T> rOldLocalValues(input.getLocalValues());
		scai::hmemo::ReadAccess<T> rHaloData(haloData);

		scai::hmemo::WriteOnlyAccess<T> wNewLocalValues(newLocalValues, newLocalN);
		for (IndexType i = 0; i < newLocalN; i++) {
			const IndexType globalI = newDist->local2global(i);
			if (oldDist->isLocal(globalI)) {
				wNewLocalValues[i] = rOldLocalValues[oldDist->global2local(globalI)];
			} else {
				const IndexType localI = halo.global2halo(globalI);
				assert(localI != nIndex);
				wNewLocalValues[i] = rHaloData[localI];
			}
		}
	}

	input.swap(newLocalValues, newDist);
}

template<typename IndexType, typename ValueType>
void ITI::ParcoRepart<IndexType, ValueType>::redistributeFromHalo(CSRSparseMatrix<ValueType>& matrix, scai::dmemo::DistributionPtr newDist, Halo& halo, CSRStorage<ValueType>& haloStorage) {
	SCAI_REGION( "ParcoRepart.redistributeFromHalo" )

	scai::dmemo::DistributionPtr oldDist = matrix.getRowDistributionPtr();

	using scai::utilskernel::LArray;

	const IndexType sourceNumRows = oldDist->getLocalSize();
	const IndexType targetNumRows = newDist->getLocalSize();

	const IndexType globalN = oldDist->getGlobalSize();
	if (newDist->getGlobalSize() != globalN) {
		throw std::runtime_error("Old Distribution has " + std::to_string(globalN) + " values, new distribution has " + std::to_string(newDist->getGlobalSize()));
	}

	scai::hmemo::HArray<IndexType> targetIA;
	scai::hmemo::HArray<IndexType> targetJA;
	scai::hmemo::HArray<ValueType> targetValues;

	const CSRStorage<ValueType>& localStorage = matrix.getLocalStorage();

	matrix.setDistributionPtr(newDist);

	//check for equality
	if (sourceNumRows == targetNumRows) {
		SCAI_REGION( "ParcoRepart.redistributeFromHalo.equalityCheck" )
		bool allLocal = true;
		for (IndexType i = 0; i < targetNumRows; i++) {
			if (!oldDist->isLocal(newDist->local2global(i))) allLocal = false;
		}
		if (allLocal) {
			//nothing to redistribute and no communication to do either.
			return;
		}
	}

	scai::hmemo::HArray<IndexType> sourceSizes;
	{
		SCAI_REGION( "ParcoRepart.redistributeFromHalo.sourceSizes" )
		scai::hmemo::ReadAccess<IndexType> sourceIA(localStorage.getIA());
		scai::hmemo::WriteOnlyAccess<IndexType> wSourceSizes( sourceSizes, sourceNumRows );
	    scai::sparsekernel::OpenMPCSRUtils::offsets2sizes( wSourceSizes.get(), sourceIA.get(), sourceNumRows );
	    //allocate
	    scai::hmemo::WriteOnlyAccess<IndexType> wTargetIA( targetIA, targetNumRows + 1 );
	}

	scai::hmemo::HArray<IndexType> haloSizes;
	{
		SCAI_REGION( "ParcoRepart.redistributeFromHalo.haloSizes" )
		scai::hmemo::WriteOnlyAccess<IndexType> wHaloSizes( haloSizes, halo.getHaloSize() );
		scai::hmemo::ReadAccess<IndexType> rHaloIA( haloStorage.getIA() );
		scai::sparsekernel::OpenMPCSRUtils::offsets2sizes( wHaloSizes.get(), rHaloIA.get(), halo.getHaloSize() );
	}

	std::vector<IndexType> localTargetIndices;
	std::vector<IndexType> localSourceIndices;
	std::vector<IndexType> localHaloIndices;
	std::vector<IndexType> additionalLocalNodes;
	IndexType numValues = 0;
	{
		SCAI_REGION( "ParcoRepart.redistributeFromHalo.targetIA" )
		scai::hmemo::ReadAccess<IndexType> rSourceSizes(sourceSizes);
		scai::hmemo::ReadAccess<IndexType> rHaloSizes(haloSizes);
	    scai::hmemo::WriteAccess<IndexType> wTargetIA( targetIA );

		for (IndexType i = 0; i < targetNumRows; i++) {
			IndexType newGlobalIndex = newDist->local2global(i);
			IndexType size;
			if (oldDist->isLocal(newGlobalIndex)) {
				localTargetIndices.push_back(i);
				const IndexType oldLocalIndex = oldDist->global2local(newGlobalIndex);
				localSourceIndices.push_back(oldLocalIndex);
				size = rSourceSizes[oldLocalIndex];
			} else {
				additionalLocalNodes.push_back(i);
				const IndexType haloIndex = halo.global2halo(newGlobalIndex);
				assert(haloIndex != nIndex);
				localHaloIndices.push_back(haloIndex);
				size = rHaloSizes[haloIndex];
			}
			wTargetIA[i] = size;
			numValues += size;
		}
		scai::sparsekernel::OpenMPCSRUtils::sizes2offsets( wTargetIA.get(), targetNumRows );

		//allocate
		scai::hmemo::WriteOnlyAccess<IndexType> wTargetJA( targetJA, numValues );
		scai::hmemo::WriteOnlyAccess<ValueType> wTargetValues( targetValues, numValues );
	}

	scai::hmemo::ReadAccess<IndexType> rTargetIA(targetIA);
	assert(rTargetIA.size() == targetNumRows + 1);
	IndexType numLocalIndices = localTargetIndices.size();
	IndexType numHaloIndices = localHaloIndices.size();

	for (IndexType i = 0; i < targetNumRows; i++) {
		assert(rTargetIA[i] <= rTargetIA[i+1]);
		assert(rTargetIA[i] < numValues);
	}

	{
		SCAI_REGION( "ParcoRepart.redistributeFromHalo.copy" )
		//copying JA array from local matrix and halo
		scai::dmemo::Redistributor::copyV( targetJA, targetIA, LArray<IndexType>(numLocalIndices, localTargetIndices.data()), localStorage.getJA(), localStorage.getIA(), LArray<IndexType>(numLocalIndices, localSourceIndices.data()) );
		scai::dmemo::Redistributor::copyV( targetJA, targetIA, LArray<IndexType>(additionalLocalNodes.size(), additionalLocalNodes.data()), haloStorage.getJA(), haloStorage.getIA(), LArray<IndexType>(numHaloIndices, localHaloIndices.data()) );

		//copying Values array from local matrix and halo
		scai::dmemo::Redistributor::copyV( targetValues, targetIA, LArray<IndexType>(numLocalIndices, localTargetIndices.data()), localStorage.getValues(), localStorage.getIA(), LArray<IndexType>(numLocalIndices, localSourceIndices.data()) );
		scai::dmemo::Redistributor::copyV( targetValues, targetIA, LArray<IndexType>(additionalLocalNodes.size(), additionalLocalNodes.data()), haloStorage.getValues(), haloStorage.getIA(), LArray<IndexType>(numHaloIndices, localHaloIndices.data()) );
	}
	rTargetIA.release();

	{
		SCAI_REGION( "ParcoRepart.redistributeFromHalo.setCSRData" )
		//setting CSR data
		matrix.getLocalStorage().setCSRDataSwap(targetNumRows, globalN, numValues, targetIA, targetJA, targetValues, scai::hmemo::ContextPtr());
	}
}

template<typename IndexType, typename ValueType>
scai::dmemo::Halo ITI::ParcoRepart<IndexType, ValueType>::buildNeighborHalo(const CSRSparseMatrix<ValueType>& input) {

	SCAI_REGION( "ParcoRepart.buildPartHalo" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	std::vector<IndexType> requiredHaloIndices = nonLocalNeighbors(input);

	scai::dmemo::Halo Halo;
	{
		scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
		scai::dmemo::HaloBuilder::build( *inputDist, arrRequiredIndexes, Halo );
	}

	return Halo;
}

template<typename IndexType, typename ValueType>
inline bool ITI::ParcoRepart<IndexType, ValueType>::hasNonLocalNeighbors(const CSRSparseMatrix<ValueType> &input, IndexType globalID) {
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

template<typename IndexType, typename ValueType>
std::vector<IndexType> ITI::ParcoRepart<IndexType, ValueType>::getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
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

template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>, std::vector<IndexType>> ITI::ParcoRepart<IndexType, ValueType>::getInterfaceNodes(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const std::vector<IndexType>& nodesWithNonLocalNeighbors, IndexType otherBlock, IndexType depth) {

	SCAI_REGION( "ParcoRepart.getInterfaceNodes" )
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();
	const IndexType thisBlock = comm->getRank();

	if (partDist->getLocalSize() != localN) {
		throw std::runtime_error("Partition has " + std::to_string(partDist->getLocalSize()) + " local nodes, but matrix has " + std::to_string(localN) + ".");
	}

	if (otherBlock > comm->getSize()) {
		throw std::runtime_error("Currently only implemented with one block per process, block " + std::to_string(thisBlock) + " invalid for " + std::to_string(comm->getSize()) + " processes.");
	}

	if (thisBlock == otherBlock) {
		throw std::runtime_error("Block IDs must be different.");
	}

	if (depth <= 0) {
		throw std::runtime_error("Depth must be positive");
	}

	scai::hmemo::HArray<IndexType> localData = part.getLocalValues();
	scai::hmemo::ReadAccess<IndexType> partAccess(localData);
	
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	/**
	 * send nodes with non-local neighbors to partner process.
	 * here we assume a 1-to-1-mapping of blocks to processes and a symmetric matrix
	 */
	std::unordered_set<IndexType> foreignNodes;
	{
		SCAI_REGION( "ParcoRepart.getInterfaceNodes.communication" )
		IndexType swapField[1];
		{
			SCAI_REGION( "ParcoRepart.getInterfaceNodes.communication.syncswap" );
			//swap number of local border nodes
			swapField[0] = nodesWithNonLocalNeighbors.size();
			comm->swap(swapField, 1, otherBlock);
		}
		const IndexType otherSize = swapField[0];
		const IndexType swapLength = std::max(otherSize, IndexType(nodesWithNonLocalNeighbors.size()));
		IndexType swapList[swapLength];
		for (IndexType i = 0; i < nodesWithNonLocalNeighbors.size(); i++) {
			swapList[i] = nodesWithNonLocalNeighbors[i];
		}
		comm->swap(swapList, swapLength, otherBlock);

		foreignNodes.reserve(otherSize);

		//the swapList array is only partially filled, the number of received nodes is found in swapField[0]
		for (IndexType i = 0; i < otherSize; i++) {
			foreignNodes.insert(swapList[i]);
		}
	}

	/**
	 * check which of the neighbors of our local border nodes are actually the partner's border nodes
	 */
	std::vector<IndexType> interfaceNodes;

	for (IndexType node : nodesWithNonLocalNeighbors) {
		SCAI_REGION( "ParcoRepart.getInterfaceNodes.getBorderToPartner" )
		IndexType localI = inputDist->global2local(node);
		assert(localI != nIndex);
		bool hasNonLocal = false;
		for (IndexType j = ia[localI]; j < ia[localI+1]; j++) {
			if (!inputDist->isLocal(ja[j])) {
				hasNonLocal = true;
				if (foreignNodes.count(ja[j])> 0) {
					interfaceNodes.push_back(node);
					break;
				}
			}
		}

		//This shouldn't happen, the list of local border nodes was incorrect!
		if (!hasNonLocal) {
			throw std::runtime_error("Node " + std::to_string(node) + " has " + std::to_string(ia[localI+1] - ia[localI]) + " neighbors, but all of them are local.");
		}
	}

	assert(interfaceNodes.size() <= localN);

	//keep track of which nodes were added at each BFS round
	std::vector<IndexType> roundMarkers({0});

	/**
	 * now gather buffer zone with breadth-first search
	 */
	if (depth > 1) {
		SCAI_REGION( "ParcoRepart.getInterfaceNodes.breadthFirstSearch" )
		std::vector<bool> touched(localN, false);

		std::queue<IndexType> bfsQueue;
		for (IndexType node : interfaceNodes) {
			bfsQueue.push(node);
			const IndexType localID = inputDist->global2local(node);
			assert(localID != nIndex);
			touched[localID] = true;
		}
		assert(bfsQueue.size() == interfaceNodes.size());

		for (IndexType round = 1; round < depth; round++) {
			roundMarkers.push_back(interfaceNodes.size());
			std::queue<IndexType> nextQueue;
			while (!bfsQueue.empty()) {
				IndexType nextNode = bfsQueue.front();
				bfsQueue.pop();

				const IndexType localI = inputDist->global2local(nextNode);
				assert(localI != nIndex);
				assert(touched[localI]);
				const IndexType beginCols = ia[localI];
				const IndexType endCols = ia[localI+1];

				for (IndexType j = beginCols; j < endCols; j++) {
					IndexType neighbor = ja[j];
					IndexType localNeighbor = inputDist->global2local(neighbor);
					//assume k=p
					if (localNeighbor != nIndex && !touched[localNeighbor]) {
						nextQueue.push(neighbor);
						interfaceNodes.push_back(neighbor);
						touched[localNeighbor] = true;
					}
				}
			}
			bfsQueue = nextQueue;
		}
	}

	assert(interfaceNodes.size() <= localN);
	assert(roundMarkers.size() == depth);
	return {interfaceNodes, roundMarkers};
}

template<typename IndexType, typename ValueType>
IndexType ITI::ParcoRepart<IndexType, ValueType>::getDegreeSum(const CSRSparseMatrix<ValueType> &input, const std::vector<IndexType>& nodes) {
	SCAI_REGION( "ParcoRepart.getDegreeSum" )
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> localIa(localStorage.getIA());

	IndexType result = 0;

	for (IndexType node : nodes) {
		IndexType localID = input.getRowDistributionPtr()->global2local(node);
		result += localIa[localID+1] - localIa[localID];
	}
	return result;
}

template<typename IndexType, typename ValueType>
IndexType ITI::ParcoRepart<IndexType, ValueType>::multiLevelStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part,
		DenseVector<IndexType> &nodeWeights, std::vector<DenseVector<ValueType>> &coordinates, Settings settings) {

	scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();
	const IndexType globalN = input.getRowDistributionPtr()->getGlobalSize();

	if (settings.multiLevelRounds > 0) {
		CSRSparseMatrix<ValueType> coarseGraph;
		DenseVector<IndexType> fineToCoarseMap;
		std::cout << "Beginning coarsening, still " << settings.multiLevelRounds << " levels to go." << std::endl;
		ParcoRepart<IndexType, ValueType>::coarsen(input, coarseGraph, fineToCoarseMap);
		std::cout << "Coarse graph has " << coarseGraph.getNumRows() << " nodes." << std::endl;

		//project coordinates and partition
		std::vector<DenseVector<ValueType> > coarseCoords(settings.dimensions);
		for (IndexType i = 0; i < settings.dimensions; i++) {
			coarseCoords[i] = projectToCoarse(coordinates[i], fineToCoarseMap);
		}

		DenseVector<IndexType> coarsePart = DenseVector<IndexType>(coarseGraph.getRowDistributionPtr(), comm->getRank());

		DenseVector<IndexType> coarseWeights = sumToCoarse(nodeWeights, fineToCoarseMap);

		assert(coarseWeights.sum().Scalar::getValue<IndexType>() == nodeWeights.sum().Scalar::getValue<IndexType>());

		Settings settingscopy(settings);
		settingscopy.multiLevelRounds--;
		multiLevelStep(coarseGraph, coarsePart, coarseWeights, coarseCoords, settingscopy);

		scai::dmemo::DistributionPtr projectedFineDist = projectToFine(coarseGraph.getRowDistributionPtr(), fineToCoarseMap);
		assert(projectedFineDist->getGlobalSize() == globalN);
		part = DenseVector<IndexType>(projectedFineDist, comm->getRank());
		std::cout << "Projected distribution back to coarse" << std::endl;

		if (settings.useGeometricTieBreaking) {
			for (IndexType dim = 0; dim < settings.dimensions; dim++) {
				coordinates[dim].redistribute(projectedFineDist);
			}
			std::cout << "Redistributed coordinates" << std::endl;
		}

		input.redistribute(projectedFineDist, input.getColDistributionPtr());

		nodeWeights.redistribute(projectedFineDist);
		std::cout << "Redistributed node weights" << std::endl;
	}

	scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getPEGraph(input);

	std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph);

	std::vector<IndexType> nodesWithNonLocalNeighbors = getNodesWithNonLocalNeighbors(input);

	std::vector<ValueType> distances;
	if (settings.useGeometricTieBreaking) {
		distances = distancesFromBlockCenter(coordinates);
	}

	IndexType numRefinementRounds = 0;

	ValueType gain = settings.minGainForNextRound;
	while (gain >= settings.minGainForNextRound) {
		std::vector<IndexType> gainPerRound = distributedFMStep(input, part, nodesWithNonLocalNeighbors, nodeWeights, communicationScheme, coordinates, distances, settings);
		gain = 0;
		for (IndexType roundGain : gainPerRound) gain += roundGain;

		if (settings.skipNoGainColors) {
			IndexType i = 0;
			while (i < gainPerRound.size()) {
				if (gainPerRound[i] == 0) {
					//remove this color from future rounds. This is not terribly efficient, since it causes multiple copies, but all vectors are small and this method isn't called too often.
					communicationScheme.erase(communicationScheme.begin()+i);
					gainPerRound.erase(gainPerRound.begin()+i);
				} else {
					i++;
				}
			}
		}

		ValueType cut = comm->getSize() == 1 ? computeCut(input, part) : comm->sum(localSumOutgoingEdges(input, true)) / 2;
		if (comm->getRank() == 0) {
			std::cout << "After " << numRefinementRounds + 1 << " refinement rounds, cut is " << cut << std::endl;
		}
		numRefinementRounds++;
	}
}

template<typename IndexType, typename ValueType>
std::vector<IndexType> ITI::ParcoRepart<IndexType, ValueType>::distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, std::vector<DenseVector<ValueType>> &coordinates, Settings settings) {
	/**
	 * This is a wrapper function to allow calls without precomputing a communication schedule..
	 */

	std::vector<IndexType> nodesWithNonLocalNeighbors = getNodesWithNonLocalNeighbors(input);

	//get block graph
	scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getBlockGraph( input, part, settings.numBlocks);

	//color block graph and get a communication schedule
	std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph);

	//get uniform node weights
	DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(input.getRowDistributionPtr(), 1);
	//DenseVector<IndexType> nonWeights = DenseVector<IndexType>(0, 1);

	//get distances
	std::vector<double> distances = distancesFromBlockCenter(coordinates);

	//call distributed FM-step
	return distributedFMStep(input, part, nodesWithNonLocalNeighbors, uniformWeights, communicationScheme, coordinates, distances, settings);
}

template<typename IndexType, typename ValueType>
std::vector<IndexType> ITI::ParcoRepart<IndexType, ValueType>::distributedFMStep(CSRSparseMatrix<ValueType>& input, DenseVector<IndexType>& part, std::vector<IndexType>& nodesWithNonLocalNeighbors,
		DenseVector<IndexType> &nodeWeights, const std::vector<DenseVector<IndexType>>& communicationScheme, std::vector<DenseVector<ValueType>> &coordinates,
		std::vector<ValueType> &distances, Settings settings) {
	SCAI_REGION( "ParcoRepart.distributedFMStep" )
	const IndexType globalN = input.getRowDistributionPtr()->getGlobalSize();
	scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();

	if (part.getDistributionPtr()->getLocalSize() != input.getRowDistributionPtr()->getLocalSize()) {
		throw std::runtime_error("Distributions of input matrix and partitions must be equal, for now.");
	}

	if (!input.getColDistributionPtr()->isReplicated()) {
		throw std::runtime_error("Column distribution needs to be replicated.");
	}

	if (settings.useGeometricTieBreaking) {
		for (IndexType dim = 0; dim < coordinates.size(); dim++) {
			if (coordinates[dim].getDistributionPtr()->getLocalSize() != input.getRowDistributionPtr()->getLocalSize()) {
				throw std::runtime_error("Coordinate distribution must be equal to matrix distribution");
			}
		}
		assert(distances.size() == input.getRowDistributionPtr()->getLocalSize());
	}

	if (settings.epsilon < 0) {
		throw std::runtime_error("Epsilon must be >= 0, not " + std::to_string(settings.epsilon));
	}

    if (settings.numBlocks != comm->getSize()) {
    	throw std::runtime_error("Called with " + std::to_string(comm->getSize()) + " processors, but " + std::to_string(settings.numBlocks) + " blocks.");
    }

    //block sizes TODO: adapt for weighted case
    const IndexType optSize = ceil(double(globalN) / settings.numBlocks);
    const IndexType maxAllowableBlockSize = optSize*(1+settings.epsilon);

    //for now, we are assuming equal numbers of blocks and processes
    const IndexType localBlockID = comm->getRank();

    const bool nodesWeighted = nodeWeights.getDistributionPtr()->getGlobalSize() > 0;
    if (nodesWeighted && nodeWeights.getDistributionPtr()->getLocalSize() != input.getRowDistributionPtr()->getLocalSize()) {
    	throw std::runtime_error("Node weights have " + std::to_string(nodeWeights.getDistributionPtr()->getLocalSize()) + " local values, should be "
    			+ std::to_string(input.getRowDistributionPtr()->getLocalSize()));
    }

    IndexType gainSum = 0;
    std::vector<IndexType> gainPerRound(communicationScheme.size(), 0);

	//copy into usable data structure with iterators
    //TODO: we only need those if redistribution happens.
    //Maybe hold off on creating the vector until then? On the other hand, the savings would be in those processes that are faster anyway and probably have to wait.
	std::vector<IndexType> myGlobalIndices(input.getRowDistributionPtr()->getLocalSize());
	{
		const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
		for (IndexType j = 0; j < myGlobalIndices.size(); j++) {
			myGlobalIndices[j] = inputDist->local2global(j);
		}
	}

	//main loop, one iteration for each color of the graph coloring
	for (IndexType color = 0; color < communicationScheme.size(); color++) {
		SCAI_REGION( "ParcoRepart.distributedFMStep.loop" )

		const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
		const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
		const scai::dmemo::DistributionPtr commDist = communicationScheme[color].getDistributionPtr();

		const IndexType localN = inputDist->getLocalSize();

		if (!communicationScheme[color].getDistributionPtr()->isLocal(comm->getRank())) {
			throw std::runtime_error("Scheme value for " + std::to_string(comm->getRank()) + " must be local.");
		}
		
		scai::hmemo::ReadAccess<IndexType> commAccess(communicationScheme[color].getLocalValues());
		IndexType partner = commAccess[commDist->global2local(comm->getRank())];

		//check symmetry of communication scheme
		assert(partner < comm->getSize());
		if (commDist->isLocal(partner)) {
			IndexType partnerOfPartner = commAccess[commDist->global2local(partner)];
			if (partnerOfPartner != comm->getRank()) {
				throw std::runtime_error("Process " + std::to_string(comm->getRank()) + ": Partner " + std::to_string(partner) + " has partner "
						+ std::to_string(partnerOfPartner) + ".");
			}
		}

		/**
		 * check for validity of partition
		 */
		{
			SCAI_REGION( "ParcoRepart.distributedFMStep.loop.checkPartition" )
			scai::hmemo::ReadAccess<IndexType> partAccess(part.getLocalValues());
			for (IndexType j = 0; j < localN; j++) {
				if (partAccess[j] != localBlockID) {
					throw std::runtime_error("Block ID "+std::to_string(partAccess[j])+" found on process "+std::to_string(localBlockID)+".");
				}
			}
			for (IndexType node : nodesWithNonLocalNeighbors) {
				assert(inputDist->isLocal(node));
			}
		}

		IndexType gainThisRound = 0;

		scai::dmemo::Halo graphHalo;
		CSRStorage<ValueType> haloMatrix;
		scai::utilskernel::LArray<IndexType> nodeWeightHaloData;

		if (partner != comm->getRank()) {
			//processor is active this round

			/**
			 * get indices of border nodes with breadth-first search
			 */
			std::vector<IndexType> interfaceNodes;
			std::vector<IndexType> roundMarkers;
			std::tie(interfaceNodes, roundMarkers)= getInterfaceNodes(input, part, nodesWithNonLocalNeighbors, partner, settings.borderDepth+1);
			const IndexType lastRoundMarker = roundMarkers[roundMarkers.size()-1];
			const IndexType secondRoundMarker = roundMarkers[1];

			/**
			 * now swap indices of nodes in border region with partner processor.
			 * For this, first find out the length of the swap array.
			 */

			SCAI_REGION_START( "ParcoRepart.distributedFMStep.loop.prepareSets" )
			//swap size of border region and total block size. Block size is only needed as sanity check, could be removed in optimized version
			IndexType blockSize = localBlockSize(part, localBlockID);
			if (blockSize != localN) {
				throw std::runtime_error(std::to_string(localN) + " local nodes, but only " + std::to_string(blockSize) + " of them belong to block " + std::to_string(localBlockID) + ".");
			}

			IndexType swapField[5];
			swapField[0] = interfaceNodes.size();
			swapField[1] = secondRoundMarker;
			swapField[2] = lastRoundMarker;
			swapField[3] = blockSize;
			swapField[4] = getDegreeSum(input, interfaceNodes);
			comm->swap(swapField, 5, partner);
			//want to isolate raw array accesses as much as possible, define named variables and only use these from now
			const IndexType otherSize = swapField[0];
			const IndexType otherSecondRoundMarker = swapField[1];
			const IndexType otherLastRoundMarker = swapField[2];
			const IndexType otherBlockSize = swapField[3];
			const IndexType otherDegreeSum = swapField[4];

			if (interfaceNodes.size() == 0) {
				if (otherSize != 0) {
					throw std::runtime_error("Partner PE has a border region, but this PE doesn't. Looks like the block indices were allocated inconsistently.");
				} else {
					/*
					 * These processes don't share a border and thus have no communication to do with each other. How did they end up in a communication scheme?
					 * We could skip the loop entirely.
					 */
				}
			}

			//the two border regions might have different sizes. Swapping array is sized for the maximum of the two.
			const IndexType swapLength = std::max(otherSize, int(interfaceNodes.size()));
			IndexType swapNodes[swapLength];
			for (IndexType i = 0; i < swapLength; i++) {
				if (i < interfaceNodes.size()) {
					swapNodes[i] = interfaceNodes[i];
				} else {
					swapNodes[i] = -1;
				}
			}

			//now swap border region
			comm->swap(swapNodes, swapLength, partner);

			//read interface nodes of partner process from swapped array.
			std::vector<IndexType> requiredHaloIndices(otherSize);
			std::copy(swapNodes, swapNodes+otherSize, requiredHaloIndices.begin());

			//if we need more halo indices than there are non-local indices at all, something went wrong.
			assert(requiredHaloIndices.size() <= globalN - inputDist->getLocalSize());

			//swap distances used for tie breaking
			ValueType distanceSwap[swapLength];
			if (settings.useGeometricTieBreaking) {
				for (IndexType i = 0; i < interfaceNodes.size(); i++) {
					distanceSwap[i] = distances[inputDist->global2local(interfaceNodes[i])];
				}
				comm->swap(distanceSwap, swapLength, partner);
			}

			/*
			 * Build Halo to cover border region of other PE.
			 * This uses a special halo builder method that doesn't require communication, since the required and provided indices are already known.
			 */
			IndexType numValues = input.getLocalStorage().getValues().size();
			{
				scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
				scai::hmemo::HArrayRef<IndexType> arrProvidedIndexes( interfaceNodes );
				scai::dmemo::HaloBuilder::build( *inputDist, arrRequiredIndexes, arrProvidedIndexes, partner, graphHalo );
			}

			//all required halo indices are in the halo
			for (IndexType node : requiredHaloIndices) {
				assert(graphHalo.global2halo(node) != nIndex);
			}

			/**
			 * Exchange Halo. This only requires communication with the partner process.
			 */
			haloMatrix.exchangeHalo( graphHalo, input.getLocalStorage(), *comm );
			//local part should stay unchanged, check edge number as proxy for that
			assert(input.getLocalStorage().getValues().size() == numValues);
			//halo matrix should have as many edges as the degree sum of the required halo indices
			assert(haloMatrix.getValues().size() == otherDegreeSum);

			//Here we only exchange one BFS-Round less than gathered, to make sure that all neighbors of the considered edges are still in the halo.
			std::vector<IndexType> borderRegionIDs(interfaceNodes.begin(), interfaceNodes.begin()+lastRoundMarker);
			std::vector<bool> assignedToSecondBlock(lastRoundMarker, 0);//nodes from own border region are assigned to first block
			std::copy(requiredHaloIndices.begin(), requiredHaloIndices.begin()+otherLastRoundMarker, std::back_inserter(borderRegionIDs));
			assignedToSecondBlock.resize(borderRegionIDs.size(), 1);//nodes from other border region are assigned to second block
			assert(borderRegionIDs.size() == lastRoundMarker + otherLastRoundMarker);

			const IndexType borderRegionSize = borderRegionIDs.size();

			/**
			 * If nodes are weighted, exchange Halo for node weights
			 */
			std::vector<IndexType> borderNodeWeights = {};
			if (nodesWeighted) {
				const scai::utilskernel::LArray<IndexType>& localWeights = nodeWeights.getLocalValues();
				assert(localWeights.size() == localN);
				comm->updateHalo( nodeWeightHaloData, localWeights, graphHalo );
				borderNodeWeights.resize(borderRegionSize);
				for (IndexType i = 0; i < borderRegionSize; i++) {
					const IndexType globalI = borderRegionIDs[i];
					if (inputDist->isLocal(globalI)) {
						const IndexType localI = inputDist->global2local(globalI);
						borderNodeWeights[i] = localWeights[localI];
					} else {
						const IndexType localI = graphHalo.global2halo(globalI);
						assert(localI != nIndex);
						borderNodeWeights[i] = nodeWeightHaloData[localI];
					}
					assert(borderNodeWeights[i] > 0);
				}
			}

			//block sizes and capacities
			std::pair<IndexType, IndexType> blockSizes = {blockSize, otherBlockSize};
			std::pair<IndexType, IndexType> maxBlockSizes = {maxAllowableBlockSize, maxAllowableBlockSize};

			//second round markers
			std::pair<IndexType, IndexType> secondRoundMarkers = {secondRoundMarker, otherSecondRoundMarker};

			//tie breaking keys
			std::vector<ValueType> tieBreakingKeys(borderRegionSize, 0);

			if (settings.useGeometricTieBreaking) {
				for (IndexType i = 0; i < lastRoundMarker; i++) {
					tieBreakingKeys[i] = -distances[inputDist->global2local(interfaceNodes[i])];
				}
				for (IndexType i = lastRoundMarker; i < borderRegionSize; i++) {
					tieBreakingKeys[i] = -distanceSwap[i-lastRoundMarker];
				}
			}

			if (settings.useDiffusionTieBreaking) {
				std::vector<ValueType> load = twoWayLocalDiffusion(input, haloMatrix, graphHalo, borderRegionIDs, secondRoundMarkers, assignedToSecondBlock, settings);
				for (IndexType i = 0; i < borderRegionSize; i++) {
					tieBreakingKeys[i] = std::abs(load[i]);
				}
			}

			SCAI_REGION_END( "ParcoRepart.distributedFMStep.loop.prepareSets" )

			/**
			 * execute FM locally
			 */
			IndexType gain = twoWayLocalFM(input, haloMatrix, graphHalo, borderRegionIDs, borderNodeWeights, secondRoundMarkers, assignedToSecondBlock, maxBlockSizes, blockSizes, tieBreakingKeys, settings);

			{
				SCAI_REGION( "ParcoRepart.distributedFMStep.loop.swapFMResults" )
				/**
				 * Communicate achieved gain.
				 * Since only two values are swapped, the tracing results measure the latency and synchronization overhead,
				 * the difference in running times between the two local FM implementations.
				 */
				swapField[0] = gain;
				swapField[1] = blockSizes.second;
				comm->swap(swapField, 2, partner);
			}
			const IndexType otherGain = swapField[0];
			const IndexType otherSecondBlockSize = swapField[1];

			if (otherSecondBlockSize > maxBlockSizes.first) {
				//If a block is too large after the refinement, it is only because it was too large to begin with.
				assert(otherSecondBlockSize <= blockSize);
			}

			if (otherGain <= 0 && gain <= 0) {
				//Oh well. None of the processors managed an improvement. No need to update data structures.

			}	else {
				SCAI_REGION_START( "ParcoRepart.distributedFMStep.loop.prepareRedist" )

				gainThisRound = std::max(IndexType(otherGain), IndexType(gain));

				assert(gainThisRound > 0);
				gainPerRound[color] = gainThisRound;

				gainSum += gainThisRound;

				//partition must be consistent, so if gains are equal, pick one of lower index.
				bool otherWasBetter = (otherGain > gain || (otherGain == gain && partner < comm->getRank()));

				//swap result of local FM
				IndexType resultSwap[borderRegionIDs.size()];
				std::copy(assignedToSecondBlock.begin(), assignedToSecondBlock.end(), resultSwap);
				comm->swap(resultSwap, borderRegionIDs.size(), partner);

				//keep best solution. Since the two processes used different offsets, we can't copy them directly
				if (otherWasBetter) {
					for (IndexType i = 0; i < lastRoundMarker; i++) {
						assignedToSecondBlock[i] = !(bool(resultSwap[i+otherLastRoundMarker]));//got bool array from partner, need to invert everything.
					}
					for (IndexType i = lastRoundMarker; i < borderRegionSize; i++) {
						assignedToSecondBlock[i] = !(bool(resultSwap[i-lastRoundMarker]));//got bool array from partner, need to invert everything.
					}
				}

				/**
				 * remove nodes
				 */
				for (IndexType i = 0; i < lastRoundMarker; i++) {
					if (assignedToSecondBlock[i]) {
						auto deleteIterator = std::lower_bound(myGlobalIndices.begin(), myGlobalIndices.end(), interfaceNodes[i]);
						assert(*deleteIterator == interfaceNodes[i]);
						myGlobalIndices.erase(deleteIterator);
					}
				}

				/**
				 * add new nodes
				 */
				for (IndexType i = 0; i < otherLastRoundMarker; i++) {
					if (!assignedToSecondBlock[lastRoundMarker + i]) {
						assert(requiredHaloIndices[i] == borderRegionIDs[lastRoundMarker + i]);
						myGlobalIndices.push_back(requiredHaloIndices[i]);
					}
				}

				//sort indices
				std::sort(myGlobalIndices.begin(), myGlobalIndices.end());
				SCAI_REGION_END( "ParcoRepart.distributedFMStep.loop.prepareRedist" )

				SCAI_REGION_START( "ParcoRepart.distributedFMStep.loop.redistribute" )

				SCAI_REGION_START( "ParcoRepart.distributedFMStep.loop.redistribute.generalDistribution" )
				scai::utilskernel::LArray<IndexType> indexTransport(myGlobalIndices.size(), myGlobalIndices.data());
				scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, indexTransport, comm));
				SCAI_REGION_END( "ParcoRepart.distributedFMStep.loop.redistribute.generalDistribution" )

				{
					SCAI_REGION( "ParcoRepart.distributedFMStep.loop.redistribute.updateDataStructures" )
					redistributeFromHalo(input, newDistribution, graphHalo, haloMatrix);
					part = DenseVector<IndexType>(newDistribution, localBlockID);
					if (nodesWeighted) {
						redistributeFromHalo<IndexType>(nodeWeights, newDistribution, graphHalo, nodeWeightHaloData);
					}
				}
				assert(input.getRowDistributionPtr()->isEqual(*part.getDistributionPtr()));
				SCAI_REGION_END( "ParcoRepart.distributedFMStep.loop.redistribute" )

				/**
				 * update local border. This could probably be optimized by only updating the part that could have changed in the last round.
				 */
				{
					SCAI_REGION( "ParcoRepart.distributedFMStep.loop.updateLocalBorder" )
					nodesWithNonLocalNeighbors = getNodesWithNonLocalNeighbors(input);
				}

				/**
				 * update coordinates and block distances
				 */
				if (settings.useGeometricTieBreaking)
				{
					SCAI_REGION( "ParcoRepart.distributedFMStep.loop.updateBlockDistances" )

					for (IndexType dim = 0; dim < coordinates.size(); dim++) {
						scai::utilskernel::LArray<ValueType>& localCoords = coordinates[dim].getLocalValues();
						scai::utilskernel::LArray<ValueType> haloData;
						comm->updateHalo( haloData, localCoords, graphHalo );
						redistributeFromHalo<ValueType>(coordinates[dim], newDistribution, graphHalo, haloData);
					}

					distances = distancesFromBlockCenter(coordinates);
				}
			}
		}
	}
	comm->synchronize();
	for (IndexType color = 0; color < gainPerRound.size(); color++) {
		gainPerRound[color] = comm->sum(gainPerRound[color]) / 2;
	}
	return gainPerRound;
}

template<typename IndexType, typename ValueType>
void ITI::ParcoRepart<IndexType, ValueType>::checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input) {
	SCAI_REGION( "ParcoRepart.checkLocalDegreeSymmetry" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType localN = inputDist->getLocalSize();

	const CSRStorage<ValueType>& storage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
	const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());

	std::vector<IndexType> inDegree(localN, 0);
	std::vector<IndexType> outDegree(localN, 0);
	for (IndexType i = 0; i < localN; i++) {
		IndexType globalI = inputDist->local2global(i);
		const IndexType beginCols = localIa[i];
		const IndexType endCols = localIa[i+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType globalNeighbor = localJa[j];

			if (globalNeighbor != globalI && inputDist->isLocal(globalNeighbor)) {
				IndexType localNeighbor = inputDist->global2local(globalNeighbor);
				outDegree[i]++;
				inDegree[localNeighbor]++;
			}
		}
	}

	for (IndexType i = 0; i < localN; i++) {
		if (inDegree[i] != outDegree[i]) {
			//now check in detail:
			IndexType globalI = inputDist->local2global(i);
			for (IndexType j = localIa[i]; j < localIa[i+1]; j++) {
				IndexType globalNeighbor = localJa[j];
				if (inputDist->isLocal(globalNeighbor)) {
					IndexType localNeighbor = inputDist->global2local(globalNeighbor);
					bool foundBackEdge = false;
					for (IndexType y = localIa[localNeighbor]; y < localIa[localNeighbor+1]; y++) {
						if (localJa[y] == globalI) {
							foundBackEdge = true;
						}
					}
					if (!foundBackEdge) {
						throw std::runtime_error("Local node " + std::to_string(globalI) + " has edge to local node " + std::to_string(globalNeighbor)
											+ " but no back edge found.");
					}
				}
			}
		}
	}
}

template<typename IndexType, typename ValueType>
ValueType ITI::ParcoRepart<IndexType, ValueType>::twoWayLocalFM(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage,
		const Halo &matrixHalo, const std::vector<IndexType>& borderRegionIDs, const std::vector<IndexType>& nodeWeights, std::pair<IndexType, IndexType> secondRoundMarkers,
		std::vector<bool>& assignedToSecondBlock, const std::pair<IndexType, IndexType> blockCapacities, std::pair<IndexType, IndexType>& blockSizes,
		std::vector<ValueType> tieBreakingKeys, Settings settings) {
	SCAI_REGION( "ParcoRepart.twoWayLocalFM" )

	IndexType magicStoppingAfterNoGainRounds;
	if (settings.stopAfterNoGainRounds > 0) {
		magicStoppingAfterNoGainRounds = settings.stopAfterNoGainRounds;
	} else {
		magicStoppingAfterNoGainRounds = borderRegionIDs.size();
	}

	assert(blockCapacities.first == blockCapacities.second);
	const bool nodesWeighted = (nodeWeights.size() != 0);
	const bool edgesWeighted = nodesWeighted;//TODO: adapt this, change interface

	if (edgesWeighted) {
		ValueType maxWeight = input.getLocalStorage().getValues().max();
		if (maxWeight == 0) {
			throw std::runtime_error("Edges were given as weighted, but maximum weight is zero.");
		}
	}

	const bool gainOverBalance = settings.gainOverBalance;

	if (blockSizes.first >= blockCapacities.first && blockSizes.second >= blockCapacities.second) {
		//cannot move any nodes, all blocks are overloaded already.
		return 0;
	}

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType globalN = inputDist->getGlobalSize();
	scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();

	//the size of this border region
	const IndexType veryLocalN = borderRegionIDs.size();
	assert(tieBreakingKeys.size() == veryLocalN);
	if (nodesWeighted) {
		assert(nodeWeights.size() == veryLocalN);
	}

	const IndexType firstBlockSize = std::distance(assignedToSecondBlock.begin(), std::lower_bound(assignedToSecondBlock.begin(), assignedToSecondBlock.end(), 1));
	assert(secondRoundMarkers.first <= firstBlockSize);
	assert(secondRoundMarkers.second <= veryLocalN - firstBlockSize);

	//this map provides an index from 0 to b-1 for each of the b indices in borderRegionIDs
	//globalToVeryLocal[borderRegionIDs[i]] = i
	std::map<IndexType, IndexType> globalToVeryLocal;

	for (IndexType i = 0; i < veryLocalN; i++) {
		IndexType globalIndex = borderRegionIDs[i];
		globalToVeryLocal[globalIndex] = i;
	}

	assert(globalToVeryLocal.size() == veryLocalN);

	auto isInBorderRegion = [&](IndexType globalID){return globalToVeryLocal.count(globalID) > 0;};

	/**
	 * This lambda computes the initial gain of each node.
	 * Inlining to reduce the overhead of read access locks didn't give any performance benefit.
	 */
	auto computeInitialGain = [&](IndexType veryLocalID){
		SCAI_REGION( "ParcoRepart.twoWayLocalFM.computeGain" )
		ValueType result = 0;
		IndexType globalID = borderRegionIDs[veryLocalID];
		IndexType isInSecondBlock = assignedToSecondBlock[veryLocalID];
		/**
		 * neighborhood information is either in local matrix or halo.
		 */
		const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2local(globalID) : matrixHalo.global2halo(globalID);
		assert(localID != nIndex);

		//get locks
		const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
		const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());
		const scai::hmemo::ReadAccess<ValueType> values(storage.getValues());

		//get indices for CSR structure
		const IndexType beginCols = localIa[localID];
		const IndexType endCols = localIa[localID+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType globalNeighbor = localJa[j];
			if (globalNeighbor == globalID) {
				//self-loop, not counted
				continue;
			}

			const IndexType weight = edgesWeighted ? values[j] : 1;

			if (inputDist->isLocal(globalNeighbor)) {
				//neighbor is in local block,
				result += isInSecondBlock ? weight : -weight;
			} else if (matrixHalo.global2halo(globalNeighbor) != nIndex) {
				//neighbor is in partner block
				result += !isInSecondBlock ? weight : -weight;
			} else {
				//neighbor is from somewhere else, no effect on gain.
			}
		}

		return result;
	};

	/**
	 * construct and fill gain table and priority queues. Since only one target block is possible, gain table is one-dimensional.
	 * One could probably optimize this by choosing the PrioQueueForInts, but it only supports positive keys and requires some adaptations
	 */
	PrioQueue<std::pair<IndexType, ValueType>, IndexType> firstQueue(veryLocalN);
	PrioQueue<std::pair<IndexType, ValueType>, IndexType> secondQueue(veryLocalN);

	std::vector<IndexType> gain(veryLocalN);

	for (IndexType i = 0; i < veryLocalN; i++) {
		gain[i] = computeInitialGain(i);
		const ValueType tieBreakingKey = tieBreakingKeys[i];
		if (assignedToSecondBlock[i]) {
			//the queues only support extractMin, since we want the maximum gain each round, we multiply it with -1
			secondQueue.insert(std::make_pair(-gain[i], tieBreakingKey), i);
		} else {
			firstQueue.insert(std::make_pair(-gain[i], tieBreakingKey), i);
		}
	}

	//whether a node was already moved
	std::vector<bool> moved(veryLocalN, false);
	//which node was transfered in each round
	std::vector<IndexType> transfers;
	transfers.reserve(veryLocalN);

	ValueType gainSum = 0;
	std::vector<IndexType> gainSumList, sizeList;
	gainSumList.reserve(veryLocalN);
	sizeList.reserve(veryLocalN);

	IndexType iter = 0;
	IndexType iterWithoutGain = 0;
	while (firstQueue.size() + secondQueue.size() > 0 && iterWithoutGain < magicStoppingAfterNoGainRounds) {
		SCAI_REGION( "ParcoRepart.twoWayLocalFM.queueloop" )
		IndexType bestQueueIndex = -1;

		assert(firstQueue.size() + secondQueue.size() == veryLocalN - iter);
		/*
		 * first check break situations
		 */
		if ((firstQueue.size() == 0 && blockSizes.first >= blockCapacities.first)
				 ||	(secondQueue.size() == 0 && blockSizes.second >= blockCapacities.second)) {
			//cannot move any nodes
			break;
		}

		/*
		 * now check situations where we have no choice, for example because one queue is empty or one block is already full
		 */
		if (firstQueue.size() == 0) {
			assert(blockSizes.first < blockCapacities.first);
			bestQueueIndex = 1;
		} else if (secondQueue.size() == 0) {
			assert(blockSizes.second < blockCapacities.second);
			bestQueueIndex = 0;
		} else if (blockSizes.first >= blockCapacities.first) {
			bestQueueIndex = 1;
		} else if (blockSizes.second >= blockCapacities.second) {
			bestQueueIndex = 0;
		}
		else {
			/**
			 * now the rest
			 */
			SCAI_REGION( "ParcoRepart.twoWayLocalFM.queueloop.queueselection" )
			std::pair<IndexType, IndexType> firstComparisonPair;
			std::pair<IndexType, IndexType> secondComparisonPair;

			if (gainOverBalance) {
				firstComparisonPair = {-firstQueue.inspectMin().first.first, blockSizes.first};
				firstComparisonPair = {-secondQueue.inspectMin().first.first, blockSizes.second};
			} else {
				firstComparisonPair = {blockSizes.first, -firstQueue.inspectMin().first.first};
				secondComparisonPair = {blockSizes.second, -secondQueue.inspectMin().first.first};
			}

			if (firstComparisonPair > secondComparisonPair) {
				bestQueueIndex = 0;
			} else if (secondComparisonPair > firstComparisonPair) {
				bestQueueIndex = 1;
			} else {
				//tie, break randomly
				bestQueueIndex = (double(rand()) / RAND_MAX < 0.5);
			}

			assert(bestQueueIndex == 0 || bestQueueIndex == 1);
		}

		PrioQueue<std::pair<IndexType, ValueType>, IndexType>& currentQueue = bestQueueIndex == 0 ? firstQueue : secondQueue;

		//Now, we have selected a Queue. Get best vertex and gain
		IndexType veryLocalID;
		std::tie(std::ignore, veryLocalID) = currentQueue.extractMin();
		ValueType topGain = gain[veryLocalID];
		IndexType topVertex = borderRegionIDs[veryLocalID];

		//here one could assert some consistency

		if (topGain > 0) iterWithoutGain = 0;
		else iterWithoutGain++;

		//move node
		transfers.push_back(veryLocalID);
		assignedToSecondBlock[veryLocalID] = !bestQueueIndex;
		moved[veryLocalID] = true;

		//update gain sum
		gainSum += topGain;
		gainSumList.push_back(gainSum);

		//update sizes
		const IndexType nodeWeight = nodesWeighted ? nodeWeights[veryLocalID] : 1;
		blockSizes.first += bestQueueIndex == 0 ? -nodeWeight : nodeWeight;
		blockSizes.second += bestQueueIndex == 0 ? nodeWeight : -nodeWeight;
		sizeList.push_back(std::max(blockSizes.first, blockSizes.second));

		/**
		 * update gains of neighbors
		 */
		SCAI_REGION_START("ParcoRepart.twoWayLocalFM.queueloop.acquireLocks")
		const CSRStorage<ValueType>& storage = inputDist->isLocal(topVertex) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(topVertex) ? inputDist->global2local(topVertex) : matrixHalo.global2halo(topVertex);
		assert(localID != nIndex);

		const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
		const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());
		const scai::hmemo::ReadAccess<ValueType> localValues(storage.getValues());
		const IndexType beginCols = localIa[localID];
		const IndexType endCols = localIa[localID+1];
		SCAI_REGION_END("ParcoRepart.twoWayLocalFM.queueloop.acquireLocks")

		for (IndexType j = beginCols; j < endCols; j++) {
			SCAI_REGION( "ParcoRepart.twoWayLocalFM.queueloop.gainupdate" )
			IndexType neighbor = localJa[j];
			//here we only need to update gain of neighbors in border regions
			if (isInBorderRegion(neighbor)) {
				IndexType veryLocalNeighborID = globalToVeryLocal.at(neighbor);
				if (moved[veryLocalNeighborID]) {
					continue;
				}
				bool wasInSameBlock = (bestQueueIndex == assignedToSecondBlock[veryLocalNeighborID]);
				const ValueType edgeWeight = edgesWeighted ? localValues[j] : 1;
				const ValueType oldGain = gain[veryLocalNeighborID];
				//gain change is twice the value of the affected edge. Direction depends on block assignment.
				gain[veryLocalNeighborID] = oldGain + 2*(2*wasInSameBlock - 1)*edgeWeight;

				const ValueType tieBreakingKey = tieBreakingKeys[veryLocalNeighborID];
				const std::pair<IndexType, ValueType> oldKey = std::make_pair(-oldGain, tieBreakingKey);
				const std::pair<IndexType, ValueType> newKey = std::make_pair(-gain[veryLocalNeighborID], tieBreakingKey);

				if (assignedToSecondBlock[veryLocalNeighborID]) {
					secondQueue.updateKey(oldKey, newKey, veryLocalNeighborID);
				} else {
					firstQueue.updateKey(oldKey, newKey, veryLocalNeighborID);
				}
			}
		}
		iter++;
	}

	/**
	* now find best partition among those tested
	*/
	ValueType maxGain = 0;
	const IndexType testedNodes = gainSumList.size();
	if (testedNodes == 0) return 0;

	SCAI_REGION_START( "ParcoRepart.twoWayLocalFM.recoverBestCut" )
	IndexType maxIndex = -1;

	for (IndexType i = 0; i < testedNodes; i++) {
		if (gainSumList[i] > maxGain && sizeList[i] <= blockCapacities.first) {
			maxIndex = i;
			maxGain = gainSumList[i];
		}
	}
	assert(testedNodes >= maxIndex);
	assert(testedNodes-1 < transfers.size());

	/**
	 * apply partition modifications in reverse until best is recovered
	 */
	for (int i = testedNodes-1; i > maxIndex; i--) {
		assert(transfers[i] < globalN);
		IndexType veryLocalID = transfers[i];
		bool previousBlock = !assignedToSecondBlock[veryLocalID];

		//apply movement in reverse
		assignedToSecondBlock[veryLocalID] = previousBlock;
		const IndexType nodeWeight = nodesWeighted ? nodeWeights[veryLocalID] : 1;
		blockSizes.first += previousBlock == 0 ? nodeWeight : -nodeWeight;
		blockSizes.second += previousBlock == 0 ? -nodeWeight : nodeWeight;

	}
	SCAI_REGION_END( "ParcoRepart.twoWayLocalFM.recoverBestCut" )

	return maxGain;
}

template<typename IndexType, typename ValueType>
IndexType ITI::ParcoRepart<IndexType, ValueType>::twoWayLocalCut(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage,
		const Halo &matrixHalo, const std::vector<IndexType>& borderRegionIDs, const std::vector<bool>& assignedToSecondBlock) {

	//initialize map
	std::map<IndexType, IndexType> globalToVeryLocal;
	for (IndexType i = 0; i < borderRegionIDs.size(); i++) {
		IndexType globalIndex = borderRegionIDs[i];
		globalToVeryLocal[globalIndex] = i;
	}

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	auto isInBorderRegion = [&](IndexType globalID){return globalToVeryLocal.count(globalID) > 0;};

	//compute cut
	IndexType cut = 0;
	for (IndexType i = 0; i < borderRegionIDs.size(); i++) {
		const IndexType globalID = borderRegionIDs[i];
		const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2local(globalID) : matrixHalo.global2halo(globalID);
		assert(localID != nIndex);

		const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
		const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());

		const IndexType beginCols = localIa[localID];
		const IndexType endCols = localIa[localID+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			const IndexType globalNeighbor = localJa[j];
			if (globalNeighbor == globalID) {
				//self-loop, not counted
				continue;
			}

			bool neighborIsInOtherBlock;
			if (isInBorderRegion(globalNeighbor)) {
				const IndexType veryLocalNeighbor = globalToVeryLocal.at(globalNeighbor);
				if (veryLocalNeighbor < i) {
					continue;//only count edges once
				}
				neighborIsInOtherBlock = (assignedToSecondBlock[i] != assignedToSecondBlock[veryLocalNeighbor]);
			} else {
				if (assignedToSecondBlock[i]) {
					neighborIsInOtherBlock = inputDist->isLocal(globalNeighbor);
				} else {
					neighborIsInOtherBlock = (matrixHalo.global2halo(globalNeighbor) != nIndex);
				}
			}

			cut += neighborIsInOtherBlock;
		}
	}
	return cut;
}

template<typename IndexType, typename ValueType>
ValueType ITI::ParcoRepart<IndexType, ValueType>::twoWayLocalDiffusion(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage,
		const Halo &matrixHalo, const std::vector<IndexType>& borderRegionIDs, std::pair<IndexType, IndexType> secondRoundMarkers,
		std::vector<bool>& assignedToSecondBlock, const std::pair<IndexType, IndexType> blockCapacities, std::pair<IndexType, IndexType>& blockSizes,
		Settings settings) {

	//get old cut and block sizes
	const IndexType veryLocalN = borderRegionIDs.size();
	const IndexType oldCut = twoWayLocalCut(input, haloStorage, matrixHalo, borderRegionIDs, assignedToSecondBlock);
	const IndexType firstBlockSize = std::distance(assignedToSecondBlock.begin(), std::lower_bound(assignedToSecondBlock.begin(), assignedToSecondBlock.end(), 1));
	const IndexType secondBlockSize = veryLocalN - firstBlockSize;

	//call diffusion
	std::vector<ValueType> load = twoWayLocalDiffusion(input, haloStorage, matrixHalo, borderRegionIDs, secondRoundMarkers, assignedToSecondBlock, settings);

	//update cut, block sizes and result
	IndexType newSecondBlockSize = 0;
	for (IndexType i = 0; i < veryLocalN; i++) {
		if ((load[i] < 0) != assignedToSecondBlock[i]) {
			//std::cout << i << " has load " << load[i] << ", assigned to block " << assignedToSecondBlock[i]+1 << std::endl;
		}
		assignedToSecondBlock[i] = load[i] < 0;
		newSecondBlockSize += assignedToSecondBlock[i];
	}

	//get new cut and block sizes
	const IndexType newCut = twoWayLocalCut(input, haloStorage, matrixHalo, borderRegionIDs, assignedToSecondBlock);
	const IndexType newFirstBlockSize = veryLocalN - secondBlockSize;

	blockSizes.first += newFirstBlockSize - firstBlockSize;
	blockSizes.second += newSecondBlockSize - secondBlockSize;

	return oldCut - newCut;
}

template<typename IndexType, typename ValueType>
std::vector<ValueType> ITI::ParcoRepart<IndexType, ValueType>::twoWayLocalDiffusion(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage,
		const Halo &matrixHalo, const std::vector<IndexType>& borderRegionIDs, std::pair<IndexType, IndexType> secondRoundMarkers,
		const std::vector<bool>& assignedToSecondBlock, Settings settings) {

	SCAI_REGION( "ParcoRepart.twoWayLocalDiffusion" )
	//settings and constants
	const IndexType magicNumberDiffusionSteps = settings.diffusionRounds;
	const ValueType degreeEstimate = ValueType(haloStorage.getNumValues()) / matrixHalo.getHaloSize();

	const ValueType magicNumberDiffusionLoad = 1;
	const IndexType veryLocalN = borderRegionIDs.size();

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	const IndexType firstBlockSize = std::distance(assignedToSecondBlock.begin(), std::lower_bound(assignedToSecondBlock.begin(), assignedToSecondBlock.end(), 1));
	const IndexType secondBlockSize = veryLocalN - firstBlockSize;
	const ValueType initialLoadPerNodeInFirstBlock = magicNumberDiffusionLoad / firstBlockSize;
	const ValueType initialLoadPerNodeInSecondBlock = -magicNumberDiffusionLoad / secondBlockSize;

	//assign initial diffusion load
	std::vector<ValueType> result(veryLocalN);
	for (IndexType i = 0; i < veryLocalN; i++) {
		result[i] = assignedToSecondBlock[i] ? initialLoadPerNodeInSecondBlock : initialLoadPerNodeInFirstBlock;
	}

	//mark nodes in border as active, rest as inactive
	std::vector<bool> active(veryLocalN, false);
	for (IndexType i = 0; i < secondRoundMarkers.first; i++) {
		active[i] = true;
	}
	for (IndexType i = firstBlockSize; i < firstBlockSize+secondRoundMarkers.second; i++) {
		active[i] = true;
	}


	//this map provides an index from 0 to b-1 for each of the b indices in borderRegionIDs
	//globalToVeryLocal[borderRegionIDs[i]] = i
	std::map<IndexType, IndexType> globalToVeryLocal;

	for (IndexType i = 0; i < veryLocalN; i++) {
		IndexType globalIndex = borderRegionIDs[i];
		globalToVeryLocal[globalIndex] = i;
	}

	IndexType maxDegree = 0;
	{
		const scai::hmemo::ReadAccess<IndexType> localIa(input.getLocalStorage().getIA());
		for (IndexType i = 0; i < firstBlockSize; i++) {
			const IndexType localI = inputDist->global2local(borderRegionIDs[i]);
			if (localIa[localI+1]-localIa[localI] > maxDegree) maxDegree = localIa[localI+1]-localIa[localI];
		}
	}
	{
		const scai::hmemo::ReadAccess<IndexType> localIa(haloStorage.getIA());
		for (IndexType i = 0; i < localIa.size()-1; i++) {
			if (localIa[i+1]-localIa[i] > maxDegree) maxDegree = localIa[i+1]-localIa[i];
		}
	}


	const ValueType magicNumberAlpha = 1.0/(maxDegree+1);

	//assert that all indices were unique
	assert(globalToVeryLocal.size() == veryLocalN);

	auto isInBorderRegion = [&](IndexType globalID){return globalToVeryLocal.count(globalID) > 0;};

	//perform diffusion
	for (IndexType round = 0; round < magicNumberDiffusionSteps; round++) {
		std::vector<ValueType> nextDiffusionValues(result);
		std::vector<bool> nextActive(veryLocalN, false);

		//diffusion update
		for (IndexType i = 0; i < veryLocalN; i++) {
			if (!active[i]) {
				continue;
			}
			nextActive[i] = active[i];

			const ValueType oldDiffusionValue = result[i];
			const IndexType globalID = borderRegionIDs[i];
			const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
			const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2local(globalID) : matrixHalo.global2halo(globalID);
			assert(localID != nIndex);

			const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
			const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());

			const IndexType beginCols = localIa[localID];
			const IndexType endCols = localIa[localID+1];

			double delta = 0.0;
			for (IndexType j = beginCols; j < endCols; j++) {
				const IndexType neighbor = localJa[j];
				if (isInBorderRegion(neighbor)) {
					const IndexType veryLocalNeighbor = globalToVeryLocal.at(neighbor);
					const ValueType difference = result[veryLocalNeighbor] - oldDiffusionValue;
					delta += difference;
					if (difference != 0 && !active[veryLocalNeighbor]) {
						throw std::logic_error("Round " + std::to_string(round) + ": load["+std::to_string(i)+"]="+std::to_string(oldDiffusionValue)
						+", load["+std::to_string(veryLocalNeighbor)+"]="+std::to_string(result[veryLocalNeighbor])
						+", but "+std::to_string(veryLocalNeighbor)+" marked as inactive.");
					}
					nextActive[veryLocalNeighbor] = true;
				}
			}

			nextDiffusionValues[i] = oldDiffusionValue + delta * magicNumberAlpha;
			assert (std::abs(nextDiffusionValues[i]) <= magicNumberDiffusionLoad);
		}

		result.swap(nextDiffusionValues);
		active.swap(nextActive);
	}

	return result;
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {

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

	scai::dmemo::Halo partHalo = buildNeighborHalo(adjM);
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

//----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> ParcoRepart<IndexType, ValueType>::getPEGraph( const CSRSparseMatrix<ValueType> &adjM) {
    SCAI_REGION("ParcoRepart.getPEGraph");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr(); 
    const IndexType numPEs = comm->getSize();
    
    const std::vector<IndexType> nonLocalIndices = nonLocalNeighbors(adjM);
    
    SCAI_REGION_START("ParcoRepart.getPEGraph.getOwners");
    scai::utilskernel::LArray<IndexType> indexTransport(nonLocalIndices.size(), nonLocalIndices.data());
    // find the PEs that own every non-local index
    scai::hmemo::HArray<IndexType> owners(nonLocalIndices.size() , -1);
    dist->computeOwners( owners, indexTransport);
    SCAI_REGION_END("ParcoRepart.getPEGraph.getOwners");
    
    scai::hmemo::ReadAccess<IndexType> rOwners(owners);
    std::vector<IndexType> neighborPEs(rOwners.get(), rOwners.get()+rOwners.size());
    rOwners.release();
    std::sort(neighborPEs.begin(), neighborPEs.end());
    //remove duplicates
    neighborPEs.erase(std::unique(neighborPEs.begin(), neighborPEs.end()), neighborPEs.end());
    const IndexType numNeighbors = neighborPEs.size();

    // create the PE adjacency matrix to be returned
    scai::dmemo::DistributionPtr distPEs ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, numPEs) );
    assert(distPEs->getLocalSize() == 1);
    scai::dmemo::DistributionPtr noDistPEs (new scai::dmemo::NoDistribution( numPEs ));

    SCAI_REGION_START("ParcoRepart.getPEGraph.buildMatrix");
    scai::utilskernel::LArray<IndexType> ia(2, 0, numNeighbors);
    scai::utilskernel::LArray<IndexType> ja(numNeighbors, neighborPEs.data());
    scai::utilskernel::LArray<ValueType> values(numNeighbors, 1);
    scai::lama::CSRStorage<ValueType> myStorage(1, numPEs, neighborPEs.size(), ia, ja, values);
    SCAI_REGION_END("ParcoRepart.getPEGraph.buildMatrix");
    
    //could be optimized with move semantics
    scai::lama::CSRSparseMatrix<ValueType> PEgraph(myStorage, distPEs, noDistPEs);

    return PEgraph;
}

//-----------------------------------------------------------------------------------------

//return: there is an edge in the block graph between blocks ret[0][i]-ret[1][i]
template<typename IndexType, typename ValueType>
std::vector<std::vector<IndexType>> ParcoRepart<IndexType, ValueType>::getLocalBlockGraphEdges( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {
    SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges");
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.initialise");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::utilskernel::LArray<IndexType>& localPart= part.getLocalValues();
    IndexType N = adjM.getNumColumns();
    IndexType max = part.max().Scalar::getValue<IndexType>();
   
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.initialise");
    
    
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.addLocalEdge_newVersion");
    
    scai::hmemo::HArray<IndexType> nonLocalIndices( dist->getLocalSize() ); 
    scai::hmemo::WriteAccess<IndexType> writeNLI(nonLocalIndices, dist->getLocalSize() );
    IndexType actualNeighbours = 0;

    const CSRStorage<ValueType> localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());
    
    // we do not know the size of the non-local indices that is why we use an std::vector
    // with push_back, then convert that to a DenseVector in order to call DenseVector::gather
    // TODO: skip the std::vector to DenseVector conversion. maybe use HArray or LArray
    std::vector< std::vector<IndexType> > edges(2);
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
                                add_edge= false;
                                break;      // the edge (u,v) already exists
                            }
                        }
                        if( add_edge== true){       //if this edge does not exist, add it
                            edges[0].push_back(u);
                            edges[1].push_back(v);
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
    
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.theRest")
    // TODO: this seems to take quite a long !
    // take care of all the non-local indices found
    assert( localInd.size() == nonLocalInd.size() );
    DenseVector<IndexType> nonLocalDV( nonLocalInd.size(), 0 );
    DenseVector<IndexType> gatheredPart( nonLocalDV.size(),0 );
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.theRest")
    
    //get a DenseVector from a vector
    for(IndexType i=0; i<nonLocalInd.size(); i++){
        SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges.vector2DenseVector");
        nonLocalDV.setValue(i, nonLocalInd[i]);
    }
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.gatherNonLocal")
        //gather all non-local indexes
        gatheredPart.gather(part, nonLocalDV , scai::utilskernel::binary::COPY );
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.gatherNonLocal")
    
    assert( gatheredPart.size() == nonLocalInd.size() );
    assert( gatheredPart.size() == localInd.size() );
    
    for(IndexType i=0; i<gatheredPart.size(); i++){
        SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges.addNonLocalEdge");
        IndexType u = localPart[ localInd[i] ];         
        IndexType v = gatheredPart.getValue(i).Scalar::getValue<IndexType>();
        assert( u < max +1);
        assert( v < max +1);
        if( u != v){    // the nodes belong to different blocks                  
            bool add_edge = true;
            for(IndexType k=0; k<edges[0].size(); k++){ //check that this edge is not already in
                if( edges[0][k]==u && edges[1][k]==v ){
                    add_edge= false;
                    break;      // the edge (u,v) already exists
                }
            }
            if( add_edge== true){       //if this edge does not exist, add it
                edges[0].push_back(u);
                edges[1].push_back(v);
            }
        }
    }
    return edges;
}

//-----------------------------------------------------------------------------------------

// in this version the graph is an HArray with size k*k and [i,j] = i*k+j
//
// Not distributed.
//
template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> ParcoRepart<IndexType, ValueType>::getBlockGraph( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, const int k) {
    SCAI_REGION("ParcoRepart.getBlockGraph");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    const scai::utilskernel::LArray<IndexType>& localPart= part.getLocalValues();
    
    // there are k blocks in the partition so the adjecency matrix for the block graph has dimensions [k x k]
    scai::dmemo::DistributionPtr distRowBlock ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, k) );  
    scai::dmemo::DistributionPtr distColBlock ( new scai::dmemo::NoDistribution( k ));
    
    // TODO: memory costly for big k
    IndexType size= k*k;
    
    // get, on each processor, the edges of the blocks that are local
    std::vector< std::vector<IndexType> > blockEdges = ParcoRepart<int, double>::getLocalBlockGraphEdges( adjM, part);
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
                sendPartWrite[ u*k + v ] = 1;
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
    scai::hmemo::HArray<ValueType> csrValues; 
    {
        IndexType numNZ = numEdges;     // this equals the number of edges of the graph
        scai::hmemo::WriteOnlyAccess<IndexType> ia( csrIA, k +1 );
        scai::hmemo::WriteOnlyAccess<IndexType> ja( csrJA, numNZ );
        scai::hmemo::WriteOnlyAccess<ValueType> values( csrValues, numNZ );   
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
                    values[nnzCounter] = 1;
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
std::vector< std::vector<IndexType>> ParcoRepart<IndexType, ValueType>::getGraphEdgeColoring_local(CSRSparseMatrix<ValueType> &adjM, IndexType &colors) {
    SCAI_REGION("ParcoRepart.coloring");
    using namespace boost;
    IndexType N= adjM.getNumRows();
    assert( N== adjM.getNumColumns() ); // numRows = numColumns
    
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));
    if (!adjM.getRowDistributionPtr()->isReplicated()) {
    	adjM.redistribute(noDist, noDist);
    	//throw std::runtime_error("Input matrix must be replicated.");
    }

    // use boost::Graph and boost::edge_coloring()
    typedef adjacency_list<vecS, vecS, undirectedS, no_property, size_t, no_property> Graph;
    typedef std::pair<std::size_t, std::size_t> Pair;
    Graph G(N);
    
    // retG[0][i] the first node, retG[1][i] the second node, retG[2][i] the color of the edge
    std::vector< std::vector<IndexType>> retG(3);
    
	const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    // create graph G from the input adjacency matrix
    for(IndexType i=0; i<N; i++){
    	//we replicated the matrix, so global indices are local indices
    	const IndexType globalI = i;
    	for (IndexType j = ia[i]; j < ia[i+1]; j++) {
    		if (globalI < ja[j]) {
				boost::add_edge(globalI, ja[j], G);
				retG[0].push_back(globalI);
				retG[1].push_back(ja[j]);
    		}
    	}
    }
    
    colors = boost::edge_coloring(G, boost::get( boost::edge_bundle, G));
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    for (size_t i = 0; i <retG[0].size(); i++) {
        retG[2].push_back( G[ boost::edge( retG[0][i],  retG[1][i], G).first] );
    }
    
    return retG;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<DenseVector<IndexType>> ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( CSRSparseMatrix<ValueType> &adjM) {
    IndexType N= adjM.getNumRows();
    SCAI_REGION("ParcoRepart.getCommunicationPairs_local");
    // coloring.size()=3: coloring(i,j,c) means that edge with endpoints i and j is colored with color c.
    // and coloring[i].size()= number of edges in input graph

    assert(adjM.getNumColumns() == adjM.getNumRows() );

    IndexType colors;
    std::vector<std::vector<IndexType>> coloring = getGraphEdgeColoring_local( adjM, colors );
    std::vector<DenseVector<IndexType>> retG(colors);
    
    if (adjM.getNumRows()==2) {
    	assert(colors<=1);
    	assert(coloring[0].size()<=1);
    }
    
    for(IndexType i=0; i<colors; i++){        
        retG[i].allocate(N);
        // TODO: although not distributed maybe try to avoid setValue
        // initialize so retG[i][j]=j instead of -1
        for( IndexType j=0; j<N; j++){
            retG[i].setValue( j, j );                               
        }
    }
    
    // for all the edges:
    // coloring[0][i] = the first block , coloring[1][i] = the second block,
    // coloring[2][i]= the color/round in which the two blocks shall communicate
    for(IndexType i=0; i<coloring[0].size(); i++){
        IndexType color = coloring[2][i]; // the color/round of this edge
        assert(color<colors);
        IndexType firstBlock = coloring[0][i];
        IndexType secondBlock = coloring[1][i];
        retG[color].setValue( firstBlock, secondBlock);
        retG[color].setValue( secondBlock, firstBlock );
    }
    
    return retG;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::pair<IndexType,IndexType>> ParcoRepart<IndexType, ValueType>::maxLocalMatching(const scai::lama::CSRSparseMatrix<ValueType>& adjM){
	SCAI_REGION("ParcoRepart.maxLocalMatching");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    
    // get local data of the adjacency matrix
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia( localStorage.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localStorage.getJA() );
    scai::hmemo::ReadAccess<ValueType> values( localStorage.getValues() );

    // localN= number of local nodes
    const IndexType localN= adjM.getLocalNumRows();
    
    // ia must have size localN+1
    assert(ia.size()-1 == localN );
    
    //mainly for debugging reasons
    IndexType totalNbrs= 0;
    
    // the vector<vector> to return
    // matching[0][i]-matching[1][i] are the endopoints of an edge that is matched
    //std::vector<std::vector<IndexType>> matching(2);
    std::vector<std::pair<IndexType,IndexType>> matching;
    
    // keep track of which nodes are already matched
    std::vector<bool> matched(localN, false);
    
    // localNode is the local index of a node
    for(IndexType localNode=0; localNode<localN; localNode++){
        // if the node is already matched go to the next one;
        if(matched[localNode]){
            continue;
        }
        
        IndexType bestTarget = -1;
        const IndexType endCols = ia[localNode+1];
        for (IndexType j = ia[localNode]; j < endCols; j++) {
        	IndexType localNeighbor = distPtr->global2local(ja[j]);
        	if (localNeighbor != nIndex && localNeighbor != localNode && !matched[localNeighbor]) {
        		//neighbor is local and unmatched, possible partner
        		if (bestTarget < 0 || values[j] > values[bestTarget]) {
        			//either we haven't found any target yet, or the current one is better
        			bestTarget = j;
        		}
        	}
        }

        if (bestTarget > 0) {
        	IndexType globalNgbr = ja[bestTarget];

			// at this point -globalNgbr- is the local node with the heaviest edge
			// and should be matched with -localNode-.
			// So, actually, globalNgbr is also local....
			assert( distPtr->isLocal(globalNgbr));
			IndexType localNgbr = distPtr->global2local(globalNgbr);

			matching.push_back( std::pair<IndexType,IndexType> (localNode, localNgbr) );

			// mark nodes as matched
			matched[localNode]= true;
			matched[localNgbr]= true;
			//PRINT(*comm << ", contracting nodes (local indices): "<< localNode <<" - "<< localNgbr );
        }
    }
    
    assert(ia[ia.size()-1] >= totalNbrs);
    
    return matching;
}

template<typename IndexType, typename ValueType>
DenseVector<ValueType> ParcoRepart<IndexType, ValueType>::projectToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("ParcoRepart.projectToCoarse.interpolate");
	const scai::dmemo::DistributionPtr inputDist = input.getDistributionPtr();

	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();
	scai::dmemo::DistributionPtr coarseDist = projectToCoarse(fineToCoarse);
	IndexType coarseLocalN = coarseDist->getLocalSize();
	assert(inputDist->getLocalSize() == fineLocalN);

	//add values in preparation for interpolation
	std::vector<ValueType> sum(coarseLocalN, 0);
	std::vector<IndexType> numFineNodes(coarseLocalN, 0);
	{
		scai::hmemo::ReadAccess<ValueType> rInput(input.getLocalValues());
		scai::hmemo::ReadAccess<IndexType> rFineToCoarse(fineToCoarse.getLocalValues());
		for (IndexType i = 0; i < fineLocalN; i++) {
			const IndexType coarseTarget = coarseDist->global2local(rFineToCoarse[i]);
			sum[coarseTarget] += rInput[i];
			numFineNodes[coarseTarget] += 1;
		}
	}

	DenseVector<ValueType> result(coarseDist, 0);
	scai::hmemo::WriteAccess<ValueType> wResult(result.getLocalValues());
	for (IndexType i = 0; i < coarseLocalN; i++) {
		assert(numFineNodes[i] > 0);
		wResult[i] = sum[i] / numFineNodes[i];
	}
	wResult.release();
	return result;
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::sumToCoarse(const DenseVector<IndexType>& input, const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("ParcoRepart.sumToCoarse");
	const scai::dmemo::DistributionPtr inputDist = input.getDistributionPtr();

	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();
	scai::dmemo::DistributionPtr coarseDist = projectToCoarse(fineToCoarse);
	IndexType coarseLocalN = coarseDist->getLocalSize();
	assert(inputDist->getLocalSize() == fineLocalN);

	DenseVector<IndexType> result(coarseDist, 0);
	scai::hmemo::WriteAccess<IndexType> wResult(result.getLocalValues());
	{
		scai::hmemo::ReadAccess<IndexType> rInput(input.getLocalValues());
		scai::hmemo::ReadAccess<IndexType> rFineToCoarse(fineToCoarse.getLocalValues());
		for (IndexType i = 0; i < fineLocalN; i++) {
			const IndexType coarseTarget = coarseDist->global2local(rFineToCoarse[i]);
			assert(coarseTarget < coarseLocalN);
			wResult[coarseTarget] += rInput[i];
		}
	}

	wResult.release();
	return result;
}

template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr ParcoRepart<IndexType, ValueType>::projectToCoarse(const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("ParcoRepart.projectToCoarse.distribution");
	const IndexType newGlobalN = fineToCoarse.max().Scalar::getValue<IndexType>() +1;
	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();

	//get set of coarse local indices, without repetitions
	scai::utilskernel::LArray<IndexType> myCoarseGlobalIndices(fineToCoarse.getLocalValues());
	scai::hmemo::WriteAccess<IndexType> wIndices(myCoarseGlobalIndices);
	assert(wIndices.size() == fineLocalN);
	std::sort(wIndices.get(), wIndices.get() + fineLocalN);
	auto newEnd = std::unique(wIndices.get(), wIndices.get() + fineLocalN);
	wIndices.resize(std::distance(wIndices.get(), newEnd));
	IndexType coarseLocalN = wIndices.size();
	wIndices.release();

	scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(newGlobalN, myCoarseGlobalIndices, fineToCoarse.getDistributionPtr()->getCommunicatorPtr()));
	return newDist;
}

template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr ParcoRepart<IndexType, ValueType>::projectToFine(scai::dmemo::DistributionPtr dist, const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("ParcoRepart.projectToFine");
	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();
	scai::dmemo::DistributionPtr coarseDist = projectToCoarse(fineToCoarse);
	IndexType coarseLocalN = coarseDist->getLocalSize();
	scai::dmemo::CommunicatorPtr comm = fineDist->getCommunicatorPtr();

	scai::utilskernel::LArray<IndexType> myCoarseGlobalIndices;
	coarseDist->getOwnedIndexes(myCoarseGlobalIndices);

	scai::utilskernel::LArray<IndexType> owners(coarseLocalN);
	dist->computeOwners(owners, myCoarseGlobalIndices);

	//get send quantities and array
	std::vector<IndexType> quantities(comm->getSize(), 0);
	std::vector<std::vector<IndexType> > sendIndices(comm->getSize());
	{
		scai::hmemo::ReadAccess<IndexType> rOwners(owners);
		scai::hmemo::ReadAccess<IndexType> rFineToCoarse(fineToCoarse.getLocalValues());
		for (IndexType i = 0; i < fineLocalN; i++) {
			IndexType targetRank = rOwners[coarseDist->global2local(rFineToCoarse[i])];
			assert(targetRank < comm->getSize());
			sendIndices[targetRank].push_back(fineDist->local2global(i));
			quantities[targetRank]++;
		}
	}

	std::vector<IndexType> flatIndexVector;
	for (IndexType i = 0; i < sendIndices.size(); i++) {
		std::copy(sendIndices[i].begin(), sendIndices[i].end(), std::back_inserter(flatIndexVector));
	}

	assert(flatIndexVector.size() == fineLocalN);

    scai::dmemo::CommunicationPlan sendPlan;

    sendPlan.allocate( quantities.data(), comm->getSize() );

    assert(sendPlan.totalQuantity() == fineLocalN);

    scai::dmemo::CommunicationPlan recvPlan;

	recvPlan.allocateTranspose( sendPlan, *comm );

	scai::utilskernel::LArray<IndexType> newValues;

	IndexType newLocalSize = recvPlan.totalQuantity();

	{
		scai::hmemo::WriteOnlyAccess<IndexType> recvVals( newValues, newLocalSize );
		comm->exchangeByPlan( recvVals.get(), recvPlan, flatIndexVector.data(), sendPlan );
	}
	assert(comm->sum(newLocalSize) == fineDist->getGlobalSize());

	{
		scai::hmemo::WriteAccess<IndexType> wValues(newValues);
		std::sort(wValues.get(), wValues.get()+newLocalSize);
	}

	scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(fineDist->getGlobalSize(), newValues, comm));
	return newDist;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
DenseVector<T> ParcoRepart<IndexType, ValueType>::computeGlobalPrefixSum(DenseVector<T> input, T globalOffset) {
	SCAI_REGION("ParcoRepart.computeGlobalPrefixSum");
	scai::dmemo::CommunicatorPtr comm = input.getDistributionPtr()->getCommunicatorPtr();

	const IndexType p = comm->getSize();

	//first, check that the input is some block distribution
	const IndexType localN = input.getDistributionPtr()->getBlockDistributionSize();
    if (localN == nIndex) {
    	throw std::logic_error("Global Prefix sum only implemented for block distribution.");
    }

    //get local prefix sum
    scai::hmemo::ReadAccess<T> localValues(input.getLocalValues());
    std::vector<T> localPrefixSum(localN);
    std::partial_sum(localValues.get(), localValues.get()+localN, localPrefixSum.begin());

    T localSum[1] = {localPrefixSum[localN-1]};

    //communicate local sums
    T allOffsets[p];
    comm->gather(allOffsets, 1, 0, localSum);

    //compute prefix sum of offsets.
    std::vector<T> offsetPrefixSum(p+1, 0);
    if (comm->getRank() == 0) {
    	//shift begin of output by one, since the first offset is 0
    	std::partial_sum(allOffsets, allOffsets+p, offsetPrefixSum.begin()+1);
    }

    //remove last value, since it would be the offset for the p+1th processor
    offsetPrefixSum.resize(p);

    //communicate offsets
    T myOffset[1];
    comm->scatter(myOffset, 1, 0, offsetPrefixSum.data());

    //get results by adding local sums and offsets
    DenseVector<T> result(input.getDistributionPtr());
    scai::hmemo::WriteOnlyAccess<T> wResult(result.getLocalValues(), localN);
    for (IndexType i = 0; i < localN; i++) {
    	wResult[i] = localPrefixSum[i] + myOffset[0] + globalOffset;
    }

    return result;
}

//-------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void ParcoRepart<IndexType, ValueType>::coarsen(const CSRSparseMatrix<ValueType>& adjM, CSRSparseMatrix<ValueType>& coarseGraph, DenseVector<IndexType>& fineToCoarse, IndexType iterations) {
	SCAI_REGION("ParcoRepart.coarsen");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    
    // get local data of the adjacency matrix
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia( localStorage.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localStorage.getJA() );
    scai::hmemo::ReadAccess<ValueType> values( localStorage.getValues() );

    // localN= number of local nodes
    IndexType localN= adjM.getLocalNumRows();
    IndexType globalN = adjM.getNumColumns();
    
    // ia must have size localN+1
    assert(ia.size()-1 == localN );
     
    //get a matching, the returned indices are from 0 to localN
    std::vector<std::pair<IndexType,IndexType>> matching = ParcoRepart<IndexType, ValueType>::maxLocalMatching( adjM );
    
    std::vector<IndexType> localMatchingPartner(localN, -1);

    // number of new local nodes
    IndexType newLocalN = localN - matching.size();

    //sort the matching according to its first element
    std::sort(matching.begin(), matching.end());

    //get new global indices by computing a prefix sum over the preserved nodes
    DenseVector<IndexType> preserved(distPtr, 1);
    {
		scai::hmemo::WriteAccess<IndexType> localPreserved(preserved.getLocalValues());

		for (IndexType i = 0; i < matching.size(); i++) {
			assert(matching[i].first != matching[i].second);
			assert(matching[i].first < localN);
			assert(matching[i].second < localN);
			assert(matching[i].first >= 0);
			assert(matching[i].second >= 0);
			localMatchingPartner[matching[i].first] = matching[i].second;
			localMatchingPartner[matching[i].second] = matching[i].first;
			if (matching[i].first < matching[i].second) {
				localPreserved[matching[i].second] = 0;
			} else if (matching[i].second < matching[i].first) {
				localPreserved[matching[i].first] = 0;
			}
		}
    }

    //fill gaps in index list. This might be expensive.
    scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(globalN, comm));
    preserved.redistribute(blockDist);
    fineToCoarse = computeGlobalPrefixSum(preserved, -1);
    const IndexType newGlobalN = fineToCoarse.max().Scalar::getValue<IndexType>() + 1;
    fineToCoarse.redistribute(distPtr);

    //set new indices for contracted nodes
    {
		scai::hmemo::WriteAccess<IndexType> wFineToCoarse(fineToCoarse.getLocalValues());
		assert(wFineToCoarse.size() == localN);
		IndexType nonMatched = 0;

		for (IndexType i = 0; i < localN; i++) {
			if (localMatchingPartner[i] < 0) {
				localMatchingPartner[i] = i;
				nonMatched++;
			}

			if (localMatchingPartner[i] < i) {
				wFineToCoarse[i] = wFineToCoarse[localMatchingPartner[i]];
			}
		}

	    for (IndexType i = 0; i < localN; i++) {
	    	//assert(distPtr->isLocal(wFineToCoarse[i]));
	    	//assert(distPtr->global2local(wFineToCoarse[i]) < newLocalN);
	    	assert(localMatchingPartner[i] < localN);
	    }
	    assert(nonMatched == localN - matching.size()*2);
    }
    assert(fineToCoarse.max().Scalar::getValue<IndexType>() + 1 == newGlobalN);
    assert(newGlobalN <= globalN);
    assert(newGlobalN == comm->sum(newLocalN));

    //build halo of new global indices
    Halo halo = buildNeighborHalo(adjM);
    scai::utilskernel::LArray<IndexType> haloData;
    comm->updateHalo(haloData, fineToCoarse.getLocalValues(), halo);
    
    scai::hmemo::ReadAccess<IndexType> localFineToCoarse(fineToCoarse.getLocalValues());

    //create new coarsened CSR matrix
    scai::hmemo::HArray<IndexType> newIA;
    scai::hmemo::HArray<IndexType> newJA;
    scai::hmemo::HArray<ValueType> newValues;
    
    // this is larger, need to resize afterwards
    IndexType nnzValues = values.size() - matching.size();

    {
        SCAI_REGION("ParcoRepart.coarsen.getCSRMatrix");
        // this is larger, need to resize afterwards
        scai::hmemo::WriteOnlyAccess<IndexType> newIAWrite( newIA, newLocalN +1 );
        scai::hmemo::WriteOnlyAccess<IndexType> newJAWrite( newJA, nnzValues);
        scai::hmemo::WriteOnlyAccess<ValueType> newValuesWrite( newValues, nnzValues);
        newIAWrite[0] = 0;
        IndexType iaIndex = 1;
        IndexType jaIndex = 0;
        
        //for all rows before the coarsening
        for(IndexType i=0; i<localN; i++){
            IndexType matchingPartner = localMatchingPartner[i];
            //duplicate code is evil. Maybe use a lambda instead?
            if (matchingPartner >= i) {
            	std::map<IndexType, ValueType> outgoingEdges;

            	//add edges for this node
            	for (IndexType j = ia[i]; j < ia[i+1]; j++) {
            		IndexType oldNeighbor = ja[j];
            		IndexType newGlobalNeighbor;
            		IndexType localID = distPtr->global2local(oldNeighbor);
            		if (localID == matchingPartner) {
            			continue;//no self loops!
            		}
            		if (localID != nIndex) {
            			newGlobalNeighbor = localFineToCoarse[localID];
            		} else {
            			IndexType haloID = halo.global2halo(oldNeighbor);
            			assert(haloID != nIndex);
            			newGlobalNeighbor = haloData[haloID];
            		}
            		//make sure entry exists
            		if (outgoingEdges.count(newGlobalNeighbor) == 0) {
            			outgoingEdges[newGlobalNeighbor] = 0;
            		}
            		outgoingEdges[newGlobalNeighbor] += values[j];
            	}

            	//add edges for matching partner
				if (matchingPartner > i) {
					for (IndexType j = ia[matchingPartner]; j < ia[matchingPartner+1]; j++) {
						IndexType oldNeighbor = ja[j];
						IndexType localID = distPtr->global2local(oldNeighbor);
						IndexType newGlobalNeighbor;
						if (localID == i) {
							continue;//no self loops!
						}
						if (localID != nIndex) {
							newGlobalNeighbor = localFineToCoarse[localID];
						} else {
							IndexType haloID = halo.global2halo(oldNeighbor);
							assert(haloID != nIndex);
							newGlobalNeighbor = haloData[haloID];
						}
						//make sure entry exists
						if (outgoingEdges.count(newGlobalNeighbor) == 0) {
							outgoingEdges[newGlobalNeighbor] = 0;
						}
						outgoingEdges[newGlobalNeighbor] += values[j];
					}
				}

				//write new matrix entries. Since entries in std::map are sorted by their keys, we can just iterate over them
				for (std::pair<IndexType, ValueType> entry : outgoingEdges) {
					assert(jaIndex < newJAWrite.size());
					newJAWrite[jaIndex] = entry.first;
					newValuesWrite[jaIndex] = entry.second;
					jaIndex++;
				}
				assert(iaIndex < newIAWrite.size());
				newIAWrite[iaIndex] = jaIndex;
				iaIndex++;
            }
        }
        
        newJA.resize(jaIndex);
        newValues.resize(jaIndex);
        nnzValues = jaIndex;
    }
    
    //create distribution object for coarse graph
    scai::utilskernel::LArray<IndexType> myGlobalIndices(fineToCoarse.getLocalValues());
    scai::hmemo::WriteAccess<IndexType> wIndices(myGlobalIndices);
    assert(wIndices.size() == localN);
    std::sort(wIndices.get(), wIndices.get() + localN);
    auto newEnd = std::unique(wIndices.get(), wIndices.get() + localN);
    wIndices.resize(std::distance(wIndices.get(), newEnd));
    assert(wIndices.size() == newLocalN);
    wIndices.release();


	const scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(newGlobalN, myGlobalIndices, comm));
	const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(newGlobalN));

    CSRStorage<ValueType> storage;
    storage.setCSRDataSwap(newLocalN, newGlobalN, nnzValues, newIA, newJA, newValues, scai::hmemo::ContextPtr());
    coarseGraph = CSRSparseMatrix<ValueType>(newDist, noDist);
    coarseGraph.swapLocalStorage(storage);
 }


//---------------------------------------------------------------------------------------

//to force instantiation
template DenseVector<int> ParcoRepart<int, double>::partitionGraph(CSRSparseMatrix<double> &input, std::vector<DenseVector<double>> &coordinates, struct Settings);
			     
template double ParcoRepart<int, double>::computeImbalance(const DenseVector<int> &partition, int k, const DenseVector<int> &nodeWeights);

template double ParcoRepart<int, double>::computeCut(const CSRSparseMatrix<double> &input, const DenseVector<int> &part, bool ignoreWeights);

template void ParcoRepart<int, double>::checkLocalDegreeSymmetry(const CSRSparseMatrix<double> &input);

template double ParcoRepart<int, double>::replicatedMultiWayFM(const CSRSparseMatrix<double> &input, DenseVector<int> &part, int k, double epsilon, bool unweighted);

template std::vector<int> ParcoRepart<int, double>::distributedFMStep(CSRSparseMatrix<double> &input, DenseVector<int> &part, std::vector<DenseVector<double>> &coordinates, Settings settings);

template std::vector<DenseVector<int>> ParcoRepart<int, double>::computeCommunicationPairings(const CSRSparseMatrix<double> &input, const DenseVector<int> &part,	const DenseVector<int> &blocksToPEs);

template std::vector<int> ITI::ParcoRepart<int, double>::nonLocalNeighbors(const CSRSparseMatrix<double>& input);

template std::vector<double> ITI::ParcoRepart<int, double>::distancesFromBlockCenter(const std::vector<DenseVector<double>> &coordinates);

template scai::dmemo::Halo ITI::ParcoRepart<int, double>::buildNeighborHalo(const CSRSparseMatrix<double> &input);

template std::pair<std::vector<int>, std::vector<int>> ITI::ParcoRepart<int, double>::getInterfaceNodes(const CSRSparseMatrix<double> &input, const DenseVector<int> &part, const std::vector<int>& nodesWithNonLocalNeighbors, int otherBlock, int depth);

template DenseVector<int> ParcoRepart<int, double>::getBorderNodes( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::lama::CSRSparseMatrix<double> ParcoRepart<int, double>::getPEGraph( const CSRSparseMatrix<double> &adjM);

template std::vector<std::vector<IndexType>> ParcoRepart<int, double>::getLocalBlockGraphEdges( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::lama::CSRSparseMatrix<double> ParcoRepart<int, double>::getBlockGraph( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part, const int k );

template std::vector< std::vector<int>>  ParcoRepart<int, double>::getGraphEdgeColoring_local( CSRSparseMatrix<double> &adjM, int& colors);

template std::vector<DenseVector<int>> ParcoRepart<int, double>::getCommunicationPairs_local( CSRSparseMatrix<double> &adjM);

template std::vector<std::pair<int,int>> ParcoRepart<int, double>::maxLocalMatching(const scai::lama::CSRSparseMatrix<double>& graph);

template void ParcoRepart<int, double>::coarsen(const CSRSparseMatrix<double>& inputGraph, CSRSparseMatrix<double>& coarseGraph, DenseVector<int>& fineToCoarse, int iterations);

template DenseVector<int> ParcoRepart<int, double>::computeGlobalPrefixSum(DenseVector<int> input, int offset);

}
