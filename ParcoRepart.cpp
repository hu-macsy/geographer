/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include <scai/dmemo/HaloBuilder.hpp>
#include <scai/dmemo/Distribution.hpp>
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

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "HilbertCurve.h"

namespace ITI {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, IndexType k,  double epsilon)
{
	SCAI_REGION( "ParcoRepart.partitionGraph" )
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

	if (coordDist->getLocalSize() != localN) {
		throw std::runtime_error(std::to_string(coordDist->getLocalSize()) + " point coordinates, "
				+ std::to_string(localN) + " rows present.");
	}

	if( !coordDist->isEqual( *inputDist) ){
		throw std::runtime_error( "Distributions should be equal.");
	}
	
	/**
	*	gather information for space-filling curves
	*/
	std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
	DenseVector<IndexType> result(inputDist);

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
		const IndexType recursionDepth = std::min(std::log2(n), double(21));
	
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
		scai::lama::DenseVector<IndexType> permutation, inversePermutation;
        {
			SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition.sorting" )
			hilbertIndices.sort(permutation, true);
			//permutation.redistribute(inputDist);
			DenseVector<IndexType> tmpPerm = permutation;
			tmpPerm.sort( inversePermutation, true);
        }
                
		/**
		 * The permutations given by DenseVector.sort are distributed by BlockDistributions.
		 * However, the sorting does not guarantee that each processor has the same number of values.
		 * Without a redistribution step, the line result.getLocalValues()[i] = int( inversePermutation.getLocalValues()[i] *k/n);
		 * sometimes segfaults. We can't have that.
		 */
		inversePermutation.redistribute(inputDist);
		assert(inversePermutation.getDistributionPtr()->getLocalSize() == localN);

		/**
		* initial partitioning with sfc. Upgrade to chains-on-chains-partitioning later
		*/

		for (IndexType i = 0; i < localN; i++) {
			result.getLocalValues()[i] = int( inversePermutation.getLocalValues()[i] *k/n);
		}

		if (!inputDist->isReplicated() && comm->getSize() == k) {
			SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition.redistribute" )
			//TODO: this won't scale well.
		        result.redistribute(noDist);

			scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(result.getLocalValues(), comm));
			//assert(newDistribution->getLocalSize() == subsetSizes[comm->getRank()]);

			input.redistribute(newDistribution, input.getColDistributionPtr());
			result.redistribute(newDistribution);
		}
        }
	IndexType numRefinementRounds = 0;

	if (comm->getSize() == 1 || comm->getSize() == k) {
		ValueType gain = 1;
		ValueType cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input)) / 2;

		while (gain > 0) {
			if (inputDist->isReplicated()) {
				gain = replicatedMultiWayFM(input, result, k, epsilon);
			} else {
				gain = distributedFMStep(input, result, k, epsilon);
			}
			ValueType oldCut = cut;
			cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input)) / 2;
			if (cut != oldCut - gain) {
				IndexType sumOutgoingEdges = comm->sum(localSumOutgoingEdges(input)) / 2;

				std::cout << std::string("Old cut was " + std::to_string(oldCut) + ", new cut is " + std::to_string(cut) + " with "
						+ std::to_string(sumOutgoingEdges) + " outgoing edges, but gain is " + std::to_string(gain)+".") << std::endl;
			}

			assert(oldCut - gain == cut);
			//std::cout << "Last FM round yielded gain of " << gain << ", for total cut of " << computeCut(input, result) << std::endl;
			numRefinementRounds++;
		}
	} else {
		std::cout << "Local refinement only implemented sequentially and for one block per process. Called with " << comm->getSize() << " process and " << k << " blocks." << std::endl;
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
		Scalar partID = part.getValue(v);

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
ValueType ParcoRepart<IndexType, ValueType>::computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const bool ignoreWeights) {
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

	scai::dmemo::Halo partHalo = buildPartHalo(input, part);
	scai::utilskernel::LArray<IndexType> haloData;
	partDist->getCommunicatorPtr()->updateHalo( haloData, localData, partHalo );

	/**
	 * first pass, compute local cut and build list of required halo indices
	 */
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
				if (ignoreWeights) {
					result++;
				} else {
					result += values[j];
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
ValueType ParcoRepart<IndexType, ValueType>::computeCutTwoWay(const CSRSparseMatrix<ValueType> &input,
		const CSRStorage<ValueType> &haloStorage, const Halo &halo,
		const std::set<IndexType> &firstregion,  const std::set<IndexType> &secondregion, const bool ignoreWeights) {

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	ValueType cut = 0;
	for (IndexType node : firstregion) {

		const CSRStorage<ValueType>& storage = inputDist->isLocal(node) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(node) ? inputDist->global2local(node) : halo.global2halo(node);
		assert(localID != nIndex);

		const scai::hmemo::ReadAccess<IndexType> ia(storage.getIA());
		const scai::hmemo::ReadAccess<IndexType> ja(storage.getJA());
		const scai::hmemo::ReadAccess<ValueType> values(storage.getValues());

		for (IndexType j = ia[localID]; j < ia[localID+1]; j++) {
			ValueType edgeweight = ignoreWeights ? 1 : values[j];
			if (secondregion.count(ja[j]) > 0) cut += edgeweight;
		}
	}
	return cut;
}

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(const CSRSparseMatrix<ValueType> &input) {
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	IndexType numOutgoingEdges = 0;
	for (IndexType j = 0; j < ja.size(); j++) {
		if (!input.getRowDistributionPtr()->isLocal(ja[j])) numOutgoingEdges++;
	}

	return numOutgoingEdges;
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
ValueType ParcoRepart<IndexType, ValueType>::computeImbalance(const DenseVector<IndexType> &part, IndexType k) {
	SCAI_REGION( "ParcoRepart.computeImbalance" )
	const IndexType n = part.getDistributionPtr()->getGlobalSize();
	std::vector<IndexType> subsetSizes(k, 0);
	scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());
	const Scalar maxK = part.max();
	if (maxK.getValue<IndexType>() >= k) {
		throw std::runtime_error("Block id " + std::to_string(maxK.getValue<IndexType>()) + " found in partition with supposedly" + std::to_string(k) + " blocks.");
	}
 	
	for (IndexType i = 0; i < localPart.size(); i++) {
		IndexType partID = localPart[i];
		subsetSizes[partID] += 1;
	}
	IndexType optSize = std::ceil(n / k);

	//if we don't have the full partition locally, 
	scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
	if (!part.getDistribution().isReplicated()) {
	  //sum block sizes over all processes
	  for (IndexType partID = 0; partID < k; partID++) {
	    subsetSizes[partID] = comm->sum(subsetSizes[partID]);
	  }
	}
	
	IndexType maxBlockSize = *std::max_element(subsetSizes.begin(), subsetSizes.end());
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
		scai::hmemo::ReadAccess<IndexType> sourceIA(localStorage.getIA());
		scai::hmemo::WriteOnlyAccess<IndexType> wSourceSizes( sourceSizes, sourceNumRows );
	    scai::sparsekernel::OpenMPCSRUtils::offsets2sizes( wSourceSizes.get(), sourceIA.get(), sourceNumRows );
	    //allocate
	    scai::hmemo::WriteOnlyAccess<IndexType> wTargetIA( targetIA, targetNumRows + 1 );
	}

	scai::hmemo::HArray<IndexType> haloSizes;
	{
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

	{
		SCAI_REGION( "ParcoRepart.redistributeFromHalo.setCSRData" )
		//setting CSR data
		matrix.getLocalStorage().setCSRData(targetNumRows, globalN, numValues, targetIA, targetJA, targetValues);
	}
}

template<typename IndexType, typename ValueType>
scai::dmemo::Halo ITI::ParcoRepart<IndexType, ValueType>::buildPartHalo(
		const CSRSparseMatrix<ValueType>& input, const DenseVector<IndexType> &part) {

	SCAI_REGION( "ParcoRepart.buildPartHalo" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();

	if (inputDist->getLocalSize() != partDist->getLocalSize()) {
		throw std::runtime_error("Input matrix and partition must have the same distribution.");
	}

	std::vector<IndexType> requiredHaloIndices = nonLocalNeighbors(input);

	assert(requiredHaloIndices.size() <= partDist->getGlobalSize() - partDist->getLocalSize());

	scai::dmemo::Halo Halo;
	{
		scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
		scai::dmemo::HaloBuilder::build( *partDist, arrRequiredIndexes, Halo );
	}

	return Halo;
}

template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>, IndexType> ITI::ParcoRepart<IndexType, ValueType>::getInterfaceNodes(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, IndexType thisBlock, IndexType otherBlock, IndexType depth) {

	SCAI_REGION( "ParcoRepart.getInterfaceNodes" )
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();

	if (partDist->getLocalSize() != localN) {
		throw std::runtime_error("Partition has " + std::to_string(partDist->getLocalSize()) + " local nodes, but matrix has " + std::to_string(localN) + ".");
	}

	if (thisBlock != comm->getRank()) {
		throw std::runtime_error("Currently only implemented with one block per process, block " + std::to_string(thisBlock) + " invalid for process " + std::to_string(comm->getRank()));
	}


	if (otherBlock > comm->getSize()) {
		throw std::runtime_error("Currently only implemented with one block per process, block " + std::to_string(thisBlock) + " invalid for " + std::to_string(comm->getSize()) + " processes.");
	}

	if (thisBlock == otherBlock) {
		throw std::runtime_error("Block IDs must be different.");
	}

	if (depth < 0) {
		throw std::runtime_error("Depth must be non-negative");
	}


	scai::hmemo::HArray<IndexType> localData = part.getLocalValues();
	scai::hmemo::ReadAccess<IndexType> partAccess(localData);
	
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	/**
	 * first get internal interface nodes and nodes with non-local neighbors
	 */
	std::vector<IndexType> interfaceNodes;
	std::vector<IndexType> nodesWithNonLocalNeighbors;

	for (IndexType localI = 0; localI < localN; localI++) {
		if (partAccess[localI] == thisBlock) {
			const IndexType globalI = inputDist->local2global(localI);
			for (IndexType j = ia[localI]; j < ia[localI+1]; j++) {
				IndexType neighbor = ja[j];
				IndexType neighborBlock;
				if (partDist->isLocal(neighbor)) {
					if (partAccess[partDist->global2local(neighbor)] == otherBlock) {
						interfaceNodes.push_back(globalI);
						break;
					}
				} else {
					nodesWithNonLocalNeighbors.push_back(globalI);
					break;
				}
			}
		}
	}

	/**
	 * send nodes with non-local neighbors to partner process.
	 * here we assume a 1-to-1-mapping of blocks to processes and a symmetric matrix
	 */
	std::unordered_set<IndexType> foreignNodes;
	{
		SCAI_REGION( "ParcoRepart.getInterfaceNodes.communication" )
		IndexType swapField[1];
		swapField[0] = nodesWithNonLocalNeighbors.size();
		comm->swap(swapField, 1, otherBlock);

		const IndexType swapLength = std::max(swapField[0], IndexType(nodesWithNonLocalNeighbors.size()));
		IndexType swapList[swapLength];
		for (IndexType i = 0; i < nodesWithNonLocalNeighbors.size(); i++) {
			swapList[i] = nodesWithNonLocalNeighbors[i];
		}
		comm->swap(swapList, swapLength, otherBlock);

		//the swapList array is only partially filled, the number of received nodes is found in swapField[0]
		for (IndexType i = 0; i < swapField[0]; i++) {
			foreignNodes.insert(swapList[i]);
		}
	}

	for (IndexType node : nodesWithNonLocalNeighbors) {
		IndexType localI = inputDist->global2local(node);
		for (IndexType j = ia[localI]; j < ia[localI+1]; j++) {
			if (foreignNodes.count(ja[j])> 0) {
				interfaceNodes.push_back(node);
				break;
			}
		}
	}

	assert(interfaceNodes.size() <= localN);

	IndexType lastRoundMarker = 0;
	/**
	 * now gather buffer zone with breadth-first search
	 */
	if (depth > 1) {
		std::vector<bool> touched(localN, false);

		std::queue<IndexType> bfsQueue;
		for (IndexType node : interfaceNodes) {
			bfsQueue.push(node);
			touched[inputDist->global2local(node)] = true;
		}
		assert(bfsQueue.size() == interfaceNodes.size());

		for (IndexType round = 1; round < depth; round++) {
			lastRoundMarker = interfaceNodes.size();
			std::queue<IndexType> nextQueue;
			while (!bfsQueue.empty()) {
				IndexType nextNode = bfsQueue.front();
				bfsQueue.pop();

				const IndexType localI = inputDist->global2local(nextNode);
				assert(touched[localI]);
				const IndexType beginCols = ia[localI];
				const IndexType endCols = ia[localI+1];

				for (IndexType j = beginCols; j < endCols; j++) {
					IndexType neighbor = ja[j];
					if (inputDist->isLocal(neighbor) && partAccess[partDist->global2local(neighbor)] == thisBlock &&
							!touched[inputDist->global2local(neighbor)]) {
						nextQueue.push(neighbor);
						interfaceNodes.push_back(neighbor);
						touched[inputDist->global2local(neighbor)] = true;
					}
				}
			}
			bfsQueue = nextQueue;
		}
	}

	assert(interfaceNodes.size() <= localN);
	return {interfaceNodes, lastRoundMarker};
}

template<typename IndexType, typename ValueType>
IndexType ITI::ParcoRepart<IndexType, ValueType>::getDegreeSum(const CSRSparseMatrix<ValueType> &input, std::vector<IndexType> nodes) {
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
ValueType ITI::ParcoRepart<IndexType, ValueType>::distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted) {
	const IndexType magicBorderRegionDepth = 4;
	SCAI_REGION( "ParcoRepart.distributedFMStep" )
	const IndexType globalN = input.getRowDistributionPtr()->getGlobalSize();
	scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();

	if (part.getDistributionPtr()->getLocalSize() != input.getRowDistributionPtr()->getLocalSize()) {
		throw std::runtime_error("Distributions of input matrix and partitions must be equal, for now.");
	}

	if (!input.getColDistributionPtr()->isReplicated()) {
		throw std::runtime_error("Column distribution needs to be replicated.");
	}

	if (epsilon < 0) {
		throw std::runtime_error("Epsilon must be >= 0, not " + std::to_string(epsilon));
	}

	//std::cout << "Thread " << comm->getRank() << ", entered distributed FM." << std::endl;

        /* test communication with coloring. get the block graph and then the communication pairs for all the rounds
         * */
        scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getBlockGraph( input, part, k);
        
        std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph);
        

    const Scalar maxBlockScalar = part.max();
    const IndexType maxBlockID = maxBlockScalar.getValue<IndexType>();

    if (k != maxBlockID + 1) {
    	throw std::runtime_error("Should have " + std::to_string(k) + " blocks, has maximum ID " + std::to_string(maxBlockID));
    }

    if (k != comm->getSize()) {
    	throw std::runtime_error("Called with " + std::to_string(comm->getSize()) + " processors, but " + std::to_string(k) + " blocks.");
    }

    //block sizes
    const IndexType optSize = ceil(double(globalN) / k);
    const IndexType maxAllowableBlockSize = optSize*(1+epsilon);

    //for now, we are assuming equal numbers of blocks and processes
    const IndexType localBlockID = comm->getRank();

    ValueType gainSum = 0;

	//copy into usable data structure with iterators
	std::vector<IndexType> myGlobalIndices(input.getRowDistributionPtr()->getLocalSize());
	{
		const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
		for (IndexType j = 0; j < myGlobalIndices.size(); j++) {
			myGlobalIndices[j] = inputDist->local2global(j);
		}
	}

	for (IndexType i = 0; i < communicationScheme.size(); i++) {
		SCAI_REGION( "ParcoRepart.distributedFMStep.loop" )

		const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
		const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
		const scai::dmemo::DistributionPtr commDist = communicationScheme[i].getDistributionPtr();

		const IndexType localN = inputDist->getLocalSize();
		assert(comm->sum(localN) == globalN);

		if (!communicationScheme[i].getDistributionPtr()->isLocal(comm->getRank())) {
			throw std::runtime_error("Scheme value for " + std::to_string(comm->getRank()) + " must be local.");
		}
		
		scai::hmemo::ReadAccess<IndexType> commAccess(communicationScheme[i].getLocalValues());
		IndexType partner = commAccess[commDist->global2local(comm->getRank())];
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
			scai::hmemo::ReadAccess<IndexType> partAccess(part.getLocalValues());
			for (IndexType j = 0; j < localN; j++) {
				if (partAccess[j] != localBlockID) {
					throw std::runtime_error("Block ID "+std::to_string(partAccess[j])+" found on process "+std::to_string(localBlockID)+".");
				}
			}
		}

		ValueType gainThisRound = 0;

		scai::dmemo::Halo graphHalo;
		CSRStorage<ValueType> haloMatrix;


		if (partner != comm->getRank()) {
			//processor is active this round

			/**
			 * get indices of border nodes with breadth-first search
			 */
			std::vector<IndexType> interfaceNodes;
			IndexType lastRoundMarker;
			std::tie(interfaceNodes, lastRoundMarker)= getInterfaceNodes(input, part, localBlockID, partner, magicBorderRegionDepth+1);

			std::cout << "Thread " << comm->getRank() << ", round " << i << ", gathered interface nodes." << std::endl;

			/**
			 * now swap indices of nodes in border region with partner processor.
			 * For this, first find out the length of the swap array.
			 */

			//swap size of border region and total block size
			IndexType blockSize = localBlockSize(part, localBlockID);
			if (blockSize != localN) {
				throw std::runtime_error(std::to_string(localN) + " local nodes, but only " + std::to_string(blockSize) + " of them belong to block " + std::to_string(localBlockID) + ".");
			}

			IndexType swapField[4];
			swapField[0] = interfaceNodes.size();
			swapField[1] = lastRoundMarker;
			swapField[2] = blockSize;
			swapField[3] = getDegreeSum(input, interfaceNodes);
			comm->swap(swapField, 4, partner);
			const IndexType otherLastRoundMarker = swapField[1];
			const IndexType otherBlockSize = swapField[2];
			const IndexType otherDegreeSum = swapField[3];
			IndexType swapLength = std::max(int(swapField[0]), int(interfaceNodes.size()));

			if (interfaceNodes.size() == 0) {
				if (swapLength != 0) {
					throw std::runtime_error("Partner PE has a border region, but this PE doesn't. Looks like the block indices were allocated badly.");
				} else {
					/*
					 * These processes don't share a border and thus have no communication to do with each other.
					 * We could skip the loop entirely.
					 */
				}
			}

			ValueType swapNodes[swapLength];
			for (IndexType i = 0; i < swapLength; i++) {
				if (i < interfaceNodes.size()) {
					swapNodes[i] = interfaceNodes[i];
				} else {
					swapNodes[i] = -1;
				}
			}

			comm->swap(swapNodes, swapLength, partner);

			//the number of interface nodes was stored in swapField[0] and then swapped.
			//swapField[0] now contains the number of nodes in the partner's border region
			std::vector<IndexType> requiredHaloIndices(swapField[0]);
			for (IndexType i = 0; i < swapField[0]; i++) {
				assert(swapNodes[i] >= 0);
				requiredHaloIndices[i] = swapNodes[i];
			}

			//std::cout << "Thread " << comm->getRank() << ", round " << i << ", swapped " << swapLength << " interface nodes with thread " << partner << std::endl;

			assert(requiredHaloIndices.size() <= globalN - inputDist->getLocalSize());

			/*
			 * extend halos to cover border region of other PE.
			 * This is probably very inefficient in a general distribution where all processors need to be contacted to gather the halo.
			 * Possible Improvements: Assemble arrays describing the subgraph, swap that.
			 * Exchanging the partHalo is actually unnecessary, since all indices in requiredHaloIndices have the same block.
			 */
			IndexType numValues = input.getLocalStorage().getValues().size();
			{
				scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
				scai::hmemo::HArrayRef<IndexType> arrProvidedIndexes( interfaceNodes );
				scai::dmemo::HaloBuilder::build( *inputDist, arrRequiredIndexes, arrProvidedIndexes, partner, graphHalo );
			}

			haloMatrix.exchangeHalo( graphHalo, input.getLocalStorage(), *comm );
			assert(input.getLocalStorage().getValues().size() == numValues);//number of edges in local part stays unchanged.
			assert(haloMatrix.getValues().size() == otherDegreeSum);
			for (IndexType node : requiredHaloIndices) {
				assert(graphHalo.global2halo(node) != nIndex);
			}

			//why not use vectors in the FM step or use sets to begin with? Might be faster.
			//here we only exchange one round less than gathered. The last round forms a dummy border layer.
			std::set<IndexType> firstRegion(interfaceNodes.begin(), interfaceNodes.begin()+lastRoundMarker);
			std::set<IndexType> secondRegion(requiredHaloIndices.begin(), requiredHaloIndices.begin()+otherLastRoundMarker);
			const std::set<IndexType> firstRegionOld = firstRegion;

			std::set<IndexType> firstDummyLayer(interfaceNodes.begin()+lastRoundMarker, interfaceNodes.end());
			std::set<IndexType> secondDummyLayer(requiredHaloIndices.begin()+otherLastRoundMarker, requiredHaloIndices.end());

			assert(firstRegion.size() == lastRoundMarker);
			assert(secondRegion.size() == otherLastRoundMarker);
			assert(firstRegion.size() + firstDummyLayer.size() == interfaceNodes.size());
			assert(secondRegion.size() + secondDummyLayer.size() == requiredHaloIndices.size());

			//block sizes and capacities
			std::pair<IndexType, IndexType> blockSizes = {blockSize, otherBlockSize};
			std::pair<IndexType, IndexType> maxBlockSizes = {maxAllowableBlockSize, maxAllowableBlockSize};

			/**
			 * execute FM locally
			 */
			ValueType gain = twoWayLocalFM(input, haloMatrix, graphHalo, firstRegion, secondRegion, firstDummyLayer, secondDummyLayer, blockSizes, maxBlockSizes, unweighted);

			//communicate achieved gain. PE with better solution should send their secondRegion.
			assert(unweighted); //if this assert fails, you need to change the type of swapField back to ValueType before removing it.
			swapField[0] = secondRegion.size();
			swapField[1] = gain;
			swapField[2] = blockSizes.second;
			comm->swap(swapField, 3, partner);

			if (swapField[2] > maxBlockSizes.first) {
				//If a block is too large after the refinement, it is only because it was too large to begin with.
				assert(swapField[2] <= blockSize);
			}

			if (swapField[1] == 0 && gain == 0) {
				//Oh well. None of the processors managed an improvement. No need to update data structures.
				//std::cout << "Thread " << comm->getRank() << ", round " << i << " no gain here and at " << partner << "." << std::endl;

			}	else {
				SCAI_REGION( "ParcoRepart.distributedFMStep.loop.prepareRedist" )

				gainThisRound = std::max(ValueType(swapField[1]), ValueType(gain));

				assert(gainThisRound > 0);

				gainSum += gainThisRound;

				bool otherWasBetter = (swapField[1] > gain || (swapField[1] == gain && partner < comm->getRank()));

				if (otherWasBetter) {
					swapLength = swapField[0];
				} else {
					swapLength = secondRegion.size();
				}

				ValueType resultSwap[swapLength];
				if (!otherWasBetter) {
					IndexType j = 0;
					for (IndexType nodeID : secondRegion) {
						resultSwap[j] = nodeID;
						j++;
					}
					assert(j == secondRegion.size());
				}

				comm->swap(resultSwap, swapLength, partner);

				//keep best solution
				if (otherWasBetter) {
					firstRegion.clear();
					for (IndexType j = 0; j < swapLength; j++) {
						firstRegion.insert(resultSwap[j]);
					}
					assert(firstRegion.size() == swapLength);
				}

				std::vector<IndexType> additionalNodes;
				std::vector<IndexType> deletedNodes;
				std::vector<IndexType> newIndices(myGlobalIndices);

				/**
				 * remove nodes
				 */
				for (IndexType node : firstRegionOld) {
					if (firstRegion.count(node) == 0) {
						deletedNodes.push_back(node);
						part.setValue(node, partner);

						/**
						 * we want to delete the value from our local indices.
						 * We can assume that the index vector is unique and sorted and that the unwanted index is present.
						 */
						auto deleteIterator = std::lower_bound(newIndices.begin(), newIndices.end(), node);
						newIndices.erase(deleteIterator);
					}
				}

				/**
				 * add new nodes
				 */
				for (IndexType node : firstRegion) {
					if (firstRegionOld.count(node) == 0) {
						additionalNodes.push_back(node);
						newIndices.push_back(node);
						//the following statement probably doesn't do anything.
						part.setValue(node, localBlockID);
					}
				}

				/**
				 * update indices
				 */
				std::sort(newIndices.begin(), newIndices.end());
				assert(newIndices.size() == myGlobalIndices.size()-deletedNodes.size()+additionalNodes.size());
				myGlobalIndices = newIndices;
			}
		}

		{
			SCAI_REGION( "ParcoRepart.distributedFMStep.loop.redistribute" )
			{
				SCAI_REGION( "ParcoRepart.distributedFMStep.loop.redistribute.sync" )
				IndexType participating = comm->sum(1);
				if (participating != comm->getSize()) {
					std::cout << participating << " of " << comm->getSize() << " threads in redistribution." << std::endl;
				}
			}

			SCAI_REGION_START( "ParcoRepart.distributedFMStep.loop.redistribute.generalDistribution" )
			scai::utilskernel::LArray<IndexType> indexTransport(myGlobalIndices.size(), myGlobalIndices.data());
			scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, indexTransport, comm));
			SCAI_REGION_END( "ParcoRepart.distributedFMStep.loop.redistribute.generalDistribution" )

			{
				SCAI_REGION( "ParcoRepart.distributedFMStep.loop.redistribute.updateDataStructures" )
				redistributeFromHalo(input, newDistribution, graphHalo, haloMatrix);
				part = DenseVector<IndexType>(newDistribution, localBlockID);
			}
		}
		assert(input.getRowDistributionPtr()->isEqual(*part.getDistributionPtr()));
	}
	return comm->sum(gainSum) / 2;
}

template<typename IndexType, typename ValueType>
ValueType ITI::ParcoRepart<IndexType, ValueType>::twoWayLocalFM(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage,
		const Halo &matrixHalo, std::set<IndexType> &firstregion,  std::set<IndexType> &secondregion,
		const std::set<IndexType> &firstDummyLayer, const std::set<IndexType> &secondDummyLayer,
		std::pair<IndexType, IndexType> blockSizes,
		const std::pair<IndexType, IndexType> blockCapacities, const bool unweighted) {
	SCAI_REGION( "ParcoRepart.twoWayLocalFM" )

	if (blockSizes.first >= blockCapacities.first && blockSizes.second >= blockCapacities.second) {
		//cannot move any nodes, all blocks are overloaded already.
		return 0;
	}

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType globalN = inputDist->getGlobalSize();
	scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();

	/**
	 * These maps are mapping the b global indices received in the border region sets onto the interval [0,b-1].
	 */
	std::map<IndexType, IndexType> globalToVeryLocal;
	std::map<IndexType, IndexType> veryLocalToGlobal;
	IndexType i = 0;
	for (IndexType index : firstregion) {
		//edge information about elements must be either local or in halo
		assert(inputDist->isLocal(index) || matrixHalo.global2halo(index) != std::numeric_limits<IndexType>::max());
		//no overlap between input regions or dummy layers
		assert(secondregion.count(index) == 0);
		assert(firstDummyLayer.count(index) == 0);

		assert(globalToVeryLocal.count(index) == 0);
		globalToVeryLocal[index] = i;
		veryLocalToGlobal[i] = index;
		i++;
	}

	for (IndexType index : secondregion) {
		//edge information about elements must be either local or in halo
		assert(inputDist->isLocal(index) || matrixHalo.global2halo(index) != std::numeric_limits<IndexType>::max());
		assert(secondDummyLayer.count(index) == 0);

		assert(globalToVeryLocal.count(index) == 0);
		globalToVeryLocal[index] = i;
		veryLocalToGlobal[i] = index;
		i++;
	}

	assert(i == firstregion.size() + secondregion.size());
	assert(globalToVeryLocal.size() == i);
	assert(veryLocalToGlobal.size() == i);

	const IndexType veryLocalN = i;

	auto isVeryLocal = [&](IndexType globalID){return globalToVeryLocal.count(globalID) > 0;};
	auto isInFirstBlock = [&](IndexType globalID){return firstregion.count(globalID) + firstDummyLayer.count(globalID) > 0;};
	auto isInSecondBlock = [&](IndexType globalID){return secondregion.count(globalID) + secondDummyLayer.count(globalID) > 0;};

	/**
	 * check degree symmetry
	 */
	std::vector<IndexType> inDegree(veryLocalN, 0);
	std::vector<IndexType> outDegree(veryLocalN, 0);
	for (IndexType i = 0; i < veryLocalN; i++) {
		IndexType globalID = veryLocalToGlobal.at(i);
		const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2local(globalID) : matrixHalo.global2halo(globalID);
		assert(localID != nIndex);

		const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
		const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());
		const IndexType beginCols = localIa[localID];
		const IndexType endCols = localIa[localID+1];
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType globalNeighbor = localJa[j];
			std::string locationString = inputDist->isLocal(globalID) ? "locally" : "in halo";
			std::string countString = globalNeighbor != globalID && isVeryLocal(globalNeighbor) ? "counted" : "not counted";
			//std::cout << "Thread " << comm->getRank() << ": (" << globalID << "," << globalNeighbor << ") found " << locationString << ", " << countString << std::endl;

			if (globalNeighbor != globalID && isVeryLocal(globalNeighbor)) {
				IndexType veryLocalNeighbor = globalToVeryLocal.at(globalNeighbor);
				outDegree[i]++;
				inDegree[veryLocalNeighbor]++;
			}
		}
	}

	for (IndexType i = 0; i < veryLocalN; i++) {
		if (inDegree[i] != outDegree[i]) {
			throw std::runtime_error("Process " + std::to_string(comm->getRank()) + ": Node " + std::to_string(veryLocalToGlobal[i]) + " has " + std::to_string(inDegree[i]) + " incoming local edges but "
					+ std::to_string(outDegree[i]) + " outgoing local edges.");
		}
	}


	auto computeGain = [&](IndexType globalID){
		SCAI_REGION( "ParcoRepart.computeGain" )
		bool firstBlock = isInFirstBlock(globalID);
		assert(firstBlock != isInSecondBlock(globalID));
		assert(isVeryLocal(globalID));

		ValueType result = 0;
		/**
		 * neighborhood information is either in local matrix or halo.
		 */
		const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2local(globalID) : matrixHalo.global2halo(globalID);
		assert(localID != nIndex);

		const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
		const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());
		const scai::hmemo::ReadAccess<ValueType> localValues(storage.getValues());

		const IndexType beginCols = localIa[localID];
		const IndexType endCols = localIa[localID+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType globalNeighbor = localJa[j];
			if (globalNeighbor == globalID) {
				//self-loop, not counted
				continue;
			}
			const ValueType edgeweight = unweighted ? 1 : localValues[j];
			bool same;
			if (isInSecondBlock(globalNeighbor)) {
				same = !firstBlock;
			} else if (isInFirstBlock(globalNeighbor)) {
				same = firstBlock;
			} else {
				//neighbor in other block, not relevant for gain
				continue;
			}

			result += same ? -edgeweight : edgeweight;
		}

		return result;
	};

	/**
	 * construct and fill gain table and priority queues. Since only one target block is possible, gain table is one-dimensional.
	 */
	PrioQueue<ValueType, IndexType> firstQueue(veryLocalN);
	PrioQueue<ValueType, IndexType> secondQueue(veryLocalN);

	std::vector<ValueType> gain(veryLocalN);

	for (IndexType globalIndex : firstregion) {
		IndexType veryLocalID = globalToVeryLocal.at(globalIndex);
		gain[veryLocalID] = computeGain(globalIndex);
		assert(!firstQueue.contains(veryLocalID));
		firstQueue.insert(-gain[veryLocalID], veryLocalID);
	}

	for (IndexType globalIndex : secondregion) {
		IndexType veryLocalID = globalToVeryLocal.at(globalIndex);
		gain[veryLocalID] = computeGain(globalIndex);
		assert(!firstQueue.contains(veryLocalID));
		assert(!secondQueue.contains(veryLocalID));
		secondQueue.insert(-gain[veryLocalID], veryLocalID);
	}

	std::vector<bool> moved(veryLocalN, false);
	std::vector<std::pair<IndexType, IndexType>> transfers;
	transfers.reserve(veryLocalN);

	ValueType gainSum = 0;
	std::vector<ValueType> gainSumList, fillFactorList;
	gainSumList.reserve(veryLocalN);
	fillFactorList.reserve(veryLocalN);

	IndexType iter = 0;
	while (firstQueue.size() + secondQueue.size() > 0) {
		SCAI_REGION( "ParcoRepart.twoWayLocalFM.queueloop" )
		IndexType bestQueueIndex;

		/*
		 * first check break situations
		 */

		//only one block overfull,
		if ((firstQueue.size() == 0 && blockSizes.first >= blockCapacities.first)
				 ||	(secondQueue.size() == 0 && blockSizes.second >= blockCapacities.second)) {
			//cannot move any nodes
			break;
		}

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
			std::vector<ValueType> fullness = {double(blockSizes.first) / blockCapacities.first, double(blockSizes.second) / blockCapacities.second};
			std::vector<ValueType> gains = {firstQueue.inspectMin().first, secondQueue.inspectMin().first};

			assert(fullness[0] <= 1);
			assert(fullness[1] <= 1);
			assert(fullness[0] >= 0);
			assert(fullness[1] >= 0);

			//decide first by fullness, if fullness is equal, decide by gain. Since gain was inverted before inserting into queue, smaller gain is better.
			if (fullness[0] > fullness[1] || (fullness[0] == fullness[1] && gains[0] < gains[1])) {
				bestQueueIndex = 0;
			} else if (fullness[1] > fullness[0] || (fullness[0] == fullness[1] && gains[1] < gains[0])) {
				bestQueueIndex = 1;
			} else {
				//tie, break randomly
				assert(fullness[0] == fullness[1] && gains[0] == gains[1]);

				if ((rand() / RAND_MAX) < 0.5) {
					bestQueueIndex = 0;
				} else {
					bestQueueIndex = 1;
				}
			}
			//std::cout << "Fullness: (" << fullness[0] << "," << fullness[1] << ")" << ", gains (" << gains[0] << "," << gains[1] << "), picked queue " << bestQueueIndex << std::endl;
		}

		PrioQueue<ValueType, IndexType>& currentQueue = bestQueueIndex == 0 ? firstQueue : secondQueue;
		PrioQueue<ValueType, IndexType>& otherQueue = bestQueueIndex == 0 ? secondQueue : firstQueue;

		std::set<IndexType>& currentRegion = bestQueueIndex == 0 ? firstregion : secondregion;
		std::set<IndexType>& targetRegion = bestQueueIndex == 0 ? secondregion : firstregion;

		//now, we have selected a Queue.
		IndexType veryLocalID;
		ValueType topGain;
		std::tie(topGain, veryLocalID) = currentQueue.extractMin();
		assert(!currentQueue.contains(veryLocalID));
		assert(!otherQueue.contains(veryLocalID));
		topGain *= -1;

		IndexType topVertex = veryLocalToGlobal.at(veryLocalID);
		assert(isVeryLocal(topVertex));
		assert(!moved[veryLocalID]);
		if (topGain != computeGain(topVertex)) {
			throw std::runtime_error("iter " + std::to_string(transfers.size()) + ", queue key for " + std::to_string(topVertex) +
					": " + std::to_string(topGain) + ", computed gain: " + std::to_string(computeGain(topVertex))
					+ ", stored gain: " + std::to_string(gain[veryLocalID]));
		}
		assert(topGain == gain[veryLocalID]);

		//move node
		transfers.emplace_back(bestQueueIndex, topVertex);
		gainSum += topGain;
		gainSumList.push_back(gainSum);

		currentRegion.erase(topVertex);
		targetRegion.insert(topVertex);
		moved[veryLocalID] = true;

		blockSizes.first += bestQueueIndex == 0 ? -1 : 1;
		blockSizes.second += bestQueueIndex == 0 ? 1 : -1;
		//std::cout << "Moving " << topVertex << " from " << bestQueueIndex << ", bringing block sizes to " << blockSizes.first << ", " << blockSizes.second << std::endl;

		fillFactorList.push_back(std::max(double(blockSizes.first) / blockCapacities.first, double(blockSizes.second) / blockCapacities.second));

		//update gains of neighbors
		const CSRStorage<ValueType>& storage = inputDist->isLocal(topVertex) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(topVertex) ? inputDist->global2local(topVertex) : matrixHalo.global2halo(topVertex);
		assert(localID != nIndex);

		const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
		const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());
		const scai::hmemo::ReadAccess<ValueType> localValues(storage.getValues());
		const IndexType beginCols = localIa[localID];
		const IndexType endCols = localIa[localID+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			SCAI_REGION( "ParcoRepart.twoWayLocalFM.queueloop.gainupdate" )
			IndexType neighbor = localJa[j];
			ValueType edgeweight = unweighted ? 1 : localValues[j];
			if (isVeryLocal(neighbor)) {
				IndexType veryLocalNeighborID = globalToVeryLocal.at(neighbor);
				if (moved[veryLocalNeighborID]) {
					continue;
				}
				bool wasInSameBlock = bestQueueIndex == 0 ? isInFirstBlock(neighbor) : isInSecondBlock(neighbor);
				assert(isInFirstBlock(neighbor) != isInSecondBlock(neighbor));

				gain[veryLocalNeighborID] += wasInSameBlock ? 2*edgeweight : -2*edgeweight;

				if (wasInSameBlock) {
					assert(currentQueue.contains(veryLocalNeighborID));
					currentQueue.decreaseKey(-gain[veryLocalNeighborID], veryLocalNeighborID);
				} else {
					assert(otherQueue.contains(veryLocalNeighborID));
					otherQueue.decreaseKey(-gain[veryLocalNeighborID], veryLocalNeighborID);
				}
			}
		}
		iter++;
	}

	/**
	* now find best partition among those tested
	*/
	const IndexType testedNodes = gainSumList.size();
	if (testedNodes == 0) return 0;

	IndexType maxIndex = -1;
	ValueType maxGain = 0;

	for (IndexType i = 0; i < testedNodes; i++) {
		if (gainSumList[i] > maxGain && fillFactorList[i] <= 1) {
			maxIndex = i;
			maxGain = gainSumList[i];
		}
	}
	assert(testedNodes >= maxIndex);
	//assert(maxIndex >= 0);
	assert(testedNodes-1 < transfers.size());

	/**
	 * apply partition modifications in reverse until best is recovered
	 */
	for (int i = testedNodes-1; i > maxIndex; i--) {
		assert(transfers[i].second < globalN);
		std::set<IndexType>& sourceRegion = transfers[i].first == 0 ? firstregion : secondregion;
		std::set<IndexType>& targetRegion = transfers[i].first == 0 ? secondregion : firstregion;

		//apply movement in reverse
		sourceRegion.insert(transfers[i].second);
		targetRegion.erase(transfers[i].second);

		blockSizes.first += transfers[i].first == 0 ? 1 : -1;
		blockSizes.second += transfers[i].first == 0 ? -1 : 1;

	}

	for (IndexType node : firstregion) {
		assert(secondregion.count(node) == 0);
		assert(firstDummyLayer.count(node) == 0);
		assert(secondDummyLayer.count(node) == 0);
	}

	for (IndexType node : secondregion) {
		assert(firstDummyLayer.count(node) == 0);
		assert(secondDummyLayer.count(node) == 0);
	}

	assert(firstregion.size() + secondregion.size() == veryLocalN);
	return maxGain;
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

	//scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	scai::dmemo::Halo partHalo = buildPartHalo(adjM, part);
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
   
    //border.setValues(localBorder);
    assert(border.getDistributionPtr()->getLocalSize() == localN);
    return border;
}

//----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> ParcoRepart<IndexType, ValueType>::getPEGraph( const CSRSparseMatrix<ValueType> &adjM) {
    SCAI_REGION("ParcoRepart.getPEGraph");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr(); 
    
    
    SCAI_REGION_START("ParcoRepart.getPEGraph.findNonLocalNeighbours");
    // get all the non-local indices in an HArray
    scai::hmemo::HArray<IndexType> nonLocalIndices( dist->getLocalSize() ); 
    // dist->getLocalSize() is a "safe" (see below) upper bound of the neighbours if this PE
    // TODO: localSize is a way upper bound. replace with better estimate
    // ### in special cases, eg. 8x8 grid and k=8, a block , eg. block 1 in the example,
    // ### can have local size less than its not-local neighbours, eg, block 1 has 
    // ### size 8 but 10 non-local neighbours !!!
    scai::hmemo::WriteAccess<IndexType> writeNLI(nonLocalIndices, dist->getLocalSize() );
    // count the actual neighbours and resize the array
    IndexType actualNeighbours = 0;

    const CSRStorage<ValueType> localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    //scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

    for(IndexType i=0; i<dist->getLocalSize(); i++){        // for all local nodes
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){             // for all the edges of a node
            if( !dist->isLocal(ja[j]) ){                    // if ja[j] is not a local node
                // TODO: this check is needed because in small instances the "safe" upper
                //        bound is no safe and we must enlarge the array
                if( actualNeighbours >= nonLocalIndices.size() ){
                    writeNLI.resize(nonLocalIndices.size()+1);
                    writeNLI[actualNeighbours] = ja[j];         // store the non local index
                    ++actualNeighbours;
                }else {
                    writeNLI[actualNeighbours] = ja[j];         // store the non local index
                    ++actualNeighbours;
                }
            }
    	}
    }

    // if needed resize-shrink the array
    if(actualNeighbours != nonLocalIndices.size() ){
        writeNLI.resize(actualNeighbours);
    }

    //nonLocalIndices.resize(actualNeighbours);
    writeNLI.release();
    SCAI_REGION_END("ParcoRepart.getPEGraph.findNonLocalNeighbours");
    
    SCAI_REGION_START("ParcoRepart.getPEGraph.getOwners");
    // find the PEs that own every non-local index
        scai::hmemo::HArray<IndexType> owners(nonLocalIndices.size() , -1);
        dist->computeOwners( owners, nonLocalIndices);
    SCAI_REGION_END("ParcoRepart.getPEGraph.getOwners");
    
    // create the PE adjacency matrix to be returned
    IndexType numPEs = comm->getSize();
    scai::dmemo::DistributionPtr distPEs ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, numPEs) );  
    scai::dmemo::DistributionPtr noDistPEs (new scai::dmemo::NoDistribution( numPEs ));
    // every PE must have one row of the matrix since we have numPes and the matrix is [numPes x numPEs]

    SCAI_REGION_START("ParcoRepart.getPEGraph.buildMatrix");
    // TODO: this takes a significant amount of time! ### must reduce
        scai::lama::SparseAssemblyStorage<ValueType> myStorage( distPEs->getLocalSize(), numPEs);
        //scai::lama::MatrixStorage<ValueType> myStorage( distPEs->getLocalSize(), numPEs);
        scai::hmemo::ReadAccess<IndexType> readI(owners);
        for(IndexType i=0; i<readI.size(); i++){
            myStorage.setValue(0, readI[i], 1);
        }
        readI.release();
    SCAI_REGION_END("ParcoRepart.getPEGraph.buildMatrix");
    
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
    
    IndexType size= k*k;
    // get, on each processor, the edges of the blocks that are local
    std::vector< std::vector<IndexType> > blockEdges = ParcoRepart<int, double>::getLocalBlockGraphEdges( adjM, part);
    
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
    {
        SCAI_REGION("ParcoRepart.getBlockGraph.getNumEdges");
        scai::hmemo::ReadAccess<IndexType> recvPartRead( recvPart );
        for(IndexType i=0; i<recvPartRead.size(); i++){
            if( recvPartRead[i]>0 )
                ++numEdges;
        }
    }
    
    //convert the k*k HArray to a [k x k] CSRSparseMatrix
    scai::lama::CSRStorage<ValueType> localMatrix;
    localMatrix.allocate( k ,k );
    
    scai::hmemo::HArray<IndexType> csrIA;
    scai::hmemo::HArray<IndexType> csrJA;
    scai::hmemo::HArray<ValueType> csrValues; 
    {
        SCAI_REGION("ParcoRepart.getBlockGraph.HArray2CSR");
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
std::vector< std::vector<IndexType>> ParcoRepart<IndexType, ValueType>::getGraphEdgeColoring_local( const CSRSparseMatrix<ValueType> &adjM, IndexType &colors) {
    using namespace boost;
    IndexType N= adjM.getNumRows();
    assert( N== adjM.getNumColumns() ); // numRows = numColumns
    
    // use boost::Graph and boost::edge_coloring()
    typedef adjacency_list<vecS, vecS, undirectedS, no_property, size_t, no_property> Graph;
    typedef std::pair<std::size_t, std::size_t> Pair;
    //std::vector<std::vector<IndexType>> edges(2);
    Graph G(N);
    
    // retG[0][i] the first node, retG[1][i] the second node, retG[2][i] the color of the edge
    std::vector< std::vector<IndexType>> retG(3);
    
    // create graph G by the input adjacency matrix
    // TODO: get ia and ja values, do not do adjM(i,j) !!!
    for(IndexType i=0; i<N; i++){
        for(IndexType j=i; j<N; j++){
            if(adjM(i, j)== 1){ // there is an edge between nodes i and j. add the edge to G
                boost::add_edge(i, j, G).first;
                retG[0].push_back(i);  
                retG[1].push_back(j);
                //PRINT("adding edge ("<< i <<", "<< j<<")");
            }
        }
    }
    
    colors = boost::edge_coloring(G, boost::get( boost::edge_bundle, G));
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    PRINT( *comm << ", Colored using " << colors << " colors");
    for (size_t i = 0; i <retG[0].size(); i++) {
        //std::cout << "  " <<  retG[0][i] << "-" << retG[1][i] << ": " << \
        G[ boost::edge( retG[0][i],  retG[1][i], G).first] << std::endl;
        retG[2].push_back( G[ boost::edge( retG[0][i],  retG[1][i], G).first] );
    }
    
    return retG;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> ParcoRepart<IndexType, ValueType>::getGraphEdgeColoring_local( const std::vector<std::vector<IndexType>> &edgeList) {
    SCAI_REGION("ParcoRepart.getGraphEdgeColoring_local");
    using namespace boost;
    assert( edgeList.size() == 2);
    IndexType N= edgeList[0].size();
    
    // use boost::Graph and boost::edge_coloring()
    typedef adjacency_list<vecS, vecS, undirectedS, no_property, size_t, no_property> Graph;
    typedef std::pair<std::size_t, std::size_t> Pair;
    Graph G(N);
    
    // create the graph from the input edge list
    for(IndexType i=0; i<N; i++){
        boost::add_edge(edgeList[0][i], edgeList[1][i], G).first;
    }
    
    size_t colors = boost::edge_coloring(G, boost::get( boost::edge_bundle, G));
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    std::vector<IndexType> ret;
    PRINT( *comm << ", Colored using " << colors << " colors");
    for (size_t i = 0; i <edgeList[0].size(); i++) {
        //std::cout << "  " <<  edgeList[0][i] << "-" << edgeList[1][i] << ": " << \
        G[ boost::edge( edgeList[0][i],  edgeList[1][i], G).first] << std::endl;
        ret.push_back( G[ boost::edge( edgeList[0][i],  edgeList[1][i], G).first] );
    }
    
    return ret;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<DenseVector<IndexType>> ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( const CSRSparseMatrix<ValueType> &adjM) {
    IndexType N= adjM.getNumRows();
    SCAI_REGION("ParcoRepart.getCommunicationPairs_local");
    // coloring.size()=3: coloring(i,j,c) means that edge with endpoints i and j is colored with color c.
    // and coloring[i].size()= number of edges in input graph

    assert(adjM.getNumColumns() == adjM.getNumRows() );
    // TODO: coloring seems problematic when adjM has 2 nodes and only one edge
    // handle individually
    if(adjM.getNumRows()==2){           // graph has 2 nodes
        std::vector<DenseVector<IndexType>> retG(1);
        //TODO: setNumVlaues returns number of non-zero elements plus adjM.numRows() (?!?!)
        // use l1Norm but does not considers weighted edges
        //TODO: aparently CSRSparseMatrix.getNumValues() counts also 0 when setting via a setRawDenseData despite
        // the documentation claiming otherwise
        if(adjM.l1Norm()==2){     // and one edge
            retG[0].allocate(2);
            retG[0].setValue(0,1);
            retG[0].setValue(1,0);
        }
        return retG;
    }

    IndexType colors;
    std::vector<std::vector<IndexType>> coloring = getGraphEdgeColoring_local( adjM, colors );
    std::vector<DenseVector<IndexType>> retG(colors);
    
    // retG.size()= number of colors used in graph edge coloring
    // retG[i].size()= N , all nodes not communicating have -1
    
    for(IndexType i=0; i<colors; i++){        
        retG[i].allocate(N);
        // retG[i] = static_cast<IndexType> ( -1 );    // set value -1 for blocks NOT communicating
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


//to force instantiation
template DenseVector<int> ParcoRepart<int, double>::partitionGraph(CSRSparseMatrix<double> &input, std::vector<DenseVector<double>> &coordinates, int k,  double epsilon);
			     
template double ParcoRepart<int, double>::computeImbalance(const DenseVector<int> &partition, int k);

template double ParcoRepart<int, double>::computeCut(const CSRSparseMatrix<double> &input, const DenseVector<int> &part, bool ignoreWeights);

template double ParcoRepart<int, double>::replicatedMultiWayFM(const CSRSparseMatrix<double> &input, DenseVector<int> &part, int k, double epsilon, bool unweighted);

template std::vector<DenseVector<int>> ParcoRepart<int, double>::computeCommunicationPairings(const CSRSparseMatrix<double> &input, const DenseVector<int> &part,	const DenseVector<int> &blocksToPEs);

template std::vector<int> ITI::ParcoRepart<int, double>::nonLocalNeighbors(const CSRSparseMatrix<double>& input);

template scai::dmemo::Halo ITI::ParcoRepart<int, double>::buildPartHalo(const CSRSparseMatrix<double> &input,  const DenseVector<int> &part);

template std::pair<std::vector<int>, int> ITI::ParcoRepart<int, double>::getInterfaceNodes(const CSRSparseMatrix<double> &input, const DenseVector<int> &part, int thisBlock, int otherBlock, int depth);

template DenseVector<int> ParcoRepart<int, double>::getBorderNodes( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::lama::CSRSparseMatrix<double> ParcoRepart<int, double>::getPEGraph( const CSRSparseMatrix<double> &adjM);

template std::vector<std::vector<IndexType>> ParcoRepart<int, double>::getLocalBlockGraphEdges( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::lama::CSRSparseMatrix<double> ParcoRepart<int, double>::getBlockGraph( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part, const int k );

template std::vector< std::vector<int>>  ParcoRepart<int, double>::getGraphEdgeColoring_local( const CSRSparseMatrix<double> &adjM, int& colors);

template std::vector<IndexType> ParcoRepart<int, double>::getGraphEdgeColoring_local( const std::vector<std::vector<IndexType>> &edgeList );

template std::vector<DenseVector<int>> ParcoRepart<int, double>::getCommunicationPairs_local( const CSRSparseMatrix<double> &adjM);
}
