/*
 * KMeans.h
 *
 *  Created on: 19.07.2017
 *      Author: moritz
 */

#pragma once

#include <vector>
#include <numeric>
#include <scai/lama/DenseVector.hpp>
#include <scai/tracing.hpp>

#include "quadtree/QuadNodeCartesianEuclid.h"
#include "GraphUtils.h"

using scai::lama::DenseVector;

namespace ITI {
namespace KMeans {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &nodeWeights,
		const std::vector<IndexType> &blockSizes, const ValueType epsilon = 0.05);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCenters(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &nodeWeights);

template<typename IndexType, typename ValueType, typename Iterator>
std::vector<std::vector<ValueType> > findCenters(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const IndexType k,
		const Iterator firstIndex, const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(const std::vector<DenseVector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers);

template<typename IndexType, typename ValueType, typename Iterator>
DenseVector<IndexType> assignBlocks(const std::vector<std::vector<ValueType>> &coordinates, const std::vector<std::vector<ValueType> > &centers,
		const Iterator firstIndex, const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights, const DenseVector<IndexType> &previousAssignment,
		const std::vector<IndexType> &blockSizes,  const SpatialCell &boundingBox, const ValueType epsilon,
		std::vector<ValueType> &upperBoundOwnCenter, std::vector<ValueType> &lowerBoundNextCenter,
		std::vector<ValueType> &influence);

template<typename ValueType>
ValueType biggestDelta(const std::vector<std::vector<ValueType>> &firstCoords, const std::vector<std::vector<ValueType>> &secondCoords);

/**
 * Implementations
 */
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &  nodeWeights, const std::vector<IndexType> &blockSizes, const ValueType epsilon) {
	SCAI_REGION( "KMeans.computePartition" );

	std::vector<std::vector<ValueType> > centers = findInitialCenters(coordinates, k, nodeWeights);
	std::vector<ValueType> influence(k,1);
	const IndexType dim = coordinates.size();
	assert(dim > 0);
	const IndexType localN = nodeWeights.getLocalValues().size();
	assert(nodeWeights.getLocalValues().size() == coordinates[0].getLocalValues().size());
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	const IndexType p = comm->getSize();
	const ValueType blocksPerProcess = ValueType(k)/p;

	std::vector<ValueType> minCoords(dim);
	std::vector<ValueType> maxCoords(dim);
	std::vector<std::vector<ValueType> > convertedCoords(dim);
	for (IndexType d = 0; d < dim; d++) {
		scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
		assert(rAccess.size() == localN);
		convertedCoords[d] = std::vector<ValueType>(rAccess.get(), rAccess.get()+localN);
		assert(convertedCoords[d].size() == localN);
		minCoords[d] = *std::min_element(convertedCoords[d].begin(), convertedCoords[d].end());
		maxCoords[d] = *std::max_element(convertedCoords[d].begin(), convertedCoords[d].end());
	}

	QuadNodeCartesianEuclid boundingBox(minCoords, maxCoords);
	std::cout << "Process " << comm->getRank() << ": ( ";
	for (auto coord : minCoords) std::cout << coord << " ";
	std::cout << ") , ( ";
	for (auto coord : maxCoords) std::cout << coord << " ";
	std::cout << ")" << std::endl;

	DenseVector<IndexType> result(coordinates[0].getDistributionPtr(), 0);
	std::vector<ValueType> upperBoundOwnCenter(localN, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> lowerBoundNextCenter(localN, 0);

	//prepare sampling
	std::vector<IndexType> localIndices(localN);
	const typename std::vector<IndexType>::iterator firstIndex = localIndices.begin();
	typename std::vector<IndexType>::iterator lastIndex = localIndices.end();;
	std::iota(firstIndex, lastIndex, 0);

	IndexType minNodes = 100*blocksPerProcess;
	IndexType samplingRounds = 0;
	std::vector<IndexType> samples;
	std::vector<IndexType> adjustedBlockSizes(blockSizes);
	if (localN > minNodes) {
		ITI::GraphUtils::FisherYatesShuffle(firstIndex, lastIndex, localN);

		samplingRounds = std::ceil(std::log2(ValueType(localN) / minNodes))+1;
		samples.resize(samplingRounds);
		samples[0] = minNodes;
	}

	for (IndexType i = 1; i < samplingRounds; i++) {
		samples[i] = std::min(IndexType(samples[i-1]*2), localN);
	}
	assert(samples[samplingRounds-1] == localN);

	scai::hmemo::ReadAccess<ValueType> rWeight(nodeWeights.getLocalValues());

	IndexType iter = 0;
	ValueType delta = 0;
	bool balanced = false;
	const ValueType threshold = 2;
	do {

		if (iter < samplingRounds) {
			lastIndex = firstIndex + samples[iter];
			std::sort(firstIndex, lastIndex);//sorting not really necessary, but increases locality
			ValueType ratio = ValueType(samples[iter]) / localN;
			for (IndexType j = 0; j < k; j++) {
				adjustedBlockSizes[j] = ValueType(blockSizes[j]) * ratio;
			}
		} else {
			assert(lastIndex == localIndices.end());
		}

		result = assignBlocks(convertedCoords, centers, firstIndex, lastIndex, nodeWeights, result, adjustedBlockSizes, boundingBox, epsilon, upperBoundOwnCenter, lowerBoundNextCenter, influence);
		scai::hmemo::ReadAccess<IndexType> rResult(result.getLocalValues());

		std::vector<std::vector<ValueType> > newCenters = findCenters(coordinates, result, k, firstIndex, lastIndex, nodeWeights);
		std::vector<ValueType> squaredDeltas(k,0);
		std::vector<ValueType> deltas(k,0);
		for (IndexType j = 0; j < k; j++) {
			for (int d = 0; d < dim; d++) {
				ValueType diff = (centers[d][j] - newCenters[d][j]);
				squaredDeltas[j] += diff*diff;
			}
			deltas[j] = std::sqrt(squaredDeltas[j]);
		}

		delta = *std::max_element(deltas.begin(), deltas.end());
		const double deltaSq = delta*delta;
		double maxInfluence = *std::max_element(influence.begin(), influence.end());
		double minInfluence = *std::min_element(influence.begin(), influence.end());

		{
			SCAI_REGION( "KMeans.computePartition.updateBounds" );
			for (auto it = firstIndex; it != lastIndex; it++) {
				const IndexType i = *it;
				IndexType cluster = rResult[i];
				upperBoundOwnCenter[i] += (2*deltas[cluster]*std::sqrt(upperBoundOwnCenter[i]/influence[cluster]) + squaredDeltas[cluster])*(influence[cluster] + 1e-10);
				ValueType pureSqrt(std::sqrt(lowerBoundNextCenter[i]/maxInfluence));
				if (pureSqrt < delta) {
					lowerBoundNextCenter[i] = 0;
				} else {
					ValueType diff = (-2*delta*pureSqrt + deltaSq)*(maxInfluence + 1e-10);
					assert(diff < 0);
					lowerBoundNextCenter[i] += diff;
					if (!(lowerBoundNextCenter[i] > 0)) lowerBoundNextCenter[i] = 0;
				}
				assert(std::isfinite(lowerBoundNextCenter[i]));
			}
		}
		centers = newCenters;

		std::vector<IndexType> blockWeights(k,0);
		for (auto it = firstIndex; it != lastIndex; it++) {
			const IndexType i = *it;
			IndexType cluster = rResult[i];
			blockWeights[cluster] += rWeight[i];
		}
		{
			SCAI_REGION( "KMeans.computePartition.blockWeightSum" );
			comm->sumImpl(blockWeights.data(), blockWeights.data(), k, scai::common::TypeTraits<IndexType>::stype);
		}

		balanced = true;
		for (IndexType j = 0; j < k; j++) {
			if (blockWeights[j] > blockSizes[j]*(1+epsilon)) {
				balanced = false;
			}
		}

		if (comm->getRank() == 0) {
			std::cout << "i: " << iter << ", delta: " << delta << std::endl;
		}
		iter++;
	} while (iter < samplingRounds || (iter < 50 && (delta > threshold || !balanced)));
	return result;
}
}
} /* namespace ITI */
