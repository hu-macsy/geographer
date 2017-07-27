/*
 * KMeans.h
 *
 *  Created on: 19.07.2017
 *      Author: moritz
 */

#pragma once

#include <vector>

#include <scai/lama/DenseVector.hpp>
#include <scai/tracing.hpp>

using scai::lama::DenseVector;

namespace ITI {
namespace KMeans {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights,
		const std::vector<IndexType> &blockSizes, const ValueType epsilon = 0.05);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCenters(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findCenters(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const IndexType k,
		const DenseVector<IndexType> &nodeWeights);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(const std::vector<DenseVector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(const std::vector<std::vector<ValueType>> &coordinates, const std::vector<std::vector<ValueType> > &centers,
		const DenseVector<IndexType> &nodeWeights, const DenseVector<IndexType> &previousAssignment,
		const std::vector<IndexType> &blockSizes,  const ValueType epsilon,
		std::vector<ValueType> &upperBoundOwnCenter, std::vector<ValueType> &lowerBoundNextCenter, std::vector<ValueType> &influence);

template<typename ValueType>
ValueType biggestDelta(const std::vector<std::vector<ValueType>> &firstCoords, const std::vector<std::vector<ValueType>> &secondCoords);

/**
 * Implementations
 */


template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights,
		const std::vector<IndexType> &blockSizes, const ValueType epsilon) {
	SCAI_REGION( "KMeans.computePartition" );

	std::vector<std::vector<ValueType> > centers = findInitialCenters(coordinates, k, nodeWeights);
	DenseVector<IndexType> result;
	std::vector<ValueType> influence(k,1);
	const IndexType dim = coordinates.size();
	const IndexType localN = nodeWeights.getLocalValues().size();

	std::vector<std::vector<ValueType> > convertedCoords(dim);
	for (IndexType d = 0; d < dim; d++) {
		scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
		assert(rAccess.size() == localN);
		convertedCoords[d] = std::vector<ValueType>(rAccess.get(), rAccess.get()+localN);
		assert(convertedCoords[d].size() == localN);
		for (IndexType i = 0; i < localN; i++) {
			assert(convertedCoords[d][i] == rAccess[i]);
		}
		ValueType convertedSum = std::accumulate(convertedCoords[d].begin(), convertedCoords[d].end(), 0);
		ValueType nativeSum = coordinates[d].getLocalValues().sum();
		std::cout << convertedSum << " | " << nativeSum << std::endl;
	}

	result = assignBlocks<IndexType, ValueType>(coordinates, centers);
	std::vector<ValueType> upperBoundOwnCenter(localN, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> lowerBoundNextCenter(localN, 0);
	IndexType i = 0;
	ValueType delta = 0;
	ValueType threshold = 2;
	do {
		result = assignBlocks(convertedCoords, centers, nodeWeights, result, blockSizes, epsilon, upperBoundOwnCenter, lowerBoundNextCenter, influence);
		scai::hmemo::ReadAccess<IndexType> rResult(result.getLocalValues());

		std::vector<std::vector<ValueType> > newCenters = findCenters(coordinates, result, k, nodeWeights);
		std::vector<ValueType> squaredDeltas(k,0);
		std::vector<ValueType> deltas(k,0);
		for (IndexType j = 0; j < k; j++) {
			for (int d = 0; d < dim; d++) {
				ValueType diff = (centers[d][j] - newCenters[d][j]);
				squaredDeltas[j] +=  diff*diff;
			}
			deltas[j] = std::sqrt(squaredDeltas[j]);
		}

		delta = *std::max_element(deltas.begin(), deltas.end());
		double maxInfluence = *std::max_element(influence.begin(), influence.end());
		double minInfluence = *std::min_element(influence.begin(), influence.end());

		for (IndexType i = 0; i < localN; i++) {
			IndexType cluster = rResult[i];
			upperBoundOwnCenter[i] += (2*deltas[cluster]*std::sqrt(upperBoundOwnCenter[i]/influence[cluster]) + squaredDeltas[cluster])*(influence[cluster] + 1e-10);
			lowerBoundNextCenter[i] -= (2*delta*std::sqrt(lowerBoundNextCenter[i]/maxInfluence) + delta*delta)*(maxInfluence + 1e-10);
		}
		centers = newCenters;

		std::cout << "i: " << i << ", delta: " << delta << std::endl;
		i++;
	} while (i < 50 && delta > threshold);
	return result;
}
/**
 * DenseVector<IndexType> assignBlocks(
		const std::vector<std::vector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers,
		const DenseVector<IndexType> &nodeWeights,
		const DenseVector<IndexType> &previousAssignment,
		const std::vector<IndexType> &targetBlockSizes,
		const ValueType epsilon,
		std::vector<ValueType> &upperBoundOwnCenter,
		std::vector<ValueType> &lowerBoundNextCenter,
		std::vector<ValueType> &influence) {
 */

}
} /* namespace ITI */
