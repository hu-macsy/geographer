/*
 * KMeans.h
 *
 *  Created on: 19.07.2017
 *      Author: moritz
 */

#pragma once

#include <vector>

#include <scai/lama/DenseVector.hpp>

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
DenseVector<IndexType> assignBlocks(const std::vector<DenseVector<ValueType>> &coordinates, const std::vector<std::vector<ValueType> > &centers,
		const DenseVector<IndexType> &nodeWeights, const std::vector<IndexType> &blockSizes,  const ValueType epsilon, std::vector<ValueType> &influence);

template<typename ValueType>
ValueType biggestDelta(const std::vector<std::vector<ValueType>> &firstCoords, const std::vector<std::vector<ValueType>> &secondCoords);

/**
 * Implementations
 */

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(const std::vector<DenseVector<ValueType>> &coordinates, const std::vector<std::vector<ValueType> > &centers,
		const DenseVector<IndexType> &nodeWeights, const std::vector<IndexType> &blockSizes,  const ValueType epsilon = 0.05) {
	std::vector<ValueType> influence(blockSizes.size(), 1);
	return assignBlocks(coordinates, centers, nodeWeights, blockSizes, epsilon, influence);
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights,
		const std::vector<IndexType> &blockSizes, const ValueType epsilon) {

	std::vector<std::vector<ValueType> > centers = findInitialCenters(coordinates, k, nodeWeights);
	DenseVector<IndexType> result;
	std::vector<ValueType> influence(k,1);

	IndexType i = 0;
	ValueType delta = 0;
	ValueType threshold = 2;
	do {
		result = assignBlocks(coordinates, centers, nodeWeights, blockSizes, epsilon, influence);
		std::vector<std::vector<ValueType> > newCenters = findCenters(coordinates, result, k, nodeWeights);
		delta = biggestDelta(centers, newCenters);
		centers = newCenters;
		std::cout << "i: " << i << ", delta: " << delta << std::endl;
		i++;
	} while (i < 50 && delta > threshold);
	return result;
}

}
} /* namespace ITI */
