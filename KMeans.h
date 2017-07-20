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
std::vector<std::vector<ValueType> > findInitialCenters(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findCenters(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const IndexType k,
		const DenseVector<IndexType> &nodeWeights);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(const std::vector<DenseVector<ValueType>> &coordinates, const std::vector<std::vector<ValueType> > &centers,
		const DenseVector<IndexType> &nodeWeights, const std::vector<IndexType> &blockSizes,  const ValueType epsilon = 0.05);

}
} /* namespace ITI */
