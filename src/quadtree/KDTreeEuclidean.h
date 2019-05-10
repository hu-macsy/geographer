/*
 * KDTreeEuclidean.h
 *
 *  Created on: 12.11.2016
 *      Author: moritzl
 */

#pragma once

#include <vector>

#include "SpatialTree.h"
#include "KDNodeEuclidean.h"

namespace ITI {

template <bool cartesian=true>
class KDTreeEuclidean: public ITI::SpatialTree {
public:
	KDTreeEuclidean() = default;
	virtual ~KDTreeEuclidean() = default;
	KDTreeEuclidean(const Point<ValueType> &minCoords, const Point<ValueType> &maxCoords, count capacity=1000) {
		this->root = std::shared_ptr<KDNodeEuclidean<cartesian> >(new KDNodeEuclidean<cartesian>(minCoords, maxCoords, capacity));
	}
};

} /* namespace ITI */
