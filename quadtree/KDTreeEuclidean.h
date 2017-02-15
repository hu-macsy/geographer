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

template <class T, bool cartesian=true>
class KDTreeEuclidean: public ITI::SpatialTree<T> {
public:
	KDTreeEuclidean() = default;
	virtual ~KDTreeEuclidean() = default;
	KDTreeEuclidean(const Point<double> &minCoords, const Point<double> &maxCoords, count capacity=1000) {
		this->root = std::shared_ptr<KDNodeEuclidean<T, cartesian> >(new KDNodeEuclidean<T, cartesian>(minCoords, maxCoords, capacity));
	}
};

} /* namespace ITI */
