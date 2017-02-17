/*
 * Quadtree.h
 *
 *  Created on: 21.05.2014
 *      Author: Moritz v. Looz (moritz.looz-corswarem@kit.edu)
 */

#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <omp.h>
#include <functional>

#include "QuadNodeCartesianEuclid.h"
#include "SpatialTree.h"

namespace ITI {

template <class T>
class QuadTreeCartesianEuclid : public ITI::SpatialTree<T> {
	friend class QuadTreeCartesianEuclidTest;
public:
	/**
	 * @param lower Minimal coordinates of region
	 * @param upper Maximal coordinates of region (excluded)
	 * @param capacity Number of points a leaf cell can store before splitting
	 * @param splitTheoretical Whether to split in a theoretically optimal way or in a way to decrease measured running times
	 *
	 */
	QuadTreeCartesianEuclid(Point<double> lower = Point<double>({0.0, 0.0}), Point<double> upper = Point<double>({1.0, 1.0}), bool theoreticalSplit=true, count capacity=1000) {
		this->root = std::shared_ptr<QuadNodeCartesianEuclid<T> >(new QuadNodeCartesianEuclid<T>(lower, upper, capacity, theoreticalSplit));
	}
	
	void extractCoordinates(vector<Point<double> > &posContainer) const {
		this->root->getCoordinates(posContainer);
	}

	void getElementsInEuclideanCircle(const Point<double> circleCenter, const double radius, vector<T> &circleDenizens) const {
		this->getElementsInCircle(circleCenter, radius, circleDenizens);
	}

	void recount() {
		this->root->recount();
	}

	count countLeaves() const {
		return this->root->countLeaves();
	}

	index indexSubtree(index nextID) {
		return this->root->indexSubtree(nextID);
	}

	index getCellID(Point<double> pos) const {
		return this->root->getCellID(pos);
	}
};
}
