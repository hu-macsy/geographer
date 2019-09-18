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

/** A cartesian, euclidean tree.

*/

template <typename ValueType>
class QuadTreeCartesianEuclid : public ITI::SpatialTree<ValueType> {
    //friend class QuadTreeCartesianEuclidTest;
public:
    /**
     * @param lower Minimal coordinates of region
     * @param upper Maximal coordinates of region (excluded)
     * @param capacity Number of points a leaf cell can store before splitting
     * @param splitTheoretical Whether to split in a theoretically optimal way or in a way to decrease measured running times
     *
     */
    QuadTreeCartesianEuclid(Point<ValueType> lower = Point<ValueType>({0.0, 0.0}), Point<ValueType> upper = Point<ValueType>({1.0, 1.0}), bool theoreticalSplit=true, count capacity=1000) {
        this->root = std::shared_ptr<QuadNodeCartesianEuclid<ValueType>>(new QuadNodeCartesianEuclid<ValueType>(lower, upper, capacity, theoreticalSplit));
    }

//	void extractCoordinates(std::vector<Point<ValueType> > &posContainer) const {
//		this->root->getCoordinates(posContainer);
//	}

    void getElementsInEuclideanCircle(const Point<ValueType> circleCenter, const ValueType radius, std::vector<index> &circleDenizens) const {
        this->getElementsInCircle(circleCenter, radius, circleDenizens);
    }

    void recount() {
        this->root->recount();
    }
};
}
