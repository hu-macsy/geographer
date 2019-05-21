/*
 * Quadtree.h
 *
 *  Created on: 21.05.2014
 *      Author: Moritz v. Looz (moritz.looz-corswarem@kit.edu)
 */

#ifndef QUADTREEPOLAREUCLID_H_
#define QUADTREEPOLAREUCLID_H_

#include <vector>
#include <memory>
#include <cmath>
#include <omp.h>
#include <functional>
#include "SpatialTree.h"
#include "QuadNodePolarEuclid.h"

namespace ITI {

class QuadTreePolarEuclid : public SpatialTree{

public:
	~QuadTreePolarEuclid() = default;

	/**
	 * @param maxR Radius of the managed area. Must be smaller than 1.
	 * @param theoreticalSplit If true, split cells to get the same area in each child cell. Default is false
	 * @param alpha dispersion Parameter of the point distribution. Only has an effect if theoretical split is true
	 * @param capacity How many points can inhabit a leaf cell before it is split up?
	 *
	 */
	QuadTreePolarEuclid(Point<ValueType> minCoords = {0,0}, Point<ValueType> maxCoords = {2*M_PI, 1}, bool theoreticalSplit=false, count capacity=1000, ValueType balance = 0.5)
	{
		this->root = std::shared_ptr<QuadNodePolarEuclid>(new QuadNodePolarEuclid(minCoords, maxCoords, capacity, theoreticalSplit, balance));
	}

	QuadTreePolarEuclid(const std::vector<ValueType> &angles, const std::vector<ValueType> &radii, const std::vector<index > &content, bool theoreticalSplit=false, count capacity=1000, ValueType balance = 0.5) {
		const count n = angles.size();
		assert(angles.size() == radii.size());
		assert(radii.size() == content.size());
		ValueType minRadius, maxRadius, minAngle, maxAngle;

		auto angleMinMax = std::minmax_element(angles.begin(), angles.end());
		auto radiiMinMax = std::minmax_element(radii.begin(), radii.end());
		minAngle = *angleMinMax.first;
		minRadius = *radiiMinMax.first;
		maxAngle = std::nextafter(*angleMinMax.second, std::numeric_limits<ValueType>::max());
		maxRadius = std::nextafter(*radiiMinMax.second, std::numeric_limits<ValueType>::max());
		this->root = std::shared_ptr<QuadNodePolarEuclid>(new QuadNodePolarEuclid({minAngle, minRadius}, {maxAngle, maxRadius}, capacity, theoreticalSplit, balance));

		for (index i = 0; i < n; i++) {
			assert(content[i] < n);
			this->root->addContent(content[i], {angles[i], radii[i]});
		}
	}

	/**
	 * @param newcomer content to be removed at point x
	 * @param angle angular coordinate of x
	 * @param R radial coordinate of x
	 */
	bool removeContent(int toRemove, ValueType angle, ValueType r) {
		return this->root->removeContent(toRemove, {angle, r});
	}

	void getElementsInEuclideanCircle(const Point<ValueType> circleCenter, const ValueType radius, std::vector<index> &circleDenizens) const {
		this->root->getElementsInCircle(circleCenter, radius, circleDenizens);
	}

	void recount() {
		this->root->recount();
	}

	index getCellID(ValueType phi, ValueType r) const {
		return this->root->getCellID({phi, r});
	}
};
}

#endif /* QUADTREE_H_ */
