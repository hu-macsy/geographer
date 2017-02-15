/*
 * KDNodeCartesianEuclid.h
 *
 *  Created on: 12.11.2016
 *      Author: moritzl
 */

#pragma once

#include "SpatialCell.h"

namespace ITI {

template <class T, bool cartesian=true>
class KDNodeEuclidean: public ITI::SpatialCell<T> {
public:
	KDNodeEuclidean() = default;
	virtual ~KDNodeEuclidean() = default;

	KDNodeEuclidean(const Point<double> &minCoords, const Point<double> &maxCoords, count capacity=1000)
	: SpatialCell<T>(minCoords, maxCoords, capacity) {
		if (!cartesian) {
			if (minCoords.getDimensions() != 2) {
				throw std::runtime_error("Polar Coordinates only supported for 2 dimensions");
			}

			assert(this->minCoords[0] >= 0);
			assert(this->maxCoords[0] <= 2*M_PI);

			assert(this->minCoords[1] >= 0);
		}
	}

	void split() override {
		assert(this->isLeaf);
		const index numPoints = this->positions.size();
		if (numPoints == 0) {
			throw std::runtime_error("Cannot split empty cell.");
		}

		const count dimension = this->minCoords.getDimensions();
		index mostSpreadDimension = dimension;
		double maximumSpread = 0;

		//find dimension in which to split
		for (index d = 0; d < dimension; d++) {
			double maxElement = this->minCoords[d] - 1;
			double minElement = this->maxCoords[d] + 1;
			for (index i = 0; i < numPoints; i++) {
				double pos = this->positions[i][d];
				if (pos < minElement) minElement = pos;
				if (pos > maxElement) maxElement = pos;
			}
			double spread = maxElement - minElement;
			if (spread > maximumSpread) {
				maximumSpread = spread;
				mostSpreadDimension = d;
			}
		}

		//find median
		vector<double> sorted(numPoints);
		for (index i = 0; i < numPoints; i++) {
			sorted[i] = this->positions[i][mostSpreadDimension];
		}

		std::sort(sorted.begin(), sorted.end());
		double middle = sorted[numPoints/2];
		assert(middle <= this->maxCoords[mostSpreadDimension]);
		assert(middle >= this->minCoords[mostSpreadDimension]);

		Point<double> newLower(this->minCoords);
		Point<double> newUpper(this->maxCoords);
		newLower[mostSpreadDimension] = middle;
		newUpper[mostSpreadDimension] = middle;

		std::shared_ptr<KDNodeEuclidean<T, cartesian> > firstChild(new KDNodeEuclidean<T, cartesian>(this->minCoords, newUpper, this->capacity));
		std::shared_ptr<KDNodeEuclidean<T, cartesian> > secondChild(new KDNodeEuclidean<T, cartesian>(newLower, this->maxCoords, this->capacity));

		this->children = {firstChild, secondChild};
		this->isLeaf = false;
	}

	std::pair<double, double> distances(const Point<double> &query) const override {
		if (cartesian) {
			return this->EuclideanCartesianDistances(query);
		} else {
			return this->EuclideanPolarDistances(query);
		}
	}

	double distance(const Point<double> &query, index k) const override {
		if (cartesian) {
			return query.distance(this->positions[k]);
		}	else {
			return this->euclidDistancePolar(query[0], query[1], this->positions[k][0], this->positions[k][1]);
		}
	}
};

} /* namespace ITI */
