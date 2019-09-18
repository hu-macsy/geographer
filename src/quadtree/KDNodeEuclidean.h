/*
 * KDNodeCartesianEuclid.h
 *
 *  Created on: 12.11.2016
 *      Author: moritzl
 */

#pragma once

#include "SpatialCell.h"

namespace ITI {

template <typename ValueType, bool cartesian=true>
class KDNodeEuclidean: public ITI::SpatialCell<ValueType> {
public:
    KDNodeEuclidean() = default;
    virtual ~KDNodeEuclidean() = default;

    KDNodeEuclidean(const Point<ValueType> &minCoords, const Point<ValueType> &maxCoords, count capacity=1000)
        : SpatialCell<ValueType>(minCoords, maxCoords, capacity) {
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
        ValueType maximumSpread = 0;

        //find dimension in which to split
        for (index d = 0; d < dimension; d++) {
            ValueType maxElement = this->minCoords[d] - 1;
            ValueType minElement = this->maxCoords[d] + 1;
            for (index i = 0; i < numPoints; i++) {
                ValueType pos = this->positions[i][d];
                if (pos < minElement) minElement = pos;
                if (pos > maxElement) maxElement = pos;
            }
            ValueType spread = maxElement - minElement;
            if (spread > maximumSpread) {
                maximumSpread = spread;
                mostSpreadDimension = d;
            }
        }

        //find median
        std::vector<ValueType> sorted(numPoints);
        for (index i = 0; i < numPoints; i++) {
            sorted[i] = this->positions[i][mostSpreadDimension];
        }

        std::sort(sorted.begin(), sorted.end());
        ValueType middle = sorted[numPoints/2];
        assert(middle <= this->maxCoords[mostSpreadDimension]);
        assert(middle >= this->minCoords[mostSpreadDimension]);

        Point<ValueType> newLower(this->minCoords);
        Point<ValueType> newUpper(this->maxCoords);
        newLower[mostSpreadDimension] = middle;
        newUpper[mostSpreadDimension] = middle;

        std::shared_ptr<KDNodeEuclidean<ValueType,cartesian> > firstChild(new KDNodeEuclidean<ValueType,cartesian>(this->minCoords, newUpper, this->capacity));
        std::shared_ptr<KDNodeEuclidean<ValueType,cartesian> > secondChild(new KDNodeEuclidean<ValueType,cartesian>(newLower, this->maxCoords, this->capacity));

        this->children = {firstChild, secondChild};
        this->isLeaf = false;
    }

    bool isConsistent() const override {
        if (this->isLeaf) {
            return (this->children.size() == 0);
        } else {
            return (this->children.size() == 2);
        }
        //TODO: check for region coverage
    }

    std::pair<ValueType, ValueType> distances(const Point<ValueType> &query) const override {
        if (cartesian) {
            return this->EuclideanCartesianDistances(query);
        } else {
            return this->EuclideanPolarDistances(query);
        }
    }

    ValueType distance(const Point<ValueType> &query, index k) const override {
        if (cartesian) {
            return query.distance(this->positions[k]);
        }	else {
            return this->euclidDistancePolar(query[0], query[1], this->positions[k][0], this->positions[k][1]);
        }
    }
};

} /* namespace ITI */
