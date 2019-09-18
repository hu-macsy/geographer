/*
 * QuadNodePolarEuclid.h
 *
 *  Created on: 21.05.2014
 *      Author: Moritz v. Looz (moritz.looz-corswarem@kit.edu)
 *
 *  Note: This is similar enough to QuadNode.h that one could merge these two classes.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <assert.h>

#include "SpatialCell.h"

namespace ITI {

template <typename ValueType>
class QuadNodePolarEuclid : public ITI::SpatialCell<ValueType> {
    //friend class QuadTreeTest<ValueType>;
private:
    static const long unsigned sanityNodeLimit = 10E15; //just assuming, for debug purposes, that this algorithm never runs on machines with more than 4 Petabyte RAM
    bool splitTheoretical;
    ValueType balance;

public:
    /**
     * Construct a QuadNode for polar coordinates.
     *
     *
     */
    QuadNodePolarEuclid(Point<ValueType> minCoords = {0,0}, Point<ValueType> maxCoords = {2*M_PI, 1}, unsigned capacity = 1000, bool splitTheoretical = false, ValueType balance = 0.5)
        : SpatialCell<ValueType>(minCoords, maxCoords, capacity) {
        if (balance <= 0 || balance >= 1) throw std::runtime_error("Quadtree balance parameter must be between 0 and 1.");
        if (minCoords.getDimensions() != 2) throw std::runtime_error("Currently only supported for two dimensions");
        this->balance = balance;
        this->splitTheoretical = splitTheoretical;
    }

    virtual ~QuadNodePolarEuclid() = default;

    void split() {
        assert(this->isLeaf);
        assert(this->minCoords.getDimensions() == 2);
        const ValueType leftAngle = this->minCoords[0];
        const ValueType rightAngle = this->maxCoords[0];
        const ValueType minR = this->minCoords[1];
        const ValueType maxR = this->maxCoords[1];
        //heavy lifting: split up!
        ValueType middleAngle, middleR;
        if (splitTheoretical) {
            //Euclidean space is distributed equally
            middleAngle = (rightAngle - leftAngle) / 2 + leftAngle;
            middleR = pow(maxR*maxR*(1-balance)+minR*minR*balance, 0.5);
        } else {
            //median of points
            const index n = this->positions.size();
            std::vector<ValueType> angles(n);
            std::vector<ValueType> radii(n);
            for (index i = 0; i < n; i++) {
                angles[i] = this->positions[i][0];
                radii[i] = this->positions[i][1];
            }

            std::sort(angles.begin(), angles.end());
            middleAngle = angles[n/2];
            std::sort(radii.begin(), radii.end());
            middleR = radii[n/2];
        }
        assert(middleR < maxR);
        assert(middleR > minR);

        std::shared_ptr<QuadNodePolarEuclid > southwest(new QuadNodePolarEuclid({leftAngle, minR}, {middleAngle, middleR}, this->capacity, splitTheoretical, balance));
        std::shared_ptr<QuadNodePolarEuclid > southeast(new QuadNodePolarEuclid({middleAngle, minR}, {rightAngle, middleR}, this->capacity, splitTheoretical, balance));
        std::shared_ptr<QuadNodePolarEuclid > northwest(new QuadNodePolarEuclid({leftAngle, middleR}, {middleAngle, maxR}, this->capacity, splitTheoretical, balance));
        std::shared_ptr<QuadNodePolarEuclid > northeast(new QuadNodePolarEuclid({middleAngle, middleR}, {rightAngle, maxR}, this->capacity, splitTheoretical, balance));
        this->children = {southwest, southeast, northwest, northeast};
        this->isLeaf = false;
    }

    bool isConsistent() const override {
        if (this->children.size() != 0 && this->children.size() != 4) return false;
        //TODO: check for region coverage
        return true;
    }

    virtual std::pair<ValueType, ValueType> distances(const Point<ValueType> &query) const override {
        return this->EuclideanPolarDistances(query);
    }

    virtual ValueType distance(const Point<ValueType> &query, index k) const override {
        return this->euclidDistancePolar(query[0], query[1], this->positions[k][0], this->positions[k][1]);
    }
};
}
