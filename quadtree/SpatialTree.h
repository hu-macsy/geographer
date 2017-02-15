/*
 * SpatialTree.h
 *
 *  Created on: 10.11.2016
 *      Author: moritzl
 */

#pragma once

#include "SpatialCell.h"

namespace ITI {

template <class T>
class SpatialTree {
public:
	SpatialTree() = default;
	virtual ~SpatialTree() = default;

	void addContent(T content, const Point<double> &coords) {
		root->addContent(content, coords);
	}

	bool removeContent(T content, const Point<double> &coords) {
		return root->removeContent(content, coords);
	}

	void getElementsInCircle(const Point<double> query, const double radius, std::vector<T> &circleDenizens) const {
		root->getElementsInCircle(query, radius, circleDenizens);
	}

	count getElementsProbabilistically(Point<double> query, std::function<double(double)> prob, std::vector<T> &circleDenizens) {
		return root->getElementsProbabilistically(query, prob, circleDenizens);
	}

	count size() const {
		return root->size();
	}

	count height() const {
		return root->height();
	}

	void trim() {
		root->trim();
	}

	void reindex() {
		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				root->reindex(0);
			}
		}
	}

	/**
	 * Get all elements, regardless of position
	 *
	 * @return vector<T> of elements
	 */
	std::vector<T> getElements() const {
		return root->getElements();
	}

protected:
	std::shared_ptr<SpatialCell<T>> root;
};

} /* namespace ITI */
