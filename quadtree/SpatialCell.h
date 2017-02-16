/*
 * SpatialCell.h
 *
 *  Created on: 10.11.2016
 *      Author: moritzl
 */

#pragma once

#include <vector>
#include <functional>
#include <memory>

#include "Point.h"

namespace ITI {

class SpatialCell {
	friend class QuadTreeTest;
public:
	SpatialCell() = default;
	virtual ~SpatialCell() = default;
	SpatialCell(const Point<double> &minCoords, const Point<double> &maxCoords, count capacity=1000) : minCoords(minCoords), maxCoords(maxCoords), capacity(capacity) {
		const count dimension = minCoords.getDimensions();
		if (maxCoords.getDimensions() != dimension) {
			throw std::runtime_error("minCoords has dimension " + std::to_string(dimension) + ", maxCoords " + std::to_string(maxCoords.getDimensions()));
		}
		for (index i = 0; i < dimension; i++) {
			if (!(minCoords[i] < maxCoords[i])) {
				throw std::runtime_error("minCoords["+std::to_string(i)+("]="+std::to_string(minCoords[i])+" not < "+std::to_string(maxCoords[i])+"=maxCoords["+std::to_string(i)+"]"));
			}
		}
		isLeaf = true;
		subTreeSize = 0;
		ID = 0;
		queries = 0;
	}

	virtual void split() = 0;
	virtual std::pair<double, double> distances(const Point<double> &query) const = 0;
	virtual double distance(const Point<double> &query, index k) const = 0;

	virtual void coarsen() {
		assert(this->height() == 2);
		assert(content.size() == 0);
		assert(positions.size() == 0);

		std::vector<int> allContent;
		std::vector<Point<double> > allPositions;
		for (index i = 0; i < this->children.size(); i++) {
			allContent.insert(allContent.end(), children[i]->content.begin(), children[i]->content.end());
			allPositions.insert(allPositions.end(), children[i]->positions.begin(), children[i]->positions.end());
		}
		assert(this->subTreeSize == allContent.size());
		assert(this->subTreeSize == allPositions.size());

		this->children.clear();
		this->content.swap(allContent);
		this->positions.swap(allPositions);
		this->isLeaf = true;
		this->subTreeSize = 0;
	}

	/**
	 * Remove content. May cause coarsening of the quadtree
	 *
	 * @return True if content was found and removed, false otherwise
	 */
	bool removeContent(int input, const Point<double> &pos) {
		if (!this->responsible(pos)) return false;
		if (this->isLeaf) {
			index i = 0;
			for (; i < this->content.size(); i++) {
				if (this->content[i] == input) break;
			}
			if (i < this->content.size()) {
				//remove element
				this->content.erase(this->content.begin()+i);
				this->positions.erase(this->positions.begin()+i);
				return true;
			} else {
				return false;
			}
		}
		else {
			bool removed = false;
			bool allLeaves = true;
			assert(this->children.size() > 0);
			for (index i = 0; i < children.size(); i++) {
				if (!children[i]->isLeaf) allLeaves = false;
				if (children[i]->removeContent(input, pos)) {
					assert(!removed);
					removed = true;
				}
			}
			if (removed) this->subTreeSize--;
			//coarsen?
			if (removed && allLeaves && this->size() < this->coarsenLimit) {
				this->coarsen();
			}

			return removed;
		}
	}

	void recount() {
		this->subTreeSize = 0;
		for (index i = 0; i < children.size(); i++) {
			this->children[i]->recount();
			this->subTreeSize += this->children[i]->size();
		}
	}

	virtual bool outOfReach(Point<double> query, double radius) const {
		return distances(query).first > radius;
	}

	virtual void getElementsInCircle(Point<double> center, double radius, std::vector<int> &result) const {
		if (outOfReach(center, radius)) {
			return;
		}

		if (this->isLeaf) {
			const count cSize = content.size();

			for (index i = 0; i < cSize; i++) {
				if (distance(center, i) < radius) {
					result.push_back((content[i]));
				}
			}
		} else {
			for (index i = 0; i < children.size(); i++) {
				children[i]->getElementsInCircle(center, radius, result);
			}
		}
	}

	virtual void addContent(int input, const Point<double> &coords) {
		assert(content.size() == positions.size());
		assert(this->responsible(coords));
		if (isLeaf) {
			content.push_back(input);
			positions.push_back(coords);

			//if now overfull, split up
			if (content.size() > capacity) {
				split();

				for (index i = 0; i < content.size(); i++) {
					this->addContent(content[i], positions[i]);
				}
				assert(subTreeSize == content.size());//we have added everything twice

				content.clear();
				positions.clear();
			}
		}
		else {
			assert(children.size() > 0);
			bool foundResponsibleChild = false;
			for (index i = 0; i < children.size(); i++) {
				if (children[i]->responsible(coords)) {
					foundResponsibleChild = true;
					children[i]->addContent(input, coords);
					break;
				}
			}
			assert(foundResponsibleChild);
			subTreeSize++;
		}
	}

	virtual bool responsible(const Point<double> &point) const {
		const index d = minCoords.getDimensions();
		assert(point.getDimensions() == d);
		for (index i = 0; i < d; i++) {
			if (point[i] < minCoords[i] || point[i] >= maxCoords[i]) return false;
		}
		return true;
	}

	virtual count size() const {
		return isLeaf ? content.size() : subTreeSize;
	}

	virtual void trim() {
		content.shrink_to_fit();
		positions.shrink_to_fit();

		for (index i = 0; i < children.size(); i++) {
			children[i]->trim();
		}
	}

	virtual count getElementsProbabilistically(const Point<double> &query, std::function<double(double)> prob, std::vector<int> &result) {
		auto distancePair = distances(query);
		double probUB = prob(distancePair.first);
		double probLB = prob(distancePair.second);
		if (probUB > 1) {
			throw std::runtime_error("f("+std::to_string(distancePair.first)+")="+std::to_string(probUB)+" is not a probability!");
		}
		assert(probLB <= probUB);
		if (probUB > 0.5) probUB = 1;//if we are going to take every second element anyway, no use in calculating expensive jumps
		if (probUB == 0) return 0;
		//TODO: return whole if probLB == 1
		double probdenom = std::log(1-probUB);
		if (probdenom == 0) {
			return 0;
		}

		count expectedNeighbours = probUB*size();
		count candidatesTested = 0;

		if (isLeaf) {
			queries++;
			const count lsize = content.size();
			for (index i = 0; i < lsize; i++) {
				//jump!
				if (probUB < 1) {
					double random = double(rand()) / RAND_MAX;
					double delta = std::log(random) / probdenom;
					assert(delta == delta);
					assert(delta >= 0);
					i += delta;
					if (i >= lsize) break;
				}

				//see where we've arrived
				candidatesTested++;
				double dist = distance(query, i);
				assert(dist >= distancePair.first);
				assert(dist <= distancePair.second);

				double q = prob(dist);
				if (q > probUB) {
					throw std::runtime_error("Probability function is not monotonically decreasing: f(" + std::to_string(dist) + ")="+std::to_string(q)+">"+std::to_string(probUB)+"=f("+std::to_string(distancePair.first)+").");
				}
				q = q / probUB; //since the candidate was selected by the jumping process, we have to adjust the probabilities
				assert(q <= 1);
				assert(q >= 0);

				//accept?
				double acc = double(rand()) / RAND_MAX;
				if (acc < q) {
					result.push_back(content[i]);
				}
			}
		}	else {
			if (expectedNeighbours < 1) {//select candidates directly instead of calling recursively
				assert(probUB < 1);
				const count stsize = size();
				for (index i = 0; i < stsize; i++) {
					double delta = std::log(double(rand()) / RAND_MAX) / probdenom;
					assert(delta >= 0);
					i += delta;
					if (i < size()) maybeGetKthElement(probUB, query, prob, i, result);//this could be optimized. As of now, the offset is subtracted separately for each point
					else break;
					candidatesTested++;
				}
			} else {//carry on as normal
				for (index i = 0; i < children.size(); i++) {
					candidatesTested += children[i]->getElementsProbabilistically(query, prob, result);
				}
			}
		}
		return candidatesTested;
	}

	virtual void maybeGetKthElement(double upperBound, Point<double> query, std::function<double(double)> prob, index k, std::vector<int> &circleDenizens) {
			assert(k < size());
			if (isLeaf) {
				queries++;
				double dist = distance(query, k);

				double acceptance = prob(dist)/upperBound;
				if (double(rand()) / RAND_MAX < acceptance) circleDenizens.push_back(content[k]);
			} else {
				index offset = 0;
				for (index i = 0; i < children.size(); i++) {
					count childsize = children[i]->size();
					if (k - offset < childsize) {
						children[i]->maybeGetKthElement(upperBound, query, prob, k - offset, circleDenizens);
						break;
					}
					offset += childsize;
				}
			}
		}

	static double euclidDistancePolar(double phi_a, double r_a, double phi_b, double r_b){
			return pow(r_a*r_a+r_b*r_b-2*r_a*r_b*cos(phi_a-phi_b), 0.5);
		}

	std::pair<double, double> EuclideanPolarDistances(Point<double> query) const {
		assert(query.getDimensions() == 2);
		assert(minCoords.getDimensions() == 2);
		const double phi = query[0];
		const double r = query[1];
		const double leftAngle = this->minCoords[0];
		const double rightAngle = this->maxCoords[0];
		const double minR = this->minCoords[1];
		const double maxR = this->maxCoords[1];
		/**
		 * If the query point is not within the quadnode, the distance minimum is on the border.
		 * Need to check whether extremum is between corners.
		 */
		double maxDistance = 0;
		double minDistance = std::numeric_limits<double>::max();

		if (this->responsible(query)) minDistance = 0;

		auto updateMinMax = [&minDistance, &maxDistance, phi, r](double phi_b, double r_b){
			double extremalValue = euclidDistancePolar(phi, r, phi_b, r_b);
			//assert(extremalValue <= r + r_b);
			maxDistance = std::max(extremalValue, maxDistance);
			minDistance = std::min(minDistance, extremalValue);
		};

		/**
		 * angular boundaries
		 */
		//left
		double extremum = r*cos(leftAngle - phi);
		if (extremum < maxR && extremum > minR) {
			updateMinMax(leftAngle, extremum);
		}

		//right
		extremum = r*cos(rightAngle - phi);
		if (extremum < maxR && extremum > minR) {
			updateMinMax(rightAngle, extremum);
		}


		/**
		 * radial boundaries.
		 */
		if (phi > leftAngle && phi < rightAngle) {
			updateMinMax(phi, maxR);
			updateMinMax(phi, minR);
		}
		if (phi + M_PI > leftAngle && phi + M_PI < rightAngle) {
			updateMinMax(phi + M_PI, maxR);
			updateMinMax(phi + M_PI, minR);
		}
		if (phi - M_PI > leftAngle && phi -M_PI < rightAngle) {
			updateMinMax(phi - M_PI, maxR);
			updateMinMax(phi - M_PI, minR);
		}

		/**
		 * corners
		 */
		updateMinMax(leftAngle, maxR);
		updateMinMax(rightAngle, maxR);
		updateMinMax(leftAngle, minR);
		updateMinMax(rightAngle, minR);

		//double shortCutGainMax = maxR + r - maxDistance;
		//assert(minDistance <= minR + r);
		//assert(maxDistance <= maxR + r);
		assert(minDistance < maxDistance);
		return std::pair<double, double>(minDistance, maxDistance);
	}

	/**
	 * @param query Position of the query point
	 */
	std::pair<double, double> EuclideanCartesianDistances(Point<double> query) const {
		/**
		 * If the query point is not within the quadnode, the distance minimum is on the border.
		 * Need to check whether extremum is between corners.
		 */
		double maxDistance = 0;
		double minDistance = std::numeric_limits<double>::max();
		const count dimension = this->minCoords.getDimensions();

		if (this->responsible(query)) minDistance = 0;

		auto updateMinMax = [&minDistance, &maxDistance, query](Point<double> pos){
			double extremalValue = pos.distance(query);
			maxDistance = std::max(extremalValue, maxDistance);
			minDistance = std::min(minDistance, extremalValue);
		};

		std::vector<double> closestValues(dimension);
		std::vector<double> farthestValues(dimension);

		for (index d = 0; d < dimension; d++) {
			if (std::abs(query[d] - this->minCoords.at(d)) < std::abs(query[d] - this->maxCoords.at(d))) {
				closestValues[d] = this->minCoords.at(d);
				farthestValues[d] = this->maxCoords.at(d);
			} else {
				farthestValues[d] = this->minCoords.at(d);
				closestValues[d] = this->maxCoords.at(d);
			}
			if (query[d] >= this->minCoords.at(d) && query[d] <= this->maxCoords.at(d)) {
				closestValues[d] = query[d];
			}
		}
		updateMinMax(Point<double>(closestValues));
		updateMinMax(Point<double>(farthestValues));

		assert(minDistance < query.length() + this->maxCoords.length());
		assert(minDistance < maxDistance);
		return std::pair<double, double>(minDistance, maxDistance);
	}

	/**
	 * Get all Elements in this QuadNode or a descendant of it
	 *
	 * @return std::vector of content type T
	 */
	std::vector<int> getElements() const {
		if (isLeaf) {
			return content;
		} else {
			assert(content.size() == 0);
			assert(positions.size() == 0);

			std::vector<int> result;
			for (index i = 0; i < children.size(); i++) {
				std::vector<int> subresult = children[i]->getElements();
				result.insert(result.end(), subresult.begin(), subresult.end());
			}
			return result;
		}
	}

	count height() const {
		count result = 1;//if leaf node, the children loop will not execute
		for (auto child : children) result = std::max(result, child->height()+1);
		return result;
	}

	/**
	 * Leaf cells in the subtree hanging from this QuadNode
	 */
	count countLeaves() const {
		if (isLeaf) return 1;
		count result = 0;
		for (index i = 0; i < children.size(); i++) {
			result += children[i]->countLeaves();
		}
		return result;
	}

	index getID() const {
			return ID;
		}


	index indexSubtree(index nextID) {
		index result = nextID;
		for (int i = 0; i < this->children.size(); i++) {
			result = this->children[i]->indexSubtree(result);
		}
		this->ID = result;
		return result+1;
	}

	index getCellID(const Point<double> query) const {
		if (!this->responsible(query)) return none;
		if (this->isLeaf) return getID();
		else {
			for (int i = 0; i < children.size(); i++) {
				index childresult = this->children[i]->getCellID(query);
				if (childresult != none) return childresult;
			}
			throw std::runtime_error("Tree structure inconsistent: No responsible child found.");
		}
	}

	index getMaxIDInSubtree() const {
		if (this->isLeaf) return getID();
		else {
			index result = -1;
			for (int i = 0; i < children.size(); i++) {
				result = std::max(this->children[i]->getMaxIDInSubtree(), result);
			}
			return std::max(result, getID());
		}
	}

	count reindex(count offset) {
		if (this->isLeaf)
		{
			#pragma omp task
			{
				index p = offset;
				std::generate(this->content.begin(), this->content.end(), [&p](){return p++;});
			}
			offset += this->size();
		} else {
			for (int i = 0; i < children.size(); i++) {
				offset = this->children[i]->reindex(offset);
			}
		}
		return offset;
	}

protected:
	Point<double> minCoords;
	Point<double> maxCoords;
	std::vector<int> content;
	std::vector<Point<double> > positions;
	std::vector<std::shared_ptr<SpatialCell> > children;
	bool isLeaf;
	count capacity;
	index subTreeSize;
	index ID;
	count queries;

private:
	static const unsigned coarsenLimit = 4;
	static const index none = std::numeric_limits<index>::max();

};

} /* namespace ITI */
