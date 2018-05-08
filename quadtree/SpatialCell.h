/*
 * SpatialCell.h
 *
 *  Created on: 10.11.2016
 *      Author: moritzl
 */

#pragma once

#include <scai/lama/matrix/all.hpp>

#include <vector>
#include <functional>
#include <memory>
#include <set>
#include <list>
#include <queue> 

#include "Point.h"
//#include "../Settings.h"

#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

namespace ITI {

class SpatialCell : public std::enable_shared_from_this<SpatialCell>{
	friend class QuadTreeTest;
public:
	SpatialCell() = default;
	virtual ~SpatialCell() = default;
	SpatialCell(const Point<double> &minCoords, const Point<double> &maxCoords, count capacity=1000) : minCoords(minCoords), maxCoords(maxCoords), indexed(false), capacity(capacity) {
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
	virtual bool isConsistent() const = 0;

	void addChild(std::shared_ptr<SpatialCell> child) {
		assert(this->distances(child->maxCoords).first == 0);
		if (!this->responsible(child->minCoords)) {
			throw std::runtime_error("Node with corners " + std::to_string(this->minCoords[0]) + ", "
					+ std::to_string(this->minCoords[1]) + ", " + std::to_string(this->minCoords[2])
					+ " not responsible for " + std::to_string(child->minCoords[0]) + ", " + std::to_string(child->minCoords[1]) + ", " + std::to_string(child->minCoords[2]));
		}

		for (std::shared_ptr<SpatialCell> previousChild : this->children) {
			assert(!previousChild->responsible(child->minCoords));
			assert(!child->responsible(previousChild->minCoords));
		}
		if (isLeaf) {
			assert(children.size() == 0);
		} else {
			assert(children.size() > 0);
		}
		children.push_back(child);
		isLeaf = false;
	}

	virtual count getNumberOfChildren() const {
		return children.size();
	}

	virtual void coarsen() {
		assert(this->height() == 2);
		assert(content.size() == 0);
		assert(positions.size() == 0);

		std::vector<index> allContent;
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
	bool removeContent(index input, const Point<double> &pos) {
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

	virtual void getElementsInCircle(Point<double> center, double radius, std::vector<index> &result) const {
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

	virtual void addContent(index input, const Point<double> &coords) {
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

	virtual count getElementsProbabilistically(const Point<double> &query, std::function<double(double)> prob, std::vector<index> &result) {
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

	virtual void maybeGetKthElement(double upperBound, Point<double> query, std::function<double(double)> prob, index k, std::vector<index> &circleDenizens) {
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
		assert(std::isfinite(query.length()));

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

		assert(minDistance <= query.length() + this->maxCoords.length());//triangle inequality
		assert(minDistance < maxDistance);
		return std::pair<double, double>(minDistance, maxDistance);
	}

	/**
	 * Get all Elements in this QuadNode or a descendant of it
	 *
	 * @return std::vector of content type T
	 */
	std::vector<index> getElements() const {
		if (isLeaf) {
			return content;
		} else {
			assert(content.size() == 0);
			assert(positions.size() == 0);

			std::vector<index> result;
			for (index i = 0; i < children.size(); i++) {
				std::vector<index> subresult = children[i]->getElements();
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
		count result = 0;
		for (index i = 0; i < children.size(); i++) {
			result += children[i]->countLeaves();
		}
		if( this->isLeaf ) ++result;
		return result;
	}

	count countNodes() const {
		count result = 1;
		for (auto child : children) result += child->countNodes();
		return result;
	}

	index getID() const {
		return ID;
	}

	void setID(index newID) {
		ID = newID;
	}

	index indexSubtree(index nextID) {
		index result = nextID;
		for (unsigned int i = 0; i < this->children.size(); i++) {
			result = this->children[i]->indexSubtree(result);
		}
                this->ID = result;
                indexed = true;
		return result+1;
	}

	bool isIndexed() const {
		return indexed;
	}

	index getCellID(const Point<double> query) const {
		if (!this->responsible(query)) return none;
		if (this->isLeaf) return getID();
		else {
			for (unsigned int i = 0; i < children.size(); i++) {
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
			for (unsigned int i = 0; i < children.size(); i++) {
				result = std::max(this->children[i]->getMaxIDInSubtree(), result);
			}
			return std::max(result, getID());
		}
	}

	count reindexContent(count offset) {
		if (this->isLeaf)
		{
			#pragma omp task
			{
				index p = offset;
				std::generate(this->content.begin(), this->content.end(), [&p](){return p++;});
			}
			offset += this->size();
		} else {
			for (unsigned int i = 0; i < children.size(); i++) {
				offset = this->children[i]->reindexContent(offset);
			}
		}
		return offset;
	}
	
//-------------------------------------------------------------------------------------------
    
	/** Returns the sub tree starting from this node as the adjacency list/matrix of a graph.
         * Whenever I use "graph" I mean in the final graph, not the tree.
         * 
         * @param[in] graphNgbrsCells Keeps pointers to neighbours of every node in the graph.
         *  graphNgbrsPtrs[i]= a set with pointers to the neighbours of -i- in the CSR matrix/graph.
         *  graphNgbrsPtrs.size() == size of the forest or tree = all nodes on every tree, the global sum of nodes
         * @param[out] coords The coordinates of the output graph.
         * @param[in] frontier Used in the case we have a forest and is initialised inside SpatialTree.h while called
         *  in SpatialTree::getGraphFromForest(...).
         */
        //TODO: graphNgbrsCells, is initialised only in case of forest. If we have a tree is empty. right???
	template<typename IndexType, typename ValueType>
	scai::lama::CSRSparseMatrix<ValueType> getSubTreeAsGraph(
                        std::vector< std::set<std::shared_ptr<const SpatialCell>>>& graphNgbrsCells,
			std::vector<std::vector<ValueType>>& coords,
			std::queue<std::shared_ptr<const SpatialCell>> frontier = std::queue<std::shared_ptr<const SpatialCell>>()) const {
            SCAI_REGION("SpatialCell.getSubTreeAsGraph");
            unsigned int treeSize= graphNgbrsCells.size();
            
            // graphNeighbours[i]: the indices of the neighbours of node i in the final graph, not of the tree
            //std::vector<std::set<index>> graphNgbrsID(treeSize);
            
            const IndexType dimension = minCoords.getDimensions();
            
            assert(coords.size() == dimension);
            
            // not recursive, keep a frontier of the nodes to be checked
            // start with this and add every child
            
            //WARNING: not sure if this is the right way to use shared_from_this. At least it works for now
            if( frontier.empty() ){
                frontier.push( this->shared_from_this() );
            }

            // coordsTree[dim].size()== treeSize, initialize with dummy value -1, 
            // might be a problem if you can have negative coords.            
            std::vector<std::vector<ValueType>> coordsTree( dimension , std::vector<ValueType>( treeSize, -1));
            
            while( !frontier.empty() ){
                SCAI_REGION("SpatialCell.getSubTreeAsGraph.inFrontier");
                std::shared_ptr<const SpatialCell> thisNode = frontier.front();
                                   
                // if not indexed
                if(thisNode->getID() == -1){
                    PRINT("Got cell ID = -1.");
                    throw std::logic_error("Tree not indexed?");
                }
                
                // for all children - connect children in the graph
                for(unsigned int c=0; c<thisNode->children.size(); c++){
                    
                    std::shared_ptr<const SpatialCell> child = thisNode->children[c];
                    
                    // for all siblings
                    for(unsigned int s=c+1; s<thisNode->children.size(); s++){
                        std::shared_ptr<const SpatialCell> sibling = thisNode->children[s];
                        // if cells are adjacent add -it- to your graph neighbours list
                        if( child->isAdjacent( *sibling) ){        
                            assert( child->getID() < graphNgbrsCells.size() );
                            assert( sibling->getID() < graphNgbrsCells.size() );
                            graphNgbrsCells[child->getID()].insert(sibling);
                            graphNgbrsCells[sibling->getID()].insert(child);
                        }
                    }
                    
                    // check this child with all the neighbours of father in the graph
                    // graphNgb is a neighbouring cell of this node as a shared_ptr
                    for(std::shared_ptr<const SpatialCell> graphNgb : graphNgbrsCells[thisNode->ID]) {
                        if( child->isAdjacent(*graphNgb) ){
                            assert( child->getID() < graphNgbrsCells.size() );
                            assert( graphNgb->getID() < graphNgbrsCells.size() );
                            graphNgbrsCells[child->getID()].insert(graphNgb );
                            graphNgbrsCells[graphNgb->getID()].insert(child);
                        } 
                    }

                    //when finished with this child, push it to frontier
                    frontier.push(child);
                }
            
                // now all children are checked and we set the pointers, if this node is not a leaf
                // then it is not needed anymore. Reset/delete pointers pointing to this and from
                // this to others.
                
                if( !thisNode->isLeaf ){
                    //PRINT("Node "<< thisNode->getID() << " is not a leaf node");   
                    assert(thisNode->getID() < graphNgbrsCells.size() );
                    
                    // first remove this->ID from others in the graphNgbrsCells vector<set>
                    for(typename std::set<std::shared_ptr<const SpatialCell>>::iterator graphNgb= graphNgbrsCells[thisNode->getID()].begin(); graphNgb!=graphNgbrsCells[thisNode->getID()].end(); graphNgb++){
            
                        assert( graphNgb->get()->getID() < graphNgbrsCells.size());
                        // the neigbours of thisNode->graphNgb. this->ID must be in there somewhere
                        std::set<std::shared_ptr<const SpatialCell>>& ngbSet = graphNgbrsCells[graphNgb->get()->getID()];
                        
                        typename std::set<std::shared_ptr<const SpatialCell>>::iterator fnd = ngbSet.find(std::shared_ptr<const SpatialCell>(thisNode) ) ;

                        if( fnd != ngbSet.end() ){
                            // erase the shared_ptr from the set
                            ngbSet.erase(fnd);
                        }else{
                            // in principle this must never occur.
                            // TODO: change the warning to an assertion or error
                            PRINT("\n WARNING:\nNode ID: "<< thisNode->getID() << " was NOT found in the set of node "<< graphNgb->get()->getID());
                        }
                    }
                    // empty this set too 
                    graphNgbrsCells[thisNode->getID()].clear();
                }
                else{ //thisNode is a leaf, then store its coords
                    //PRINT("Node "<< thisNode->getID() << " IS a leaf node"); 
                    for(int d=0; d<dimension; d++){
                        assert(d<coordsTree.size());
                        assert(d< maxCoords.getDimensions());
                        ValueType thisCoord = thisNode->minCoords[d] + double(thisNode->maxCoords[d] - thisNode->minCoords[d])/ 2;
                        assert( thisNode->getID()< coordsTree[d].size() );
                        coordsTree[d][thisNode->getID()] = thisCoord;
                    }                    
                }
                frontier.pop();
            } //for(unsigned int frontierI=0; frontierI<frontier.size(); frontierI++)
            
            /*
             * Before making the CSR sparse matrix must reindex because leaves do not have
             * sequential indices.
             * Also set the coordinates so they correspond only to leaves.
             * REMEMBER: graphNgbrsCells.size()== treeSize, not leafSize
             */
            
            // start from end so we can remove non-leaf coordinates too
            index leafIndex = 0, non_leaves= 0;
            std::vector<index> leafIndexMapping(treeSize);
            for(index i=0; i<graphNgbrsCells.size(); i++){
                //if a set has size 0 then it is not a leaf so set -1
                if( graphNgbrsCells[i].size() != 0){
                    leafIndexMapping[i]= leafIndex++;
                    for(int d=0; d<dimension; d++){
                        coords[d].push_back( coordsTree[d][i] );
                    }                
                }else{
                    leafIndexMapping[i]= -1;
                    ++non_leaves;
                }
            }
    
            index numLeaves = leafIndex;
            SCAI_ASSERT_EQ_ERROR(coords[0].size(), numLeaves, "Wrong leaf size." )
            SCAI_ASSERT( numLeaves+ non_leaves == treeSize, "Wrong leaves or non-leaves count.");
            
            // this assertion is not correct in the case where we get a forest
            //SCAI_ASSERT( leafIndex == countLeaves(), leafIndex << " >< "<< countLeaves() );

            /* 
             * from the graphNgbrsCells vector set the CSR sparse matrix
             */

            IndexType nnzValues= 0;
            for(int i=0; i<graphNgbrsCells.size(); i++){
                nnzValues += graphNgbrsCells[i].size();
            }
            
            IndexType N = numLeaves;
            
            //create the adjacency matrix
            scai::hmemo::HArray<IndexType> csrIA;
            scai::hmemo::HArray<IndexType> csrJA;
            scai::hmemo::HArray<ValueType> csrValues;
            
            {
                SCAI_REGION("SpatialCell.getSubTreeAsGraph.getCSRMatrix");
                scai::hmemo::WriteOnlyAccess<IndexType> ia( csrIA, N +1 );
                scai::hmemo::WriteOnlyAccess<IndexType> ja( csrJA, nnzValues);
                scai::hmemo::WriteOnlyAccess<ValueType> values( csrValues, nnzValues);
                ia[0] = 0;
                
                // count non-zero elements
                IndexType nnzCounter = 0; 
                index leafIndex= -1;
                
                // since we do not have distribution traverse all rows
                for(IndexType i=0; i<graphNgbrsCells.size(); i++){
                    // TODO:
                    // should be numRowElems == graphNgbrsCells[i].size(), if this is correct then no need to count and
                    // we just do after the for: ia[i+1] = ia[i] + graphNgbrsCells[i].size() 
                    IndexType numRowElems= 0;
                    // the index of the leaves since -i- traverses also non-leaf nodes
                    
                    assert( i< leafIndexMapping.size() );
                    if(leafIndexMapping[i]==-1){    // is not a leaf
                        continue;
                    }else{
                        // leaf indices are sequenrial
                        SCAI_ASSERT_EQUAL_ERROR(leafIndex , leafIndexMapping[i]-1);
                        leafIndex = leafIndexMapping[i];
                    }
                    
                    // graphNgb is a neighbouring cell of this node as a shared_ptr
                    for(std::shared_ptr<const SpatialCell> graphNgb : graphNgbrsCells[i]){
                        // all nodes must be leaves                
                        assert( graphNgb->isLeaf );
                        // not -i- since it also includes non-leaf nodes, use leafIndex instead
                        assert( graphNgb->getID() < leafIndexMapping.size() );
                        IndexType ngbGlobalInd = leafIndexMapping[ graphNgb->getID() ];
                        assert( ngbGlobalInd>= 0);
                        SCAI_ASSERT( ngbGlobalInd<= numLeaves, "Global indexing: " << ngbGlobalInd << " shoould be less () at this point) than the number of leaves: "<< numLeaves );
                        assert( nnzCounter< ja.size() );
                        SCAI_ASSERT( nnzCounter < nnzValues, __FILE__<<" ,"<<__LINE__<< ": nnzValues not calculated properly")                   
                        ja[nnzCounter] = ngbGlobalInd;
                        values[nnzCounter] = 1;
                        ++nnzCounter;
                        ++numRowElems;
                    }
                    SCAI_ASSERT(leafIndex < ia.size(), "Leaf index: "<<leafIndex << " , ia.size: "<< ia.size()  );
                    ia[leafIndex+1] = ia[leafIndex] + numRowElems;
                    SCAI_ASSERT(numRowElems == graphNgbrsCells[i].size(),  __FILE__<<" ,"<<__LINE__<<"something is wrong");
                }
                //TODO: probably resize is never needed (if everything is counted correctly)
                assert(ja.size() == values.size() );
                if( ja.size()!=nnzCounter){
                    PRINT("resizing since ja.size= "<< ja.size() << " and nnzCounter= "<< nnzCounter);                    
                    ja.resize(nnzCounter);
                    values.resize(nnzCounter);
                }
                //PRINT("nnz afterwards= " << nnzCounter << " should be == "<< nnzValues);
                SCAI_ASSERT_EQUAL_ERROR( nnzCounter, nnzValues);
                
            }
                    
            scai::lama::CSRStorage<ValueType> localMatrix( N, N, csrIA, csrJA, csrValues );            
            scai::lama::CSRSparseMatrix<ValueType> ret( std::move(localMatrix));

            return ret;
        }
        
        
        static int getVSsize(std::vector< std::set<std::shared_ptr<SpatialCell>>> input){
            int size= 0;
            for(unsigned int i=0; i<input.size(); i++){
                for(typename std::set<std::shared_ptr<SpatialCell>>::iterator graphNgb= input[i].begin(); graphNgb!=input[i].end(); graphNgb++){
                    ++size;
                    //PRINT(graphNgb->get()->getID());
                }
            }
            return size;
        }
        
        
        /* Checks if two cells are adjacent and share an area. If they have an edge or a corner in common
         * then the test is false.
         * TODO: it does not test if on cell is inside another (well, this cannot happen in a quadTree).
         * TODO: remove the dimension warning if there nothing to warn about.
         * */
        
        bool isAdjacent(const SpatialCell& other) const {
            int dim = minCoords.getDimensions();
            if(dim!=3){
                //std::cout<<"Dimension != 3: WARNING, It could work but not sure...."<< std::endl;
            }

            // if 0 or 1 OK, if 2 cells share just an edge, if 3 they share a corner
            int strictEqualities= 0;
            for(int d=0; d<dim; d++){
                // this ensures that share a face but not sure if edge or corner
                if(maxCoords[d]< other.minCoords[d]){
                    return false;
                }else if(minCoords[d]> other.maxCoords[d]){
                    return false;
                }
                
                // this rules out if they share only an edge or a corner
                if( maxCoords[d]== other.minCoords[d] or minCoords[d]== other.maxCoords[d]){
                    ++strictEqualities;
                }
            }
            
            // for arbitrary dimension this can be >d-2 (?)
            if(dim==2){
                if( strictEqualities > 1){
                    return false;
                }
            }else {
                if( strictEqualities > dim-2){
                    return false;
                }
            }
            // if none of the above failed
            
            return true;
        }
        
        
        /* Returns the 2^dim corners of the cell.
         */
        std::vector<Point<double>> getCorners() const {
        
            int dim = minCoords.getDimensions();
            std::vector<Point<double>> corners(pow(2,dim));
            assert(corners.size() % 2 == 0);//if this assert fails, we had some floating point bug in the exponentiation
            for(unsigned int i=0; i<corners.size(); i++){
                int bitCopy= i;
                std::vector<double> point(dim, 0.0);
                for(int d=0; d<dim; d++){
                    point[d]= minCoords[d]+ (bitCopy & 1)* maxCoords[d];
                    bitCopy = bitCopy >> 1;
                }
                corners[i] = Point<double>(point);
            }
            return corners;
        }

        count getDimensions() const {
        	return minCoords.getDimensions();
        }
	
	
//-------------------------------------------------------------------------------------

protected:
	Point<double> minCoords;
	Point<double> maxCoords;
	std::vector<index> content;
	std::vector<Point<double> > positions;
	std::vector<std::shared_ptr<SpatialCell> > children;
	bool isLeaf;
	bool indexed;
	count capacity;
	index subTreeSize;
	index ID;
	count queries;

private:
	static const unsigned coarsenLimit = 4;
	static const index none = std::numeric_limits<index>::max();

};

} /* namespace ITI */

