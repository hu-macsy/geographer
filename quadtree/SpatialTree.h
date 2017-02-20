/*
 * SpatialTree.h
 *
 *  Created on: 10.11.2016
 *      Author: moritzl
 */

#pragma once

#include "SpatialCell.h"

namespace ITI {

class SpatialTree {
public:
	SpatialTree() = default;
	virtual ~SpatialTree() = default;

	void addContent(index content, const Point<double> &coords) {
		root->addContent(content, coords);
	}

	bool removeContent(index content, const Point<double> &coords) {
		return root->removeContent(content, coords);
	}

	void getElementsInCircle(const Point<double> query, const double radius, std::vector<index> &circleDenizens) const {
		root->getElementsInCircle(query, radius, circleDenizens);
	}

	count getElementsProbabilistically(Point<double> query, std::function<double(double)> prob, std::vector<index> &circleDenizens) {
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

	void reindexContent() {
		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				root->reindexContent(0);
			}
		}
	}
	
	std::shared_ptr<SpatialCell> getRoot(){
            return root;
        }

	/**
	 * Get all elements, regardless of position
	 *
	 * @return vector<T> of elements
	 */
	std::vector<index> getElements() const {
		return root->getElements();
	}
        
	template<typename IndexType, typename ValueType>
	scai::lama::CSRSparseMatrix<ValueType>  getTreeAsGraph( std::vector< std::set<std::shared_ptr<SpatialCell>>>& graphNgbrsCells, std::vector<std::vector<ValueType>>& coords ){
		return root->getSubTreeAsGraph<IndexType, ValueType>( graphNgbrsCells, coords );
	}
        
        
        template<typename IndexType, typename ValueType>
	static scai::lama::CSRSparseMatrix<ValueType>  getGraphFromForest( std::vector< std::set<std::shared_ptr<SpatialCell>>>& graphNgbrsCells, std::vector<std::shared_ptr<SpatialCell>>& treePtrVector,  std::vector<std::vector<ValueType>>& coords){
            IndexType numTrees = treePtrVector.size();
            //  both vectors must have the sime size = forestSize
            IndexType forestSize = treePtrVector[numTrees-1]->getID()+1;
            //PRINT("graphNgbrsCells.size()= " << graphNgbrsCells.size() << ", forest size= " << forestSize);      
            assert( forestSize == graphNgbrsCells.size() );
            
            std::shared_ptr<SpatialCell> onlyChild;
            if(treePtrVector.size()!=0){
                onlyChild= treePtrVector[0];
            }else{
                throw std::runtime_error("Input vector is empty.");
            }
            
            int maxHeight= 0;
            for(IndexType i=0; i<numTrees; i++){
                std::shared_ptr<SpatialCell> thisNode = treePtrVector[i];
                if( thisNode->height() > maxHeight){
                    onlyChild = thisNode;
                    maxHeight = thisNode->height();
                }
            }
            PRINT("numTrees= "<< numTrees);            
            std::shared_ptr<SpatialCell> dummyRoot= onlyChild;

            // convert the tree vector to a queue for the starting frontier
            std::queue<std::shared_ptr<SpatialCell>> frontier;
            for(int i=0; i< numTrees; i++){
                frontier.push( treePtrVector[i] );
            }
            /* since we use a frontier as an input, dummy node should not be needed at all
             * maybe only to call getSubTreeAsGraph.
             * TODO: turn getSubTreeAsGraph to static ??
             * */
            
            return dummyRoot->getSubTreeAsGraph<IndexType, ValueType>( graphNgbrsCells , coords, frontier);
	}
        
        
protected:
	std::shared_ptr<SpatialCell> root;
};

} /* namespace ITI */
