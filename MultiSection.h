#pragma once

#include <scai/lama.hpp>
#include <scai/lama/Vector.hpp>
//#include <scai/lama/Scalar.hpp>

#include <climits>

#include "Settings.h"

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define BUILD_COMMIT_STRING STR_VALUE(BUILD_COMMIT)
#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

        

namespace ITI {

    /* A d-dimensional rectangle represented by two points: the bottom and the top corner.
     * For all i: 0<i<d, it must be bottom[i]<top[i]
     * */
    //template <typename IndexType, typename ValueType>
    struct rectangle{
        
        double weight;
        
        // for all i: 0<i<dimension, bottom[i]<top[i]
        std::vector<double> bottom;
        std::vector<double> top;
        
        void print(){
            std::cout<< "rectangle bottom: ";
            for(int i=0; i<bottom.size(); i++){
                std::cout<< bottom[i] << ", ";
            }
            std::cout<< std::endl;
            
            std::cout<< "             top: ";
            for(int i=0; i<top.size(); i++){
                std::cout<< top[i] << ", ";
            }
            std::cout<< std::endl;
        }
        /*
         *     static bool heavier(rectangle& a, rectangle& b){
         *         return a.weight > b.weight;
    }
    */
        /** Checks if this rectangle resides entirelly in the given rectangle: 
         * for all dimensions i< d:  
         * this.bottom[i]> outer.bottom[i] and this.top[i]< outer.top[i]
         */
        bool inside(rectangle outer){
            SCAI_REGION("rectangle.inside");
            
            bool ret = true;
            for(int d=0; d<bottom.size(); d++){
                //TODO: test if we need < or <= for bottom or/and top
                if( this->bottom[d]<outer.bottom[d] or this->top[d]>outer.top[d]){
                    ret= false;
                    break;
                }
            }
            return ret;
        }
        
        /** Checks if the given point is inside this rectangle.
         */
        bool owns( const std::vector<double>& point){
            SCAI_REGION("rectangle.owns");
            
            IndexType dim= point.size();
            SCAI_ASSERT( dim==this->top.size(), "Wrong dimensions: point.dim= " << dim << ", this->top.dim= "<< this->top.size() );
            
            bool ret= true;
            for(int d=0; d<dim; d++){
                if( point[d]<this->bottom[d] or point[d]>=this->top[d]){
                    ret= false;
                    break;
                }
            }
            return ret;
        }
        
        bool operator()(rectangle& a, rectangle& b){
            return a.weight < b.weight;
        }
        
        bool operator==(rectangle& r){
            for(int d=0; d<this->top.size(); d++){
                if( this->top[d]!=r.top[d] ){
                    return false;
                }
                if( this->bottom[d]!=r.bottom[d] ){
                    return false;
                }
            }
            return true;
        }
        
    };
        
        
//---------------------------------------------------------------------------------------
    template <typename IndexType, typename ValueType>
    struct rectCell{
    public:
        rectCell( const rectangle rect ){
            myRect = rect;
            weight = -1;
            isLeaf = true;
        }
        
        rectCell( const rectangle rect , const ValueType w){
            myRect = rect;
            weight = w;
            isLeaf = true;
        }
        
        /**Insert a rectangle in the tree
         */
        //TODO: not handling the case where a rectangle intersects to rectangles allready in the tree
        void insert(rectangle& rect){
            SCAI_REGION("rectCell.insert");
            
            //check dimension is correct
            SCAI_ASSERT( rect.top.size()==myRect.top.size(), "Dimensions do not agree."); 
            
            // check if coordinates of the rectangle are wrong
            if( !rect.inside(this->myRect) ){
                PRINT("Input rectangle should be inside this:");
                rect.print();
                myRect.print();
                throw std::runtime_error("Wrong rectangle size: too big or wrong coordinates.");
            }
            
            if( this->isLeaf ){
                std::shared_ptr<rectCell>  tmp (new rectCell( rect ) );
                this->children.push_back( tmp );
                this->isLeaf = false;
            }else{
                bool inserted = false;
                // check if rect can fit inside some of the children
                for(int c=0; c<children.size(); c++){
                    if( rect.inside(children[c]->myRect) ){                        
                        inserted = true;
                        children[c]->insert(rect);
                        //TODO: with break it inserts rect to the first it fits
                        break;
                    }
                }
                // does not fit in any of the children, insert as a new child
                if( !inserted ){
                    std::shared_ptr<rectCell>  tmp (new rectCell( rect ) );
                    this->children.push_back( tmp );
                }
            }
        }
        
        /** Finds and returns the smallest rectangle that contains the given point.
         *  @param[in] point The query point given as a d dimensional vector;
         *  @return A pointer to the rectangle cell that contains the point.
         */
        std::shared_ptr<rectCell> contains( const std::vector<ValueType>& point){
            SCAI_REGION("rectCell.contains");
            
            IndexType dim = point.size();
            SCAI_ASSERT( dim==myRect.top.size(), "Dimensions do not agree");
            
            // point should be inside this rectangle
            for(int d=0; d<dim; d++){
                SCAI_ASSERT( point[d]<myRect.top[d], "Point is out of bounds");
                SCAI_ASSERT( point[d]>=myRect.bottom[d], "Point is out of bounds");
                //TODO: or just return NULL
            }
            
            std::shared_ptr<rectCell> ret;
            // only leaf nodes must be returned as owners of points
            //TODO: maybe not... when leaf rectangles do not cover the whole space.
            if( !this->isLeaf ){
                bool foundOwnerChild = false;
                for(int c=0; c<this->children.size(); c++){
                    if( children[c]->myRect.owns( point ) ){                    
                        foundOwnerChild = true;
                        ret = children[c]->contains( point );
                    }
                }
               
                // this is not a leaf node and none of the childrens owns the point
                if( !foundOwnerChild ){
                    // check again if this rectangle owns the point
                    //TODO: not need to check again
                    if( myRect.owns( point ) ){
                        ret = std::make_shared<rectCell>(*this);
                    }else{
                        ret = NULL;
                    }
                }
            }else{
                SCAI_ASSERT( this->myRect.owns(point), "Should not happen")    
                ret = std::make_shared<rectCell>(*this);
            }
            return ret;
        }
        
        IndexType getSubtreeSize(){
            IndexType ret = 1;
            for(int c=0; c<children.size(); c++){
                ret += children[c]->getSubtreeSize();
            }
            return ret;
        }
        
        IndexType getLeafSize(){
            IndexType ret = 0;
            if( isLeaf ){
                ret= 1;
            }else{
                for(int c=0; c<children.size(); c++){
                    ret += children[c]->getLeafSize();
                }
            }
            return ret;
        }
        
        rectangle getRect(){
            return myRect;
        }
        
        /*
        bool areChildrenOverlapping(){
            bool ret = false;
            for(int c=0; c<children.size(); c++){
                 for(int c2=0; c<children.size(); c++){
                     }
                 }
            }
            return ret;
        }
        */
    protected:
        rectangle myRect;
        std::vector<std::shared_ptr<rectCell>> children;
        ValueType weight;
        bool isLeaf;
        
    };
        
        
        
//---------------------------------------------------------------------------------------

        
    template <typename IndexType, typename ValueType>
    class MultiSection{
    public:

        /** Get a partition of a uniform grid of side length sideLen into settings.numBlocks blocks.
         */
        static std::priority_queue< rectangle, std::vector<rectangle>, rectangle> getPartition( const scai::lama::DenseVector<ValueType>& nodeWeights, const IndexType sideLen, Settings settings);
        
        /** Calculates the projection of all points in the bounding box (bBox) in the given dimension. Every PE
         *  creates an array of appropriate length, calculates the projection for its local coords and then
         *  calls a all2all sum routine. We assume a uniform grid so the exact coordinates can be infered just
         *  by the index of the -nodeWeights- vector.
         * 
         * @param[in] nodeWeights The weights for each point.
         * @param[in] bBox The bounding box/rectangle in which we wish to calculate the projection. For all dimensions, bBox.bottom[i]< bBox.top[i].
         * @param[in] dimensiontoProject The dimension in which we wish to project the weights. Should be more or equal to 0 and less than d (where d are the total dimensions).
         * @param[in] sideLen The length of the side of the whole uniform, square grid.
         * @param[in] setting A settigns struct passing various arguments.
         * @return Return an vector where in each position is the sum of the weights of the corespondig coordianate (not the same).

         * Example: bBox={(5,10),(8,15)} and dimensionToProject=0 (=x). Then the return vector has size |8-5|=3. return[0] is the sum of the coordinates in the bBox which have their 0-coordinate equal to 5, return[1] fot he points with 0-coordinate equal to 3 etc. If dimensionToProject=1 then return vector has size |10-15|=5.
         */
        static std::vector<ValueType> projection( const scai::lama::DenseVector<ValueType>& nodeWeights, const struct rectangle& bBox, const IndexType dimensionToProject, const IndexType sideLen, Settings settings);
        
        /** Calculates the projection for a vector of rectangles.
         * @param[in] nodeWeights The weights for each point.
         * @param[in] bBoxes The bounding boxes for all the rectangles. bBoxes.size()= k
         * @param[in] dimensiontoProject The dimension in which we wish to project the weights for every rectangle. Should be more or equal to 0 and less than d (where d are the total dimensions). dimensionsToProject.size()= k.
         * @param[in] sideLen The length of the side of the whole uniform, square grid.
         * @param[in] setting A settigns struct passing various arguments.
         * 
         * @return A vector of size k that holds the projection of every rectangle in the given dimension. return.size()= k and return[i].size() depends on the size of the rectangle i.
         */
        static std::vector<std::vector<ValueType>> allProjection( const scai::lama::DenseVector<ValueType>& nodeWeights, const std::vector<struct rectangle>& bBoxes, const std::vector<IndexType> dimensionsToProject, const IndexType sideLen, Settings settings);
        
        /** Partitions the given vector into k parts of roughly equal weights.
         *
         * @param[in] array The 1 dimensional array of positive numbers to be partitioned.
         * @param[in] k The number of parts/blocks.
         * @return The first returned value is a vector of size k-1 and holds the indices of each part/block: first part is from 0 to return.first[0] (inluding the weight of array[return.first[0]]), second from return.first[0]+1 till return.first[1] ets. Last part if from return.first[k-2]+1 till return.first.size().
         * The second vector is of size k and holds the weights of each part.
         * 
         * Example: input= [ 11, 6, 8, 1, 2, 4, 11, 1, 2] and k=3
         *                        |           |
         * return.first = [ 1, 5]. Implies the partition: 0, 1 | 2, 3, 4, 5 | 6, 7, 8
         * return.second=[ 17, 15, 14]
         */
        static std::pair<std::vector<ValueType>,std::vector<ValueType>> partition1D( const std::vector<ValueType>& array, const IndexType k, Settings settings);   
        
        /**Checks if the given index is in the given bounding box. Index corresponds to a uniform matrix given
         * as a 1D array/vector. 
         * 
         * @param[in] coords The coordinates of the input point.
         * @param[in] bBox The bounding box given as two vectors; one for the bottom point and one for the top point. For all dimensions i, should be: bBox.first(i)< bBox.second(i).
         * @param[in] sideLen The length of the side of the whole uniform, square grid.
         * 
         */
        static bool inBBox( const std::vector<IndexType>& coords, const struct rectangle& bBox, const IndexType sideLen);
        
        /** Calculates the wwight of the rentangle.
         * 
         * @param[in] nodeWeights The weights for each point.
         * @param[in] bBox The bounding box/rectangle in which we wish to calculate the projection. For all dimensions, bBox.bottom[i]< bBox.top[i]
         * @param[in] sideLen The length of the side of the whole uniform, square grid.
         *
         * @return The weight of the given rectangle.
         */
        static ValueType getRectangleWeight( const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle& bBox, const IndexType sideLen, Settings settings);
        
        /** Function to transform a 1D index to 2D or 3D given the side length of the cubical grid.
         * For example, in a 4x4 grid, indexTo2D(1)=(0,1), indexTo2D(4)=(1,0) and indexTo2D(13)=(3,1)
         * 
         * @param[in] ind The index to transform.
         * @param[in] sideLen The side length of the 2D or 3D cube/grid.
         * @param[in] dimensions The dimension of the cube/grid (either 2 or 3).
         * @return A vector containing the index for every dimension. The size of the vector is equal to dimensions.
         */
        static std::vector<IndexType> indexToCoords(const IndexType ind, const IndexType sideLen, const IndexType dimensions);
        
    private:

        static std::vector<IndexType> indexTo2D(IndexType ind, IndexType sideLen);
        
        static std::vector<IndexType> indexTo3D(IndexType ind, IndexType sideLen);
    };

}