#pragma once

#include <scai/lama.hpp>
#include <scai/tracing.hpp>

#include <climits>
#include <chrono>
#include <fstream>

#include "Settings.h"

namespace ITI {


/** @cond INTERNAL
 * A d-dimensional rectangle represented by two points: the bottom and the top corner.
 * For all i: 0<i<d, it must be bottom[i]<top[i]
 * Also, the rectangle contains the points [bottom, top], so, in a 1D rectangle, [4,8] contains
 * the points 4,5,6,7 and 8.
 *
 * */
template<typename ValueType>
class rectangle {
public:
    ValueType weight;

    // for all i: 0<i<dimension, bottom[i]<top[i]
    std::vector<ValueType> bottom;
    std::vector<ValueType> top;

    void print(std::ostream& out) {
        std::cout<< "rectangle bottom: ";
        for(unsigned int i=0; i<bottom.size(); i++) {
            out<< bottom[i] << ", ";
        }
        out<< std::endl;

        out<< "             top: ";
        for(unsigned int i=0; i<top.size(); i++) {
            out<< top[i] << ", ";
        }

        out<< "          weight: "<< weight << std::endl;
    }

    /**@internal
     Checks if this rectangle resides entirely in the given rectangle:
     * for all dimensions i< d:
     * this.bottom[i]> outer.bottom[i] or this.top[i]< outer.top[i]
     */
    bool isInside(rectangle outer) {
        SCAI_REGION("rectangle.isInside");

        bool ret = true;
        for(unsigned int d=0; d<bottom.size(); d++) {
            if( this->bottom[d]<outer.bottom[d] or this->top[d]>outer.top[d]) {
                ret= false;
                break;
            }
        }
        return ret;
    }

    /** Checks if the given point is inside this rectangle.
     */
    template< typename D>
    bool owns( const std::vector<D>& point) {
        SCAI_REGION("rectangle.owns");

        IndexType dim= point.size();
        SCAI_ASSERT( dim==this->top.size(), "Wrong dimensions: point.dim= " << dim << ", this->top.dim= "<< this->top.size() );

        for(int d=0; d<dim; d++) {
            if( point[d]<bottom[d] or point[d]>top[d]) {
                return false;
            }
        }
        return true;
    }

    /*  Checks if two rectangles share a common border. In our case if top[d]=10 and bottom[d]=11
     *  then the rectangles are adjacent: their difference must be more than 1 in order NOT to be
     *  adjacent.
    */
    bool isAdjacent(const rectangle& other) const {
        int dim = bottom.size();
        // if 0 or 1 OK, if 2 cells share just an edge, if 3 they share a corner
        int strictEqualities= 0;
        for(int d=0; d<dim; d++) {
            // this ensures that share a face but not sure if edge or corner
            if(top[d]-other.bottom[d]<1) {
                return false;
            } else if(bottom[d]-other.top[d]>1) {
                return false;
            }

            // this rules out if they share only an edge or a corner
            if( top[d]== other.bottom[d] or bottom[d]== other.top[d]) {
                ++strictEqualities;
            }
        }

        // for arbitrary dimension this can be >d-2 (?)
        if(dim==2) {
            if( strictEqualities > 1) {
                return false;
            }
        } else {
            if( strictEqualities > dim-2) {
                return false;
            }
        }
        // if none of the above failed
        return true;
    }

    bool operator()(rectangle& a, rectangle& b) {
        return a.weight < b.weight;
    }

    bool operator==(rectangle& r) {
        for(unsigned int d=0; d<this->top.size(); d++) {
            if( this->top[d]!=r.top[d] ) {
                return false;
            }
            if( this->bottom[d]!=r.bottom[d] ) {
                return false;
            }
        }
        return true;
    }

};//struct rectangle



//---------------------------------------------------------------------------------------
template <typename IndexType, typename ValueType>
class rectCell {

    template <typename T, typename U>
    friend class MultiSection;

public:

    rectCell( const rectangle<ValueType> rect ) {
        myRect = rect;
        weight = -1;
        isLeaf = true;
    }

    rectCell( const rectangle<ValueType> rect, const ValueType w) {
        myRect = rect;
        weight = w;
        isLeaf = true;
    }

    /**Insert a rectangle in the tree
     */
    //TODO: not handling the case where a rectangle intersects to rectangles allready in the tree
    void insert(rectangle<ValueType>& rect) {
        SCAI_REGION("rectCell.insert");

        //check dimension is correct
        SCAI_ASSERT( rect.top.size()==myRect.top.size(), "Dimensions do not agree.");

        // check if coordinates of the rectangle are wrong
        if( !rect.isInside(this->myRect) ) {
            PRINT("Input rectangle:");
            rect.print(std::cout);
            PRINT("Should be inside this:");
            myRect.print(std::cout);
            throw std::runtime_error("Wrong rectangle size: too big or wrong coordinates.");
        }

        if( this->isLeaf ) {
            std::shared_ptr<rectCell>  tmp (new rectCell( rect ) );
            this->children.push_back( tmp );
            this->isLeaf = false;
        } else {
            bool inserted = false;
            // check if rect can fit inside some of the children
            for(int c=0; c<children.size(); c++) {
                if( rect.isInside(children[c]->myRect) ) {
                    inserted = true;
                    children[c]->insert(rect);
                    //TODO: with break it inserts rect to the first it fits
                    break;
                }
            }
            // does not fit in any of the children, insert as a new child
            if( !inserted ) {
                std::shared_ptr<rectCell>  tmp (new rectCell( rect ) );
                this->children.push_back( tmp );
            }
        }
    }

    /** Finds and returns the smallest rectangle that contains the given point.
     *  @param[in] point The query point given as a d dimensional vector;
     *  @return A pointer to the rectangle cell that contains the point.
     */
    template<typename D>
    std::shared_ptr<rectCell> getContainingLeaf( const std::vector<D>& point) {
        SCAI_REGION("rectCell.getContainingLeaf");

        IndexType dim = point.size();
        SCAI_ASSERT( dim==myRect.top.size(), "Dimensions do not agree");

        // point should be inside this rectangle
        for(int d=0; d<dim; d++) {
            SCAI_REGION("rectCell.getContainingLeaf.checkPoint");
            if(point[d]>myRect.top[d] or  point[d]<myRect.bottom[d]) {
                throw std::logic_error("Point out of bounds");
            }
        }

        if( !this->isLeaf ) {
            SCAI_REGION("rectCell.getContainingLeaf.ifNotLeaf");
            for(int c=0; c<this->children.size(); c++) {
                if( children[c]->myRect.owns( point ) ) {
                    return children[c]->getContainingLeaf( point );
                }
            }
            // this is not a leaf node but none of the childrens owns the point
            //WARNING: in our case this should never happen, but it may happen in a more general
            // case where the children rectangles do not cover the entire father rectangle
            //this->getRect().print();
            throw std::logic_error("Null pointer, the given point in not contained in the tree.");
        } else {
            SCAI_REGION("rectCell.getContainingLeaf.ifLeaf");
            //TODO: possibly a bit expensive and not needed assertion
            SCAI_ASSERT( myRect.owns(point), "Should not happen")
            return  std::make_shared<rectCell>(*this);
        }

    }

    IndexType getSubtreeSize() {
        SCAI_REGION("rectCell.getSubtreeSize");
        IndexType ret = 1;
        for(int c=0; c<children.size(); c++) {
            ret += children[c]->getSubtreeSize();
        }
        return ret;
    }

    IndexType getNumLeaves() {
        SCAI_REGION("rectCell.getNumLeaves");
        IndexType ret = 0;
        if( isLeaf ) {
            ret= 1;
        } else {
            for(int c=0; c<children.size(); c++) {
                ret += children[c]->getNumLeaves();
            }
        }
        return ret;
    }


    /** Returns a vector of size getNumLeaves() with pointers to all the leaf nodes
     *  The traversal, and thus the order that leaves appear in the returned vector, is done in a DFS way.
     */
    std::vector<std::shared_ptr<rectCell>> getAllLeaves() {
        SCAI_REGION("rectCell.getAllLeaves");

        if( isLeaf ) {
            std::vector<std::shared_ptr<rectCell>> ret(1);
            ret[0] = std::make_shared<rectCell>(*this);
            return ret;
        }
        IndexType leafIndex = this->indexLeaves(0);
        SCAI_ASSERT( leafIndex==this->getNumLeaves(), "Wrong leaf indexing");

        // we use a frontier queue and it looks like a BFS, but we insert the leaves
        // in leaves[child->leafID] and leafID is computed in a DFS way
        std::vector<std::shared_ptr<rectCell>> leaves( leafIndex );
        std::queue<std::shared_ptr<rectCell>> frontier;

        for(int c=0; c<children.size(); c++) {
            if( !children[c]->isLeaf ) {
                frontier.push( children[c] );
            } else {
                std::shared_ptr<rectCell> child = this->children[c];
                leaves[child->leafID] = child;
            }
        }

        while( !frontier.empty() ) {
            std::shared_ptr<rectCell> thisNode = frontier.front();

            for(int c=0; c<thisNode->children.size(); c++) {
                std::shared_ptr<rectCell> child = thisNode->children[c];
                if( !child->isLeaf ) {
                    frontier.push( child );
                } else {
                    leaves[child->leafID] = child;
                }
            }

            frontier.pop();
        }
        return  leaves;
    }

    /** Indexes only the leaf nodes of the tree in a DFS way.
     * */
    IndexType indexLeaves(IndexType currentIndex) {
        SCAI_REGION("rectCell.indexLeaves");
        IndexType ret = currentIndex;
        for(int c=0; c<this->children.size(); c++) {
            ret = children[c]->indexLeaves( ret );
        }
        if( this->isLeaf ) {
            this->leafID = ret;
            ret = currentIndex+1;
        } else { //do not index if this is not a leaf
            this->leafID= -1;
        }

        return ret;
    }

    void printLeavesInFile( const std::string filename, IndexType dimension ) {
        std::ofstream f(filename);
        if(f.fail())
            throw std::runtime_error("File "+ filename+ " failed.");

        std::vector<std::shared_ptr<rectCell>> allLeaves = getAllLeaves();

        const IndexType numLeaves = allLeaves.size();

        for(int l=0; l<numLeaves; l++) {
            std::shared_ptr<rectCell> thisLeaf = allLeaves[l];

            for(int d=0; d<dimension; d++) {
                f<< thisLeaf->getRect().bottom[d] << " ";
            }
            f<< std::endl;
            for(int d=0; d<dimension; d++) {
                f<< thisLeaf->getRect().top[d] << " ";
            }
            f<< std::endl;
        }
    }

    void printLeavesInFileNestler( const std::string filename, IndexType dimension ) {
        std::ofstream f(filename);
        if(f.fail())
            throw std::runtime_error("File "+ filename+ " failed.");

        std::vector<std::shared_ptr<rectCell>> allLeaves = getAllLeaves();

        const IndexType numLeaves = allLeaves.size();

        for(int l=0; l<numLeaves; l++) {
            std::shared_ptr<rectCell> thisLeaf = allLeaves[l];
            rectangle<ValueType> thisRect = thisLeaf->getRect();

            for(int d=0; d<dimension; d++) {
                f<< thisRect.bottom[d] << " ";
            }

            for(int d=0; d<dimension; d++) {
                f<< thisRect.top[d] - thisRect.bottom[d] << " ";
            }
            f<< std::endl;
        }
    }

    rectangle<ValueType> getRect() {
        return myRect;
    }

    std::shared_ptr<rectangle<ValueType>> getRectPtr() {
        return std::make_shared<rectangle>(myRect);
    }

    IndexType getLeafID() {
        return leafID;
    }

    ValueType getLeafWeight() {
        return myRect.weight;
    }

protected:
    rectangle<ValueType> myRect;
    //std::shared_ptr<rectangle> myRect;
    std::vector<std::shared_ptr<rectCell>> children;
    ValueType weight;
    bool isLeaf;
    IndexType leafID = -1;   // id value only for leaf nodes
};

// rectangle and rectCell are marked as internal in order not to produce the documentation
/** @endcond INTERNAL
*/

//---------------------------------------------------------------------------------------

/** @brief Get a partition of a point set by partitioning in rectangular regions recursively for every dimension.
*/

template<typename IndexType, typename ValueType>
class MultiSection {
public:

    typedef std::vector<ValueType> point;
    typedef std::vector<IndexType> intPoint;

    /** @brief A partition of non-uniform grid.
     *
     * @param[in] input Adjacency matrix of the input graph
     * @param[in] coordinates Coordinates of input points
     * @param[in] nodeWeights Optional node weights.
     * @param[in] settings Settings struct
     *
     * @return Distributed DenseVector of length n, partition[i] contains the block ID of node i
     */
    static scai::lama::DenseVector<IndexType> computePartition(
        const scai::lama::CSRSparseMatrix<ValueType> &input,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        struct Settings settings );

    template<typename T>
    static scai::lama::DenseVector<IndexType> computePartition(
        const scai::lama::CSRSparseMatrix<ValueType>& input,
        const std::vector<std::vector<T>>& coordinates,
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        const std::vector<T>& minCoords,
        const std::vector<T>& maxCoords,
        struct Settings settings );

    /** @brief Iterative version to get a partition
    */
    static scai::lama::DenseVector<IndexType> getPartitionIter(
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        struct Settings settings );

    /** @brief Given a tree of rectangles, sets the partition for every point.
     *
     * @param[in] root The root of the rectTree, a tree of non-overlapping rectangles
     * @param[in] distPtr The pointer of the DistributionPtr
     * @param[in] localPoint The points that are local to every processor
     *
     * @return A distributed DenseVector of length n, partition[i] contains the block ID of node i
     */
    template <typename T>
    static scai::lama::DenseVector<IndexType> setPartition(
        std::shared_ptr<rectCell<IndexType,ValueType>> root,
        const scai::dmemo::DistributionPtr distPtr,
        const std::vector<std::vector<T>>& localPoints);


    /** Get a tree of rectangles of a uniform grid with side length sideLen. The rectangles cover the whole grid and
     * do not overlap.
     *
     * @param[in] nodeWeights The node weights of the points.
     * @param[in] sideLen The length of the side of the whole uniform, square grid. The coordinates are from 0 to sideLen-1. Example: if sideLen=2, the points are (0,0),(0,1),(1,0),(1,1)
     * @param[in] setting Settings struct
     *
     * @return A pointer to the root of the tree. number of leaves = settings.numBlocks.
     */
    static std::shared_ptr<rectCell<IndexType,ValueType>> getRectangles(
                const scai::lama::DenseVector<ValueType>& nodeWeights,
                const IndexType sideLen,
                Settings settings);

    /** @brief Iterative version
    */
    static std::shared_ptr<rectCell<IndexType,ValueType>> getRectanglesIter(
                //const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
                const std::vector<std::vector<ValueType>>& coordinates,
                const scai::lama::DenseVector<ValueType>& nodeWeights,
                const std::vector<ValueType>& minCoords,
                const std::vector<ValueType>& maxCoords,
                Settings settings);

    //TODO: Let coordinates be of ValueType and round inside the function if needed.

    /** @brief Get a tree of rectangles for a non-uniform grid.
     *
     * @param[in] input The adjacency matrix of the input graph.
     * @param[in] coordinates The coordinates of each point of the graph. Here they are of type IndexType for the projections need to be in the integers.
     * @param[in] nodeWeights The weights for each point.
     * @param[in] minCoords The minimum coordinate on each dimensions. minCoords.size()= dimensions
     * @param[in] maxCoords The maximum coordinate on each dimensions. maxCoords.size()= dimensions
     * @param[in] setting Settings struct
     *
     * @return A pointer to the root of the tree. The leaves of the tree are the rewuested rectangles.
     */
    template<typename T>
    static std::shared_ptr<rectCell<IndexType,ValueType>> getRectanglesNonUniform(
                const scai::lama::CSRSparseMatrix<ValueType>& input,
                const std::vector<std::vector<T>>& coordinates,
                const scai::lama::DenseVector<ValueType>& nodeWeights,
                const std::vector<T>& minCoords,
                const std::vector<T>& maxCoords,
                Settings settings);

    /** @brief Project and partition using a optimal 1D partition algo
    */
    template<typename T>
    static IndexType projectAnd1Dpartition(
        std::shared_ptr<rectCell<IndexType,ValueType>>& treeRoot,
        const std::vector<std::vector<T>>& coordinates,
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        const std::vector<IndexType>& numCuts,
        const std::vector<T>& maxCoords);

    /** Calculates the projection of all points in the bounding box (bBox) in the given dimension. Every PE
     *  creates an array of appropriate length, calculates the projection for its local coords and then
     *  calls a all2all sum routine. We assume a uniform grid so the exact coordinates can be inferred just
     *  by the index of the -nodeWeights- vector.
     *
     * @param[in] nodeWeights The weights for each point.
     * @param[in] treeRoot The root of the tree that contains all current rectangles for which we get the projections. We only calculate the projections for the leaf nodes.
     * @param[in] dimensiontoProject A vector of size treeRoot.getNumLeaves(). dimensionsToProject[i]= the dimension in which we wish to project the weights for rectangle/leaf i. Should be more or equal to 0 and less than d (where d are the total dimensions).
     * @param[in] sideLen The length of the side of the whole uniform, square grid.
     * @param[in] setting A settings struct passing various arguments.
     * @return Return a vector where in each position is the sum of the weights of the corresponding coordinate for every leaf.
     * \p ret[i][j] is the the weight of the j-th coordinate of the i-th leaf/rectangle.

     * Example, for a rectangle i with i.Bbox={(5,10),(8,15)} and dimensionToProject=0 (=x). Then the return vector in position i has size |8-5|=3.
     The total size of the return vector is equal the number of leaves.
     return[i][0] is the sum of the coordinates in the bBox which have their 0-coordinate equal to 5,
     return[i][1] for he points with 0-coordinate equal to 6 etc. If dimensionToProject=1 then return_vector[i] has size |10-15|=5 .
     */
    static std::vector<std::vector<ValueType>> projection(
            const scai::lama::DenseVector<ValueType>& nodeWeights,
            const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
            const std::vector<IndexType>& dimensionToProject,
            const IndexType sideLen,
            Settings settings );


    /** @brief Projection for the non-uniform grid case. Coordinates must be Indextype.  \sa projection
        @param[in] coordinates The coordinates of the points.\p coordinates[i][d] the d-th coordinate of the i-th point.
     */
    template<typename T>
    static std::vector<std::vector<ValueType>> projectionNonUniform(
            const std::vector<std::vector<T>>& coordinates,
            const scai::lama::DenseVector<ValueType>& nodeWeights,
            const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
            const std::vector<IndexType>& dimensionToProject);


    /** @bried Iterative version to get the projection
    */
    template<typename T>
    static std::vector<std::vector<ValueType>> projectionIter(
            const std::vector<std::vector<T>>& coordinates,
            const scai::lama::DenseVector<ValueType>& nodeWeights,
            const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
            const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>>& allLeaves,
            const std::vector<std::vector<ValueType>>& hyperplanes,
            const std::vector<IndexType>& dimensionToProject);

    /*
    //TODO: there is not definition of this function; remove it - fix
    static std::vector<std::vector<ValueType>> projectionIter(
    //const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<std::vector<IndexType>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
    const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>>& allLeaves,
    const std::vector<std::vector<ValueType>>& hyperplanes,
    const std::vector<IndexType>& dimensionToProject);
    */
    template<typename T>
    static IndexType iterativeProjectionAndPart(
        std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
        const std::vector<std::vector<T>>& coordinates,
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        const std::vector<IndexType>& numCuts,
        Settings settings);

    /** @brief Partitions the given vector into k parts of roughly equal weights using a greedy approach.
     *
     * @param[in] array The 1 dimensional array of positive numbers to be partitioned.
     * @param[in] k The number of parts/blocks.
     * @return The first returned value is a vector of size k and holds the indices of each part/block: first part is from [return.first[0], return.first[1]) ( not inluding the weight of the last element), second from [return.first[1], return.first[2]) ets. Last part if from return.first[k-1] till return.first.size().
     * The second vector is of size k and holds the weights of each part.
     *
     * Example: input= [ 11, 6, 8, 1, 2, 4, 11, 1, 2] and k=3
     *                        |           |
     * return.first = [ 0, 2, 6]. Implies the partition: 0, 1 | 2, 3, 4, 5 | 6, 7, 8
     * return.second=[ 17, 15, 14]
     */
    static std::pair<std::vector<IndexType>,std::vector<ValueType>> partition1DGreedy( const std::vector<ValueType>& array, const IndexType k, Settings settings);

    /** @brief Partitions the given vector into k parts of roughly equal weights.
     *
     * @param[in] array The 1 dimensional array of positive numbers to be partitioned.
     * @param[in] k The number of parts/blocks.
     *
     * @return The first returned value is a vector of size k and holds the indices of each part/block: first part is from [return.first[0], return.first[1]) ( not inluding the weight of the last element), second from [return.first[1], return.first[2]) ets. Last part if from return.first[k-1] till return.first.size().
     * The second vector is of size k and holds the weights of each part.
     */
    static std::pair<std::vector<IndexType>,std::vector<ValueType>> partition1DOptimal(
                const std::vector<ValueType>& array,
                const IndexType k);

    /** @brief Searches if there is a partition of the input vector into k parts where the maximum weight of a part is <=target.
     * @param[in] input A vector of numbers.
     * @param[in] k The number of desired blocks
     * @param[in] target The maximum allowed weight for every block
     *
     * @return True if the partition can be done, false otherwise.
     */
    static bool probe(const std::vector<ValueType>& input, const IndexType k, const ValueType target);

    /** Searches if there is a partition of the input vector into k parts where the maximum weight of a part is <=target. If the desired partition cannot be achieved the vector returned must be ignored.
     *
     * @param[in] input A vector of numbers.
     * @param[in] k The number of desired blocks
     * @param[in] target The maximum allowed weight for every block
     *
     * @return return.first is true if the wanted partition can be done and false otherwise
     * return.second is a vector with the indices where the vector is partitioned. If the partition cannot be achieved then the vector must be ignored.
     */
    static std::pair<bool, std::vector<IndexType>> probeAndGetSplitters(const std::vector<ValueType>& input, const IndexType k, const ValueType target);

    /**Checks if the given point is in the given bounding box.
     *
     * @param[in] coords The coordinates of the input point.
     * @param[in] bBox The bounding box given as two vectors; one for the bottom point and one for the top point. For all dimensions i, should be: bBox.first(i)< bBox.second(i).
     *
     */
    template<typename T>
    static bool inBBox( const std::vector<T>& coords, const struct rectangle<ValueType>& bBox);

    /** Calculates the weight of the rectangle.
     *
     * @param[in] nodeWeights The weights for each point.
     * @param[in] bBox The bounding box/rectangle in which we wish to calculate the projection. For all dimensions, bBox.bottom[i]< bBox.top[i]
     * @param[in] sideLen The length of the side of the whole uniform, square grid.
     *
     * @return The weight of the given rectangle.
     */
    static ValueType getRectangleWeight(
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        const struct rectangle<ValueType>& bBox,
        const IndexType sideLen,
        Settings settings);

    /* Overloaded version for the non-uniform grid that also takes as input the coordinates.
     * \overload
     */
    template<typename T>
    static ValueType getRectangleWeight(
        const std::vector<scai::lama::DenseVector<T>> &coordinates,
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        const struct rectangle<ValueType>& bBox,
        Settings settings);

    /* Overloaded version for the non-uniform grid with different type for coordinates.
     * \overload
     */
    template<typename T>
    static ValueType getRectangleWeight(
        const std::vector<std::vector<T>> &coordinates,
        const scai::lama::DenseVector<ValueType>& nodeWeights,
        const struct rectangle<ValueType>& bBox,
        Settings settings);


    static scai::lama::CSRSparseMatrix<ValueType> getBlockGraphFromTree_local( const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot );

    /** Function to transform a 1D index to 2D or 3D given the side length of the cubical grid.
     * For example, in a 4x4 grid, indexTo2D(1)=(0,1), indexTo2D(4)=(1,0) and indexTo2D(13)=(3,1)
     *
     * @param[in] ind The index to transform.
     * @param[in] sideLen The side length of the 2D or 3D cube/grid.
     * @param[in] dimensions The dimension of the cube/grid (either 2 or 3).
     * @return A vector containing the index for every dimension. The size of the vector is equal to dimensions.
     */
    template<typename T>
    static std::vector<T> indexToCoords(const IndexType ind, const IndexType sideLen, const IndexType dimensions);

    /** @brief Overloaded function for non-cubic grids: every side has different length. sideLen.size()=dimensions
    **/
    template<typename T>
    static std::vector<T> indexToCoords(const IndexType ind, const std::vector<IndexType> sideLen );

private:
    template<typename T>
    static std::vector<T> indexTo2D(IndexType ind, IndexType sideLen);

    template<typename T>
    static std::vector<T> indexTo3D(IndexType ind, IndexType sideLen);

    template<typename T>
    static std::vector<T> indexTo2D(IndexType ind, std::vector<IndexType> sideLen);

    template<typename T>
    static std::vector<T> indexTo3D(IndexType ind, std::vector<IndexType> sideLen);
};

} //namespace ITI
