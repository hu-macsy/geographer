#pragma once

#include <vector>

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <scai/dmemo/HaloExchangePlan.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>

#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>

#include "Settings.h"
#include "Metrics.h"
#include "CommTree.h"

/** @brief Global namespace that includes all classes.
*/

namespace ITI {

using namespace scai::lama;
using scai::dmemo::HaloExchangePlan;

/** @brief Main class to partition a graph.
*/

template <typename IndexType, typename ValueType>
class ParcoRepart {
public:

    /**
     * Partitions the given input graph
     * If the number of blocks is equal to the number of processes, graph, coordinates and weights are redistributed according to the partition.
     *
     * @param[in,out] input Adjacency matrix of the input graph with n vertices. numRows=numColumns=n
     * @param[in,out] coordinates Coordinates of input points, coordinates.size()=n
     * @param[in,out] nodeWeights Optional node weights. nodeWeights.size()=n
     * @param[in] settings Settings struct
     * @param[out] metrics struct into which time measurements are written
     *
     * @return partition Distributed DenseVector of length n, partition[i] contains the block ID of node i.  partition[i]<\p settings.numBlocks
     */
    static DenseVector<IndexType> partitionGraph(
        CSRSparseMatrix<ValueType> &input,
        std::vector<DenseVector<ValueType>> &coordinates,
        std::vector<DenseVector<ValueType>> &nodeWeights,
        struct Settings settings,
        struct Metrics& metrics);

    /**
     * Repartitions the given input graph, starting from a given previous partition
     * If the number of blocks is equal to the number of processes, graph and coordinates are redistributed according to the partition.
     *
     * @param[in,out] input Adjacency matrix of the input graph
     * @param[in,out] coordinates Coordinates of input points
     * @param[in,out] nodeWeights Optional node weights.
     * @param[in] previous The previous partition
     * @param[in] commTree A tree describing the communication graph.
     * @param[in] settings Settings struct
     * @param[out] metrics Struct into which time measurements are written
     *
     * @return partition Distributed DenseVector of length n, partition[i] contains the block ID of node i
     */
    static DenseVector<IndexType> partitionGraph(
        CSRSparseMatrix<ValueType> &input,
        std::vector<DenseVector<ValueType>> &coordinates,
        std::vector<DenseVector<ValueType>> &nodeWeights,
        DenseVector<IndexType>& previous,
        CommTree<IndexType,ValueType> commTree,
        struct Settings settings,
        struct Metrics& metrics);

    /**
     * Wrapper without node weights.
     */
    static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, struct Settings settings, struct Metrics& metrics);

    /**
     * Wrapper without node weights and no metrics
     */
    static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, struct Settings settings);

    /**
     * Wrapper without metrics.
     */
    static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, std::vector<DenseVector<ValueType>> &nodeWeights, struct Settings settings);

    /**
    *	Wrapper with block sizes.

    @param[in] blockSizes \p blockSizes[i][j] is the wanted target weight of the i-th weight for the j-th block.
    blockSizes.size() = number of weights (remember: vertices can have multiple weights),
    blockSizes[i].size() = number of blocks (=\p settings.numBlocks)
    */
    static DenseVector<IndexType> partitionGraph(
        CSRSparseMatrix<ValueType> &input,
        std::vector<DenseVector<ValueType>> &coordinates,
        std::vector<DenseVector<ValueType>> &nodeWeights,
        std::vector<std::vector<ValueType>> &blockSizes,
        Settings settings,
        struct Metrics& metrics);


    /**
    * Wrapper for metis-like input. \p numPEs is the number of processors (comm->getSize()),
    \p localN  is the number of local points

    * vtxDist, size=numPEs,  is a replicated array, it is the prefix sum of the number of nodes per PE
    		e.g.: [0, 15, 25, 50], PE0 has 15 vertices, PE1 10 and PE2 25

    * xadj, size=localN+1, (= IA array of the CSR sparse matrix format), is the prefix sum of the degrees
    		of the local nodes, i.e., how many non-zero values the row has.

    * adjncy, size=localM (number of local edges = the JA array), contains numbers >0 and < N, each
    		number is the global id of the neighboring vertex

    * localM, the number of local edges

    * vwgt, size=localN, array for the node weights

    * ndims, the dimensions of the coordinates

    * xyz, size=ndims*locaN, the coordinates for every vertex. For vertex/point i, its coordinates are
    		in xyz[ndims*i], xyz[ndims*i+1], ... , xyz[ndims*i+ndims]

    \sa <a href="glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf">metis manual</a>.

    \sa <a href="https://en.wikipedia.org/wiki/Sparse_matrix">CSR matrix</a>
    */
    static std::vector<IndexType> partitionGraph(
        IndexType *vtxDist, IndexType *xadj, IndexType *adjncy, IndexType localM,
        IndexType *vwgt, IndexType ndims, ValueType *xyz,
        Settings  settings, Metrics& metrics);

    /**
     * @brief Partition a point set using the Hilbert curve, only implemented for equal number of blocks and processes.
     *
     * @param coordinates Coordinates of the input points
     * @param settings Settings struct
     *
     * @return partition DenseVector, redistributed according to the partition
     */
    static DenseVector<IndexType> hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings);

    /** \overload
    @param[in] nodeWeights Weights for the points
    */
    /*
    * Get an initial partition using the Hilbert curve.
    * TODO: This currently does nothing and isn't used. Remove?
    */
    static DenseVector<IndexType> hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings);


    /**
     * Get an initial partition using the morton curve and measuring density per square.
     */
    static DenseVector<IndexType> pixelPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings);

    /** Given the block graph, creates an edge coloring of the graph and returns a communication
     *  scheme based on the coloring
     *  TODO: This method redistributes the graph. Maybe it should not.
     *
     * @param[in] adjM The adjacency matrix of a graph.
     * @return std::vector.size()= number of colors used for coloring the graph. If D is the
     *  maximum number of edges for a node, then numbers of colors is D or D+1.
     *  vector[i].size()= number of nodes in the graph = adjM.numRows = adjMnumCols.
     *  return[i][j] = k : in round i, node j talks with node k. Must also be that return[i][k] = j.
     *  Inactive nodes have their own rank: rank[i][j] = j.
     */
    static std::vector<DenseVector<IndexType>> getCommunicationPairs_local( CSRSparseMatrix<ValueType> &adjM, Settings settings);

    //private:

    /** Finds the IDs of all the neighbors of this pixel.
    @param[in] thisPixel The pixel ID to find its neighbors.
    @param[in] sideLen The side length of a uniform, homogeneous grid.
    @param[in] dimensions The dimensions of the grid
    @return The IDs of \p thisPixel neighboring pixels in the grid.
    */
    static std::vector<IndexType> neighbourPixels(const IndexType thisPixel,const IndexType sideLen, const IndexType dimensions);

private:


    /** The initial (geometric) partition of a graph. 

    Attention, for metis, and methods using the multilevel approach, the term 'initial partition' usually refers to the first
    partition in the coarsest level of the multilevel cycle. Here, we obtain an initial partition without coarsening, by
    using the coordinates of the graph.
    */
    static DenseVector<IndexType> initialPartition(
        CSRSparseMatrix<ValueType> &input,
        std::vector<DenseVector<ValueType>> &coordinates,
        std::vector<DenseVector<ValueType>> &nodeWeights,
        DenseVector<IndexType>& previous,
        CommTree<IndexType,ValueType> commTree,
        scai::dmemo::CommunicatorPtr comm,
        struct Settings settings,
        struct Metrics& metrics); 
};
} //namespace ITI
