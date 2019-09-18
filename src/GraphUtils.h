/*
 * @file GraphUtils.h
 *
 *  @authors: Moritz von Looz, Charilaos Tzovas
 *  @date 29.06.2017
 */

#pragma once

#include <set>
#include <tuple>
#include <vector>

#include <scai/lama/matrix/CSRSparseMatrix.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GeneralDistribution.hpp>

#include "Settings.h"

namespace ITI {

/** @brief This class holds all the functionality related to graphs.
*/

template <typename IndexType, typename ValueType>
class GraphUtils {
public:
    /**
     * Reindexes the nodes of the input graph to form a BlockDistribution. No redistribution of the graph happens, only the indices are changed.
     * After this method is run, the input graph has a BlockDistribution.
     *
     * @param[in,out] the graph
     *
     * @return A block-distributed vector containing the old indices.
     */
    static scai::lama::DenseVector<IndexType> reindex(scai::lama::CSRSparseMatrix<ValueType> &graph);

    /**
     * @brief Perform a BFS on the local subgraph.
     *
     * @param[in] graph (may be distributed)
     * @param[in] u local index of starting node
     *
     * @return vector with (local) distance to u for each local node
     */
    static std::vector<IndexType> localBFS(const scai::lama::CSRSparseMatrix<ValueType> &graph, IndexType u);

    /**
    	@brief Single source shortest path, a Dijkstra implementation

    	* @param[in] graph (may be distributed)
    	* @param[in] u local index of starting node
    	* @param[out] predecessor A vector with the predecessor of a node in
    	the shortest path.

    	Example, predecessor[4] = 5 means that in the shortest path from u to 4,
    	the previous vertex is 5. If also, predecessor[5] = 10 and predecessor[10] = u
    	then the shortest path to 4 is: u--10--5--4

    	* @return vector with (local) distance to u for each local node
    */
    static std::vector<ValueType> localDijkstra(const scai::lama::CSRSparseMatrix<ValueType> &graph, const IndexType u, std::vector<IndexType>& predecessor);

    /**
     * @brief Computes the diameter of the local subgraph using the iFUB algorithm.
     *
     * @param[in] graph
     * @param[in] u local index of starting node. Should be central.
     * @param[in] lowerBound of diameter. Can be 0. A good lower bound might speed up the computation
     * @param[in] k tolerance Algorithm aborts if upperBound - lowerBound <= k
     * @param[in] maxRounds Maximum number of diameter rounds.
     *
     * @return new lower bound
     */
    static IndexType getLocalBlockDiameter(const scai::lama::CSRSparseMatrix<ValueType> &graph, const IndexType u, IndexType lowerBound, const IndexType k, IndexType maxRounds);

    /**
     * This method takes a (possibly distributed) partition and computes its global cut.
     *
     * @param[in] input The adjacency matrix of the graph.
     * @param[in] part The partition vector for the input graph.
     * @param[in] weighted If edges are weighted or not.

     * @return The value of the cut, i.e., the weigh of the edges if the edges have weights or the number of edges otherwise.
     */
    static ValueType computeCut(const scai::lama::CSRSparseMatrix<ValueType> &input, const scai::lama::DenseVector<IndexType> &part, bool weighted = false);

    /**
     * This method takes a (possibly distributed) partition and computes its imbalance.
     * The number of blocks is also a required input, since it cannot be guessed accurately from the partition vector if a block is empty.
     *
     * @param[in] part partition
     * @param[in] k number of blocks in partition.
     * @param[in] nodeWeights The weight of every point/vertex if available
     * @param[in] blockSizes The optimum size/weight for every block

     * @return The value of the maximum imbalance; this is calculated as: (maxBlockWeight-optimumBlockWeught)/optimumBlockWeight
     */

//TODO: this does not include the case where we can have different
//blocks sizes but no node weights; adapt
    static ValueType computeImbalance(
        const scai::lama::DenseVector<IndexType> &part,
        IndexType k,
        const scai::lama::DenseVector<ValueType> &nodeWeights = scai::lama::DenseVector<ValueType>(0,0),
        const std::vector<ValueType> &blockSizes = std::vector<ValueType>(0,0));

    /**
     * @brief Builds a halo containing all non-local neighbors.
     *
     * @param[in] input Adjacency Matrix
     *
     * @return HaloExchangePlan
     */
    static scai::dmemo::HaloExchangePlan buildNeighborHalo(const scai::lama::CSRSparseMatrix<ValueType> &input);

    /**
     * Returns true if the node identified with globalID has a neighbor that is not local on this process.
     * Since this method acquires reading locks on the CSR structure, it might be expensive to call often
     *
     * @param[in] input The adjacency matrix of the graph.
     * @param[in] globalID The global ID of the vertex to be checked
     * @return True if the vertex has a neighbor that resides in a different PE, false if all the neighbors are local.
     *
     */
    static bool hasNonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType> &input, IndexType globalID);

    /**
     * Returns a vector of global indices of nodes which are local on this process, but have neighbors that are not local.
     * These non-local neighbors may or may not be in the same block.
     * No communication required, iterates once over the local adjacency matrix

     * @param[in] input Adjacency matrix of the input graph

     * @return The global vertex ids of the local vertices that have non-local neighboring vertices.
     */
    static  std::vector<IndexType> getNodesWithNonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input);

    /**
     * Returns a vector of global indices of nodes which are local on this process, but have neighbors that are not local.
     * This method differs from the other method with the same name by accepting a list of candidates.
     * Only those are checked for non-local neighbors speeding up the process.
     *
     * @param[in] input Adjacency matrix of the input graph
     * @param[in] candidates A list of non-local neighbor vertices; only those are checked as possible non-local neighbors of the local vertices.
     *
     * @return vector of nodes with non-local neighbors
     */
    static std::vector<IndexType> getNodesWithNonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input, const std::set<IndexType>& candidates);

    /**
     * Computes a list of global IDs of nodes which are adjacent to nodes local on this processor, but are themselves not local.
     * @param[in] input Adjacency matrix of the input graph
     * @return The vector with the global IDs
     */
    static std::vector<IndexType> nonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input);

    /** Get the borders nodes of each block. Border node: one that has at least one neighbor in different block.

    * @param[in] adjM Adjacency matrix of the input graph
    * @param[in] part A partition of the graph.

    * @return A distributed vector that contains 0 or 1: of retunr[i]=1 then the i-th vertex is a border node, if 0 it is not.
    */
    static scai::lama::DenseVector<IndexType> getBorderNodes( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part);

    /** Returns two vectors each of size k: the first contains the number of border nodes per block and the second one the number of inner nodes per blocks.

     * @param[in] adjM Adjacency matrix of the input graph
     * @param[in] part A partition of the graph.
     * @param[in] settings Settings struct

     * @return First vector is the number of border vertices per block, second vector is the number of inner vertices per block.
     Since these two are disjoint, ret.first[i]+ret.second[i] = size of block i.
     */
    static std::pair<std::vector<IndexType>,std::vector<IndexType>> getNumBorderInnerNodes( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, Settings settings);

    /** Computes the communication volume for every block.
     * @param[in] adjM Adjacency matrix of the input graph
     * @param[in] part A partition of the graph.
     * @param[in] settings Settings struct

     * @return Vector of size k (=numBlocks) with the communication volume for every part. Note that the vector is replicated in every PE.
    */
// TODO: Should the result is gathered in the root PE and not be replicated?
    static std::vector<IndexType> computeCommVolume( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, Settings settings );

    /**Computes the communication volume, boundary and inner nodes in one pass to save time.
     *
     * @return A tuple with three vectors each of size numBlocks: first vector is the communication volume, second is the number of boundary nodes and third
     * the number of inner nodes per block.

     @sa  getNumBorderInnerNodes(), computeCommVolume()
     */
    static std::tuple<std::vector<IndexType>, std::vector<IndexType>, std::vector<IndexType>> computeCommBndInner( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, Settings settings );


    /** Builds the block (aka, communication) graph of the given partition. Every vertex corresponds to a block and two vertices u and v
     * are adjacent in the block graph if there is a vertex in (block) u and one in (block) v that are adjacent in the input graph.
     * Creates an HArray that is passed around in numPEs (=comm->getSize()) rounds and every time
     * a processor writes in the array its part.
     *
     * The returned matrix is replicated in all PEs.
     *
     * @param[in] adjM The adjacency matrix of the input graph.
     * @param[in] part The partition of the input graph.
     * @param[in] k Number of blocks.
     *
     * @return The adjacency matrix of the block graph; the returned graph has as many vertices as the blocks of the partition,
     * i.e., return.numRRows()=part.max()-1
     */
    static scai::lama::CSRSparseMatrix<ValueType> getBlockGraph( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k);

    /** Constructs the block (aka, communication)  graph of the partition (@sa getBlockGraph()).
    The difference with getBlockGraph() is that now, every PE sends its local adjacent list to a root PE, the block is constructed
    there and then it is broadcast back to all PEs. This avoids the k*k space needed in getBlockGraph().

    Input parameter and return value are the same as for getBlockGraph().
    */

    static scai::lama::CSRSparseMatrix<ValueType>  getBlockGraph_dist( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k);

    /** @brief Get the maximum degree of a graph.
    */
    static IndexType getGraphMaxDegree( const scai::lama::CSRSparseMatrix<ValueType>& adjM);


    /** Computes maximum communication, i.e., max degree of the block graph, and total communication, i.e., sum of all edges
    of the block graph.

      @return first: maximum communication = max degree of the block graph.

       second: total communication = sum of all edges of the block graph.
     */
    static std::pair<IndexType, IndexType> computeBlockGraphComm( const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType> &part);

    /** @deprecated Not used anymore, use computeCommVolume().

      Compute maximum and total communication volume= max and sum of number of border nodes
     * WARNING: this works properly only when k=p and the nodes are redistributed so each PE owns nodes from one block.
     */
    static std::pair<IndexType,IndexType> computeCommVolume_p_equals_k( const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType> &part);

    /**Returns the process graph. Every processor traverses its local part of adjM and for every
     * edge (u,v) that one node, say u, is not local it gets the owner processor of u.
     * The returned graph is distributed with a BLOCK distribution where each PE owns one row.
     *
     * @param[in] adjM The adjacency matrix of the input graph.
     * @return A [#PE x #PE] adjacency matrix of the process graph, distributed with a Block distribution
     */
    static scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const scai::lama::CSRSparseMatrix<ValueType> &adjM);

    /**Returns the process graph, as calculated from the local halos.
     * The edges of the process graph has weights that indicate the number of vertices (not edges) that
     * are not local. eg: w(0,1)=10 means than PE 0 has 10 neighboring vertices in PE 1
     * and if w(1,0)=12, then PE 1 has 12 neighboring vertices in PE 0. The graph is not symmetric.
     *
     * @param halo HaloExchangePlan objects in which all non-local neighbors are present
     * @return A [#PE x #PE] adjacency matrix of the process graph, distributed with a Block distribution
     */
    static scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const scai::dmemo::HaloExchangePlan& halo);

    /** Convert a set of unweighted adjacency lists into a CSR matrix
     *
     * @param[in] adjList For each node, a possibly empty set of neighbors
     * @return The distributed adjacency matrix
     */
    static scai::lama::CSRSparseMatrix<ValueType> getCSRmatrixFromAdjList_NoEgdeWeights( const std::vector<std::set<IndexType>>& adjList);

    /** Given a list of edges (i.e., a graph in edge list format), convert to CSR adjacency matrix.
     Go over the vector of the local edges, sort the edges, construct the local part of CSR sparse matrix and
     build the global matrix in the end.
     * @param[in] edgeList The local list of edges for this PE; edgeList[i].first is one vertex of the edge and .second the other.
     * @return The distributed adjacency matrix.
     */
    static scai::lama::CSRSparseMatrix<ValueType> edgeList2CSR( std::vector< std::pair<IndexType, IndexType>>& edgeList );


    /** Given a CSR sparse matrix, it calculates its edge list representations.
    	For every tuple, the first two numbers are the vertex IDs for this edge and the	third is the edge weight.
    	@warning Assumes the graph is undirected; it will implicitly duplicate all edges turning a directed graph to undirected.

    	@param[in] adjM The input graph (ignores direction)
    	@param[out] maxDegree The maximum degree of the graph
    	@return The local part of an edge list representation. return.size()==adjM.getLocalNumValues()/2. Global size==graph.getNumValues()/2.
    */
    static std::vector<std::tuple<IndexType,IndexType,ValueType>> CSR2EdgeList_local(const scai::lama::CSRSparseMatrix<ValueType>& adjM, IndexType& maxDegree=0);

    /** @brief Construct the Laplacian of the input matrix. May contain parallel communication.
     *
     * @param adjM Input matrix, must have a (general) block distribution or be replicated.
     *
     * @return laplacian with same distribution as input
     */
    static scai::lama::CSRSparseMatrix<ValueType> constructLaplacian(const scai::lama::CSRSparseMatrix<ValueType>& adjM);

    /** @brief Construct a replicated projection matrix for a fast Johnson-Lindenstrauß-Transform
     *
     * @param epsilon Desired accuracy of transform
     * @param n
     * @param origDimension Dimension of original space
     *
     * @return FJLT matrix
     */
    static scai::lama::CSRSparseMatrix<ValueType> constructFJLTMatrix(ValueType epsilon, IndexType n, IndexType origDimension);

    /** @brief Construct a replicated Hadamard matrix
     *
     * @param d Dimension
     *
     * @return Hadamard matrix
     */
    static scai::lama::DenseMatrix<ValueType> constructHadamardMatrix(IndexType d);

    /** Returns an edge coloring of the graph using the Max Edge Color (MEC) algorithm
    <a href=https://www.sciencedirect.com/science/article/pii/S0304397510002483>MEC algorithm</a> by Bourgeois et al.
    The edges of the graph are grouped in colors so that heavier edge are group together.

     * @param[in] adjM The adjacency matrix of the input graph.
     * @param[out] colors The number of colors (i.e. groups) used.

     * @return A vector of size colors where ret[i] contains all the vertices that belong to color/group i.
    */
    static std::vector< std::vector<IndexType>> mecGraphColoring( const scai::lama::CSRSparseMatrix<ValueType> &adjM, IndexType &colors);


    /**
    * @brief Sum of weights of local outgoing edges. An edge (u,v) is an outgoing if u is
    local to this PE but v belongs to some other PE.

    * @param[in] input The adjacency matrix of the graph. Possibly distributed among PEs.
    * @param[in] weighted If the edges have weights or not.
    * @return The total weight of the edges that have an endpoint to another PE.
    */
    static ValueType localSumOutgoingEdges(const scai::lama::CSRSparseMatrix<ValueType> &input, const bool weighted);

    /** A random permutation of the elements in the given range. Randomly select elements and move them to the front.
     *
     * @param begin Begin of range
     * @param end End of range
     * @param num_random Number of selected elements
     */
//taken from https://stackoverflow.com/a/9345144/494085
    template<class BidiIter >
    static BidiIter FisherYatesShuffle(BidiIter begin, BidiIter end, size_t num_random) {
        size_t left = std::distance(begin, end);
        for (IndexType i = 0; i < num_random; i++) {
            BidiIter r = begin;
            std::advance(r, rand()%left);
            std::swap(*begin, *r);
            ++begin;
            --left;
        }
        return begin;
    }

    /**	Reordering a sequence of numbers from 0 to maxIndex.
     * The order is: maxIndex/2, maxIdnex/4, maxIndex*3/4, maxIndex/8, maxIndex*3/8, maxIndex*5/8, ...
     * @param[in] maxIndex The maximum number, the implied sequence is [0, 1, 2, 3, ..., maxIndex]
     * @return The permuted numbers. return.size()=maxIdnex and 0< return[i]< maxIndex.
     */
    static std::vector<IndexType> indexReorderCantor(const IndexType maxIndex);


}; //class GraphUtils

} /* namespace ITI */
