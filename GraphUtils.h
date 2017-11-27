/*
 * GraphUtils.h
 *
 *  Created on: 29.06.2017
 *      Author: moritzl
 */

#ifndef GRAPHUTILS_H_
#define GRAPHUTILS_H_

#include <set>

#include <scai/lama/matrix/CSRSparseMatrix.hpp>
#include "Settings.h"

namespace ITI {

namespace GraphUtils {

template<typename IndexType, typename ValueType>
IndexType getFarthestLocalNode(const scai::lama::CSRSparseMatrix<ValueType> graph, std::vector<IndexType> seedNodes);

/**
 * This method takes a (possibly distributed) partition and computes its global cut.
 *
 * @param[in] input The adjacency matrix of the graph.
 * @param[in] part The partition vector for the input graph.
 * @param[in] weighted If edges are weighted or not.
 */
template<typename IndexType, typename ValueType>
ValueType computeCut(const scai::lama::CSRSparseMatrix<ValueType> &input, const scai::lama::DenseVector<IndexType> &part, bool weighted = false);

/**
 * This method takes a (possibly distributed) partition and computes its imbalance.
 * The number of blocks is also a required input, since it cannot be guessed accurately from the partition vector if a block is empty.
 *
 * @param[in] part partition
 * @param[in] k number of blocks in partition.
 */
template<typename IndexType, typename ValueType>
ValueType computeImbalance(const scai::lama::DenseVector<IndexType> &part, IndexType k, const scai::lama::DenseVector<ValueType> &nodeWeights = {});

/**
 * Builds a halo containing all non-local neighbors.
 */
template<typename IndexType, typename ValueType>
scai::dmemo::Halo buildNeighborHalo(const scai::lama::CSRSparseMatrix<ValueType> &input);

/**
 * Returns true if the node identified with globalID has a neighbor that is not local on this process.
 * Since this method acquires reading locks on the CSR structure, it might be expensive to call often
 */
template<typename IndexType, typename ValueType>
bool hasNonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType> &input, IndexType globalID);

/**
 * Returns a vector of global indices of nodes which are local on this process, but have neighbors that are not local. They non local neighbors may or may not be in the same block.
 * No communication required, iterates once over the local adjacency matrix
 * @param[in] input Adjacency matrix of the input graph
 */
template<typename IndexType, typename ValueType>
std::vector<IndexType> getNodesWithNonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input);

/**
 * Returns a vector of global indices of nodes which are local on this process, but have neighbors that are not local. They non local neighbors may or may not be in the same block.
 * No communication required, iterates once over the local adjacency matrix
 * @param[in] input Adjacency matrix of the input graph
 */
template<typename IndexType, typename ValueType>
std::vector<IndexType> getNodesWithNonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input, const std::set<IndexType>& candidates);

/**
 * Computes a list of global IDs of nodes which are adjacent to nodes local on this processor, but are themselves not local.
 */
template<typename IndexType, typename ValueType>
std::vector<IndexType> nonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input);

/** Get the borders nodes of each block. Border node: one that has at least one neighbor in different block.
*/
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> getBorderNodes( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part);

/* Returns two vectors each of size k: the first contains the number of border nodes per block and the second one the number of inner nodes per blocks.
 *
 */
template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>,std::vector<IndexType>> getNumBorderInnerNodes( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part);

/* Computes the communication volume for every block. 
 * TODO: Should the result is gathered in the root PE and not be replicated?
 * */
template<typename IndexType, typename ValueType>
std::vector<IndexType> computeCommVolume( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part /*, const IndexType root*/ );


/** Returns the edges of the block graph only for the local part. Eg. if blocks 1 and 2 are local
 * in this processor it finds the edge (1,2) ( and the edge (2,1)).
 * Also if the other endpoint is in another processor it finds this edge: block 1 is local, it
 * shares an edge with block 3 that is not local, this edge is found and returned.
 *
 * @param[in] adjM The adjacency matrix of the input graph.
 * @param[in] part The partition of the input graph.
 *
 * @return A 2 dimensional vector with the edges of the local parts of the block graph:
 * edge (u,v) is (ret[0][i], ret[1][i]) if block u and block v are connected.
 */
template<typename IndexType, typename ValueType>
std::vector<std::vector<IndexType>> getLocalBlockGraphEdges( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part);

/** Builds the block graph of the given partition.
 * Creates an HArray that is passed around in numPEs (=comm->getSize()) rounds and every time
 * a processor writes in the array its part.
 *
 * Not distributed.
 *
 * @param[in] adjM The adjacency matric of the input graph.
 * @param[in] part The partition of the input garph.
 * @param[in] k Number of blocks.
 *
 * @return The "adjacency matrix" of the block graph. In this version is a 1-dimensional array
 * with size k*k and [i,j]= i*k+j.
 */
template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getBlockGraph( const scai::lama::CSRSparseMatrix<ValueType> &adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k);


/** Get the maximum degree of a graph.
 * */
template<typename IndexType, typename ValueType>
IndexType getGraphMaxDegree( const scai::lama::CSRSparseMatrix<ValueType>& adjM);


/** first = Compute maximum communication = max degree of the block graph.
 *  second = Compute total communication = sum of all edges of the block graph.
 */
template<typename IndexType, typename ValueType>
std::pair<IndexType, IndexType> computeBlockGraphComm( const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType> &part, const IndexType k);

/** Compute maximum and total communication volume= max and sum of number of border nodes
 * WARNING: this works properly only when k=p and the nodes are redistributed so each PE owns nodes from one block.
 */
template<typename IndexType, typename ValueType>
std::pair<IndexType,IndexType> computeCommVolume_p_equals_k( const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType> &part);
    
/**Returns the processor graph. Every processor traverses its local part of adjM: and for every
 * edge (u,v) that one node, say u, is not local it gets the owner processor of u. The returned graph is distributed with a BLOCK distribution.
 *
 * @param[in] adjM The adjacency matrix of the input graph.
 * @return A [#PE x #PE] adjacency matrix of the processor graph.
 */
template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const scai::lama::CSRSparseMatrix<ValueType> &adjM);

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const scai::dmemo::Halo& halo);


template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> getCSRmatrixNoEgdeWeights( const std::vector<std::set<IndexType>> adjList);

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

} /*namespace GraphUtils*/

} /* namespace ITI */
#endif /* GRAPHUTILS_H_ */
