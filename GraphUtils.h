/*
 * GraphUtils.h
 *
 *  Created on: 29.06.2017
 *      Author: moritzl
 */

#ifndef GRAPHUTILS_H_
#define GRAPHUTILS_H_

#include <scai/lama/matrix/CSRSparseMatrix.hpp>

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
ValueType computeImbalance(const scai::lama::DenseVector<IndexType> &part, IndexType k, const scai::lama::DenseVector<IndexType> &nodeWeights = {});

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
 * Returns a vector of global indices of nodes which are local on this process, but have neighbors that are node.
 * No communication required, iterates once over the local adjacency matrix
 * @param[in] input Adjacency matrix of the input graph
 */
template<typename IndexType, typename ValueType>
std::vector<IndexType> getNodesWithNonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input);

/**
 * Computes a list of global IDs of nodes which are adjacent to nodes local on this processor, but are themselves not local.
 */
template<typename IndexType, typename ValueType>
std::vector<IndexType> nonLocalNeighbors(const scai::lama::CSRSparseMatrix<ValueType>& input);

}

} /* namespace ITI */
#endif /* GRAPHUTILS_H_ */
