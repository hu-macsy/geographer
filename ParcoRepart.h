#pragma once

#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>

#include <scai/lama/Vector.hpp>
#include <scai/dmemo/Halo.hpp>

#include <set>

using namespace scai::lama;
using scai::dmemo::Halo;

namespace ITI {
	template <typename IndexType, typename ValueType>
	class ParcoRepart {
		public:
			/**
	 		* Partitions the given input graph with a space-filling curve and (in future versions) local refinement
	 		*
	 		* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		* @param[in] k Number of desired blocks
	 		* @param[in] epsilon Tolerance of block size
	 		*
	 		* @return DenseVector with the same distribution as the input matrix, at position i is the block node i is assigned to
	 		*/
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, IndexType k,  double epsilon = 0.05);

			static ValueType computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, bool ignoreWeights = true);

			/**
			 * This method takes a (possibly distributed) partition and computes its imbalance.
			 * The number of blocks is also a required input, since it cannot be guessed accurately from the partition vector if a block is empty.
			 *
			 * @param[in] part partition
			 * @param[in] k number of blocks in partition.
			 */
			static ValueType computeImbalance(const DenseVector<IndexType> &part, IndexType k);

			/**
			* Returns the minimum distance between two neighbours
			*
			* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		*
	 		* @return The spatial distance of the closest pair of neighbours
			*/
			static ValueType getMinimumNeighbourDistance(const CSRSparseMatrix<ValueType> &input, const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions);

			/**
			* Performs local refinement of a given partition
			*
	 		* @param[in] input Adjacency matrix of the input graph
			* @param[in,out] part Partition which is to be refined
	 		* @param[in] k Number of desired blocks. Must be the same as in the previous partition.
	 		* @param[in] epsilon Tolerance of block size
			*
			* @return The difference in cut weight between this partition and the previous one
			*/
			static ValueType replicatedMultiWayFM(const CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted = true);

			static std::vector<DenseVector<IndexType>> computeCommunicationPairings(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const DenseVector<IndexType> &blocksToPEs);

			/**
			 * Computes a list of global IDs of nodes which are adjacent to nodes local on this processor, but are themselves not local.
			 */
			static std::vector<IndexType> nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input);

			/**
			 * Builds a halo containing all matrix entries of non-local neighbors.
			 */
			static scai::dmemo::Halo buildMatrixHalo(const CSRSparseMatrix<ValueType> &input);

			/**
			 * Builds a halo containing all partition entries of non-local neighbors.
			 */
			static scai::dmemo::Halo buildPartHalo(const CSRSparseMatrix<ValueType> &input,  const DenseVector<IndexType> &part);

			/**
			 * Computes the border region within one block, adjacent to another block
			 * @param[in] input Adjacency matrix of the input graph
			 * @param[in] part Partition vector
			 * @param[in] thisBlock block in which the border region is required
			 * @param[in] otherBlock block to which the border region should be adjacent
			 * @param[in] depth Width of the border region, measured in hops
			 */
			static std::pair<std::vector<IndexType>, IndexType> getInterfaceNodes(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, IndexType thisBlock, IndexType otherBlock, IndexType depth);

			static ValueType distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted = true);

			static ValueType twoWayLocalFM(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage, const Halo &halo,
					std::set<IndexType> &firstregion,  std::set<IndexType> &secondregion,
					const std::set<IndexType> &firstDummyLayer, const std::set<IndexType> &secondDummyLayer,
					std::pair<IndexType, IndexType> blockSizes,	const std::pair<IndexType, IndexType> blockCapacities, ValueType epsilon, const bool unweighted = true);

			static IndexType localBlockSize(const DenseVector<IndexType> &part, IndexType blockID);
	};
}
