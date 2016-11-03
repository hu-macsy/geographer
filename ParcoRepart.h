#pragma once

#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>


#include <scai/lama/Vector.hpp>


using namespace scai::lama;

namespace ITI {
	template <typename IndexType, typename ValueType>
	class ParcoRepart {
		public:
			/**
	 		* Partitions the given input graph with a space-filling curve and (in future versions) local refinement
	 		*
	 		* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		*	In principle arbitrary dimensions, currently only 2 are supported. 
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		* @param[in] k Number of desired blocks
	 		* @param[in] epsilon Tolerance of block size
	 		*
	 		* @return DenseVector with the same distribution as the input matrix, at position i is the block node i is assigned to
	 		*/
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, DenseVector<ValueType> &coordinates, IndexType dimensions,	IndexType k,  double epsilon = 0.05);

			/**
			* Returns the minimum distance between two neighbours
			*
			* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		*
	 		* @return The spatial distance of the closest pair of neighbours
			*/
			static ValueType getMinimumNeighbourDistance(const CSRSparseMatrix<ValueType> &input, const DenseVector<ValueType> &coordinates, IndexType dimensions);

			/**
			* Accepts a point and calculates where along the hilbert curve it lies.
			*
			* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		* @param[in] index The index of the points whose hilbert index is desired
	 		* @param[in] recursionDepth The number of refinement levels the hilbert curve should have
	 		* @param[in] minCoords A vector containing the minimal value for each dimension
	 		* @param[in] maxCoords A vector containing the maximal value for each dimension
			*
	 		* @return A value in the unit interval [0,1]
			*/
			static ValueType getHilbertIndex(const DenseVector<ValueType> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
			 const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);

			/**
			* Performs local refinement of a given partition
			*
	 		* @param[in] input Adjacency matrix of the input graph
			* @param[in] part Previous partition
	 		* @param[in] k Number of desired blocks. Must be the same as in the previous partition.
	 		* @param[in] epsilon Tolerance of block size
			*
			* @return The difference in cut weight between this partition and the previous one
			*/
			static ValueType fiducciaMattheysesRound(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon);
	};
}
