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
			* Partition the input graph given by the edge lists. 
			*/
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, DenseVector<ValueType> &coordinates, IndexType dimensions,	IndexType k,  double epsilon = 0.05);
			static ValueType getMinimumNeighbourDistance(const CSRSparseMatrix<ValueType> &input, const DenseVector<ValueType> &coordinates, IndexType dimensions);
			static ValueType getHilbertIndex(const DenseVector<ValueType> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
			 const std::vector<ValueType> &minCoord, const std::vector<ValueType> &maxCoords);
	};
}
