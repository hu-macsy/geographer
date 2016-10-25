#pragma once

#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>

#include <scai/lama/Vector.hpp>


using namespace scai::lama;

namespace ITI {
	template <typename IndexType>
	class ParcoRepart {
		public:
			/**
			* Partition the input graph given by the edge lists. 
			*/
			static DenseVector<IndexType> partitionGraph(Matrix &input, Vector &coordinates, IndexType dimensions,	IndexType k,  double epsilon = 0.05);
	};
}
