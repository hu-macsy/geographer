#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <scai/dmemo/Halo.hpp>
#include <scai/dmemo/HaloBuilder.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>

#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>
#include <scai/tracing.hpp>

#include "AuxiliaryFunctions.h"
#include "GraphUtils.h"
#include "Metrics.h"

//WARNING and TODO: error if Wrappers.h is the last include !!
#include "Wrappers.h"


//using namespace scai::lama;
//using scai::dmemo::Halo;
//using scai::dmemo::Halo;

namespace ITI {

	template <typename IndexType, typename ValueType>
	class Repartition {
	public:
		
		/** Create node weights.
		 * @param[in] coordinates The input coordinates.
		 * @param[in] seed A random seed.
		 * @param[in] diverg Divergence, how different are the node weigths. For 0 all weights are 1, the larger
		 * the value more diverse the node weights.
		 * @param[in] dimensions The dimension of the coordinates.
		 */
		static scai::lama::DenseVector<ValueType> sNW(  const std::vector<scai::lama::DenseVector<ValueType> >& coordinates, const IndexType seed, const ValueType diverg, const IndexType dimensions);
		
		
		static void getImbalancedDistribution(
			scai::lama::CSRSparseMatrix<ValueType> &graph,
			std::vector<scai::lama::DenseVector<ValueType>> &coords, 
			scai::lama::DenseVector<ValueType> &nodeWeights,
			ITI::Tool tool,
			struct Settings &settings,
			struct Metrics &metrics);
	};
	
}
