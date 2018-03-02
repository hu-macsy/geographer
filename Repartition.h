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
#ifndef SETTINGS_H
#include "Settings.h"
#endif


//using namespace scai::lama;
//using scai::dmemo::Halo;
//using scai::dmemo::Halo;

namespace ITI {

	template <typename IndexType, typename ValueType>
	class ParcoRepart {
	public:
		
		static scai::lama::DenseVector<ValueType> sNW( const scai::dmemo::DistributionPtr distPtr, const IndexType seed, const Settings settings);
	};
}
