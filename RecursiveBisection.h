#pragma once

#include <scai/lama.hpp>
#include <scai/lama/Vector.hpp>

#include "Settings.h"



namespace ITI {

    template <typename IndexType, typename ValueType>
    class RecursiveBisection{
        public:

        void getPartition(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings);

        scai::dmemo::CommunicatorPtr bisection(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings);
    };

}