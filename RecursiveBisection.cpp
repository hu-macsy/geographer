/*
 * RecursiveBisection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "RecursiveBisection.h"

namespace ITI {

template<typename IndexType, typename ValueType>
void RecursiveBisection<IndexType, ValueType>::getPartition(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings) {


}
//---------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
scai::dmemo::CommunicatorPtr RecursiveBisection<IndexType, ValueType>::bisection(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings) {


}
//---------------------------------------------------------------------------------------

template void RecursiveBisection<int, double>::getPartition(scai::lama::DenseVector<int>& nodeWeights, int k,  scai::dmemo::CommunicatorPtr comm, Settings settings);

template scai::dmemo::CommunicatorPtr RecursiveBisection<int, double>::bisection(scai::lama::DenseVector<int>& nodeWeights, int k, scai::dmemo::CommunicatorPtr comm, Settings settings);


};