/*
 * RecursiveKMeans.cpp
 *
 *  Created on: 07.11.2018
 *      Author: tzovas
 */

#include <assert.h>
#include <queue>
#include <unordered_set>
#include <chrono>

#include "HierarchicalKMeans.h"

namespace ITI {

template<typename IndexType, typename ValueType>
static scai::lama::DenseVector<IndexType> computePartition( 
	const std::vector<DenseVector<ValueType>> &coordinates, 
	const DenseVector<ValueType> &nodeWeights, 
	const std::vector<IndexType> &blockSizes, 
	std::vector<std::vector<ValueType> > centers, 
	scai::dmemo::CommunicatorPtr comm,
	const Settings settings,
	struct Metrics &metrics ){


	return scai::lama::DenseVector<IndexType>(10,0);
}



}//namespace ITI