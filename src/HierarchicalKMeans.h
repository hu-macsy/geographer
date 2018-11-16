/*
 * RecursiveKMeans.h
 *
 *  Created on: 07.11.2018
 *      Author: tzovas
 */

#pragma once

#include <vector>
#include <numeric>
#include <scai/lama/DenseVector.hpp>
#include <scai/tracing.hpp>
#include <chrono>

#include "quadtree/QuadNodeCartesianEuclid.h"
#include "Settings.h"
#include "Metrics.h"
#include "GraphUtils.h"
#include "HilbertCurve.h"
#include "AuxiliaryFunctions.h"


namespace ITI {

template <typename IndexType, typename ValueType>
class RecursiveKMeans {	
public:


static scai::lama::DenseVector<IndexType> computePartition( 
	const std::vector<DenseVector<ValueType>> &coordinates, 
	const DenseVector<ValueType> &nodeWeights, 
	//const std::vector<IndexType> &blockSizes,
	const Settings settings,
	struct Metrics &metrics );


static scai::lama::DenseVector<IndexType> computePartition( 
	const std::vector<DenseVector<ValueType>> &coordinates, 
	const DenseVector<ValueType> &nodeWeights, 
	const std::vector<IndexType> &blockSizes, 
	std::vector<std::vector<ValueType> > centers, 
	scai::dmemo::CommunicatorPtr comm,
	const Settings settings,
	struct Metrics &metrics );


};//class RecursiveKMeans

}//namespace ITI