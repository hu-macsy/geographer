/*
 * Wrappers.h
 *
 *  Created on: 02.02.2018
 *      Author: tzovas
 */

#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include "Metrics.h"
#ifndef SETTINGS_H
#include "Settings.h"
#endif

#include <parmetis.h>

//for zoltan
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_InputTraits.hpp>

namespace ITI {

        
template <typename IndexType, typename ValueType>
class Wrappers {

public:

	static scai::lama::DenseVector<IndexType> metisWrapper (
		const scai::lama::CSRSparseMatrix<ValueType> &graph,
		const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, 
		const scai::lama::DenseVector<ValueType> &nodeWeights,
		int parMetisGeom,
		struct Settings &settings,
		struct Metrics &metrics);
		
	
	static scai::lama::DenseVector<IndexType> zoltanWrapper (
		const scai::lama::CSRSparseMatrix<ValueType> &graph,
		const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
		const scai::lama::DenseVector<ValueType> &nodeWeights,
		std::string algo,
		struct Settings &settings,
		struct Metrics &metrics);
		
};
} /* namespace ITI */
