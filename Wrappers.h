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

#include "FileIO.h"
#include "Metrics.h"
#include "MeshGenerator.h"
#ifndef SETTINGS_H
#include "Settings.h"
#endif

#include <parmetis.h>


namespace ITI {

        
template <typename IndexType, typename ValueType>
class Wrappers {

public:
	
	//static void writeGraph (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);

	static scai::lama::DenseVector<IndexType> metisWrapper (
		const CSRSparseMatrix<ValueType> &graph,
		const std::vector<DenseVector<ValueType>> &coordinates, 
		const DenseVector<ValueType> &nodeWeights,
		int parMetisGeom,
		struct Settings &settings,
		struct Metrics &metrics);
		
};
} /* namespace ITI */
