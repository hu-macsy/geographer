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
	
	//metis wrapper
	
	/** Returns a partition with one of the metis methods
	 * 
	 * @param[in] graph The adjacency matrix of the graph
	 * @param[in] coordinates The coordinates of the mesh. Not always used by parMetis
	 * @param[in] nodeWeights Weights for every node, used only is nodeWeightFlag is true
	 * @param[in] nodeWeightsFlag If true the node weigts are used, if false they are ignored
	 * @param[in] parMetisGeom A flag for which version should be used: 0 is for ParMETIS_V3_PartKway which does not
	 * uses geometry, 1 is for ParMETIS_V3_PartGeom which uses both graph inforamtion and geometry and
	 *  2 is for ParMETIS_V3_PartSfc which uses only geometry.
	 * @param[in] settings A Settings structure to pass various settings
	 * @param[out] metrics Structure to store/return timing info
	 * 
	 * @return A DenseVector of size N with the partition calcualted: 0<= return[i] < k with the block that point i belongs to
	 */
	static scai::lama::DenseVector<IndexType> metisPartition (
		const scai::lama::CSRSparseMatrix<ValueType> &graph,
		const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, 
		const scai::lama::DenseVector<ValueType> &nodeWeights, 
		bool nodeWeightsFlag,
		int parMetisGeom,
		struct Settings &settings,
		struct Metrics &metrics);	
	
//
//TODO: parMetis assumes that vertices are stores in a consecutive manner. This is not true for a
//		general distribution. Must reindex vertices for parMetis repartition
//
	static scai::lama::DenseVector<IndexType> metisRepartition (
		const scai::lama::CSRSparseMatrix<ValueType> &graph,
		const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, 
		const scai::lama::DenseVector<ValueType> &nodeWeights, 
		bool nodeWeightsFlag,
		struct Settings &settings,
		struct Metrics &metrics);	

/*	//TODO: or create function like that?
	static scai::lama::DenseVector<IndexType> parmetisGraphPartition
	static scai::lama::DenseVector<IndexType> parmetisGeomPartition
	static scai::lama::DenseVector<IndexType> parmetisSfcPartition
	static scai::lama::DenseVector<IndexType> parmetiRepartition
*/	
	// zoltan wrappers
	
	static scai::lama::DenseVector<IndexType> zoltanPartition (
		const scai::lama::CSRSparseMatrix<ValueType> &graph,
		const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
		const scai::lama::DenseVector<ValueType> &nodeWeights, 
		bool nodeWeightsFlag,
		std::string algo,
		struct Settings &settings,
		struct Metrics &metrics);
	
	static scai::lama::DenseVector<IndexType> zoltanRepartition (
		const scai::lama::CSRSparseMatrix<ValueType> &graph,
		const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
		const scai::lama::DenseVector<ValueType> &nodeWeights, 
		bool nodeWeightsFlag,
		std::string algo,
		struct Settings &settings,
		struct Metrics &metrics);
	
private:
	
	static scai::lama::DenseVector<IndexType> zoltanCore (
		const scai::lama::CSRSparseMatrix<ValueType> &graph,
		const std::vector<scai::lama::DenseVector<ValueType>> &coords, 
		const scai::lama::DenseVector<ValueType> &nodeWeights, 
		bool nodeWeightsFlag,
		std::string algo,
		bool repart,
		struct Settings &settings,
		struct Metrics &metrics);
	
};
} /* namespace ITI */
