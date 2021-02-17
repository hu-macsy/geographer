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

#include <scai/dmemo/BlockDistribution.hpp>

#include "Metrics.h"
#include "CommTree.h"
#include "system_info.h"


namespace ITI {

static const IndexType HARD_TIME_LIMIT= 600;     // hard limit in seconds to stop execution if exceeded

/** @brief Class for external partitioning tools like zoltan and metis.
*/
template <typename IndexType, typename ValueType>
class Wrappers {

public:

    /** Returns a partition with one of the supported tools
     *
     * @param[in] graph The adjacency matrix of the graph of size NxN, where N is the number of vertices of the graph.
     * @param[in] coordinates The coordinates of the mesh. Not always needed by all tools.
     * @param[in] nodeWeights Weights for every node, used only is nodeWeightFlag is true.
     * @param[in] nodeWeightsFlag If true the node weights are used, if false they are ignored.
     * @param[in] tool One of the supported tools. 
     * @param[in] settings A Settings structure to pass various settings.
     * @param[out] metrics Structure to store/return timing info.
     *
     * @return A DenseVector of size N with the partition calculated: 0<= return[i] < k with the block that point i belongs to
     */
    virtual scai::lama::DenseVector<IndexType> partition(
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const Tool tool,
        const ITI::CommTree<IndexType,ValueType> &commTree,
        const struct Settings &settings,
        Metrics<ValueType> &metrics
    ) = 0; 


    /** Returns a partition with one of the supported tools based on the current distribution of the data.
     *
     * @param[in] graph The adjacency matrix of the graph of size NxN
     * @param[in] coordinates The coordinates of the mesh. Not always needed by all tools
     * @param[in] nodeWeights Weights for every node, used only is nodeWeightFlag is true
     * @param[in] nodeWeightsFlag If true the node weights are used, if false they are ignored
     * @param[in] tool One of the supported tools.
     * @param[in] settings A Settings structure to pass various settings
     * @param[out] metrics Structure to store/return timing info
     *
     * @return A DenseVector of size N with the partition calculated: 0<= return[i] < k with the block that point i belongs to
     */
    virtual scai::lama::DenseVector<IndexType> repartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const Tool tool,
        const struct Settings &settings,
        Metrics<ValueType> &metrics
    ) = 0;

};
} /* namespace ITI */
