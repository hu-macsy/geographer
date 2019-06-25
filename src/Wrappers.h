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


namespace ITI {

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
    static scai::lama::DenseVector<IndexType> partition(
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        Tool tool,
        struct Settings &settings,
        struct Metrics &metrics
    );


    /** @brief Version for tools that do not need the graph as input.
     */
    static scai::lama::DenseVector<IndexType> partition(
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        Tool tool,
        struct Settings &settings,
        struct Metrics &metrics
    );

    /** Returns a partition with one of the supported tools based on the current distribution of the data.
     *
     * @param[in] graph The adjacency matrix of the graph of size NxN
     * @param[in] coordinates The coordinates of the mesh. Not always needed by all tools
     * @param[in] nodeWeights Weights for every node, used only is nodeWeightFlag is true
     * @param[in] nodeWeightsFlag If true the node weigts are used, if false they are ignored
     * @param[in] tool One of the supported tools.
     * @param[in] settings A Settings structure to pass various settings
     * @param[out] metrics Structure to store/return timing info
     *
     * @return A DenseVector of size N with the partition calcualted: 0<= return[i] < k with the block that point i belongs to
     */
    static scai::lama::DenseVector<IndexType> repartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        Tool tool,
        struct Settings &settings,
        struct Metrics &metrics
    );


private:

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
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
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
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        struct Settings &settings,
        struct Metrics &metrics);


    // zoltan wrappers

    static scai::lama::DenseVector<IndexType> zoltanPartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        struct Settings &settings,
        struct Metrics &metrics);

    static scai::lama::DenseVector<IndexType> zoltanRepartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        struct Settings &settings,
        struct Metrics &metrics);

    static scai::lama::DenseVector<IndexType> zoltanCore (
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        bool repart,
        struct Settings &settings,
        struct Metrics &metrics);

};
} /* namespace ITI */
