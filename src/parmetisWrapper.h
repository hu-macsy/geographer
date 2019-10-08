#pragma once
#include "Wrappers.h"

namespace ITI {

template <typename IndexType, typename ValueType>
class parmetisWrapper {

/** @brief Class for external partitioning parmetis tool.
*/

    /** Given the input (graph, coordinates, node weights) and a partition
    of the input, apply local refinement.

    The input and the partition DenseVector should have the same distribution.

    Returns the new, refined partition;
    */
    static scai::lama::DenseVector<IndexType> refine(
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const scai::lama::DenseVector<IndexType> partition,
        struct Settings &settings,
        Metrics<ValueType> &metrics
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
        Metrics<ValueType> &metrics);

//
//TODO: parMetis assumes that vertices are stores in a consecutive manner. This is not true for a
//      general distribution. Must reindex vertices for parMetis repartition
//
    static scai::lama::DenseVector<IndexType> metisRepartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        struct Settings &settings,
        Metrics<ValueType> &metrics);
}; //class parmetisWrapper
}//namespace ITI