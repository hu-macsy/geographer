#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <scai/dmemo/Halo.hpp>
#include <scai/dmemo/HaloBuilder.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>

#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>
#include <scai/tracing.hpp>

#include "AuxiliaryFunctions.h"
#include "GraphUtils.h"
#include "Metrics.h"

//WARNING and TODO: error if Wrappers.h is the last include !!
#include "Wrappers.h"


namespace ITI {

/** @brief Class for testing/benchmarking repartition
*/
template <typename IndexType, typename ValueType>
class Repartition {
public:

    /** Create node weights.
     * @param[in] coordinates The input coordinates.
     * @param[in] seed A random seed.
     * @param[in] diverg Divergence, how different are the node weigths. For 0 all weights are 1, the larger
     * the value more diverse the node weights.
     * @param[in] dimensions The dimension of the coordinates.
     * @return The weights of the nodes. This has the same distribution as the coordinates.
     */
    static scai::lama::DenseVector<ValueType> setNodeWeights(  const std::vector<scai::lama::DenseVector<ValueType> >& coordinates, const IndexType seed, const ValueType diverg, const IndexType dimensions);

    /** Given the input (graph, coordinates, node weights) and a tool, it produces and imbalanced
     * distribution of the input by repeatedly partitioning with the given tool, with different parameters,
     * until it gets an imbalanced partition. After that, all input data are redistributed based on that
     * imbalanced partition.
     *
     * @param[in,out] graph The adjacency matrix of the graph.
     * @param[in,out] coordinates The input coordinates.
     * @param[in,out] nodeWeights The weights for each point/vertex.
     * @param[in] tool The tool to partition with.
     * @param[in] setting A settings struct passing various arguments.
     * @param[out] metrics A metrics struct to store different metrics.
     */

    static void getImbalancedDistribution(
        scai::lama::CSRSparseMatrix<ValueType> &graph,
        std::vector<scai::lama::DenseVector<ValueType>> &coords,
        scai::lama::DenseVector<ValueType> &nodeWeights,
        ITI::Tool tool,
        struct Settings &settings,
        struct Metrics &metrics);
};

}
