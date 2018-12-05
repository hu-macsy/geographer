#pragma once

#include <assert.h>
#include <queue>
#include <string>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <chrono>

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/tracing.hpp>

#include "ParcoRepart.h"
#include "LocalRefinement.h"
#include "Settings.h"

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::dmemo::Halo;

namespace ITI{
    
    template <typename IndexType, typename ValueType>
    class MultiLevel{
    public:
        /**
         * Implementation of the multi-level heuristic.
         * Performs coarsening, recursive call of multiLevelStep, uncoarsening and local refinement.
         * Only works if number of blocks is equal to the number of processes and the distribution is aligned with the partition.
         * Input data are redistributed to match the changed partition.
         *
         * @param[in,out] input Adjacency matrix of input graph
         * @param[in,out] part Partition
         * @param[in,out] nodeWeights Weights of input points
         * @param[in,out] coordinates of input points
         * @param[in] halo for non-local neighbors
         * @param[in] settings
         *
         * @return origin DenseVector that specifies for each element the original process before the multiLevelStep. Only needed when used to speed up redistribution.
         */
        static DenseVector<IndexType> multiLevelStep(scai::lama::CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, DenseVector<ValueType> &nodeWeights, std::vector<DenseVector<ValueType>> &coordinates, const Halo& halo, Settings settings, Metrics& metrics);

        /**
         * Given the origin array resulting from a multi-level step on a coarsened graph, compute where local elements on the current level have to be sent to recreate the coarse distribution on the current level.
         * Involves communication.
         * Used in uncoarsening to accelerate redistribution.
         *
         * @param[in] coarseOrigin
         * @param[in] fineToCoarseMap
         *
         * @return fineTargets
         */
        static DenseVector<IndexType> getFineTargets(const DenseVector<IndexType> &coarseOrigin, const DenseVector<IndexType> &fineToCoarseMap);

        /**
         * Coarsen the input graph with edge matchings and contractions.
         * For an input graph with n nodes, the coarse graph will contain roughly 2^{-i}*n nodes, where i is the number of iterations.
         *
         * @param[in] inputGraph Adjacency matrix of input graph
         * @param[in] nodeWeights
         * @param[in] halo Halo of non-local neighbors
         * @param[out] coarseGraph Adjacency matrix of coarsened graph
         * @param[out] fineToCoarse DenseVector with as many entries as uncoarsened nodes. For each uncoarsened node, contains the corresponding coarsened node.
         * @param[in] iterations Number of contraction iterations
         */
        static void coarsen(const CSRSparseMatrix<ValueType>& inputGraph, const DenseVector<ValueType> &nodeWeights, const Halo& halo, CSRSparseMatrix<ValueType>& coarseGraph, DenseVector<IndexType>& fineToCoarse, IndexType iterations = 1);

        /**
         * @brief Perform a local maximum matching
         *
         * @param[in] graph Adjacency matrix of input graph
         * @param[in] nodeWeights
         *
         * @return vector of edges in maximum matching
         */
        static std::vector<std::pair<IndexType,IndexType>> maxLocalMatching(const scai::lama::CSRSparseMatrix<ValueType>& graph, const DenseVector<ValueType> &nodeWeights);
        
        /**
         * @brief Project a fine DenseVector to a coarse DenseVector. Values are interpolated linearly.
         *
         * @param[in] input
         * @param[in] fineToCoarse
         *
         * @return coarse representation
         */
        static DenseVector<ValueType> projectToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse);
        
        /**
         * @brief Project a fine DenseVector to a coarse DenseVector. Values are summed.
         *
         * @param[in] input
         * @param[in] fineToCoarse
         *
         * @return coarse representation
         */
        static DenseVector<ValueType> sumToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse);
        
        /**
         * @brief Compute coarse distribution from fineToCoarse map
         *
         * @param[in] fineToCoarse
         *
         * @return coarse distribution
         */
        static scai::dmemo::DistributionPtr projectToCoarse(const DenseVector<IndexType>& fineToCoarse);
        
        /**
         * @brief Compute a global prefix sum of a block-distributed input
         */
        template<typename T>
        static DenseVector<T> computeGlobalPrefixSum(const DenseVector<T> &input, T offset = 0);
        
        /**
         * Creates a coarsened graph using geometric information. Rounds every point according to settings.pixeledDetailLevel
         * creating a grid of size 2^detailLevel x 2^detailLevel (for 2D). Every coarse node/pixel of the
         * grid has weight equal the number of points it contains. The edge between two coarse nodes/pixels is the
         * number of edges of the input graph that their endpoints belong to different pixels.
         * 
         * WARNING: can happen that pixels are empty, this would create isolated vertices in the pixeled graph 
         *          which is not so good for spectral partitioning. To avoid that, we add every edge in the isolated
         *          vertices with a small weight of 0.001. This might cause other problems though, so have it in mind.
         * 
         * @param[in] adjM The adjacency matrix of the input graph
         * @param[in] coordinates The coordinates of the input points.
         * @param[out] nodeWeights The weights for the coarse nodes/pixels of the returned graph.
         * @param[in] settings Descibe different setting for the coarsening. Here we need settings.pixeledDetailLevel.
         * @return The adjacency matric of the coarsened/pixeled graph. This has side length 2^detailLevel and the whole size is dimension^sideLength.
         */
        static scai::lama::CSRSparseMatrix<ValueType> pixeledCoarsen (const CSRSparseMatrix<ValueType>& adjM, const std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings);
    
    }; // class MultiLevel
} // namespace ITI
