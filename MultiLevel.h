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
        static IndexType multiLevelStep(scai::lama::CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, DenseVector<ValueType> &nodeWeights, std::vector<DenseVector<ValueType>> &coordinates, const Halo& halo, Settings settings);

        static void coarsen(const CSRSparseMatrix<ValueType>& inputGraph, const DenseVector<ValueType> &nodeWeights, const Halo& halo, CSRSparseMatrix<ValueType>& coarseGraph, DenseVector<IndexType>& fineToCoarse, IndexType iterations = 1);

        static std::vector<std::pair<IndexType,IndexType>> maxLocalMatching(const scai::lama::CSRSparseMatrix<ValueType>& graph, const DenseVector<ValueType> &nodeWeights);
        
        static DenseVector<ValueType> projectToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse);
        
        static DenseVector<ValueType> sumToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse);
        
        static scai::dmemo::DistributionPtr projectToCoarse(const DenseVector<IndexType>& fineToCoarse);
        
        static scai::dmemo::DistributionPtr projectToFine(scai::dmemo::DistributionPtr, const DenseVector<IndexType>& fineToCoarse);
        
        template<typename T>
        static DenseVector<T> projectToFine(const DenseVector<T>& input, const DenseVector<IndexType>& fineToCoarse);
        
        template<typename T>
        static DenseVector<T> computeGlobalPrefixSum(const DenseVector<T> &input, T offset = 0);
        
        /**
         * Creates a coarsened graph using geometric information. Rounds avery point according to settings.pixeledDetailLevel
         * creating a grid of size 2^detailLevel x 2^detailLevel (for 2D). Every coarse node/pixel of the
         * grid has weight equal the number of points it contains and the edge between two coarse nodes/pixels is the
         * number of edges of the input graph that their endpoints belinf to different pixels.
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
