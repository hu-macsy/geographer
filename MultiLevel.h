#pragma once

#include <assert.h>
#include <queue>
#include <string>
#include <numeric>
#include <iterator>
#include <algorithm>

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


using namespace scai::lama;     // for CSRSparseMatrix and DenseVector

namespace ITI{
    
    template <typename IndexType, typename ValueType>
    class MultiLevel{
    public:
        			
        static IndexType multiLevelStep(scai::lama::CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, DenseVector<IndexType> &nodeWeights, std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
        
        static void coarsen(const CSRSparseMatrix<ValueType>& inputGraph, const DenseVector<IndexType> &nodeWeights, CSRSparseMatrix<ValueType>& coarseGraph, DenseVector<IndexType>& fineToCoarse, IndexType iterations = 1);
        
        static std::vector<std::pair<IndexType,IndexType>> maxLocalMatching(const scai::lama::CSRSparseMatrix<ValueType>& graph, const DenseVector<IndexType> &nodeWeights);
        
        static DenseVector<ValueType> projectToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse);
        
        static DenseVector<IndexType> sumToCoarse(const DenseVector<IndexType>& input, const DenseVector<IndexType>& fineToCoarse);
        
        static scai::dmemo::DistributionPtr projectToCoarse(const DenseVector<IndexType>& fineToCoarse);
        
        static scai::dmemo::DistributionPtr projectToFine(scai::dmemo::DistributionPtr, const DenseVector<IndexType>& fineToCoarse);
        
        template<typename T>
        static DenseVector<T> projectToFine(const DenseVector<T>& input, const DenseVector<IndexType>& fineToCoarse);
        
        template<typename T>
        static DenseVector<T> computeGlobalPrefixSum(DenseVector<T> input, T offset = 0);
        
    
    
    }; // class MultiLevel
} // namespace ITI