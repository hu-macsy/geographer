#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <scai/dmemo/Halo.hpp>
#include <scai/dmemo/HaloBuilder.hpp>

#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>

#include <scai/tracing.hpp>

#include <assert.h>

#include "Settings.h"
#include "ParcoRepart.h"
#include "PrioQueue.h"

using namespace scai::lama;     // for CSRSparseMatrix and DenseVector


namespace ITI {
    
    template <typename IndexType, typename ValueType>
    class LocalRefinement{
    public:
        
        static std::vector<IndexType> distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
        
        static std::vector<IndexType> distributedFMStep(
            CSRSparseMatrix<ValueType> &input, 
            DenseVector<IndexType> &part, 
            std::vector<IndexType>& nodesWithNonLocalNeighbors,
            DenseVector<IndexType> &nodeWeights, 
            const std::vector<DenseVector<IndexType>>& communicationScheme, 
            std::vector<DenseVector<ValueType>> &coordinates,
            std::vector<ValueType> &distances, 
            Settings settings
        );

        /**
         * Computes the border region within one block, adjacent to another block
         * @param[in] input Adjacency matrix of the input graph
         * @param[in] part Partition vector
         * @param[in] otherBlock block to which the border region should be adjacent
         * @param[in] depth Width of the border region, measured in hops
         */
        static std::pair<std::vector<IndexType>, std::vector<IndexType>> getInterfaceNodes(
            const CSRSparseMatrix<ValueType> &input, 
            const DenseVector<IndexType> &part, 
            const std::vector<IndexType>& nodesWithNonLocalNeighbors, 
            IndexType otherBlock, 
            IndexType minNodes
        );
        
        /**
         * redistributes a matrix from a local halo object without communication. It that is impossible, throw an error.
         */
        static void redistributeFromHalo(CSRSparseMatrix<ValueType>& matrix, scai::dmemo::DistributionPtr newDistribution, scai::dmemo::Halo& halo, CSRStorage<ValueType>& haloMatrix);
        
        template<typename T>
        static void redistributeFromHalo(DenseVector<T>& input, scai::dmemo::DistributionPtr newDist, scai::dmemo::Halo& halo, scai::utilskernel::LArray<T>& haloData);
        
        
    private:
        
        static ValueType twoWayLocalFM(
            const CSRSparseMatrix<ValueType> &input, 
            const CSRStorage<ValueType> &haloStorage,
            const scai::dmemo::Halo &matrixHalo, 
            const std::vector<IndexType>& borderRegionIDs,
            const std::vector<IndexType>& nodeWeights, 
            std::pair<IndexType, IndexType> secondRoundMarkers,
            std::vector<bool>& assignedToSecondBlock,
            const std::pair<IndexType, 
            IndexType> blockCapacities,
            std::pair<IndexType, 
            IndexType>& blockSizes,
            std::vector<ValueType> tieBreakingKeys,
            Settings settings
        );
        
        /* Compute tie breaking rules when diffusion is enabled.
         * */
        static std::vector<ValueType> twoWayLocalDiffusion(
            const CSRSparseMatrix<ValueType> &input, 
            const CSRStorage<ValueType> &haloStorage,
            const scai::dmemo::Halo &matrixHalo, 
            const std::vector<IndexType>& borderRegionIDs, 
            std::pair<IndexType, IndexType> secondRoundMarkers,
            const std::vector<bool>& assignedToSecondBlock, 
            Settings settings
        );
        
        static IndexType localBlockSize(const DenseVector<IndexType> &part, IndexType blockID);
        
        static IndexType getDegreeSum(const CSRSparseMatrix<ValueType> &input, const std::vector<IndexType> &nodes);

        
    }; //class LocalRefinement
} // namespace ITI
