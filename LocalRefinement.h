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
#include "PrioQueue.h"

using scai::lama::CSRSparseMatrix;
using scai::lama::CSRStorage;
using scai::lama::DenseVector;


namespace ITI {
    
    template <typename IndexType, typename ValueType>
    class LocalRefinement{
    public:
        /**
         * Performs a local refinement step using distributed Fiduccia-Mattheyses on the distributed input graph and partition.
         * Only works if the number of blocks is equal to the number of processes and the partition coincides with the distribution.
         * When changing the partition during the refinement step, the graph, partition and coordinates are redistributed to match.
         *
         * Note: This method is not in use any longer.
         *
         * @param[in,out] input Adjacency matrix of the input graph
         * @param[in,out] part Partition
         * @param[in,out] coordinates Coordinates of input points, only used for tie-breaking
         *
         */
        static std::vector<IndexType> distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
        
        /**
         * Performs a local refinement step using distributed Fiduccia-Mattheyses on the distributed input graph and partition.
         * Only works if the number of blocks is equal to the number of processes and the partition coincides with the distribution.
         * When changing the partition during the refinement step, the graph, partition and coordinates are redistributed to match.
         *
         * The difference to the other method with the same name is that a number of temporary variables are exposed to the caller to enable reusing them.
         *
         * @param[in,out] input Adjacency matrix of the input graph
         * @param[in,out] part Partition
         * @param[in,out] nodesWithNonLocalNeighbors Nodes that are local to this process, but have neighbors that are not.
         * @param[in,out] nodeWeights
         * @param[in,out] coordinates Coordinates of input points, only used for geometric tie-breaking
         * @param[in,out] distances For each node, distance to block center. Only used for geometric tie-breaking
         * @param[in,out] origin Indicating for each element, where it originally came from. Is redistributed during refinement, allowing to trace movements.
         * @param[in] communicationScheme As many elements as rounds, each element is a DenseVector of length p. Indicates the communication partner in each round.
         * @param[in] settings Settings struct
         *
         */
        static std::vector<IndexType> distributedFMStep(
            CSRSparseMatrix<ValueType> &input, 
            DenseVector<IndexType> &part, 
            std::vector<IndexType>& nodesWithNonLocalNeighbors,
            DenseVector<ValueType> &nodeWeights, 
            std::vector<DenseVector<ValueType>> &coordinates,
            std::vector<ValueType> &distances,
            DenseVector<IndexType> &origin,
            const std::vector<DenseVector<IndexType>>& communicationScheme,
            Settings settings
        );

        /**
         * @brief Computes the border region to another block, i.e. those local nodes that have a short distance to it.
         *
         * @param[in] input Adjacency matrix of the input graph
         * @param[in] part Partition vector
         * @param[in] nodesWithNonLocalNeighbors Nodes directly adjacent to other blocks
         * @param[in] otherBlock block to which the border region should be adjacent
         * @param[in] minNodes Minimum number nodes in the border region.
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
        static void redistributeFromHalo(DenseVector<T>& input, scai::dmemo::DistributionPtr newDist, scai::dmemo::Halo& halo, scai::hmemo::HArray<T>& haloData);
        
		static std::vector<ValueType> distancesFromBlockCenter(const std::vector<DenseVector<ValueType>> &coordinates);

    private:
        
        static ValueType twoWayLocalFM(
            const CSRSparseMatrix<ValueType> &input, 
            const CSRStorage<ValueType> &haloStorage,
            const scai::dmemo::Halo &matrixHalo, 
            const std::vector<IndexType>& borderRegionIDs,
            const std::vector<ValueType>& nodeWeights, 
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
