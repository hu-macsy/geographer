#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <scai/dmemo/HaloExchangePlan.hpp>

#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>

#include <scai/tracing.hpp>

#include <assert.h>

#include "Settings.h"
#include "PrioQueue.h"

namespace ITI {

    using scai::lama::CSRSparseMatrix;
    using scai::lama::CSRStorage;
    using scai::lama::DenseVector;

    /** @brief Improve the cut of a partition by doing local refinement.
    */
    
    template <typename IndexType, typename ValueType>
    class LocalRefinement{
    public:
        /**
         * Performs a local refinement step using distributed Fiduccia-Mattheyses on the distributed input graph and partition.
         * Only works if the number of blocks is equal to the number of processes and the partition coincides with the distribution.
         * When changing the partition during the refinement step, the graph, partition and coordinates are redistributed to match.
         * Internally calls twoWayLocalFM.
         *
         * @deprecated This method is not in use any longer.
         *
         * @param[in,out] input Adjacency matrix of the input graph
         * @param[in,out] part Partition
         * @param[in,out] coordinates Coordinates of input points, only used for tie-breaking
         *
         */
        static std::vector<ValueType> distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
        
        /**
         * Performs a local refinement step using distributed Fiduccia-Mattheyses on the distributed input graph and partition.
         * Only works if the number of blocks is equal to the number of processes and the partition coincides with the distribution.
         * When changing the partition during the refinement step, the graph, partition and coordinates are redistributed to match.
         * Internally calls twoWayLocalFM.
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
        static std::vector<ValueType> distributedFMStep(
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
         * Computes the border region to another block, i.e. those local nodes that have a short distance to it.
         *
         * The border region is computed by starting a breadth-first search from the nodes directly adjacent to the other block.
         * The search continues until minNodes are found, then the current BFS round is completed.
         * The result is a pair: The first element is a vector of nodes, in the order they were visited by the BFS.
         * The second element is a vector of round markers: For round i, roundMarkers[i] gives the position in the returned node set where the nodes encountered in BFS round i begin.
         *
         * @param[in] input Adjacency matrix of the input graph
         * @param[in] part Partition vector
         * @param[in] nodesWithNonLocalNeighbors Nodes directly adjacent to other blocks
         * @param[in] otherBlock block to which the border region should be adjacent
         * @param[in] minNodes Minimum number nodes in the border region.
         *
         * @return pair of interfaceNodes, roundMarkers
         */
        static std::pair<std::vector<IndexType>, std::vector<IndexType>> getInterfaceNodes(
            const CSRSparseMatrix<ValueType> &input, 
            const DenseVector<IndexType> &part, 
            const std::vector<IndexType>& nodesWithNonLocalNeighbors, 
            IndexType otherBlock, 
            IndexType minNodes
        );
        
        /**
         * Redistributes a matrix from a local halo object without communication.
         * Requires that all elements that are local in the new distribution are either local in the old distribution or present in the halo.
         *
         * @param[in,out] matrix Matrix
         * @param[in] newDistribution
         * @param[in] halo
         * @param[in] haloMatrix
         *
         */
        static void redistributeFromHalo(CSRSparseMatrix<ValueType>& matrix, scai::dmemo::DistributionPtr newDistribution, const scai::dmemo::HaloExchangePlan& halo, const CSRStorage<ValueType>& haloMatrix);
        
        /**
         * Redistributes a DenseVector from a local halo object without communication.
         *
         * @param[in,out] input
         * @param[in] newDist
         * @param[in] halo
         * @param[in] haloData
         *
         */
        template<typename T>
        static void redistributeFromHalo(DenseVector<T>& input, scai::dmemo::DistributionPtr newDist, const scai::dmemo::HaloExchangePlan& halo, const scai::hmemo::HArray<T>& haloData);
        
        /** First, it calculates the centroid of the local coordinates and then the distance of every local point to the centroid.

        @param[in] coordinates The coordinates of the points.
        @return The distance of all local points from the centroid.
        */
		static std::vector<ValueType> distancesFromBlockCenter(const std::vector<DenseVector<ValueType>> &coordinates);

    private:

        /**
         * Performs local refinement between the border region of two blocks, one of them being the local block associated with this process.
         * The non-local graph information must be given in the haloStorage.
         *
         * The vectors borderRegionIDs, nodeWeights, assignedToSecondBlock and tieBreakingKeys have the same size.
         *
         * The improved partition can be read from the assignedToSecondBlock input/output parameter.
         *
         * @param[in] input Adjacency matrix of local subgraph
         * @param[in] haloStorage Adjacency matrix of non-local border region
         * @param[in] halo Halo object to translate global IDs to elements in haloStorage
         * @param[in] borderRegionIDs global IDs of nodes in local and non-local border regions
         * @param[in] nodeWeights node weights of nodes in border region
         * @param[in,out] assignedToSecondBlock boolean array, false if node is in first (local) block, true if in second (non-local) block
         * @param[in] blockCapacities Total capacity of both blocks
         * @param[in] blockSizes Total size of both blocks, also including nodes not in the border region
         * @param[in] tieBreakingKeys When two moves would have the same gain, the node with the lower entry in tieBreakingKeys is moved
         * @param[in] settings Settings struct
         *
         * @return gain
         */
        static ValueType twoWayLocalFM(
            const CSRSparseMatrix<ValueType> &input, 
            const CSRStorage<ValueType> &haloStorage,
            const scai::dmemo::HaloExchangePlan &Halo,
            const std::vector<IndexType>& borderRegionIDs,
            const std::vector<ValueType>& nodeWeights, 
            std::vector<bool>& assignedToSecondBlock,
            const std::pair<IndexType, IndexType> blockCapacities,
            std::pair<IndexType, IndexType>& blockSizes,
            const std::vector<ValueType>& tieBreakingKeys,
            Settings settings
        );
        
        /**
         * @brief Perform a two way diffusion step, useful to generate tie breaking keys for local refinement
         *
         * @param[in] input Adjacency matrix of local subgraph
         * @param[in] haloStorage Adjacency matrix of non-local border region
         * @param[in] halo Halo object to translate global IDs to elements in haloStorage
         * @param[in] borderRegionIDs global IDs of nodes in local and non-local border regions
         * @param[in] secondRoundMarkers The number of nodes directly adjacent to the other block, for the local and non-local block
         * @param[in] assignedToSecondBlock boolean array, false if node is in first (local) block, true if in second (non-local) block
         * @param[in] settings Settings struct
         *
         * @return diffusionLoad
         */
        static std::vector<ValueType> twoWayLocalDiffusion(
            const CSRSparseMatrix<ValueType> &input, 
            const CSRStorage<ValueType> &haloStorage,
            const scai::dmemo::HaloExchangePlan &halo,
            const std::vector<IndexType>& borderRegionIDs, 
            std::pair<IndexType, IndexType> secondRoundMarkers,
            const std::vector<bool>& assignedToSecondBlock, 
            Settings settings
        );
        
        /**
         * @brief Count local nodes in block blockID
         *
         * @return Number of local nodes in block blockID
         */
        static IndexType localBlockSize(const DenseVector<IndexType> &part, IndexType blockID);
        
        /**
         * @brief Sum the degrees of a set of nodes
         *
         * @param[in] nodes global IDs of nodes
         *
         * @return degree sum
         */
        static IndexType getDegreeSum(const CSRSparseMatrix<ValueType> &input, const std::vector<IndexType> &nodes);

        
    }; //class LocalRefinement
} // namespace ITI
