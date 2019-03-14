#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <scai/dmemo/HaloExchangePlan.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>

#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edge_coloring.hpp>
#include <boost/graph/properties.hpp>

#include "Settings.h"
#include "Metrics.h"
//#include "FileIO.h"

using namespace scai::lama;
using scai::dmemo::HaloExchangePlan;

namespace ITI {

	template <typename IndexType, typename ValueType>
	class ParcoRepart {
		public:
            /**
             * Partitions the given input graph
             * If the number of blocks is equal to the number of processes, graph, coordinates and weights are redistributed according to the partition.
             *
             * @param[in,out] input Adjacency matrix of the input graph
             * @param[in,out] coordinates Coordinates of input points
             * @param[in,out] nodeWeights Optional node weights.
             * @param[in] settings Settings struct
             * @param[out] metrics struct into which time measurements are written
             *
             * @return partition Distributed DenseVector of length n, partition[i] contains the block ID of node i
             */
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, std::vector<DenseVector<ValueType>> &nodeWeights,
					struct Settings settings, struct Metrics& metrics);

			/**
			 * Repartitions the given input graph, starting from a given previous partition
			 * If the number of blocks is equal to the number of processes, graph and coordinates are redistributed according to the partition.
			 *
			 * @param[in,out] input Adjacency matrix of the input graph
			 * @param[in,out] coordinates Coordinates of input points
			 * @param[in,out] nodeWeights Optional node weights.
			 * @param[in] previous The previous partition
			 * @param[in] settings Settings struct
			 * @param[out] metrics Struct into which time measurements are written
			 *
			 * @return partition Distributed DenseVector of length n, partition[i] contains the block ID of node i
			 */
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, std::vector<DenseVector<ValueType>> &nodeWeights,
					DenseVector<IndexType>& previous, struct Settings settings, struct Metrics& metrics);

			/**
			 * Wrapper without node weights.
			 */
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, struct Settings settings, struct Metrics& metrics);
			
            /**
			 * Wrapper without node weights and no metrics
			 */
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, struct Settings settings);

            /**
			 * Wrapper without metrics.
			 */
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, std::vector<DenseVector<ValueType>> &nodeWeights, struct Settings settings);

		/**
		* Wrapper for metis-like input.
		
		* vtxDist, size=numPEs,  is a replicated array, it is the prefix sum of the number of nodes per PE
    			eg: [0, 15, 25, 50], PE0 has 15 vertices, PE1 10 and PE2 25
		* xadj, size=localN+1, (= IA array of the CSR sparse matrix format), is the prefix sum of the degrees
    			of the local nodes, ie, how many non-zero values the row has.
		* adjncy, size=localM (number of local edges = the JA array), contains numbers >0 and < N, each
    			number is the global id of the neighboring vertex
    	* localM, the number of local edges
    	* vwgt, size=localN, array for the node weights
    	* ndims, the dimensions of the coordinates
    	* xyz, size=ndims*locaN, the coordinates for every vertex. For vertex/point i, its coordinates are
    			in xyz[ndims*i], xyz[ndims*i+1], ... , xyz[ndims*i+ndims]
		*/
		static std::vector<IndexType> partitionGraph(
			IndexType *vtxDist, IndexType *xadj, IndexType *adjncy, IndexType localM,
			IndexType *vwgt, IndexType ndims, ValueType *xyz,
			Settings  settings, Metrics& metrics);

        /*
         * Get an initial partition using the Hilbert curve.
         * TODO: This currently does nothing and isn't used. Remove?
         */
        static DenseVector<IndexType> hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings);

		/*
         * @brief Partition a point set using the Hilbert curve, only implemented for equal number of blocks and processes.
         *
         * @param coordinates Coordinates of the input points
         * @param settings Settings struct
         *
         * @return partition DenseVector, redistributed according to the partition
         */
		static DenseVector<IndexType> hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings);

		/*
		 * Get an initial partition using the morton curve and measuring density per square.
		 */
		static DenseVector<IndexType> pixelPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings);

		/**
		 * Iterates over the local part of the adjacency matrix and counts local edges.
		 * If an inconsistency in the graph is detected, it tries to find the inconsistent edge and throw a runtime error.
		 * Not guaranteed to find inconsistencies. Iterates once over the edge list.
		 *
		 * @param[in] input Adjacency matrix
		 */
		static void checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input);

		/** Colors the edges of the graph using max_vertex_degree + 1 colors.
		 * TODO: This method redistributes the graph. Maybe it should not.
		 *
		 * @param[in] adjM The graph with N vertices given as an NxN adjacency matrix.
		 *
		 * @return A 3xN vector with the edges and the color of each edge: retG[0][i] the first node, retG[1][i] the second node, retG[2][i] the color of the edge.
		 */
		static std::vector< std::vector<IndexType>>  getGraphEdgeColoring_local( CSRSparseMatrix<ValueType> &adjM, IndexType& colors);

		/** Given the block graph, creates an edge coloring of the graph and returns a communication
		 *  scheme based on the coloring
		 *  TODO: This method redistributes the graph. Maybe it should not.
		 *
		 * @param[in] adjM The adjacency matrix of a graph.
		 * @return std::vector.size()= number of colors used for coloring the graph. If D is the
		 *  maximum number of edges for a node, then numbers of colors is D or D+1.
		 *  vector[i].size()= number of nodes in the graph = adjM.numRows = adjMnumCols.
		 *  return[i][j] = k : in round i, node j talks with node k. Must also be that return[i][k] = j.
		 *  Inactive nodes have their own rank: rank[i][j] = j.
		 */
		static std::vector<DenseVector<IndexType>> getCommunicationPairs_local( CSRSparseMatrix<ValueType> &adjM, Settings settings);

	//private:
		
		/**
		 * @brief Counts the number of local nodes in block blockID
		 *
		 * @param blockID
		 *
		 * @return Number of local nodes in blockID
		 */
		static IndexType localBlockSize(const DenseVector<IndexType> &part, IndexType blockID);

		/**
		 * @brief Sum of weights of local outgoing edges.
		 */
		static ValueType localSumOutgoingEdges(const CSRSparseMatrix<ValueType> &input, const bool weighted);

		static std::vector<IndexType> neighbourPixels(const IndexType thisPixel,const IndexType sideLen, const IndexType dimensions);

		//WARNING: moved function to AuxiliaryFuncions.h
		/**
		Given a partition as input, redistribute the graph, coordinates and node weights. 
		The partition vector must be a permutation of the indices from 0 to comm->getSize()-1.
		For the partition to ne valid: 0<= partition[i]<= rank,
		every entry must appear only once, i.e. there exist no i, j 
		such that partition[i]=partition[j].

		The partition vector is also redistributed so in the end, partition, graph
		coordinated and nodeWeights all ahve the same distribution.

		@param[in] partition The partition according to which we will redistribute data.
		@param[out] graph The graph to be redistributed
		@param[out} nodeWeights The coordinates of the points/vertices.
		@param[out] The weights for every vertex.
		*/
		/*
		static scai::dmemo::DistributionPtr redistributeFromPartition( 
			DenseVector<IndexType>& partition,
			CSRSparseMatrix<ValueType>& graph,
			std::vector<DenseVector<ValueType>>& coordinates,
			DenseVector<ValueType>& nodeWeights,
			Settings settings, 
			struct Metrics& metrics,
			bool useRedistributor = false );
		*/
	};
}
