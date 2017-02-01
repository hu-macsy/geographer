#pragma once

#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>

#include <scai/lama/Vector.hpp>
#include <scai/dmemo/Halo.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edge_coloring.hpp>
#include <boost/graph/properties.hpp>

#include <set>

#include "Settings.h"

using namespace scai::lama;
using scai::dmemo::Halo;

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define BUILD_COMMIT_STRING STR_VALUE(BUILD_COMMIT)

const std::string version = BUILD_COMMIT_STRING;

namespace ITI {

	template <typename IndexType, typename ValueType>
	class ParcoRepart {
		public:
			/**
	 		* Partitions the given input graph with a space-filling curve and (in future versions) local refinement
	 		*
	 		* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		* @param[in] k Number of desired blocks
	 		* @param[in] epsilon Tolerance of block size
	 		*
	 		* @return DenseVector with the same distribution as the input matrix, at position i is the block node i is assigned to
	 		*/
			//static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, IndexType k,  double epsilon = 0.05);
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, struct Settings Settings);

			static ValueType computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, bool ignoreWeights = true);

			/**
			 * This method takes a (possibly distributed) partition and computes its imbalance.
			 * The number of blocks is also a required input, since it cannot be guessed accurately from the partition vector if a block is empty.
			 *
			 * @param[in] part partition
			 * @param[in] k number of blocks in partition.
			 */
			static ValueType computeImbalance(const DenseVector<IndexType> &part, IndexType k);

			/**
			* Performs local refinement of a given partition
			*
	 		* @param[in] input Adjacency matrix of the input graph
			* @param[in,out] part Partition which is to be refined
	 		* @param[in] k Number of desired blocks. Must be the same as in the previous partition.
	 		* @param[in] epsilon Tolerance of block size
			*
			* @return The difference in cut weight between this partition and the previous one
			*/
			static ValueType replicatedMultiWayFM(const CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted = true);

			static std::vector<DenseVector<IndexType>> computeCommunicationPairings(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const DenseVector<IndexType> &blocksToPEs);

			/**
			 * Computes a list of global IDs of nodes which are adjacent to nodes local on this processor, but are themselves not local.
			 */
			static std::vector<IndexType> nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input);

			/**
			 * redistributes a matrix from a local halo object without communication. It that is impossible, throw an error.
			 */
			static void redistributeFromHalo(CSRSparseMatrix<ValueType>& matrix, scai::dmemo::DistributionPtr newDistribution, Halo& halo, CSRStorage<ValueType>& haloMatrix);

			/**
			 * Builds a halo containing all partition entries of non-local neighbors.
			 */
			static scai::dmemo::Halo buildPartHalo(const CSRSparseMatrix<ValueType> &input,  const DenseVector<IndexType> &part);

			/**
			 * Computes the border region within one block, adjacent to another block
			 * @param[in] input Adjacency matrix of the input graph
			 * @param[in] part Partition vector
			 * @param[in] thisBlock block in which the border region is required
			 * @param[in] otherBlock block to which the border region should be adjacent
			 * @param[in] depth Width of the border region, measured in hops
			 */
			static std::pair<std::vector<IndexType>, IndexType> getInterfaceNodes(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const std::vector<IndexType>& nodesWithNonLocalNeighbors, IndexType thisBlock, IndexType otherBlock, IndexType depth);

			static ValueType distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, std::vector<IndexType>& nodesWithNonLocalNeighbors, Settings settings);

			static ValueType distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, std::vector<IndexType>& nodesWithNonLocalNeighbors,
					const std::vector<DenseVector<IndexType>>& communicationScheme, Settings settings);

			static void checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input);

			static bool hasNonLocalNeighbors(const CSRSparseMatrix<ValueType> &input, IndexType globalID);

			static std::vector<IndexType> getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input);
                        
			//------------------------------------------------------------------------

			/** Get the borders nodes of each block.
			*/
			static DenseVector<IndexType> getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part);

			/**Returns the processor graph. Every processor traverses its local part of adjM: and for every
			 * edge (u,v) that one node, say u, is not local it gets the owner processor of u. The returned graph is distributed with a BLOCK distribution.
			 *
			 * @param[in] adjM The adjacency matrix of the input graph.
			 * @return A [#PE x #PE] adjacency matrix of the processor graph.
			 */
			static scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const CSRSparseMatrix<ValueType> &adjM);

			/** Returns the edges of the block graph only for the local part. Eg. if blocks 1 and 2 are local
			 * in this processor it finds the edge (1,2) ( and the edge (2,1)).
			 * Also if the other endpoint is in another processor it finds this edge: block 1 is local, it
			 * shares an edge with block 3 that is not local, this edge is found and returned.
			 *
			 * @param[in] adjM The adjacency matrix of the input graph.
			 * @param[in] part The partition of the input graph.
			 *
			 * @return A 2 dimensional vector with the edges of the local parts of the block graph:
			 * edge (u,v) is (ret[0][i], ret[1][i]) if block u and block v are connected.
			*/
			static std::vector<std::vector<IndexType> > getLocalBlockGraphEdges( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part);
                        
			/** Builds the block graph of the given partition.
			 * Creates an HArray that is passed around in numPEs (=comm->getSize()) rounds and every time
			 * a processor writes in the array its part.
			 *
			 * Not distributed.
			 *
			 * @param[in] adjM The adjacency matric of the input graph.
			 * @param[in] part The partition of the input garph.
			 * @param[in] k Number of blocks.
			 *
			 * @return The "adjacency matrix" of the block graph. In this version is a 1-dimensional array
			 * with size k*k and [i,j]= i*k+j.
			 */
			static scai::lama::CSRSparseMatrix<ValueType> getBlockGraph( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, const int k);

			/** Colors the edges of the graph using max_vertex_degree + 1 colors.
			 *
			 * @param[in] adjM The graph with N vertices given as an NxN adjacency matrix.
			 *
			 * @return A 3xN vector with the edges and the color of each edge: retG[0][i] the first node, retG[1][i] the second node, retG[2][i] the color of the edge.
			 */
			static std::vector< std::vector<IndexType>>  getGraphEdgeColoring_local( const CSRSparseMatrix<ValueType> &adjM, IndexType& colors);
                        
			/** Colors the edges of the graph using max_vertex_degree + 1 colors.
			 *
			 * @param[in] edgeList The graph given as the list of edges. It must have size 2 and an edge
			 * is (edgeList[0][i] , edgeList[1][i])
			 *
			 * @return A vector with the color for every edge. ret.size()=edgeList.size() and edge i has
			 * color ret[i].
			 */
			static std::vector<IndexType> getGraphEdgeColoring_local( const std::vector<std::vector<IndexType>> &edgeList );
                        
			/** Given the block graph, creates an edge coloring of the graph and retuns a communication
			 *  scheme based on the coloring
			 *
			 * @param[in] adjM The adjacency matrix of a graph.
			 * @return std::vector.size()= number of colors used for coloring the graph. If D is the
			 *  maximum number of edges for a node, then nubers of colors is D or D+1.
			 *  vector[i].size()= number of nodes in the graph = adjM.numRows = adjMnumCols.
			 *  return[i][j] = k : in round i, node j talks with node k. Must also be that return[i][k] = j.
			 *  Inactive nodes have their own rank: rank[i][j] = j.
			 */
			static std::vector<DenseVector<IndexType>> getCommunicationPairs_local( const CSRSparseMatrix<ValueType> &adjM);


		private:
            static ValueType twoWayLocalFM(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage,
                        		const Halo &matrixHalo, const std::vector<IndexType>& borderRegionIDs, std::vector<bool>& assignedToSecondBlock,
                        		const std::pair<IndexType, IndexType> blockCapacities, std::pair<IndexType, IndexType>& blockSizes, Settings settings);

			static IndexType localBlockSize(const DenseVector<IndexType> &part, IndexType blockID);

			static ValueType localSumOutgoingEdges(const CSRSparseMatrix<ValueType> &input);

			static IndexType getDegreeSum(const CSRSparseMatrix<ValueType> &input, std::vector<IndexType> nodes);

			static ValueType computeCutTwoWay(const CSRSparseMatrix<ValueType> &input,
					const CSRStorage<ValueType> &haloStorage, const Halo &halo,
					const std::set<IndexType> &firstregion,  const std::set<IndexType> &secondregion,
					const bool ignoreWeights = true);
	};
}
