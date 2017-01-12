#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edge_coloring.hpp>
#include <boost/graph/properties.hpp>


using namespace scai::lama;

namespace ITI {
	template <typename IndexType, typename ValueType>
	class ParcoRepart {
		public:
			/**
	 		* Partitions the given input graph with a space-filling curve and (in future versions) local refinement
	 		*
	 		* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		*	In principle arbitrary dimensions, currently only 2 are supported. 
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		* @param[in] k Number of desired blocks
	 		* @param[in] epsilon Tolerance of block size
	 		*
	 		* @return DenseVector with the same distribution as the input matrix, at position i is the block node i is assigned to
	 		*/
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions,	IndexType k,  double epsilon = 0.05);

			/**
			* Returns the minimum distance between two neighbours
			*
			* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		*
	 		* @return The spatial distance of the closest pair of neighbours
			*/
			static ValueType getMinimumNeighbourDistance(const CSRSparseMatrix<ValueType> &input, const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions);

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
			static ValueType fiducciaMattheysesRound(const CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted = true);

			static ValueType computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, bool ignoreWeights = true);       

			/**
			 * Builds a halo containing all matrix entries of non-local neighbors.
			 */
			static scai::dmemo::Halo buildMatrixHalo(const CSRSparseMatrix<ValueType> &input);

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
			static std::pair<std::vector<IndexType>, IndexType> getInterfaceNodes(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, IndexType thisBlock, IndexType otherBlock, IndexType depth);

			static ValueType distributedFMStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted = true);

			static ValueType computeImbalance(const DenseVector<IndexType> &part, IndexType k);
                        
                        static DenseVector<IndexType> getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part);
                        
                        /**Returns the processor graph. Every processor traverses its local part of adjM: and for every
                         * edge (u,v) that one node, say u, is not local it gets the owner processor of u.
                         * 
                         * @param[in] adjM The adjacency matrix of the input graph.
                         * @return A [#PE x #PE] adjacency matrix of the processor graph.
                         */
                        static scai::lama::CSRSparseMatrix<ValueType> getPEGraph( const CSRSparseMatrix<ValueType> &adjM);
                        
                        /** Returns the edges of the block graph only for the local part. Eg. if blocks 1 and 2 are local
                         * in this processor it find the edge (1,2) ( and the edge (2,1)). But if block 1 has an edge with
                         * block 3 that is not local, the function will NOT find this edge.
                         * 
                         * @param[in] adjM The adjacency matrix of the input graph.
                         * @param[in] part The partition of the input graph.
                         *
                         * @return A 2 dimensional vector with the local edges of the block graph: (ret[0][i], ret[1][i])
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
                         * @param[in] adjM The graph given as an adjacency matrix.
                         * 
                         * @return The adjacency matrix of the block graph.
                         */
                        static scai::lama::CSRSparseMatrix<ValueType>  getGraphEdgeColoring_local( const CSRSparseMatrix<ValueType> &adjM);
                        
                        /** Colors the edges of the graph using max_vertex_degree + 1 colors.
                         * 
                         * @param[in] edgeList The graph given as the list of edges. It must have size 2 and an edge
                         * is (edgeList[0][i] , edgeList[1][i])
                         * 
                         * @return A vector with the color for every edge. ret.size()=edgeList.size() and edge i has
                         * color ret[i].
                         */
                        static std::vector<IndexType> getGraphEdgeColoring_local( const std::vector<std::vector<IndexType>> &edgeList );
	};
}

