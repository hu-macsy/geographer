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

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edge_coloring.hpp>
#include <boost/graph/properties.hpp>

#include "Settings.h"
#include "Metrics.h"
#include "FileIO.h"

using namespace scai::lama;
using scai::dmemo::Halo;
using scai::dmemo::Halo;

namespace ITI {

	template <typename IndexType, typename ValueType>
	class ParcoRepart {
		public:
			/**
	 		* Partitions the given input graph
	 		*
	 		* @param[in] input Adjacency matrix of the input graph
	 		* @param[in] coordinates Node positions
                        *
	 		* @return Distributed DenseVector	, at position i is the block node i is assigned to
	 		*/
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights,
					struct Settings settings, struct Metrics& metrics);

			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights,
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
			 * Wrapper without metrics struct.
			 */
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, struct Settings settings);
                        
			/*
			 * Get an initial partition using the hilbert curve.
			 */
			static DenseVector<IndexType> hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings);

			static DenseVector<IndexType> hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings);

			static void hilbertRedistribution(std::vector<DenseVector<ValueType> >& coordinates, DenseVector<ValueType>& nodeWeights, Settings settings, struct Metrics& metrics);

			/*
			 * Get an initial partition using the morton curve and measuring density per square.
			 */
			static DenseVector<IndexType> pixelPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings);

			/**
			 * Iterates over the local part of the adjacency matrix and counts local edges.
			 * If an inconsistency in the graph is detected, it tries to find the inconsistent edge and throw a runtime error.
			 * Not guaranteed to find inconsistencies. Iterates once over the edge list.
			 */
			static void checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input);

			/** Colors the edges of the graph using max_vertex_degree + 1 colors.
			 *
			 * @param[in] adjM The graph with N vertices given as an NxN adjacency matrix.
			 *
			 * @return A 3xN vector with the edges and the color of each edge: retG[0][i] the first node, retG[1][i] the second node, retG[2][i] the color of the edge.
			 */
			static std::vector< std::vector<IndexType>>  getGraphEdgeColoring_local( CSRSparseMatrix<ValueType> &adjM, IndexType& colors);

			/** Given the block graph, creates an edge coloring of the graph and returns a communication
			 *  scheme based on the coloring
			 *
			 * @param[in] adjM The adjacency matrix of a graph.
			 * @return std::vector.size()= number of colors used for coloring the graph. If D is the
			 *  maximum number of edges for a node, then numbers of colors is D or D+1.
			 *  vector[i].size()= number of nodes in the graph = adjM.numRows = adjMnumCols.
			 *  return[i][j] = k : in round i, node j talks with node k. Must also be that return[i][k] = j.
			 *  Inactive nodes have their own rank: rank[i][j] = j.
			 */
			static std::vector<DenseVector<IndexType>> getCommunicationPairs_local( CSRSparseMatrix<ValueType> &adjM);

		//private:
			
			static IndexType localBlockSize(const DenseVector<IndexType> &part, IndexType blockID);

			static ValueType localSumOutgoingEdges(const CSRSparseMatrix<ValueType> &input, const bool weighted);

			static IndexType getDegreeSum(const CSRSparseMatrix<ValueType> &input, const std::vector<IndexType> &nodes);

			static std::vector<IndexType> neighbourPixels(const IndexType thisPixel,const IndexType sideLen, const IndexType dimensions);

			/**Returns a vector of size N (if adjM is of size NxN) with the degree for every node of
			* the input graph.
			*/
			static scai::lama::DenseVector<IndexType> getDegreeVector( const scai::lama::CSRSparseMatrix<ValueType> adjM);
	};
}
