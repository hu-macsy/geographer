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
#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

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
                        *
	 		* @return Distributed DenseVector, at position i is the block in which node i is assigned to
	 		*/
			static DenseVector<IndexType> partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, struct Settings Settings);

			/*
			 * Get an initial partition using the hilbert curve.
			 */
			static DenseVector<IndexType> hilbertPartition(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
                        
			/*
			 * Get an initial partition by measuring the density per pixel.
			 */
			static DenseVector<IndexType> pixelPartition(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
                                                        
			/**
			 * This method takes a (possibly distributed) partition and computes its global cut.
			 *
                         * @param[in] input The adjacency matrix of the graph.
			 * @param[in] part The partition vector for the input graph.
			 * @param[in] weighted If edges are weighted or not.
			 */
			static ValueType computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, bool weighted = false);

			/**
			 * This method takes a (possibly distributed) partition and computes its imbalance.
			 * The number of blocks is also a required input, since it cannot be guessed accurately from the partition vector if a block is empty.
			 *
			 * @param[in] part partition
			 * @param[in] k number of blocks in partition.
			 */
			static ValueType computeImbalance(const DenseVector<IndexType> &part, IndexType k, const DenseVector<IndexType> &nodeWeights = {});

			/**
			 * Computes a list of global IDs of nodes which are adjacent to nodes local on this processor, but are themselves not local.
			 */
			static std::vector<IndexType> nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input);

			static std::vector<ValueType> distancesFromBlockCenter(const std::vector<DenseVector<ValueType>> &coordinates);

			/**
			 * Builds a halo containing all non-local neighbors.
			 */
			static scai::dmemo::Halo buildNeighborHalo(const CSRSparseMatrix<ValueType> &input);

                        /**
			 * Returns true if the node identified with globalID has a neighbor that is not local on this process.
			 * Since this method acquires reading locks on the CSR structure, it might be expensive to call often
			 */
			static bool hasNonLocalNeighbors(const CSRSparseMatrix<ValueType> &input, IndexType globalID);

			/**
			 * Returns a vector of global indices of nodes which are local on this process, but have neighbors that are node.
			 * No communication required, iterates once over the local adjacency matrix
			 * @param[in] input Adjacency matrix of the input graph
			 */
			static std::vector<IndexType> getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input);

			/**
			 * Iterates over the local part of the adjacency matrix and counts local edges.
			 * If an inconsistency in the graph is detected, it tries to find the inconsistent edge and throw a runtime error.
			 * Not guaranteed to find inconsistencies. Iterates once over the edge list.
			 */
			static void checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input);

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
			 * the inout graph.
			 */
			static scai::lama::DenseVector<IndexType> getDegreeVector( const scai::lama::CSRSparseMatrix<ValueType> adjM);
	};
}
