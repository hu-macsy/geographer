/*
 * IO.h
 *
 *  Created on: 15.02.2017
 *      Author: moritzl
 */

#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/CSRSparseMatrix.hpp>
#include <scai/lama/DenseVector.hpp>
#include <scai/lama/io/MatrixMarketIO.hpp>

#include "quadtree/QuadTreeCartesianEuclid.h"

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;

#include <vector>
#include <set>
#include <memory>

#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

namespace ITI {
    
       /** METIS format: for graphs: first line are the nodes, N, and edges, E, of the graph
         *                            the next N lines contain the neighbours for every node. 
         *                            So if line 100 is "120 1234 8 2133" means that node 100
         *                            has edges to nodes 120, 1234, 8 and 2133.
         *                for coordinates (up to 3D): every line has 3 numbers, the real valued
         *                            coordinates. If the coordinates are in 2D the last number is 0.
         * MATRIXMARKET format: for graphs: we use the function readFromFile (or readFromSingleFile) 
         *                            provided by LAMA.
         *                for coordiantes: first line has two numbers, the number of points N and
         *                            the dimension d. Then next N*d lines contain the coordinates
         *                            for the poitns: every d lines are the coordinates for a point.
        */
	enum class Format {AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4, TEEC = 5 };
	
	
        
template <typename IndexType, typename ValueType>
class FileIO {

public:
	/** Given an adjacency matrix and a filename writes the matrix in the file using the METIS format.
	 *  Not distributed.
	 *
	 * @param[in] adjM The graph's adjacency matrix.
	 * @param[in] filename The file's name to write to
	 */
	static void writeGraph (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);

	/** Given an adjacency matrix and a filename writes the local part of matrix in the file using the METIS format.
	 *  Every proccesor adds his rank in the end of the file name.
	 * @param[in] adjM The graph's adjacency matrix.
	 * @param[in] filename The file's name to write to
	 */
	static void writeGraphDistributed (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);

	/** Given the vector of the coordinates and their dimension, writes them in file "filename".
	 * Coordinates are given as a DenseVector of size dim*numPoints.
	*/
	static void writeCoords (const std::vector<DenseVector<ValueType>> &coords, const std::string filename);

	static void writeCoordsDistributed_2D (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const std::string filename);

    /**
	 * Writes a partition to file.
	 * @param[in] part
	 * @param[in] filename The file's name to write to
	 */
	static void writePartition(const DenseVector<IndexType> &part, const std::string filename);
        
	/** Reads a graph from filename in METIS format and returns the adjacency matrix.
	 * @param[in] filename The file to read from.
         * @param[in] fileFormat The type of file to read from. 
	 * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
	 */
	static CSRSparseMatrix<ValueType> readGraph(const std::string filename, Format = Format::METIS);

	/** Reads a graph from filename in METIS format and returns the adjacency matrix.
	 * @param[in] filename The file to read from.
         * @param[in] fileFormat The type of file to read from.
	 * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
	 */
	static CSRSparseMatrix<ValueType> readGraph(const std::string filename, std::vector<DenseVector<ValueType>>& nodeWeights, Format = Format::METIS);

	/** Reads a graph in parallel that is stored in a binary file. Uses the same format as in ParHiP, the parallel version of KaHiP.
	 * @param[in] filename The file to read from.
	 * @param[in] fileFormat The type of file to read from.
	 * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
	 */
	static scai::lama::CSRSparseMatrix<ValueType> readGraphBinary(const std::string filename, std::vector<DenseVector<ValueType>>& nodeWeights);
        
	/* Reads the coordinates from file "filename" and returns then in a vector of DenseVector
	 */
	static std::vector<DenseVector<ValueType>> readCoords ( std::string filename, IndexType numberOfCoords, IndexType dimension, Format = Format::METIS);

	/*
	 * Read Coordinates in Ocean format of Vadym Aizinger
	 */
	static std::vector<DenseVector<ValueType>> readCoordsOcean ( std::string filename, IndexType dimension);

	/*
	 * Read coordinates in TEEC format
	 */
	static std::vector<DenseVector<ValueType>> readCoordsTEEC ( std::string filename, IndexType numberOfCoords, IndexType dimension, std::vector<DenseVector<ValueType>>& nodeWeights);

	/**
	 * Reads a partition from file.
	 */
	static DenseVector<IndexType> readPartition(const std::string filename);

	/**
	 * Reads a quadtree as specified in the format of Michael Selzer
	 */
	static CSRSparseMatrix<ValueType> readQuadTree( std::string filename, std::vector<DenseVector<ValueType>> &coords);

	/**
	 * Reads a quadtree as specified in the format of Michael Selzer
	 */
	static CSRSparseMatrix<ValueType> readQuadTree( std::string filename) {
		std::vector<DenseVector<ValueType>> coords;
		return readQuadTree(filename, coords);
	}

	static std::pair<IndexType, IndexType> getMatrixMarketCoordsInfos(const std::string filename);

	/** Read a file with numBLocks number of blocks. The file should contain in its first row the number of blocks and
	 *  in each line contains a number that is the size of this block.
	 *  Only PE 0 reads the given file, constructs the std::vector with the block sizes and then broadcasts the vector to
	 *  the rest of the PEs.
	 *  Example of a file with 3 blocks:
	 * 3
	 * 100
	 * 120
	 * 97
	*/
	static std::vector<IndexType> readBlockSizes(const std::string filename , const IndexType numBlocks);

	static DenseVector<IndexType> readPartition(const std::string filename, IndexType n);

private:
	/**
	 * given the central coordinates of a cell and its level, compute the bounding corners
	 */
	static std::pair<std::vector<ValueType>, std::vector<ValueType>> getBoundingCoords(std::vector<ValueType> centralCoords, 
        IndexType level);
        
        /*Reads a graph in Matrix Market format
        */
        static scai::lama::CSRSparseMatrix<ValueType> readGraphMatrixMarket(const std::string filename);
        
        /** Reads the coordinates for the MatrixMarket file format.
         */
        static std::vector<DenseVector<ValueType>> readCoordsMatrixMarket ( const std::string filename);
};

} /* namespace ITI */
