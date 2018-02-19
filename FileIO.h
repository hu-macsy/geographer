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

#include <boost/lexical_cast.hpp>

#include "quadtree/QuadTreeCartesianEuclid.h"
#include "GraphUtils.h"
#ifndef SETTINGS_H
#include "Settings.h"
#endif

#include <vector>
#include <set>
#include <memory>


using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;

namespace ITI {
    
       /** METIS format: for graphs: first line are the nodes, N, and edges, E, of the graph
         *                            the next N lines contain the neighbours for every node. 
         *                            So if line 100 is "120 1234 8 2133" means that node 100
         *                            has edges to nodes 120, 1234, 8 and 2133.
         *                for coordinates (up to 3D): every line has 3 numbers, the real valued
         *                            coordinates. If the coordinates are in 2D the last number is 0.
         * MATRIXMARKET format: for graphs: we use the function readFromFile (or readFromSingleFile) 
         *                            provided by LAMA.
         *                for coordinates: first line has two numbers, the number of points N and
         *                            the dimension d. Then next N*d lines contain the coordinates
         *                            for the points: every d lines are the coordinates for a point.
        */

        
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

    /* Write graph and partition into a .vtk file. This can be opened by paraview.
     */
    static void writeVTKCentral (const CSRSparseMatrix<ValueType> &adjM, const std::vector<DenseVector<ValueType>> &coords, const DenseVector<IndexType> &part, const std::string filename);
    
    static void writeVTKCentral_ver2 (const CSRSparseMatrix<ValueType> &adjM, const std::vector<DenseVector<ValueType>> &coords, const DenseVector<IndexType> &part, const std::string filename);
    
	/** Given the vector of the coordinates and their dimension, writes them in file "filename".
	 * Coordinates are given as a DenseVector of size dim*numPoints.
	*/
	static void writeCoords (const std::vector<DenseVector<ValueType>> &coords, const std::string filename);

    static void writeCoordsParallel(const std::vector<DenseVector<ValueType>> &coords, const std::string filename);
    
    /* Each PE writes its own part of the coordinates in a separate file.
     * */
    static void writeCoordsDistributed (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const IndexType dimensions, const std::string filename);

    /*Given the vector of the coordinates and the nodeWeights writes them both in a file in the form:
    *
    *   cood1 coord2 ... coordD weight
    * 
    * for D dimensions. Each line coresponds to one point/vertex. Each PE, one after another, writes each own part.
    * */
    static void writeInputParallel (const std::vector<DenseVector<ValueType>> &coords,const scai::lama::DenseVector<ValueType> nodeWeights, const std::string filename);
    
    /* Write a DenseVector in parallel in the filename. Each PE, one after another, write its own part.
     * */
    template<typename T>
    void writeDenseVectorParallel(const DenseVector<T> &dv, const std::string filename);
    
    /**
	 * Writes a partition to file.
	 * @param[in] part
	 * @param[in] filename The file's name to write to
	 */
	static void writePartitionParallel(const DenseVector<IndexType> &part, const std::string filename);
        
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
	static scai::lama::CSRSparseMatrix<ValueType> readGraphBinary(const std::string filename);
        
	/*Every PE reads its part of the file. The file contains all the edges of the graph: each line has two numbers indicating
	 * the vertices of the edge. 
	 * 0 1
	 * 0 2				0 - 1 - 3 - 4
	 * 3 1				  \    /  
	 * 4 3				    2 
	 * 3 2
	 */
	static scai::lama::CSRSparseMatrix<ValueType> readEdgeList(const std::string filename);
	
	
	/* Edge list format but now there are k files, one for each PE
	 * */
	static scai::lama::CSRSparseMatrix<ValueType> readEdgeListDistributed(const std::string filename);
	
	/* Reads the coordinates from file "filename" and returns then in a vector of DenseVector
	 */
	static std::vector<DenseVector<ValueType>> readCoords ( std::string filename, IndexType numberOfCoords, IndexType dimension, Format = Format::METIS);
	
	/**
     * 
    */
	static std::vector<DenseVector<ValueType>> readCoordsBinary( std::string filename, const IndexType numberOfPoints, const IndexType dimension);
        
        
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
    
    /** The partition is redistributed and printed only by root processor.
     */    
    static void writePartitionCentral( DenseVector<IndexType> &part, const std::string filename);
    
    
    /** Read graph and coordinates from a OFF file. Coordinates are (usually) in 3D.
    */
    static void readOFFTriangularCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const std::string filename);

	
    /** Read graph and coordinates from a dom.geo file of the ALYA tool. Coordinates are (usually) in 3D.
	*/
    static void readAlyaCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const IndexType N, const IndexType dimensions, const std::string filename);

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
