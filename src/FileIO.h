/*
 * FileIO.h
 *
 *  Created on: 15.02.2017
 *      Authors: Moritz v. Looz, Charilaos Tzovas
 */

#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/CSRSparseMatrix.hpp>
#include <scai/lama/DenseVector.hpp>
#include <scai/lama/io/MatrixMarketIO.hpp>

#include "quadtree/QuadTreeCartesianEuclid.h"
#include "Settings.h"
#include "GraphUtils.h"
#include "CommTree.h"

#include <vector>
#include <set>
#include <memory>
#include <sys/stat.h>

namespace ITI {

	using scai::lama::CSRSparseMatrix;
	using scai::lama::DenseVector;
    
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
	static void writeGraph (const CSRSparseMatrix<ValueType> &adjM, const std::string filename, const bool edgeWeights = false);

	/** Given an adjacency matrix and a filename writes the local part of matrix in the file using the METIS format.
	 *  Every proccesor adds his rank in the end of the file name.
	 * @param[in] adjM The graph's adjacency matrix.
	 * @param[in] filename The file's name to write to
	 */
	static void writeGraphDistributed (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);

    /** @brief Write graph and partition into a .vtk file. This can be opened by paraview.
	 * 
	 * @param[in] adjM The graph with N vertices given as an NxN adjacency matrix.
	 * @param[in] coordinates Coordinates of input points.
	 * @param[in] partition The partition of the input graph.
	 * @param[in] filename The file's name to write to
     */
    static void writeVTKCentral (const CSRSparseMatrix<ValueType> &adjM, const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const std::string filename);
    
	/** Given the vector of the coordinates and their dimension, writes them in file "filename".
	 * Coordinates are given as a DenseVector of size dim*numPoints.
	 * 
	 * @param[in] coordinates The coordinates of the points.
	 * @param[in] filename The file's name to write to
	*/
	static void writeCoords (const std::vector<DenseVector<ValueType>> &coordinates, const std::string filename);
	
	/** Given the vector of the coordinates and their dimension, writes them in file "filename".
	 * One by one the processors open the file and each PE writes its local part of the coordinates.
	 * Coordinates are given as a DenseVector of size dim*numPoints.
	 * 
	 * @param[in] coordinates The coordinates of the points.
	 * @param[in] filename The file's name to write to
	*/
    static void writeCoordsParallel(const std::vector<DenseVector<ValueType>> &coords, const std::string filename);
    
    /** @brief Each PE writes its own part of the coordinates in a separate file called filename_X.xyz where X is a number from 0 till the total number of PEs-1.
	 * @param[in] coordinates The coordinates of the points.
	 * @param[in] dimensions Number of dimensions of coordinates.
	 * @param[in] filename The file's name to write to
     * */
    static void writeCoordsDistributed (const std::vector<DenseVector<ValueType>> &coords,  const IndexType dimensions, const std::string filename);

    /** Given the vector of the coordinates and the nodeWeights writes them both in a file in the form:
    *
    *   cood1 coord2 ... coordD weight
    * 
    * for D dimensions. Each line coresponds to one point/vertex. Each PE, one after another, writes each own part.
	* 
	* @param[in] coordinates The coordinates of the points.
	* @param[in] nodeWeights The weights for each point.
	* @param[in] filename The file's name to write to
    * */
    static void writeInputParallel (const std::vector<DenseVector<ValueType>> &coords,const scai::lama::DenseVector<ValueType> nodeWeights, const std::string filename);
    
    /* Write a DenseVector in parallel in the filename. Each PE, one after another, write its own part.
     * */
    template<typename T>
    static void writeDenseVectorParallel(const DenseVector<T> &dv, const std::string filename);
	
	/*TODO: merge with writeDenseVectorParallel*/
	static void writePartitionParallel(const DenseVector<IndexType> &dv, const std::string filename);
    
    /**
	 * Writes a partition to file.
	 * @param[in] part
	 * @param[in] filename The file's name to write to
	 */
	static void writeDenseVectorCentral(DenseVector<IndexType> &part, const std::string filename);
        
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

	/** Reads a graph in parallel that is stored in a binary file. Uses the same format as in ParHiP, the parallel version of KaHiP,
	see <a href="http://algo2.iti.kit.edu/schulz/software_releases/kahipv2.00.pdf">here</a> for mode details.
	 * @param[in] filename The file to read from.
	 * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
	 */
	static scai::lama::CSRSparseMatrix<ValueType> readGraphBinary(const std::string filename);
        
	/**Every PE reads its part of the file. The file contains all the edges of the graph: each line has two numbers indicating
	 * the vertices of the edge. 
	 * 0 1
	 * 0 2				0 - 1 - 3 - 4
	 * 3 1				  \    /  
	 * 4 3				    2 
	 * 3 2
	 * @param[in] filename The file to read from.
	 * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
	 */
	static scai::lama::CSRSparseMatrix<ValueType> readEdgeList(const std::string filename, const bool binary = false);
	
	
	/** @brief Edge list format but now there are k files, one for each PE.
	 * \warning The number of mpi processes k (mpirun -np k), must be the same as the number of existing files. PE X will try
	 * to read from filenameX.
	 *
	 * @param[in] filename The prefix of the file to read from.
	 * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
	 * */
	static scai::lama::CSRSparseMatrix<ValueType> readEdgeListDistributed(const std::string filename);
	
	/** @brief Reads the coordinates from file "filename" and returns then in a vector of DenseVector
	 * 
	 * @param[in] filename The file to read from.
	 * @param[in] numberOfCoords The number of points contained in the file.
	 * @param[in] dimensions Number of dimensions of coordinates.
	 * @param[in] Format The format in which the coordinates are stored.
	 * 
	 * @return The coordinates
	 */
	static std::vector<DenseVector<ValueType>> readCoords ( std::string filename, IndexType numberOfCoords, IndexType dimension, Format = Format::METIS);
	
	/** @brief  Read coordinates from a binary file
	 * 
	 * @param[in] filename The name of the file to read the coordinates from.
	 * @param[in] numberOfCoords The number of points contained in the file.
	 * @param[in] dimension The dimension of the points
	 * 
	 * @return The coordinates
     * 
    */
	static std::vector<DenseVector<ValueType>> readCoordsBinary( std::string filename, const IndexType numberOfCoords, const IndexType dimension);
        
        
	/** @brief  Read coordinates in Ocean format of Vadym Aizinger
	 */
	static std::vector<DenseVector<ValueType>> readCoordsOcean ( std::string filename, IndexType dimension);

	/** @brief Read coordinates in TEEC format
	 * 
	 * @param[in] filename The name of the file to read the coordinates from.
	 * @param[in] numberOfCoords The number of points contained in the file.
	 * @param[in] dimension The dimension of the points
	 * @param[out] nodeWeights Weights for every coordinate
	 * 
	 * @return The coordinates
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

	 @return return.size()= number of weights, and return[i].size()= number of blocks
	*/
	static std::vector<std::vector<ValueType>> readBlockSizes(const std::string filename , const IndexType numBlocks, const IndexType numWeights = 1);

	static DenseVector<IndexType> readPartition(const std::string filename, IndexType n);
    
    /** The partition is redistributed and printed only by root processor.
     */    
    //static void writePartitionCentral( DenseVector<IndexType> &part, const std::string filename);
    
    
    /** Read graph and coordinates from a OFF file. Coordinates are (usually) in 3D.
    */
    static void readOFFTriangularCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const std::string filename);

	
    /** Read graph and coordinates from a dom.geo file of the ALYA tool. Coordinates are (usually) in 3D.
	*/
    static void readAlyaCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const IndexType N, const IndexType dimensions, const std::string filename);
	

	/** Reads a processor tree. Comments are allowed in the beginning with '#' or '%' and the first line
		after the comments must contain the number of PE in the file. Then, every line contains the 
		information of one PE: first are number indicating the label of this PE in the tree (for more details
		see CommTree.h) that ends with a '#'. Then there are two numbers, the first one is the memory of
		this PE in GB and then the relative speed of the cpu: a positive number x less than 1 indicating that
		this PE has speed x*maxCPUSpeed. Typically, at least one PE will have speed 1 (the fastest one)
		although this is not enforced or checked.
	*/
	//TODO: move to CommTree as, for example, importFromFile oder so?
	static CommTree<IndexType,ValueType> readPETree( const std::string& filename);

	
	// taken from https://stackoverflow.com/questions/4316442/stdofstream-check-if-file-exists-before-writing
	/** Check if a file exists
	 @ *param[in] filename - the name of the file to check
	 @return    true if the file exists, else false
	 */
    static bool fileExists(const std::string& filename);
	
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
