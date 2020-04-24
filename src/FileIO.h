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

/** @brief All the function to read and write in files.
  *
	 * METIS format: for graphs: first line are the nodes, N, and edges, E, of the graph
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
     * In the METIS format the first line has two numbers, first is the number on vertices and the second
     * is the number of edges. Then, row i has numbers e1, e2, e3, ... notating the edges:
     * (i, e1), (i, e2), (i, e3), ....
     *
     * Not distributed.
     *
     * @param[in] adjM The graph's adjacency matrix.
     * @param[in] filename The file's name to write to.
     */
    static void writeGraph (
        const CSRSparseMatrix<ValueType> &adjM,
        const std::string filename,
        const bool binary = false,
        const bool edgeWeights = false);

    /** Given an adjacency matrix and a filename writes the local part of matrix in the file using the METIS format.
     * Every PE writes its local data in a separate file by adding its rank in the end of the file name.
     * @param[in] adjM The graph's adjacency matrix.
     * @param[in] filename The file's name to write to
     */
    static void writeGraphDistributed (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);


    static void writeGraphAsEdgeList (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);

    /** @brief Write graph and partition into a .vtk file; this can be opened by paraview.
     *
     * @param[in] adjM The graph with N vertices given as an NxN adjacency matrix.
     * @param[in] coordinates Coordinates of input points.
     * @param[in] partition The partition of the input graph.
     * @param[in] filename The file's name to write to
     */
    static void writeVTKCentral (const CSRSparseMatrix<ValueType> &adjM, const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const std::string filename);

    /** Given the vector of the coordinates and their dimension, writes them in file "filename".
     * The coordinates are gather in one PE (or replicated in all) and only the root PE writes the file.
     * Every line holds the coordinates of one point.

     * For a more scalable version see writeCoordsParallel().
     *
     * @param[in] coordinates The coordinates of the points.
     * @param[in] filename The file's name to write to.
    */
    static void writeCoords (const std::vector<DenseVector<ValueType>> &coordinates, const std::string filename);

    /** Given the vector of the coordinates and their dimension, writes them in file "filename".
     * One by one the processors open the file and each PE writes its local part of the coordinates.
     * Every line holds the coordinates of one point.
     *
     * @param[in] coordinates The coordinates of the points.
     * @param[in] filename The file's name to write to
    */
    static void writeCoordsParallel(
        const std::vector<DenseVector<ValueType>> &coords,
        const std::string filename);

    /** Each PE writes its own part of the coordinates in a separate file called filename_X.xyz where X is the rank of the PE,
     * i.e., a number from 0 until the total number of PEs-1.
     * @param[in] coordinates The coordinates of the points.
     * @param[in] dimensions Number of dimensions of coordinates.
     * @param[in] filename The file's name to write to.
     * */
    static void writeCoordsDistributed (
        const std::vector<DenseVector<ValueType>> &coords,
        const IndexType dimensions,
        const std::string filename);

    /** Given the vector of the coordinates and the nodeWeights, writes them both in a file in the form:
    *
    *   cood1 coord2 ... coordD weight
    *
    * for D dimensions. Each line corresponds to one point/vertex. Each PE, one after another, writes each own part.
    *
    * @param[in] coordinates The coordinates of the points.
    * @param[in] nodeWeights The weights for each point.
    * @param[in] filename The file's name to write to
    * */
    static void writeInputParallel (const std::vector<DenseVector<ValueType>> &coords,const scai::lama::DenseVector<ValueType> nodeWeights, const std::string filename);


    /** Write a (possibly distributed) dense vector in a file. Each PE takes its turn and writes its local data to the file.
    @param[in] dv The dense vector to store.
    @param[] filename The file's name to write to.
    */
    /*TODO: merge with writeDenseVectorParallel*/
    static void writePartitionParallel(const DenseVector<IndexType> &dv, const std::string filename);

    /** Writes a dense vector to a file. The dense vector is replicated and only PE 0 writes it to a file.
     * @param[in] dv The dense vector to store.
     * @param[in] filename The file's name to write to
     */
    static void writeDenseVectorCentral(DenseVector<IndexType> &dv, const std::string filename);

    /**
     * Write a DenseVector in parallel in the filename. Each PE, one after another, write its own part.
     */
    template<typename T>
    static void writeDenseVectorParallel(const DenseVector<T> &dv, const std::string filename);


    /** Reads a graph from filename in a given format and returns the adjacency matrix. \sa ITI::Format
     * @param[in] filename The file to read from.
     * @param[in] fileFormat The type of file to read from.
     * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
     */
    static CSRSparseMatrix<ValueType> readGraph(const std::string filename, const scai::dmemo::CommunicatorPtr comm, Format = Format::METIS);

    /** Reads a graph from filename in the given format and returns the adjacency matrix with the node weights. \sa ITI::Format
     * @param[in] filename The file to read from.
     * @param[out] nodeWeights The weights of the nodes if they exists in the provided file.
     * @param[in] fileFormat The type of file to read from.
     * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
     */
    static CSRSparseMatrix<ValueType> readGraph(const std::string filename, std::vector<DenseVector<ValueType>>& nodeWeights, const scai::dmemo::CommunicatorPtr comm, Format = Format::METIS);

    static CSRSparseMatrix<ValueType> readGraph(const std::string filename ){
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        Format formt = Format::METIS;
        return readGraph( filename, comm, formt);
    }

    /** Reads a graph in parallel that is stored in a binary file. Uses the same format as in ParHiP, the parallel version of KaHiP,
    see <a href="http://algo2.iti.kit.edu/schulz/software_releases/kahipv2.00.pdf">here</a> for mode details.
     * @param[in] filename The file to read from.
     * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
     */
    static scai::lama::CSRSparseMatrix<ValueType> readGraphBinary(const std::string filename, const scai::dmemo::CommunicatorPtr comm);

    /**Every PE reads its part of the file. The file contains all the edges of the graph: each line has two numbers indicating
     * the vertices of the edge.
     \verbatim
     0 1				graph:
     0 2				0 - 1 - 3 - 4
     3 1				  \    /
     4 3				    2
     3 2
     \endverbatim
     * @param[in] filename The file to read from.
     * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
     */
    static scai::lama::CSRSparseMatrix<ValueType> readEdgeList(const std::string filename, const scai::dmemo::CommunicatorPtr comm, const bool binary = false);


    /** Edge list format but now there are k files, one for each PE.
     * \warning The number of mpi processes k (mpirun -np k), must be the same as the number of existing files. PE X will try
     * to read from filenameX.
     *
     * @param[in] filename The prefix of the file to read from.
     * @return The adjacency matrix of the graph. The rows of the matrix are distributed with a BlockDistribution and NoDistribution for the columns.
     * */
    static scai::lama::CSRSparseMatrix<ValueType> readEdgeListDistributed(const std::string filename, const scai::dmemo::CommunicatorPtr comm);

    /** @brief Reads the coordinates from file "filename" and returns then in a vector of DenseVector.
     *
     * @param[in] filename The file to read from.
     * @param[in] numberOfCoords The number of points contained in the file.
     * @param[in] dimensions Number of dimensions of coordinates.
     * @param[in] Format The format in which the coordinates are stored.
     *
     * @return The coordinates. ret.size()=dimension, ret[i].size()=numberOfCoords
     */
    static std::vector<DenseVector<ValueType>> readCoords ( const std::string filename, const IndexType numberOfCoords, const IndexType dimension, const scai::dmemo::CommunicatorPtr comm, Format = Format::METIS);

    static std::vector<DenseVector<ValueType>> readCoords ( const std::string filename, const IndexType numberOfCoords, const IndexType dimension, Format format = Format::METIS){
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        return readCoords( filename, numberOfCoords, dimension, comm, format);
    }

    /** @brief  Read coordinates from a binary file. Coordinates are stored in a consecutive format,

     * @warning This function can read only 2 and 3 dimensions.

     * @param[in] filename The name of the file to read the coordinates from.
     * @param[in] numberOfCoords The number of points contained in the file.
     * @param[in] dimension The dimension of the points
     *
     * @return The coordinates. ret.size()=dimension, ret[i].size()=numberOfCoords
    */
    static std::vector<DenseVector<ValueType>> readCoordsBinary( const std::string filename, const IndexType numberOfCoords, const IndexType dimension, const scai::dmemo::CommunicatorPtr comm);


    /** @brief  Read coordinates in Ocean format of Vadym Aizinger.
     */
    static std::vector<DenseVector<ValueType>> readCoordsOcean ( const std::string filename, const IndexType dimension, const scai::dmemo::CommunicatorPtr comm);

    /** @brief Read coordinates in TEEC format
     *
     * @param[in] filename The name of the file to read the coordinates from.
     * @param[in] numberOfCoords The number of points contained in the file.
     * @param[in] dimension The dimension of the points
     * @param[out] nodeWeights Weights for every coordinate
     *
     * @return The coordinates
     */
    static std::vector<DenseVector<ValueType>> readCoordsTEEC ( std::string filename, IndexType numberOfCoords, IndexType dimension, std::vector<DenseVector<ValueType>>& nodeWeights, const scai::dmemo::CommunicatorPtr comm);


    /**
     * Reads a quadtree as specified in the format of Michael Selzer.
     @param[in] filename The name of the file to read from.
     @param[out] coordinates The coordinates of the graph.
     @return The graph and coordinates are not distributed, they are replicated in all PEs.
     */
    static CSRSparseMatrix<ValueType> readQuadTree( std::string filename, std::vector<DenseVector<ValueType>> &coordinates);

    /**
     * Reads a quadtree as specified in the format of Michael Selzer
     */
    static CSRSparseMatrix<ValueType> readQuadTree( std::string filename) {
        std::vector<DenseVector<ValueType>> coords;
        return readQuadTree(filename, coords);
    }

    /** Get the number of points and dimensions from a MatrixMarket file format.
    @return first is the number of points, second is the dimensions of the points.
    */
    static std::pair<IndexType, IndexType> getMatrixMarketCoordsInfos(const std::string filename);

    /** Read a file with numBLocks number of blocks. The file should contain in its first row the number of blocks and
     *  in each line contains a number that is the size of this block.
     *  Only PE 0 reads the given file, constructs the std::vector with the block sizes and then broadcasts the vector to
     *  the rest of the PEs.
     @verbatim
     Example of a file with 3 blocks:

     3
     100
     120
     97
    @endverbatim
     @return return.size()= number of weights, and return[i].size()= number of blocks
    */
    static std::vector<std::vector<ValueType>> readBlockSizes(const std::string filename, const IndexType numBlocks, const IndexType numWeights = 1);

    /**
     * Reads a partition from file. Every line contains one number, if line i has number p, then node i belongs to block p.
     * We are using a block distribution: every PE calculates the range of indices that owns, scrolls to the line its range starts
     * and reads the respective number of indices.

     * @param[in] filename The name of the file to read the partition from.
     * @param[in] n The total number of points; also, the lines of the file.
     * @return A DenseVector distributed between PEs with a block distribution.
     */
    static DenseVector<IndexType> readPartition(const std::string filename, const IndexType n);


    /** Read graph and coordinates from a OFF file. Coordinates are (usually) in 3D.
    @param[out] graph The graph as a replicated matrix.
    @param[out] coords The coordinates of the graph, coords.size()=dimensions. Also replicated in every PE.
    @param[in] filename The name of the file to read from.
    */
    static void readOFFTriangularCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const std::string filename);


    /** Read graph and coordinates from a dom.geo file of the ALYA tool. Coordinates are (usually) in 3D.
    @param[out] graph The graph as a replicated matrix.
    @param[out] coords The coordinates of the graph, coords.size()=dimensions. Also replicated in every PE.
    @param[in] N The total number of points.
    @parm[in] dimensions The dimensions of the points.
    @param[in] filename The name of the file to read the graph and coordinates from.
    */
    static void readAlyaCentral( scai::lama::CSRSparseMatrix<ValueType>& graph, std::vector<DenseVector<ValueType>>& coords, const IndexType N, const IndexType dimensions, const std::string filename);


    /** Reads a processor tree. Comments are allowed in the beginning with '#' or '%' and the first line
    	after the comments must contain the number of PE (numPEs) in the file and the number of weights (numWeights)
    	that each PE has.
    	Next, there are numWeights bits, each bit indicating if the corresponding weight is proportional or not.
    	(In total, the first line should have numWeights+2 numbers.)
    	Then, there are numPEs lines and every line contains the
    	information of one PE: first are number indicating the label of this PE in the tree (for more details
    	see CommTree.h); this part ends with a '#'. After are numWeights numbers, the weight values for each PE.
    	For an example, see the files in meshes/processorTrees/.
    	@verbatim
    	#comments...
    	#(label), mem(GB), cpu(%)
    	28 2 0 1
    	0,0,0 # 52.5, 0.8
    	0,0,1 # 52, 0.5
    	0,1,0 # 51, 0.8
    	0,1,1 # 51.2, 0.8
    	0,1,2 # 54, 0.7
    	0,2,0 # 64, 1
    	@endverbatim

    	@return A replicated CommTree.
    */
    //TODO: move to CommTree as, for example, importFromFile oder so?
    static CommTree<IndexType,ValueType> readPETree( const std::string& filename);

    /* Read the file with the topology description; one line per compute node with 4 values: 
        name (which is used as a key for the map), CPU speed, memory in MB  and number of cores

    @param return key is the node name, vector as size 3 with the above attributes
    */

    static std::map<std::string, std::vector<ValueType>> readFlatTopology( const std::string& filename );

    /* Creates the block sizes (likely to be used to create a commTree) from a topology file.
        \sa readFlatTopology() and \sa commTree::createFlatHeterogeneous().
        This function works on;y when numBlock==number of calling processors.
    */
    static std::vector<std::vector<ValueType>>createBlockSizesFromTopology(
        const std::string filename, const std::string myName, const scai::dmemo::CommunicatorPtr comm);

    // taken from https://stackoverflow.com/questions/4316442/stdofstream-check-if-file-exists-before-writing
    /** Check if a file exists
     @param[in] filename - the name of the file to check
     @return    true if the file exists, else false
     */
    static bool fileExists(const std::string& filename);

private:
    /**
     * given the central coordinates of a cell and its level, compute the bounding corners
     */
    static std::pair<std::vector<ValueType>, std::vector<ValueType>> getBoundingCoords(std::vector<ValueType> centralCoords,
            IndexType level);

    /**
     * Reads a graph in Matrix Market format
     */
    static scai::lama::CSRSparseMatrix<ValueType> readGraphMatrixMarket(const std::string filename, const scai::dmemo::CommunicatorPtr comm);

    /**
     * Reads the coordinates for the MatrixMarket file format.
     */
    static std::vector<DenseVector<ValueType>> readCoordsMatrixMarket ( const std::string filename, const scai::dmemo::CommunicatorPtr comm);

    static void ltrim(std::string &s);

    static void rtrim(std::string &s);

    static void trim(std::string &s);

};//class FileIO

} /* namespace ITI */
