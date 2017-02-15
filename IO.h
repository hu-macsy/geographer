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

using namespace scai;
using namespace scai::lama;

#include <vector>

namespace ITI {
template <typename IndexType, typename ValueType>
class IO {

public:
	/** Given an adjacency matrix and a filename writes the matrix in the file using the METIS format.
	 *  Not distributed.
	 *
	 * @param[in] adjM The graph's adjacency matrix.
	 * @param[in] filename The file's name to write to/
	 */
	static void writeGraphToFile (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);

	/** Given an adjacency matrix and a filename writes the local part of matrix in the file using the METIS format.
	 *  Every proccesor adds his rank in the end of hte file name.
	 * @param[in] adjM The graph's adjacency matrix.
	 * @param[in] filename The file's name to write to/
	 */
	static void writeGraphToDistributedFiles (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);

	/** Given the vector of the coordinates and their dimension, writes them in file "filename".
	 * Coordinates are given as a DenseVector of size dim*numPoints.
	*/
	static void writeCoordsToFile (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const std::string filename);

	/** Reads a graph from filename in METIS format and returns the adjacency matrix.
	 * @param[in] filename The file to read from. In a METIS format.
	 * @param[out] matrix The adjacency matrix of the graph.
	 */
	static CSRSparseMatrix<ValueType> readGraphFromFile(const std::string filename);

	/* Reads the 2D coordinates from file "filename" and returns then in a DenseVector where the coordinates
	 * of point i are in [i*2][i*2+1].
	 */
	static std::vector<DenseVector<ValueType>> readCoordsFromFile ( std::string filename, IndexType numberOfCoords, IndexType dimension);
};

} /* namespace ITI */
