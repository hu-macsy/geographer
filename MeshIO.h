/*
 * Mesh3DGen.h
 *
 *  Created on: 22.11.2016
 *      Author: tzovas
 */
#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/Scalar.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/common/Math.hpp>
#include <scai/common/unique_ptr.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <iterator>

//----------------------------------------
//  for parameter input from command line
extern int my_argc;
extern char** my_argv;
//----------------------------------------
using namespace scai;
using namespace scai::lama;

namespace ITI {
	template <typename IndexType, typename ValueType>
	class MeshIO{
            public:
                /*Creates a random 3D mesh. Adjacency matrix stored in adjM and coordinates of the points in coords.
                 * 
                 */
                static void createRandom3DMesh(scai::lama::CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, int numberOfPoints, ValueType maxCoord);
                
                static void createStructured3DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, std::vector<ValueType> maxCoord, std::vector<IndexType> numPoints);

                /* Given an adjacency matrix and a filename writes the matrix in the file using the METIS format.
                 */
                static void writeInFileMetisFormat (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);
                
                /*Given the vector of the coordinates and their dimension, writes them in file "filename".
                 * Coordinates are given as a DenseVector of size dim*numPoints.
                */
                static void writeInFileCoords (const DenseVector<ValueType> &coords, IndexType dimension, const std::string filename);
                
                /* Here, coordintes are a vector of size dim and each coords[i] have numPoints.
                 */
                static void writeInFileCoords (const std::vector<DenseVector<ValueType>> &coords, IndexType dimension, IndexType numPoints, const std::string filename);
                
                /* Reads a graph from filename in METIS format and returns the adjacency matrix.
                 */
                static CSRSparseMatrix<ValueType>  readFromFile2AdjMatrix(const std::string filename);
                
                /** Reads a graph from filename in METIS format and returns the adjacency matrix.
                 * @param[in] filename The file to read from. In a METIS format.
                 * @param[out] matrix The adjacency matrix of the graph.
                 */
                static void  readFromFile2AdjMatrix( CSRSparseMatrix<ValueType> &matrix, dmemo::DistributionPtr distribution, const std::string filename);
                
                static void readFromFile2AdjMatrixDistr( lama::CSRSparseMatrix<ValueType> &matrix, const std::string filename);
                
                //static void readFromFile2AdjMatrix_Boost( lama::CSRSparseMatrix<ValueType> &matrix, dmemo::DistributionPtr  distribution, const std::string filename);

                /* Reads the 2D coordinates from file "filename" and returns then in a DenseVector where the coordiantes
                 * of point i are in [i*2][i*2+1].
                 */
                static void fromFile2Coords_2D( const std::string filename, std::vector<DenseVector<ValueType>> &coords, IndexType numberOfCoords);
                
                /* Reads the 3D coordinates form a file and stores them in coords. The coordinates of point i=(x,y,z) are in coords[0][i], coords[1][i], coords[2][i].
                */
                static void fromFile2Coords_3D( const std::string filename, std::vector<DenseVector<ValueType>> &coords, IndexType numberOfPoints);
                    
                /* Creates random points in the cube for the given dimension, points in [0,maxCoord]^dim.
                 */
                static std::vector<DenseVector<ValueType>> randomPoints(int numberOfPoints, int dimensions, ValueType maxCoord);
                
                /* Calculates the 3D distance between two points.
                 */
                static Scalar dist3D(DenseVector<ValueType> p1, DenseVector<ValueType> p2);
        };//class MeshIO
        
}//namespace ITI