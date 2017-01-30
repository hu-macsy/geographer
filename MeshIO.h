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
#include <scai/tracing.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <list>
#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>


#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

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
                /** Creates a random 3D mesh. Adjacency matrix stored in adjM and coordinates of the points in coords.
                 *  Needs O(numberOfPoints^2) time!! Every nodes adds an edge with some of its closest neighbours.
                 *  The time consuming part is to calculate the distance between all nodes.
                 * 
                 * @param[out] adjM The adjecency matrix of the graph to be created.
                 * @param[in] coords The 3D coordinates vector.
                 * @param[in] numberOfPoints The number of points.
                 * @param[in] maxCoord The maximum value a coordinate can have
                 */
                static void createRandom3DMesh( scai::lama::CSRSparseMatrix<ValueType> &adjM,  std::vector<DenseVector<ValueType>> &coords, const int numberOfPoints, const ValueType maxCoord);
                
                /** Creates a structed 3D mesh, both the adjacency matrix and the coordinates vectors.
                 * 
                 * @param[out] adjM The adjacency matrix of the output graph. Dimensions are [numPoints[0] x numPoints[1] x numPoints[2]].
                 * @param[out] coords The coordinates of every graph node. coords.size()=2 and coords[i].size()=numPoints[i], so a point i(x,y,z) has coordinates (coords[0][i], coords[1][i], coords[2][i]).
                 * @param[in] maxCoord The maximum value a coordinate can have in each dimension, maxCoord.size()=3.
                 * @param[in] numPoints The number of points in every dimension, numPoints.size()=3.
                 */
                static void createStructured3DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);
                
                /** Creates the adjacency matrix and the coordiated vector for a 3D mesh in a distributed way.
                 */
                static void createStructured3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);
                
                /* Creates a semi-random 3D mesh.
                 */
                static void createRandomStructured3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);


                /** Given an adjacency matrix and a filename writes the matrix in the file using the METIS format.
                 *  Not distributed.
                 * 
                 * @param[in] adjM The graph's adjacency matrix.
                 * @param[in] filename The file's name to write to/
                 */
                static void writeInFileMetisFormat (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);
                
                /** Given an adjacency matrix and a filename writes the local part of matrix in the file using the METIS format.
                 *  Every proccesor adds his rank in the end of hte file name.
                 * @param[in] adjM The graph's adjacency matrix.
                 * @param[in] filename The file's name to write to/
                 */
                static void writeInFileMetisFormat_dist (const CSRSparseMatrix<ValueType> &adjM, const std::string filename);
                
                /** Given the vector of the coordinates and their dimension, writes them in file "filename".
                 * Coordinates are given as a DenseVector of size dim*numPoints.
                */
                static void writeInFileCoords (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const std::string filename);
           
                /** Reads a graph from filename in METIS format and returns the adjacency matrix.
                 * @param[in] filename The file to read from. In a METIS format.
                 * @param[out] matrix The adjacency matrix of the graph.
                 */
                static void  readFromFile2AdjMatrix( CSRSparseMatrix<ValueType> &matrix, const std::string filename);
                
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
                
                static Scalar dist3D(DenseVector<ValueType> p1, ValueType* p2);
                
                static ValueType dist3DSquared(IndexType* p1, IndexType* p2);
                
                /*  Given a (global) index and the size for each dimension (numPpoints.size()=3) calculates the position
                 *  of the index in 3D. The return value is not the coordiantes of the point!
                 */
                static IndexType* index2_3DPoint(IndexType index,  std::vector<IndexType> numPoints);
        };//class MeshIO
        
}//namespace ITI