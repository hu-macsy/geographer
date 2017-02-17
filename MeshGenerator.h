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
#include <set>
#include <climits>
#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <tuple>    

            
typedef double ValueType;
typedef int IndexType;
            

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
	class MeshGenerator{
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
                
                static void createOctaTreeMesh( scai::lama::CSRSparseMatrix<ValueType> &adjM,  std::vector<DenseVector<ValueType>> &coords, const int numberOfPoints, const ValueType maxCoord);

                static void createOctaTreeMesh_2( scai::lama::CSRSparseMatrix<ValueType> &adjM,  std::vector<DenseVector<ValueType>> &coords, const int numberOfPoints, const ValueType maxCoord);
                
                /** Creates a structed 3D mesh, both the adjacency matrix and the coordinates vectors.
                 * 
                 * @param[out] adjM The adjacency matrix of the output graph. Dimensions are [numPoints[0] x numPoints[1] x numPoints[2]].
                 * @param[out] coords The coordinates of every graph node. coords.size()=2 and coords[i].size()=numPoints[i], so a point i(x,y,z) has coordinates (coords[0][i], coords[1][i], coords[2][i]).
                 * @param[in] maxCoord The maximum value a coordinate can have in each dimension, maxCoord.size()=3.
                 * @param[in] numPoints The number of points in every dimension, numPoints.size()=3.
                 */
                static void createStructured3DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);
                
                /** Creates the adjacency matrix and the coordinate vector for a 3D mesh in a distributed way.
                 */
                static void createStructured3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);

                static void createRandomStructured3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);
                
                    
                /* Creates random points in the cube for the given dimension, points in [0,maxCoord]^dim.
                 */
                static std::vector<DenseVector<ValueType>> randomPoints(int numberOfPoints, int dimensions, ValueType maxCoord);
                
                /* Calculates the 3D distance between two points.
                 */
                static Scalar dist3D(DenseVector<ValueType> p1, DenseVector<ValueType> p2);
                                
                static ValueType dist3DSquared(std::tuple<IndexType, IndexType, IndexType> p1, std::tuple<IndexType, IndexType, IndexType> p2);
                
                /*  Given a (global) index and the size for each dimension (numPpoints.size()=3) calculates the position
                 *  of the index in 3D. The return value is not the coordiantes of the point!
                 */
                static std::tuple<IndexType, IndexType, IndexType> index2_3DPoint(IndexType index,  std::vector<IndexType> numPoints);
        };//class MeshGenerator
        
}//namespace ITI
