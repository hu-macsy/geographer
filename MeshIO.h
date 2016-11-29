/*
 * Mesh3DGen.h
 *
 *  Created on: 22.11.2016
 *      Author: tzovas
 */
//#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/Scalar.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/common/Math.hpp>
#include <scai/common/unique_ptr.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>

#include <boost/tokenizer.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <list>
#include <vector>
#include <string>
#include <iostream>

using namespace scai;
using namespace scai::lama;
using namespace std;

namespace ITI {
	template <typename IndexType, typename ValueType>
	class MeshIO{
            public:
                /*Creates a random 3D mesh. Adjacency matrix stored in adjM and coordinates of the points in coords.
                 * 
                 */
                static void create3DMesh(scai::lama::CSRSparseMatrix<ValueType> &adjM, vector<DenseVector<ValueType>> &coords, int numberOfPoints, ValueType maxCoord);
                
                /* Given an adjacency matrix and a filename writes the matrix in the file using the METIS format.
                 */
                static void writeInFileMetisFormat (const CSRSparseMatrix<ValueType> &adjM, const string filename);
                
                /*Given the vector of the coordinates and their dimension, writes them in file "filename".
                */
                static void writeInFileCoords (const DenseVector<ValueType> &coords, IndexType dimension, const string filename);
                
                /* Reads a graph from filename in METIS format and returns the adjacency matrix.
                 */
                static CSRSparseMatrix<ValueType>  fromFile2AdjMatrix(const string filename);
                
                /* Reads the 2D coordinates from file "filename" and returns then in a DenseVector where the coordiantes
                 * of point i are in [i*2][i*2+1].
                 */
                static DenseVector<ValueType> fromFile2Coords_2D( const string filename, IndexType numberOfCoords);
                    
            private:
                /* Creates random points in the cube for the given dimension, points in [0,maxCoord]^dim.
                 */
                static vector<DenseVector<ValueType>> randomPoints(int numberOfPoints, int dimensions, ValueType maxCoord);
                
                /* Calculates the 3D distance between two points.
                 */
                static Scalar dist3D(DenseVector<ValueType> p1, DenseVector<ValueType> p2);
        };//class MeshIO
        
}//namespace ITI