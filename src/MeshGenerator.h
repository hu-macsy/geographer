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
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/common/Math.hpp>
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
#include <random>

/*
#include "quadtree/Point.h"
#include "quadtree/SpatialTree.h"
#include "quadtree/SpatialCell.h"
*/
#include "quadtree/QuadTreeCartesianEuclid.h"

#include "AuxiliaryFunctions.h"
#include "Settings.h"

namespace ITI {
using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;

/** @brief Create distributed, uniform meshes.
*/

template <typename IndexType, typename ValueType>
class MeshGenerator {
public:

    /** Creates a uniform 3D mesh and writes it to a file (as a graph) using the METIS format.

    @param[in] numPoints The number of points in every dimension
    @param[in] filename The name of the file to store the graph.
    */
    static void writeGraphStructured3DMesh_seq( std::vector<IndexType> numPoints, const std::string filename);

    /** Creates a structured 3D mesh sequentially, both the adjacency matrix and the coordinates vectors.
     *
     * @param[out] adjM The adjacency matrix of the output graph. The graph has numPoints[0]*numPoints[1]*numPoints[2] vertices.
     * @param[out] coords The coordinates of every graph node. coords.size()=2 and coords[i].size()=numPoints[i], so a point i(x,y,z) has coordinates (coords[0][i], coords[1][i], coords[2][i]).
     * @param[in] maxCoord The maximum value a coordinate can have in each dimension, maxCoord.size()=3.
     * @param[in] numPoints The number of points in every dimension, numPoints.size()=3.
     */
    static void createStructured3DMesh_seq(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);

    /** Creates the adjacency matrix and the coordinate vector for a 2D or 3D mesh in a distributed way. The graph is already distributed
    	according to some distribution and every PE fills its local part of the graph and the coordinates.

     @warning coords and adjM.rowDistribution() must agree.

     * @param[out] adjM The adjacency matrix of the output graph. The graph has numPoints[0]*numPoints[1]*numPoints[2]* ... number of vertices.
     * @param[out] coords The coordinates of every graph node. coords.size()=dimensions and coords[i].size()=numPoints[i], so a point i(x,y,z)
       has coordinates (coords[0][i], coords[1][i], coords[2][i]).
       @param maxCoord	The maximum coordinate in every dimension.
       @param numPoints The number of points in every dimension.
       @param dimensions The dimensions of the points. It must be:  numPoints.size() = maxCoord.size() = dimensions
    */

    static void createStructuredMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints, const IndexType dimensions);


    static void createRandomStructured3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);

    /** First, it creates points in a cube of side maxCoord around some areas and adds them in a quad tree. After constructing the quad tree
    	it converts it to a graph. The graph has as many vertices as the cells of the quad tree. Two vertices are adjacent in the graph if the
    	corresponding cells are adjacent in the quad tree.

    	@param[out] adjM The resulting graph.
    	@param[out] coords The coordinates of the vertices.
    	@param[in] dimensions The dimensions of the points.
    	@param[in] numberOfAreas The number of areas around which we will add points uniformly at random. Each area is just a point picked at random.
    	@param[in] pointsPerArea The number of points around each selected area.
    	@param[in] maxCoord The maximum of every dimension
    	@param[in] seed The random seed.
     */
    static void createQuadMesh( CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords,const int dimensions, const IndexType numberOfAreas, const IndexType pointsPerArea, const ValueType maxCoord, IndexType seed);

    /** General version for the squared distance that works for arbitrary dimensions.
    */
    template<typename T>
    static ValueType distSquared( const std::vector<T> p1, const std::vector<T> p2){
        SCAI_REGION( "MeshGenerator.distSquared" )

        const IndexType dimensions=p1.size();
        SCAI_ASSERT_EQ_ERROR( p2.size(), dimensions, "The two points must have the same dimension" );

        ValueType distanceSquared=0;

        for( int d=0; d<dimensions; d++) {
            ValueType distThisDim = p1[d]-p2[d];
            distanceSquared += distThisDim*distThisDim;
        }

        return distanceSquared;
    }

private:
    /** Creates the adjacency matrix and the coordinate vector for a 3D mesh in a distributed way. The graph is already distributed
    	according to some distribution and every PE fills its local part of the graph and the coordinates.

     @warning coords and adjM.rowDistribution() must agree.

     * @param[out] adjM The adjacency matrix of the output graph. The graph has numPoints[0]*numPoints[1]*numPoints[2] vertices.
     * @param[out] coords The coordinates of every graph node. coords.size()=2 and coords[i].size()=numPoints[i], so a point i(x,y,z) has coordinates (coords[0][i], coords[1][i], coords[2][i]).
     */
    static void createStructured2D3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const std::vector<ValueType> maxCoord, const std::vector<IndexType> numPoints);

    /**  Create a graph and coordinates from a quadtree.
    */
    static void graphFromQuadtree(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const QuadTreeCartesianEuclid<ValueType> &quad);

    /** Creates random points in the cube for the given dimension, points in [0,maxCoord]^dim.
     */
    static std::vector<DenseVector<ValueType>> randomPoints(IndexType numberOfPoints, int dimensions, ValueType maxCoord);

    /** The squared distance of two 3D points.
      */
    static ValueType dist3DSquared(std::tuple<IndexType, IndexType, IndexType> p1, std::tuple<IndexType, IndexType, IndexType> p2);

};//class MeshGenerator

}//namespace ITI
