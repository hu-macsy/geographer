/*
 * DiffusionTest.cpp
 *
 *  Created on: 26.06.2017
 *      Author: moritzl
 */
#include "gtest/gtest.h"
#include "Diffusion.h"
#include "MeshGenerator.h"
#include "FileIO.h"

namespace ITI {

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;

typedef double ValueType;
typedef int IndexType;

class DiffusionTest : public ::testing::Test {

};

TEST_F(DiffusionTest, testConstructLaplacian) {
	const IndexType nroot = 2;
	const IndexType n = nroot*nroot*nroot;
	const IndexType dimensions = 3;

	CSRSparseMatrix<ValueType> a(n, n);
	std::vector<ValueType> maxCoord(dimensions, nroot);
	std::vector<IndexType> numPoints(dimensions, nroot);

	std::vector<DenseVector<ValueType>> coordinates(dimensions);
	for(IndexType i=0; i<dimensions; i++){
	  coordinates[i] = DenseVector<ValueType>(n, 0);
	}

	MeshGenerator<IndexType, ValueType>::createStructured3DMesh(a, coordinates, maxCoord, numPoints);
	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(a);

    DenseVector<ValueType> x( n, 1 );
    DenseVector<ValueType> y( L * x );

    ValueType norm = y.maxNorm().Scalar::getValue<ValueType>();
    EXPECT_EQ(0,norm);
}

TEST_F(DiffusionTest, testPotentials) {
    std::string path = "meshes/bubbles/";
    std::string fileName = "bubbles-00010.graph";
    std::string file = path + fileName;
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType n = graph.getNumRows();
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
    graph.redistribute(noDist, noDist);

	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(graph);

	DenseVector<IndexType> nodeWeights(n,1);
	DenseVector<ValueType> potentials = Diffusion<IndexType, ValueType>::potentials(L, nodeWeights, 0);
	ASSERT_EQ(n, potentials.size());


}
} /* namespace ITI */
