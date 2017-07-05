/*
 * DiffusionTest.cpp
 *
 *  Created on: 26.06.2017
 *      Author: moritzl
 */
#include <numeric>

#include "gtest/gtest.h"
#include "Diffusion.h"
#include "MeshGenerator.h"
#include "FileIO.h"

namespace ITI {

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::lama::DenseMatrix;
using scai::dmemo::DistributionPtr;

typedef double ValueType;
typedef int IndexType;

class DiffusionTest : public ::testing::Test {

};

TEST_F(DiffusionTest, testConstructLaplacian) {
	const IndexType nroot = 4;
	const IndexType n = nroot*nroot*nroot;
	const IndexType dimensions = 3;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	DistributionPtr dist(new scai::dmemo::BlockDistribution(n, comm));
	DistributionPtr noDist(new scai::dmemo::NoDistribution(n));

	CSRSparseMatrix<ValueType> a(dist, noDist);
	std::vector<ValueType> maxCoord(dimensions, nroot);
	std::vector<IndexType> numPoints(dimensions, nroot);

	std::vector<DenseVector<ValueType>> coordinates(dimensions);
	for(IndexType i=0; i<dimensions; i++){
	  coordinates[i] = DenseVector<ValueType>(n, 0);
	}

	MeshGenerator<IndexType, ValueType>::createStructured3DMesh(a, coordinates, maxCoord, numPoints);
	a.redistribute(dist, noDist);
	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(a);

	ASSERT_TRUE(L.isConsistent());

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
	DenseVector<ValueType> potentials = Diffusion<IndexType, ValueType>::potentialsFromSource(L, nodeWeights, 0);
	ASSERT_EQ(n, potentials.size());
	ASSERT_LT(potentials.sum().Scalar::getValue<ValueType>(), 0.000001);
}

TEST_F(DiffusionTest, testMultiplePotentials) {
	const IndexType numLandmarks = 2;
	std::string path = "meshes/bubbles/";
	std::string fileName = "bubbles-00010.graph";
	std::string file = path + fileName;
	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
	const IndexType n = graph.getNumRows();
	scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
	graph.redistribute(noDist, noDist);

	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(graph);

	DenseVector<IndexType> nodeWeights(n,1);

	std::vector<IndexType> nodeIndices(n);
	std::iota(nodeIndices.begin(), nodeIndices.end(), 0);

	Diffusion<IndexType, ValueType>::FisherYatesShuffle(nodeIndices.begin(), nodeIndices.end(), numLandmarks);

	std::vector<IndexType> landmarks(numLandmarks);
	std::copy(nodeIndices.begin(), nodeIndices.begin()+numLandmarks, landmarks.begin());

	DenseMatrix<ValueType> potentials = Diffusion<IndexType, ValueType>::multiplePotentials(L, nodeWeights, landmarks, 1e-5);
	ASSERT_EQ(numLandmarks, potentials.getNumRows());
	ASSERT_EQ(n, potentials.getNumColumns());

	std::vector<DenseVector<ValueType> > convertedCoords(numLandmarks);
	for (IndexType i = 0; i < numLandmarks; i++) {
		convertedCoords[i] = DenseVector<ValueType>(n,0);
		potentials.getLocalRow(convertedCoords[i].getLocalValues(), i);
	}
	FileIO<IndexType, ValueType>::writeCoords(convertedCoords, "diffusion-coords.xyz");

}

TEST_F(DiffusionTest, testConstructFJLTMatrix) {
	const ValueType epsilon = 0.1;
	const IndexType n = 10000;
	const IndexType origDimension = 20;
	CSRSparseMatrix<ValueType> fjlt = Diffusion<IndexType, ValueType>::constructFJLTMatrix(epsilon, n, origDimension);
	EXPECT_EQ(origDimension, fjlt.getLocalNumColumns());
}
} /* namespace ITI */
