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

class DiffusionTest : public ::testing::Test {
    protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";

};

TEST_F(DiffusionTest, testConstructLaplacian) {
	std::string fileName = "bubbles-00010.graph";
	std::string file = graphPath + fileName;
	const CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
	const IndexType n = graph.getNumRows();
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));

	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(graph);

	ASSERT_EQ(L.getRowDistribution(), graph.getRowDistribution());
	ASSERT_TRUE(L.isConsistent());

    DenseVector<ValueType> x( n, 1 );
    DenseVector<ValueType> y( L * x );

    ValueType norm = y.maxNorm().Scalar::getValue<ValueType>();
    EXPECT_EQ(norm,0);

    //test consistency under distributions
    const CSRSparseMatrix<ValueType> replicatedGraph(graph, noDist, noDist);
    CSRSparseMatrix<ValueType> LFromReplicated = Diffusion<IndexType, ValueType>::constructLaplacian(replicatedGraph);
    LFromReplicated.redistribute(L.getRowDistributionPtr(), L.getColDistributionPtr());
    CSRSparseMatrix<ValueType> diff (LFromReplicated - L);
    EXPECT_EQ(0, diff.l2Norm().Scalar::getValue<ValueType>());
}

TEST_F(DiffusionTest, benchConstructLaplacian) {
	std::string fileName = "bubbles-00010.graph";
	std::string file = graphPath + fileName;
	const CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );

	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(graph);
}

TEST_F(DiffusionTest, testPotentials) {
    std::string fileName = "bubbles-00010.graph";
    std::string file = graphPath + fileName;
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType n = graph.getNumRows();
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));

	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(graph);

	DenseVector<ValueType> nodeWeights(L.getRowDistributionPtr(),1);
	DenseVector<ValueType> potentials = Diffusion<IndexType, ValueType>::potentialsFromSource(L, nodeWeights, 0);
	ASSERT_EQ(n, potentials.size());
	ASSERT_LT(potentials.sum().Scalar::getValue<ValueType>(), 0.000001);
}

TEST_F(DiffusionTest, testMultiplePotentials) {
	const IndexType numLandmarks = 2;
	std::string fileName = "bubbles-00010.graph";
	std::string file = graphPath + fileName;
	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
	scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
	const IndexType globalN = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();
	scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

	CSRSparseMatrix<ValueType> L = Diffusion<IndexType, ValueType>::constructLaplacian(graph);
	EXPECT_EQ(L.getRowDistribution(), graph.getRowDistribution());


	DenseVector<ValueType> nodeWeights(L.getRowDistributionPtr(),1);

	std::vector<IndexType> nodeIndices(globalN);
	std::iota(nodeIndices.begin(), nodeIndices.end(), 0);

	//broadcast seed value from root to ensure equal pseudorandom numbers.
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
	comm->bcast( seed, 1, 0 );
	srand(seed[0]);

	GraphUtils::FisherYatesShuffle(nodeIndices.begin(), nodeIndices.end(), numLandmarks);

	std::vector<IndexType> landmarks(numLandmarks);
	std::copy(nodeIndices.begin(), nodeIndices.begin()+numLandmarks, landmarks.begin());

	L.redistribute(noDist, noDist);
	nodeWeights.redistribute(noDist);

	DenseMatrix<ValueType> potentials = Diffusion<IndexType, ValueType>::multiplePotentials(L, nodeWeights, landmarks, 1e-5);

	ASSERT_EQ(numLandmarks, potentials.getNumRows());
	ASSERT_EQ(globalN, potentials.getNumColumns());

	std::vector<DenseVector<ValueType> > convertedCoords(numLandmarks);
	for (IndexType i = 0; i < numLandmarks; i++) {
		convertedCoords[i] = DenseVector<ValueType>(globalN,0);
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
