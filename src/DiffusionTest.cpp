/*
 * DiffusionTest.cpp
 *
 *  Created on: 26.06.2017
 *      Author: Moritz von Looz
 */
#include <numeric>


#include "gtest/gtest.h"
#include "Diffusion.h"
#include "MeshGenerator.h"
#include "FileIO.h"

#include <scai/dmemo/CyclicDistribution.hpp>

namespace ITI {

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::lama::DenseMatrix;
using scai::dmemo::DistributionPtr;

template<typename T>
class DiffusionTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";

};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(DiffusionTest, testTypes);

//-----------------------------------------------

TYPED_TEST(DiffusionTest, testPotentials) {
    using ValueType = TypeParam;

    //std::string fileName = "trace-00008.graph";
    std::string fileName = "Grid16x16";
    std::string file = DiffusionTest<ValueType>::graphPath + fileName;
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType n = graph.getNumRows();

    CSRSparseMatrix<ValueType> L = GraphUtils<IndexType, ValueType>::constructLaplacian(graph);
    SCAI_ASSERT_EQ_ERROR( graph.getRowDistributionPtr(), L.getRowDistributionPtr(), "distribution mismatch" );

    DenseVector<ValueType> nodeWeights(L.getRowDistributionPtr(),1);
    IndexType source = 0;
    ValueType epsilon = 0.01;
    DenseVector<ValueType> potentials = Diffusion<IndexType, ValueType>::potentialsFromSource(L, nodeWeights, source, epsilon);
    ASSERT_EQ(n, potentials.size());
    ASSERT_LT(potentials.sum(), 0.005);
}

TYPED_TEST(DiffusionTest, testMultiplePotentials) {
    using ValueType = TypeParam;

    const IndexType numLandmarks = 2;
    //std::string fileName = "trace-00008.graph";
    std::string fileName = "Grid16x16";
    std::string file = DiffusionTest<ValueType>::graphPath + fileName;
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const IndexType globalN = inputDist->getGlobalSize();
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

    CSRSparseMatrix<ValueType> L = GraphUtils<IndexType, ValueType>::constructLaplacian(graph);
    EXPECT_EQ(L.getRowDistribution(), graph.getRowDistribution());


    DenseVector<ValueType> nodeWeights(L.getRowDistributionPtr(),1);

    std::vector<IndexType> nodeIndices(globalN);
    std::iota(nodeIndices.begin(), nodeIndices.end(), 0);

    //broadcast seed value from root to ensure equal pseudorandom numbers.
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
    comm->bcast( seed, 1, 0 );
    srand(seed[0]);

    GraphUtils<IndexType, ValueType>::FisherYatesShuffle(nodeIndices.begin(), nodeIndices.end(), numLandmarks);

    std::vector<IndexType> landmarks(numLandmarks);
    std::copy(nodeIndices.begin(), nodeIndices.begin()+numLandmarks, landmarks.begin());

    L.redistribute(noDist, noDist);
    nodeWeights.redistribute(noDist);

    const ValueType epsilon = 0.01;
    DenseMatrix<ValueType> potentials = Diffusion<IndexType, ValueType>::multiplePotentials(L, nodeWeights, landmarks, epsilon);

    ASSERT_EQ(numLandmarks, potentials.getNumRows());
    ASSERT_EQ(globalN, potentials.getNumColumns());

    std::vector<DenseVector<ValueType> > convertedCoords(numLandmarks);
    for (IndexType i = 0; i < numLandmarks; i++) {
        convertedCoords[i] = DenseVector<ValueType>(globalN,0);
        potentials.getLocalRow(convertedCoords[i].getLocalValues(), i);
    }
    FileIO<IndexType, ValueType>::writeCoords(convertedCoords, "partResults/diffusion-coords.xyz");

}

TYPED_TEST(DiffusionTest, testConstructFJLTMatrix) {
    using ValueType = TypeParam;

    const ValueType epsilon = 0.1;
    const IndexType n = 10000;
    const IndexType origDimension = 20;
    CSRSparseMatrix<ValueType> fjlt = GraphUtils<IndexType, ValueType>::constructFJLTMatrix(epsilon, n, origDimension);
    EXPECT_EQ(origDimension, fjlt.getLocalNumColumns());
}
} /* namespace ITI */
