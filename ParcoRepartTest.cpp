#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>

#include "ParcoRepart.h"
#include "gtest/gtest.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class ParcoRepartTest : public ::testing::Test {

};

TEST_F(ParcoRepartTest, testHilbertIndexUnitSquare) {
  const IndexType dimensions = 2;
  const IndexType n = 4;
  const IndexType recursionDepth = 5;
  ValueType tempArray[8] = {0.1,0.1, 0.1, 0.6, 0.7, 0.7, 0.8, 0.1};
  DenseVector<ValueType> coordinates(n*dimensions, 0);
  coordinates.setValues(scai::hmemo::HArray<ValueType>(8, tempArray));
  const std::vector<ValueType> minCoords({0,0});
  const std::vector<ValueType> maxCoords({1,1});

  std::vector<ValueType> indices(n);
  for (IndexType i = 0; i < n; i++) {
    indices[i] = ParcoRepart<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, i, recursionDepth ,minCoords, maxCoords);
    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  EXPECT_LT(indices[0], indices[1]);
  EXPECT_LT(indices[1], indices[2]);
  EXPECT_LT(indices[2], indices[3]);
}

TEST_F(ParcoRepartTest, testHilbertIndexDistributedRandom) {
  const IndexType dimensions = 2;
  const IndexType n = 2;
  const IndexType recursionDepth = 5;
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n*dimensions) );
  DenseVector<ValueType> coordinates(dist);

  coordinates.setRandom(dist);

  scai::dmemo::DistributionPtr distIndices ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

  const IndexType localN = dist->getLocalSize()/dimensions;
  //std::cout << localN << std::endl;

  std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
  std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());

  const scai::utilskernel::LArray<ValueType> localPartOfCoords = coordinates.getLocalValues();
  assert((localPartOfCoords.size() / dimensions) == localN);
  for (IndexType i = 0; i < localN; i++) {
    for (IndexType dim = 0; dim < dimensions; dim++) {
      ValueType coord = localPartOfCoords[i*dimensions + dim];
      //std::cout << *comm << ", " << i << "." << dim << ":" << coord << std::endl;
      if (coord < minCoords[dim]) minCoords[dim] = coord;
      if (coord > maxCoords[dim]) maxCoords[dim] = coord;
    }
  }
  
  //communicate minima/maxima over processors. Not strictly necessary right now, since the RNG creates the same vector on all processors.
  for (IndexType dim = 0; dim < dimensions; dim++) {
    ValueType globalMin = comm->min(minCoords[dim]);
    ValueType globalMax = comm->max(maxCoords[dim]);
    assert(globalMin <= minCoords[dim]);
    assert(globalMax >= maxCoords[dim]);
    minCoords[dim] = globalMin;
    maxCoords[dim] = globalMax;
  }

  std::vector<ValueType> indices(localN);
  for (IndexType i = 0; i < localN; i++) {
    indices[i] = ParcoRepart<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, distIndices->local2global(i), recursionDepth ,minCoords, maxCoords);
    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
}

TEST_F(ParcoRepartTest, testMinimumNeighborDistanceDistributed) {
  IndexType nroot = 100;
  IndexType n = nroot * nroot;
  IndexType k = 10;
  IndexType dimensions = 2;
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

  scai::lama::CSRSparseMatrix<ValueType>a(dist, noDistPointer);
  scai::lama::MatrixCreator::fillRandom(a, 0.01);//TODO: make this a proper heterogenuous mesh
  
  scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n*dimensions) );
  DenseVector<ValueType> coordinates(coordDist);

  for (IndexType i = 0; i < nroot; i++) {
    for (IndexType j = 0; j < nroot; j++) {
      //this is slightly wasteful, since it also iterates over indices of other processors
      if (coordDist->isLocal(2*(i*nroot + j))) {
        coordinates.setValue(2*(i*nroot + j), i);
        coordinates.setValue(2*(i*nroot + j)+1, j);
      }
    }
  }

  const ValueType minDistance = ParcoRepart<IndexType, ValueType>::getMinimumNeighbourDistance(a, coordinates, dimensions);
  EXPECT_LE(minDistance, nroot*1.5);
  EXPECT_GE(minDistance, 1);
}

TEST_F(ParcoRepartTest, testPartitionBalanceLocal) {
  IndexType nroot = 100;
  IndexType n = nroot * nroot;
  IndexType k = 10;
  scai::lama::CSRSparseMatrix<ValueType>a(n,n);
  scai::lama::MatrixCreator::fillRandom(a, 0.01);
  IndexType dim = 2;
  ValueType epsilon = 0.05;

  scai::lama::DenseVector<ValueType> coordinates(dim*n, 0);
  for (IndexType i = 0; i < nroot; i++) {
    for (IndexType j = 0; j < nroot; j++) {
      coordinates.setValue(2*(i*nroot + j), i);
      coordinates.setValue(2*(i*nroot + j)+1, j);
    }
  }

  scai::lama::DenseVector<ValueType> partition = ParcoRepart<ValueType, ValueType>::partitionGraph(a, coordinates, dim,  k, epsilon);

  EXPECT_EQ(n, partition.size());
  EXPECT_EQ(0, partition.min().getValue<ValueType>());
  EXPECT_EQ(k-1, partition.max().getValue<ValueType>());
  EXPECT_TRUE(partition.getDistribution().isReplicated());//for now

  std::vector<IndexType> subsetSizes(k, 0);//probably replace with some Lama data structure later
  scai::utilskernel::LArray<ValueType> localPartition = partition.getLocalValues();
  for (IndexType i = 0; i < localPartition.size(); i++) {
    ValueType partID = localPartition[i];
    EXPECT_LE(partID, k);
    EXPECT_GE(partID, 0);
    subsetSizes[partID] += 1;
  }
  IndexType optSize = std::ceil(n / k);

  //in a distributed setting, this would need to be communicated and summed
  EXPECT_LE(*std::max_element(subsetSizes.begin(), subsetSizes.end()), (1+epsilon)*optSize);
}

TEST_F(ParcoRepartTest, testPartitionBalanceDistributed) {
  IndexType nroot = 100;
  IndexType n = nroot * nroot;
  IndexType k = 10;
  IndexType dimensions = 2;
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

  scai::lama::CSRSparseMatrix<ValueType>a(dist, noDistPointer);
  scai::lama::MatrixCreator::fillRandom(a, 0.01);//TODO: make this a proper heterogenuous mesh
  
  scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n*dimensions) );
  DenseVector<ValueType> coordinates(coordDist);

  for (IndexType i = 0; i < nroot; i++) {
    for (IndexType j = 0; j < nroot; j++) {
      //this is slightly wasteful, since it also iterates over indices of other processors
      if (coordDist->isLocal(2*(i*nroot + j))) {
        coordinates.setValue(2*(i*nroot + j), i);
        coordinates.setValue(2*(i*nroot + j)+1, j);
      }
    }
  }

  ValueType epsilon = 0.05;

  scai::lama::DenseVector<ValueType> partition = ParcoRepart<ValueType, ValueType>::partitionGraph(a, coordinates, dimensions,  k, epsilon);

  EXPECT_EQ(n, partition.size());
  EXPECT_EQ(0, partition.min().getValue<ValueType>());
  EXPECT_EQ(k-1, partition.max().getValue<ValueType>());
  EXPECT_EQ(a.getRowDistribution(), partition.getDistribution());

  std::vector<IndexType> subsetSizes(k, 0);
  scai::utilskernel::LArray<ValueType> localPartition = partition.getLocalValues();
  for (IndexType i = 0; i < localPartition.size(); i++) {
    ValueType partID = localPartition[i];
    EXPECT_LE(partID, k);
    EXPECT_GE(partID, 0);
    subsetSizes[partID] += 1;
  }
  IndexType optSize = std::ceil(n / k);

  //if we don't have the full partition locally, 
  if (!partition.getDistribution().isReplicated()) {
    //sum block sizes over all processes
    for (IndexType partID = 0; partID < k; partID++) {
      subsetSizes[partID] = comm->sum(subsetSizes[partID]);
    }
  }
  
  EXPECT_LE(*std::max_element(subsetSizes.begin(), subsetSizes.end()), (1+epsilon)*optSize);
}

/**
* TODO: test for correct error handling in case of inconsistent distributions
*/



} //namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}