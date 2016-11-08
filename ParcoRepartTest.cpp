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
#include "gtest.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class ParcoRepartTest : public ::testing::Test {

};

TEST_F(ParcoRepartTest, testHilbertIndexUnitSquare) {
  const IndexType dimensions = 2;
  const IndexType n = 8;
  const IndexType recursionDepth = 3;
  ValueType tempArray[2*n] = {0.1, 0.1, 0.13, 0.11, 0.1, 0.6, 0.7, 0.7, 0.55, 0.45, 0.61, 0.1, 0.76, 0.13, 0.88, 0.1};
  DenseVector<ValueType> coordinates(n*dimensions, 0);
  coordinates.setValues(scai::hmemo::HArray<ValueType>(2*n, tempArray));
  const std::vector<ValueType> minCoords({0,0});
  const std::vector<ValueType> maxCoords({1,1});

  std::vector<ValueType> indices(n);
  for (IndexType i = 0; i < n; i++) {
    indices[i] = ParcoRepart<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, i, recursionDepth, minCoords, maxCoords);
    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  for(int j=0;j<n-1;j++){
    EXPECT_LT(indices[j], indices[j+1]);	
  }
//  EXPECT_LT(indices[0], indices[1]);
//  EXPECT_LT(indices[1], indices[2]);
//  EXPECT_LT(indices[2], indices[3]);
}

//-------------------------------------------------------------------------------------------------

TEST_F(ParcoRepartTest, testHilbertIndexUnitSquare_3D) {
  const IndexType dimensions = 3;
  const IndexType n = 5;
  const IndexType recursionDepth = 3;
  ValueType tempArray[3*n] = {0.1, 0.1, 0.13, 0.61, 0.1, 0.36, 0.7, 0.7, 0.35, 0.45, 0.61, 0.71, 0.76, 0.13, 0.88};
  DenseVector<ValueType> coordinates(n*dimensions, 0);
  coordinates.setValues(scai::hmemo::HArray<ValueType>(3*n, tempArray));
  const std::vector<ValueType> minCoords({0,0,0});
  const std::vector<ValueType> maxCoords({1,1,1});

  std::vector<ValueType> indices(n);
  for (IndexType i = 0; i < n; i++) {
    indices[i] = ParcoRepart<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, i, recursionDepth, minCoords, maxCoords);
    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  for(int j=0;j<n-1;j++){
    EXPECT_LT(indices[j], indices[j+1]);	
  }

}


//-----------------------------------------------------------------

TEST_F(ParcoRepartTest, testHilbertIndexDistributedRandom_3D) {
  const IndexType dimensions = 3;
  const IndexType n = 8;
  const IndexType recursionDepth = 4;
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n*dimensions) );
  DenseVector<ValueType> coordinates(dist);

  srand(time(NULL));
  ValueType r;
  ValueType tempArray[n*dimensions];
  for(int i=0; i<n*dimensions; i++){
    r= ((double) rand()/RAND_MAX);
    tempArray[i]= r;
  }
 
  coordinates.setValues(scai::hmemo::HArray<ValueType>(n*dimensions, tempArray));
/*  for(int i=0; i<n; i++){
    for(int j=0; j<dimensions; j++)
      std::cout<< coordinates((i*dimensions+j)) <<", ";
    std::cout <<"\b"<< std::endl;
  }
*/
  scai::dmemo::DistributionPtr distIndices ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

  const IndexType localN = dist->getLocalSize()/dimensions;
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

  //without the next two lines it messes up the indexes in the distribution
  minCoords= {0,0,0};
  maxCoords= {1,1,1};

  std::vector<ValueType> indices(localN);
  for (IndexType i = 0; i < localN; i++) {
    indices[i] = ParcoRepart<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, distIndices->local2global(i), recursionDepth ,minCoords, maxCoords);

    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  
}



TEST_F(ParcoRepartTest, test2DHibertIndex2Point){

  int recursionDepth= 3;	
  int n= 65;					//how many indices we want to try
  int dimensions= 2;
  double index;					//the index of the curve
  std::vector<double> indexV(n);		//a vector of all indices
  DenseVector<ValueType> p;			//a point in 2D calculated from an index in the curve
  std::vector<DenseVector<ValueType>> pointsV(n);	//vectors of all the points
 
  //create indices at random
  srand(time(NULL));
  for(int i=1; i<n; i++){
    index= ((double) rand()/RAND_MAX);
    indexV[i]=index;  
  }

  // a non-random array of points
  //indexV[0]=0.015;
  //for(int i=1; i<n; i++)  
  //  indexV[i]=indexV[i-1]+0.015;

  ValueType hilbertI;
  std::vector<ValueType> hilbertIV(n);
  const std::vector<ValueType> minCoords({0,0});
  const std::vector<ValueType> maxCoords({1,1});  

  //sort the indices and calculate their coordinates in the unit square
  std::sort(indexV.begin(), indexV.end() );
  for(int i=0; i<n; i++){
    p= ParcoRepart<IndexType, ValueType>::Hilbert2DIndex2Point(indexV[i], recursionDepth);
    pointsV.push_back(p);
    hilbertI= ParcoRepart<IndexType, ValueType>::getHilbertIndex(p, dimensions, 0, recursionDepth, minCoords, maxCoords);
    hilbertIV[i]=hilbertI;
  }

  int error_cnt=0;
  for(int i=1; i<n; i++){
	if(hilbertIV[i-1]>hilbertIV[i])
		error_cnt++;
    EXPECT_GE(hilbertIV[i],hilbertIV[i-1]);
  }
  std::cout<<"error cnt="<<error_cnt<<std::endl;
}


//-----------------------------------------------------------------

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
