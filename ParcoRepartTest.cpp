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
#include <fstream>
#include <iostream>

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
  ValueType tempArray[3*n] = {0.1, 0.1, 0.13, 0.1, 0.61, 0.36, 0.7, 0.7, 0.35, 0.65, 0.41, 0.71, 0.4, 0.13, 0.88};
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
  //std::cout<<coordinates.size() <<"##"<< sizeof(tempArray)/sizeof(*tempArray)<< std::endl;
  //coordinates.setValues(n*dimensions, tempArray);

  for(int i=0; i<n; i++){
    for(int j=0; j<dimensions; j++){
      if(dist->isLocal(i*dimensions+j)){
	coordinates.setValue( i*dimensions+j,tempArray[i*dimensions+j]);	
        //std::cout<< coordinates((i*dimensions+j)) <<", ";
      }
    }
    //std::cout<<std::endl;
  }

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
  //minCoords= {0,0,0};
  //maxCoords= {1,1,1};

  std::vector<ValueType> indices(localN);
  for (IndexType i = 0; i < localN; i++) {
    indices[i] = ParcoRepart<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, distIndices->local2global(i), recursionDepth ,minCoords, maxCoords);

    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  
}

//-------------------------------------------------------------------------------------------------

TEST_F(ParcoRepartTest, test2DHibertIndex2Point){

  int recursionDepth= 4;	
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
  indexV[0]=0.015;
  for(int i=1; i<n; i++)  
    indexV[i]=indexV[i-1]+0.015;

  ValueType hilbertI;
  std::vector<ValueType> hilbertIV(n);
  const std::vector<ValueType> minCoords({0,0});
  const std::vector<ValueType> maxCoords({1,1});  

  //sort the indices and calculate their coordinates in the unit square
  std::sort(indexV.begin(), indexV.end() );
  for(int i=0; i<n; i++){
    p= ParcoRepart<IndexType, ValueType>::Hilbert2DIndex2Point(indexV[i], recursionDepth+2);
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

}

//-------------------------------------------------------------------------------------------------

Scalar dist3D(DenseVector<ValueType> p1, DenseVector<ValueType> p2){
  Scalar res0, res1, res2, res;
  res0= p1.getValue(0)-p2.getValue(0);
  res0= res0*res0;
  res1= p1.getValue(1)-p2.getValue(1);
  res1= res1*res1;
  res2= p1.getValue(2)-p2.getValue(2);
  res2= res2*res2;
  res = res0+ res1+ res2;
  return common::Math::sqrt( res.getValue<ScalarRepType>() );
}
//------------------------

typedef struct myclass{
  ValueType index;
  DenseVector<ValueType> p;

  bool operator()(ValueType a, ValueType b){return a<b; }
}myObject;

bool mysort(myObject a, myObject b){ return a.index<b.index; }


/*
* Create points in 3D, either random or in a structured, grid-like way (uncomment the appropriate part),
* and calculate theis hilbert index. Sorts the index and the point in a myclass vector and sorts the vector
* according to the hilbert index. Prints all the points in a file for visualisation.
* Does not work in parallel due to not distibuted sort.
*/
TEST_F(ParcoRepartTest, testHilbert3DPoint2Index){
  int recursionDepth= 4;
  const int n= 512;
  int dimensions= 3;
  ValueType index;			//a number in [0, 1], the index of the Hilbert curve

  std::vector<ValueType> indexV(n);
  DenseVector<ValueType> p;				//a point in 3D
  const std::vector<ValueType> minCoords({0,0,0});
  const std::vector<ValueType> maxCoords({1,1,1});  

  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n*dimensions) );
  DenseVector<ValueType> coordinates(dist);
  scai::dmemo::DistributionPtr distIndices ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

// points at random
/*  srand(time(NULL));
  for(int i=0; i<n*dimensions; i++){
    index= ((double) rand()/RAND_MAX);
    //if(index<0.3)       index+=0.5;
    //if(dist->isLocal(i))
    coordinates.setValue(i, index);  
    //std::cout<<i<<": "<<index<<std::endl;
  }
*/

//points in a grid-like fashion
  IndexType i=0;
  ValueType indexX, indexY, indexZ;
  for(indexZ=0.05; indexZ<=1; indexZ+=0.12)
    for(indexY=0.05; indexY<=1; indexY+=0.12)
      for(indexX=0.05; indexX<=1; indexX+=0.12){
	coordinates.setValue(i,indexX);
	coordinates.setValue(i+1,indexY);
	coordinates.setValue(i+2,indexZ);
	i=i+3;
 }

  std::vector<ValueType> hilbertIndex(n);
  std::vector<myObject> points_sorted(n);
  const IndexType localN = dist->getLocalSize()/dimensions;

  for(int i=0; i<localN; i++){
    hilbertIndex[i]= ParcoRepart<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, distIndices->local2global(i), recursionDepth ,minCoords, maxCoords);
    points_sorted[i].p = DenseVector<ValueType>(3,0);
    points_sorted[i].index= hilbertIndex[i];
    points_sorted[i].p.setValue(0,coordinates.getValue(i*3));
    points_sorted[i].p.setValue(1,coordinates.getValue(i*3+1));
    points_sorted[i].p.setValue(2,coordinates.getValue(i*3+2));
  }
  std::sort(points_sorted.begin(), points_sorted.end(), mysort );

  ValueType max_dist = sqrt(dimensions)/pow(2,recursionDepth);
  Scalar actual_max_dist=0;
  Scalar d;
  std::ofstream f;
  f.open ("hilbert3D.plt");

  int cnt=0;
  f<< points_sorted[0].p.getValue(0).getValue<ValueType>()<<" "<<points_sorted[0].p.getValue(1).getValue<ValueType>()<<" "<<points_sorted[0].p.getValue(2).getValue<ValueType>()<<std::endl;
  for(int i=1; i<localN; i++){
    //std::cout<< points_sorted[i].index <<"\t| "<< points_sorted[i].p.getValue(0)<<", "<<points_sorted[i].p.getValue(1)<<", "<<points_sorted[i].p.getValue(2)<<std::endl;
    f<< points_sorted[i].p.getValue(0).getValue<ValueType>()<<" "<<points_sorted[i].p.getValue(1).getValue<ValueType>()<<" "<<points_sorted[i].p.getValue(2).getValue<ValueType>()<<std::endl;
    if(points_sorted[i].index==points_sorted[i-1].index){
	d= dist3D(points_sorted[i].p, points_sorted[i-1].p);
	EXPECT_LT(d, Scalar( max_dist ));
    }
 }
  f.close();
}

//-------------------------------------------------------------------------------------------------

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

  scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, dim,  k, epsilon);

  EXPECT_EQ(n, partition.size());
  EXPECT_EQ(0, partition.min().getValue<IndexType>());
  EXPECT_EQ(k-1, partition.max().getValue<IndexType>());
  EXPECT_TRUE(partition.getDistribution().isReplicated());//for now

  std::vector<IndexType> subsetSizes(k, 0);//probably replace with some Lama data structure later
  scai::utilskernel::LArray<IndexType> localPartition = partition.getLocalValues();
  for (IndexType i = 0; i < localPartition.size(); i++) {
    IndexType partID = localPartition[i];
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

  scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, dimensions,  k, epsilon);

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


TEST_F(ParcoRepartTest, testImbalance) {
  const IndexType n = 10000;
  const IndexType k = 10;

  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

  //generate random partition
  scai::lama::DenseVector<IndexType> part(dist);
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = rand() % k;
    part.setValue(i, blockId);
  }

  //sanity check for partition generation
  ASSERT_GE(part.min().getValue<ValueType>(), 0);
  ASSERT_LE(part.max().getValue<ValueType>(), k-1);

  ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_GE(imbalance, 0);

  // test perfectly balanced partition
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = i % k;
    part.setValue(i, blockId);
  }
  imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_EQ(0, imbalance);

  //test maximally imbalanced partition
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = 0;
    part.setValue(i, blockId);
  }
  imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_EQ((n/std::ceil(n/k))-1, imbalance);
}

TEST_F(ParcoRepartTest, testCut) {
  const IndexType n = 1000;
  const IndexType k = 10;

  //generate random complete matrix
  scai::lama::CSRSparseMatrix<ValueType>a(n,n);
  scai::lama::MatrixCreator::fillRandom(a, 1);

  //generate balanced partition
  scai::lama::DenseVector<IndexType> part(n, 0);
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = i % k;
    part.setValue(i, blockId);
  }

  //cut should be 10*900 / 2
  const IndexType blockSize = n / k;
  const ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  EXPECT_EQ(k*blockSize*(n-blockSize) / 2, cut);
}

TEST_F(ParcoRepartTest, testFiducciaMattheysesLocal) {
  const IndexType n = 1000;
  const IndexType k = 10;
  const ValueType epsilon = 0.05;
  const IndexType iterations = 1;

  //generate random matrix
  scai::lama::CSRSparseMatrix<ValueType>a(n,n);
  scai::lama::MatrixCreator::buildPoisson(a, 3, 19, 10,10,10);

  //we want a replicated matrix
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));
  a.redistribute(noDistPointer, noDistPointer);

  //generate random partition
  scai::lama::DenseVector<IndexType> part(n, 0);
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = rand() % k;
    part.setValue(i, blockId);
  }

  ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  for (IndexType i = 0; i < iterations; i++) {
    ValueType gain = ParcoRepart<IndexType, ValueType>::fiducciaMattheysesRound(a, part, k, epsilon);

    //check correct gain calculation
    const ValueType newCut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
    EXPECT_EQ(cut - gain, newCut) << "Old cut " << cut << ", gain " << gain << " newCut " << newCut;
    EXPECT_LE(newCut, cut);
    cut = newCut;
  }
  
  //generate balanced partition
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = i % k;
    part.setValue(i, blockId);
  }

  //check correct cut with balanced partition
  cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  ValueType gain = ParcoRepart<IndexType, ValueType>::fiducciaMattheysesRound(a, part, k, epsilon);
  const ValueType newCut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  EXPECT_EQ(cut - gain, newCut);
  EXPECT_LE(newCut, cut);

  //check for balance
  ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_LE(imbalance, epsilon);
}

TEST_F(ParcoRepartTest, testFiducciaMattheysesDistributed) {
  const IndexType n = 1000;
  const IndexType k = 10;
  const ValueType epsilon = 0.05;

  //generate random matrix
  scai::lama::CSRSparseMatrix<ValueType>a(n,n);
  scai::lama::MatrixCreator::buildPoisson(a, 3, 19, 10,10,10);

  //generate balanced partition
  scai::lama::DenseVector<IndexType> part(n, 0);

  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = i % k;
    part.setValue(i, blockId);
  }

  //this object is only necessary since the Google test macro cannot handle multi-parameter templates in its comma resolution
  ParcoRepart<IndexType, ValueType> repart;

  if (!(a.getRowDistributionPtr()->isReplicated() && a.getColDistributionPtr()->isReplicated())) {
    EXPECT_THROW(repart.fiducciaMattheysesRound(a, part, k, epsilon), std::runtime_error);
  }
}


/**
* TODO: test for correct error handling in case of inconsistent distributions
*/



	
} //namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
