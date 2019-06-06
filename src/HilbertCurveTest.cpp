#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>

#include "GraphUtils.h"
#include "gtest/gtest.h"
#include "HilbertCurve.h"
#include "MeshGenerator.h"
#include "FileIO.h"


using namespace scai;

using namespace std; //should be avoided, but better here than in header file

namespace ITI {

class HilbertCurveTest : public ::testing::TestWithParam<int> {
	friend class ITI::HilbertCurve<IndexType,ValueType>;	
protected:
	// the directory of all the meshes used
	std::string graphPath = "./meshes/";

};


//-------------------------------------------------------------------------------------------------

/* Read from file and test hilbert indices. No sorting.
 * */
TEST_P(HilbertCurveTest, testHilbertIndexUnitSquare_Local) {
  const IndexType dimensions = GetParam();
  ASSERT_GE(dimensions, 2);
  ASSERT_LE(dimensions, 3);

  const IndexType recursionDepth = 11;
  IndexType N;
  
  std::vector<ValueType> maxCoords(dimensions);

  std::vector<std::vector<ValueType>> convertedCoords;

  if (dimensions == 2) {
    N=16*16;
    convertedCoords.resize(N);
    for (IndexType i = 0; i < N; i++) {
        convertedCoords[i].resize(dimensions);
    }
    std::string coordFile = graphPath + "Grid16x16.xyz";
    std::vector<DenseVector<ValueType>> coords =  FileIO<IndexType, ValueType>::readCoords( coordFile, N, dimensions);
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( N ));

    for(IndexType j=0; j<dimensions; j++){
      //replicate corrdinates
      coords[j].redistribute(noDist);
      scai::hmemo::ReadAccess<ValueType> coordAccess(coords[j].getLocalValues());
      ASSERT_EQ(coordAccess.size(), N);
      for (IndexType i = 0; i < N; i++){
        convertedCoords[i][j] = (coordAccess[i]+0.17)/8.2;
        maxCoords[j] = std::max(maxCoords[j], convertedCoords[i][j]);
      }
    }
  } else {
        N = 7;
            convertedCoords = {
        {0.1, 0.1, 0.13},
        {0.1, 0.61, 0.36},
        {0.7, 0.7, 0.35},
        {0.65, 0.41, 0.71},
        {0.4, 0.13, 0.88},
        {0.2, 0.11, 0.9},
        {0.1, 0.1, 0.95}
      };
      maxCoords = {1.0, 1.0, 1.0};
  }
    
  EXPECT_EQ(convertedCoords.size(), N);
  EXPECT_EQ(convertedCoords[0].size(), dimensions);
  
  const std::vector<ValueType> minCoords(dimensions, 0);
    
  std::vector<ValueType> indices(N);
  for (IndexType i = 0; i < N; i++){
    indices[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex( convertedCoords[i].data(), dimensions, recursionDepth, minCoords, maxCoords);
    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }

  //recover into points, check for nearness
  for (IndexType i = 0; i < N; i++) {
    std::vector<ValueType> point(dimensions,0);
    if (dimensions == 2) {
      point = HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point( indices[i], recursionDepth);
    } else {
      point = HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point( indices[i], recursionDepth);
    }

    ASSERT_EQ(dimensions, point.size());
    for (IndexType d = 0; d < dimensions; d++) {
      EXPECT_NEAR(point[d]*(maxCoords[d] - minCoords[d])+minCoords[d], convertedCoords[i][d], 0.001);
    }
  } 

}

//-------------------------------------------------------------------------------------------------

TEST_P(HilbertCurveTest, testInverseHilbertIndex_Local) {
  const IndexType dimensions = GetParam();
  const IndexType recursionDepth = 7;
  
  ValueType divisor=16;
  for(int i=0; i<divisor; i++){
    std::vector<ValueType> point(dimensions, 0);
    if (dimensions == 2) {
      point = HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point( double(i)/divisor, recursionDepth);
    } else {
      point = HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point( double(i)/divisor, recursionDepth);
    }

    ASSERT_EQ(dimensions, point.size());

    for (IndexType d = 0; d < dimensions; d++) {
      EXPECT_GE(point[d], 0);
      EXPECT_LE(point[d], 1);
    }
  }
    
}    
//-------------------------------------------------------------------------------------------------

/* Read from file and test hilbert indices.
 * */
TEST_F(HilbertCurveTest, testHilbertFromFileNew_Local_2D) {
  const IndexType dimensions = 2;
  const IndexType recursionDepth = 7;
  
  std::string fileName = graphPath + "trace-00008.graph";
  
  // get graph
  scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( fileName );

  const IndexType N = graph.getNumRows();
  const IndexType M = graph.getNumValues();

  std::vector<ValueType> maxCoords({0,0});

  //get coords
  std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( fileName+".xyz", N, dimensions);

  const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));

  for(IndexType j=0; j<dimensions; j++){
	  coords[j].redistribute(noDist);
      maxCoords[j]= coords[j].max();
  }
  EXPECT_EQ(coords[0].size(), N);
  EXPECT_EQ(coords.size(), dimensions);
  
  const std::vector<ValueType> minCoords({0,0});
  
  DenseVector<ValueType> indices(N, 0);
  {
    scai::hmemo::WriteAccess<ValueType> wIndices(indices.getLocalValues());
    scai::hmemo::ReadAccess<ValueType> rCoord0(coords[0].getLocalValues());
    scai::hmemo::ReadAccess<ValueType> rCoord1(coords[1].getLocalValues());
    for (IndexType i = 0; i < N; i++){
      ValueType point[2];
      point[0] = rCoord0[i];
      point[1] = rCoord1[i];
      
      ValueType hilbertIndex = HilbertCurve<IndexType, ValueType>::getHilbertIndex(point, dimensions, recursionDepth, minCoords, maxCoords) ;
      wIndices[i] = hilbertIndex;
      
      EXPECT_LE(wIndices[i], 1);
      EXPECT_GE(wIndices[i], 0);
    }
  }

  IndexType k=60;
  
  DenseVector<IndexType> partition( N, -1);
  DenseVector<IndexType> permutation;
  indices.sort(permutation, true);
  
  permutation.redistribute(noDist);

  //get partition by-hand
  IndexType part =0;
  {
    scai::hmemo::WriteAccess<IndexType> wPart(partition.getLocalValues());
    scai::hmemo::ReadAccess<IndexType> rPermutation(permutation.getLocalValues());

    for(IndexType i=0; i<N; i++){
      part = (int) i*k/N ;
      EXPECT_GE(part, 0);
      EXPECT_LT(part, k);

      wPart[permutation[i]] = part;
    }
  }
}

//-----------------------------------------------------------------
/*
 * Creates random coordinates for n points
*/
TEST_P(HilbertCurveTest, testHilbertIndexRandom_Distributed) {
  const IndexType dimensions = GetParam();
  const IndexType N = 200000;
  const IndexType recursionDepth = 19;
  
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
  const IndexType localN = dist->getLocalSize();
  
  std::vector<DenseVector<ValueType>> coordinates(dimensions);
  for(IndexType i=0; i<dimensions; i++){ 
      coordinates[i].allocate(dist);
      coordinates[i] = static_cast<ValueType>( 0 );
  }

  //broadcast seed value from root to ensure equal pseudorandom numbers.
  ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
  comm->bcast( seed, 1, 0 );
  srand(seed[0]);

  ValueType r;
  for(int i=0; i<localN; i++){      
    for(int d=0; d<dimensions; d++){
        r= double(rand())/RAND_MAX;
        coordinates[d].getLocalValues()[i] = r;
    }
  }

  std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
  std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());

  for (IndexType dim = 0; dim < dimensions; dim++) {
    for (IndexType i = 0; i < N; i++) {
      ValueType coord = coordinates[dim].getValue(i);
      if (coord < minCoords[dim]) minCoords[dim] = coord;
      if (coord > maxCoords[dim]) maxCoords[dim] = coord;
    }
  }

  //communicate minima/maxima over processors. Not strictly necessary right now, since the RNG creates the same vec