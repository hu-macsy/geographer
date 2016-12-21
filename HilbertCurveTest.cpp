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
#include <chrono>

#include "ParcoRepart.h"
#include "gtest/gtest.h"
#include "HilbertCurve.h"
#include "MeshIO.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class HilbertCurveTest : public ::testing::Test {

};


TEST_F(HilbertCurveTest, testHilbertIndexUnitSquare_Local_2D) {
  const IndexType dimensions = 2;
  const IndexType recursionDepth = 7;
  IndexType N=64;
  std::vector<DenseVector<ValueType>> coords(dimensions);
  for(IndexType i=0; i<dimensions; i++){ 
      coords[i].allocate(N);
      coords[i] = static_cast<ValueType>( 0 );
  }
  
  std::vector<ValueType> maxCoords({0,0});
  MeshIO<IndexType, ValueType>::fromFile2Coords_2D("./meshes/my_meshes/Grid8x8.xyz", coords,  N);
  
  for(IndexType j=0; j<dimensions; j++){
      for (IndexType i = 0; i < N; i++){
        coords[j].setValue(i, (coords[j].getValue(i)+0.17)/8.2 );
      }
      maxCoords[j]= coords[j].max().Scalar::getValue<ValueType>();
  }
  
  EXPECT_EQ(coords[0].size(), N);
  EXPECT_EQ(coords.size(), dimensions);
  
  const std::vector<ValueType> minCoords({0,0});

  DenseVector<ValueType> indices(N, 0);
  for (IndexType i = 0; i < N; i++){
    indices.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords, dimensions, i, recursionDepth, minCoords, maxCoords) );
    EXPECT_LE(indices.getValue(i).getValue<ValueType>(), 1);
    EXPECT_GE(indices.getValue(i).getValue<ValueType>(), 0);
  }

}

//-------------------------------------------------------------------------------------------------

TEST_F(HilbertCurveTest, testHilbertIndexUnitSquare_Local_3D) {
  const IndexType dimensions = 3;
  const IndexType n = 7;
  const IndexType recursionDepth = 3;
  ValueType tempArray[dimensions*n] = {0.1, 0.1, 0.13, 0.1, 0.61, 0.36, 0.7, 0.7, 0.35, 0.65, 0.41, 0.71, 0.4, 0.13, 0.88, 0.2, 0.11, 0.9, 0.1, 0.1, 0.95};
  
  std::vector<DenseVector<ValueType>> coordinates(dimensions);
  for(IndexType i=0; i<dimensions; i++){ 
      coordinates[i].allocate(n);
      coordinates[i] = static_cast<ValueType>( 0 );
  }
  for(IndexType i=0; i<n; i++){
    coordinates[0].setValue(i, tempArray[i*dimensions]);
    coordinates[1].setValue(i, tempArray[i*dimensions+1]);
    coordinates[2].setValue(i, tempArray[i*dimensions+2]);
  }
  const std::vector<ValueType> minCoords({0,0,0});
  const std::vector<ValueType> maxCoords({1,1,1});

  std::vector<ValueType> indices(n);
  for (IndexType i = 0; i < n; i++) {
    indices[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, i, recursionDepth, minCoords, maxCoords);
    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  for(int j=0;j<n-1;j++){
    EXPECT_LT(indices[j], indices[j+1]);	
  }

}


//-----------------------------------------------------------------
/*
 * Creates random coordinates for n points
*/
TEST_F(HilbertCurveTest, testHilbertIndexRandom_Distributed_3D) {
  const IndexType dimensions = 3;
  const IndexType N = 50;
  const IndexType recursionDepth = 7;
  
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
  std::vector<DenseVector<ValueType>> coordinates(dimensions);
  for(IndexType i=0; i<dimensions; i++){ 
      coordinates[i].allocate(dist);
      coordinates[i] = static_cast<ValueType>( 0 );
  }

  srand(time(NULL));
  ValueType r;
  
  for(int i=0; i<N; i++){      
    for(int j=0; j<dimensions; j++){
        r= ((double) rand()/RAND_MAX);
	coordinates[j].setValue( i, r);	
    }
  }

  std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
  std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());

  for (IndexType dim = 0; dim < dimensions; dim++) {
    for (IndexType i = 0; i < N; i++) {
      ValueType coord = coordinates[dim].getValue(i).Scalar::getValue<ValueType>();
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

  DenseVector<ValueType> indices(dist, 17);
  DenseVector<IndexType> perm(dist, 17);
  const IndexType localN = dist->getLocalSize();
  
  for (IndexType i = 0; i < localN; i++) {
    //check if the new function return the same index. seems OK.
    indices.getLocalValues()[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, dist->local2global(i), recursionDepth, minCoords, maxCoords);
    
    /*
    EXPECT_LE(indices.getLocalValues()[i], 1);
    EXPECT_GE(indices.getLocalValues()[i], 0);
    */
  }
  
  indices.sort(perm, true);
  
  for(IndexType i=0; i<N-1; i++){
    ValueType ind1 = indices.getValue(i ).Scalar::getValue<ValueType>(); 
    ValueType ind2 = indices.getValue(i+1 ).Scalar::getValue<ValueType>();
    EXPECT_LE(ind1 , ind2);
  }
  
}

//-------------------------------------------------------------------------------------------------

/*
* Create points in 3D in a structured, grid-like way and calculate theis hilbert index.
* Sorts the index and the point in a myclass vector and sorts the vector
* according to the hilbert index. Prints all the points in a file for visualisation.
*/

TEST_F(HilbertCurveTest, testStrucuturedHilbertPoint2IndexWriteInFile_Distributed_3D){
  int recursionDepth= 7;
  int dimensions= 3;
  ValueType startCoord=0, offset=0.0872;
  const int n= pow( ceil((1-startCoord)/offset), dimensions);

  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  std::vector<DenseVector<ValueType>> coordinates(dimensions);
  for(IndexType i=0; i<dimensions; i++){ 
      coordinates[i].allocate(dist);
      coordinates[i] = static_cast<ValueType>( 0 );
  }
  scai::dmemo::DistributionPtr distIndices ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

//points in a grid-like fashion
  IndexType i=0;
  ValueType indexX, indexY, indexZ;
  for(indexZ= startCoord; indexZ<=1; indexZ+=offset)
    for(indexY= startCoord; indexY<=1; indexY+=offset)
      for(indexX= startCoord; indexX<=1; indexX+=offset){
	coordinates[0].setValue(i, indexX);
	coordinates[1].setValue(i, indexY);
	coordinates[2].setValue(i, indexZ);
	++i;
 }

 DenseVector<ValueType> hilbertIndex(dist, 19);
 DenseVector<IndexType> perm(dist);

 std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
 std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
 IndexType N=i;
 
 for (IndexType dim = 0; dim < dimensions; dim++) {
    for (IndexType i = 0; i < N; i++) {
      ValueType coord = coordinates[dim].getValue(i).Scalar::getValue<ValueType>();
      if (coord < minCoords[dim]) minCoords[dim] = coord;
      if (coord > maxCoords[dim]) maxCoords[dim] = coord;
    }
 }
  
 const IndexType localN = dist->getLocalSize();

  //calculate the hilbert index of the points located in the processor and sort them
 for(int i=0; i<localN; i++)
    hilbertIndex.getLocalValues()[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, distIndices->local2global(i), recursionDepth ,minCoords, maxCoords) ;
  
  hilbertIndex.sort(perm, true);
  
  std::ofstream f;
  std::string fileName = std::string("meshes/my_meshes/hilbert3D_" + std::to_string(comm->getRank()) + ".plt");
  f.open(fileName);

  for(int i=0; i<N; i++){
      ValueType x= coordinates[0].getValue( perm.getValue(i).Scalar::getValue<IndexType>() ).Scalar::getValue<ValueType>();
      ValueType y= coordinates[1].getValue( perm.getValue(i).Scalar::getValue<IndexType>() ).Scalar::getValue<ValueType>();
      ValueType z= coordinates[2].getValue( perm.getValue(i).Scalar::getValue<IndexType>() ).Scalar::getValue<ValueType>();
      if( dist->isLocal(i)){
        f << x << " "<< y << " "<< z << std::endl;
      }
 }
 std::cout<< "Coordinates written in file: "<< fileName << " for processor #"<< comm->getRank()<< std::endl;
  f.close();
}
//-------------------------------------------------------------------------------------------------
	
} //namespace ITI



