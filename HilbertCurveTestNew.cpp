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

/* Read from file and test hilbert indices. No sorting.
 * */
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
  MeshIO<IndexType, ValueType>::fromFile2Coords_2D("./Grid8x8.xyz", coords,  N);
/*
  for(IndexType j=0; j<dimensions; j++){
      for (IndexType i = 0; i < N; i++){
        coords[j].setValue(i, (coords[j].getValue(i)+0.17)/8.2 );
      }
      maxCoords[j]= coords[j].max().Scalar::getValue<ValueType>();
  }
  */

  EXPECT_EQ(coords[0].size(), N);
  EXPECT_EQ(coords.size(), dimensions);
  
  const std::vector<ValueType> minCoords({0,0});

  DenseVector<ValueType> indices(N, 0);
  for (IndexType i = 0; i < N; i++){
    ValueType point[2] = {coords[0].getValue(i).Scalar::getValue<ValueType>(), coords[1].getValue(i).Scalar::getValue<ValueType>() };
    indices.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex( point,  dimensions, recursionDepth, minCoords, maxCoords) );
    EXPECT_LE(indices.getValue(i).getValue<ValueType>(), 1);
    EXPECT_GE(indices.getValue(i).getValue<ValueType>(), 0);
  }

}


//-------------------------------------------------------------------------------------------------
// Create and test a specific input in 3D.

TEST_F(HilbertCurveTest, testHilbertIndexUnitSquare_Local_3D) {
  const IndexType dimensions = 3;
  const IndexType n = 7;
  const IndexType recursionDepth = 3;
  ValueType tempArray[dimensions*n] = { 0.1, 0.1, 0.13,
                                        0.1, 0.61, 0.36,
                                        0.7, 0.7, 0.35, 
                                        0.65, 0.41, 0.71, 
                                        0.4, 0.13, 0.88, 
                                        0.2, 0.11, 0.9, 
                                        0.1, 0.1, 0.95};
  
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
    ValueType point[3] = {coordinates[0].getValue(i).Scalar::getValue<ValueType>(), coordinates[1].getValue(i).Scalar::getValue<ValueType>(), coordinates[2].getValue(i).Scalar::getValue<ValueType>() };
    indices[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex(point, dimensions, recursionDepth, minCoords, maxCoords);
    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  for(int j=0;j<n-1;j++){
    EXPECT_LT(indices[j], indices[j+1]);	
  }

}


//-----------------------------------------------------------------
//
//Creates random coordinates for n points in 3D
//
TEST_F(HilbertCurveTest, testHilbertIndexRandom_Distributed_3D) {
  const IndexType dimensions = 3;
  const IndexType N = 1000;
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
  
  // create own part of coordinates 
  for(IndexType i=0; i<dimensions; i++){
    SCAI_REGION("testHilbertIndexRandom_Distributed_3D.create_coords");
    scai::hmemo::WriteAccess<ValueType> coordWrite( coordinates[i].getLocalValues() );
    for(IndexType j=0; j<coordWrite.size(); j++){ 
      r= ((double) rand()/RAND_MAX);
      coordWrite.setValue( j , r) ;
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

  //the hilbert indices initiated with the dummy value 19
  DenseVector<ValueType> indices(dist, 19);
  DenseVector<ValueType> indicesOld(dist, 19);
  
  const IndexType localN = dist->getLocalSize();
/*
  std::vector<scai::hmemo::HArray<ValueType>> localCoords(3);
  localCoords[0]= coordinates[0].getLocalValues();
  localCoords[1]= coordinates[1].getLocalValues();
  localCoords[2]= coordinates[2].getLocalValues();
*/
  scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
  scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
  scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );
  
  for (IndexType i = 0; i < localN; i++) {
    //check if the new function return the same index. seems OK.
      
    SCAI_REGION_START("testHilbertIndexRandom_Distributed_3D.getPoint");
    //ValueType point[3] = {coordinates[0].getValue(i).Scalar::getValue<ValueType>(), coordinates[1].getValue(i).Scalar::getValue<ValueType>(), coordinates[2].getValue(i).Scalar::getValue<ValueType>() };    
    ValueType point[3];
    coordAccess0.getValue(point[0], i);
    coordAccess1.getValue(point[1], i);
    coordAccess2.getValue(point[2], i);
    SCAI_REGION_END("testHilbertIndexRandom_Distributed_3D.getPoint");      
    
    indices.getLocalValues()[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex(point , dimensions, recursionDepth, minCoords, maxCoords);    
    
    indicesOld.getLocalValues()[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, dist->local2global(i), recursionDepth ,minCoords, maxCoords) ;
        
    EXPECT_LE(indices.getLocalValues()[i], 1);
    EXPECT_GE(indices.getLocalValues()[i], 0);
    EXPECT_EQ(indices.getLocalValues()[i], indicesOld.getLocalValues()[i]);
  }
  
  DenseVector<IndexType> perm(dist, 19);
  indices.sort(perm, true);
  DenseVector<IndexType> perm2(dist, 19);
  indicesOld.sort(perm2, true);
  
  //check that indices are sorted
  for(IndexType i=0; i<N-1; i++){
    ValueType ind1 = indices.getValue(i ).Scalar::getValue<ValueType>(); 
    ValueType ind2 = indices.getValue(i+1 ).Scalar::getValue<ValueType>();
    EXPECT_LE(ind1 , ind2);
    EXPECT_EQ(perm.getValue(i ).Scalar::getValue<ValueType>() , perm2.getValue(i ).Scalar::getValue<ValueType>() );
  }
  
}

//-------------------------------------------------------------------------------------------------

/*
 * 
//
// Create points in 3D in a structured, grid-like way and calculate theis hilbert index.
// Sort the points by its hilbert index.
// Every processor writes its part of the coordinates in a separate file.


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

 //the hilbert indices initiated with the dummy value 19
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
 for(int i=0; i<localN; i++){
    hilbertIndex.getLocalValues()[i] = HilbertCurve<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, distIndices->local2global(i), recursionDepth ,minCoords, maxCoords) ;
    EXPECT_LE( hilbertIndex.getLocalValues()[i] , 1);
    EXPECT_GE( hilbertIndex.getLocalValues()[i] , 0);
}
  
  hilbertIndex.sort(perm, true);
  
  //test sorting: hilbertIndex(i) < hilbertIdnex(i-1)
  for(int i=1; i<localN; i++){
      EXPECT_GE( hilbertIndex.getLocalValues()[i] , hilbertIndex.getLocalValues()[i-1]); 
  }
  
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

*/

} //namespace ITI



