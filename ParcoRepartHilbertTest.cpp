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
    //check if the new function return the same index. seems OK.
    indices[i] = ParcoRepart<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, distIndices->local2global(i), recursionDepth, minCoords, maxCoords);

    EXPECT_LE(indices[i], 1);
    EXPECT_GE(indices[i], 0);
  }
  
}

//-------------------------------------------------------------------------------------------------

TEST_F(ParcoRepartTest, testHibert2DIndex2Point){

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
// Calculates the disatnce in 3D.
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
/*
* Create points in 3D in a structured, grid-like way and calculate theis hilbert index.
* Sorts the index and the point in a myclass vector and sorts the vector
* according to the hilbert index. Prints all the points in a file for visualisation.
*/
TEST_F(ParcoRepartTest, testHilbert3DPoint2Index){
  int recursionDepth= 7;
  int dimensions= 3;
  ValueType startCoord=0, offset=0.063;
  const int n= pow( ceil((1-startCoord)/offset), dimensions);
  ValueType index;			//a number in [0, 1], the index of the Hilbert curve

  std::vector<ValueType> indexV(n);
  DenseVector<ValueType> p;				//a point in 3D
  const std::vector<ValueType> minCoords({0,0,0});
  const std::vector<ValueType> maxCoords({1,1,1});  

  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n*dimensions) );
  DenseVector<ValueType> coordinates(dist);
  scai::dmemo::DistributionPtr distIndices ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

//points in a grid-like fashion
  IndexType i=0;
  ValueType indexX, indexY, indexZ;
  for(indexZ= startCoord; indexZ<=1; indexZ+=offset)
    for(indexY= startCoord; indexY<=1; indexY+=offset)
      for(indexX= startCoord; indexX<=1; indexX+=offset){
	coordinates.setValue(i,indexX);
	coordinates.setValue(i+1,indexY);
	coordinates.setValue(i+2,indexZ);
	i=i+3;
 }

  DenseVector<ValueType> hilbertIndex(n,0);
  DenseVector<IndexType> perm(dist);
  //ValueType tmp;

  const IndexType localN = dist->getLocalSize()/dimensions;

  for(int i=0; i<localN; i++){
    hilbertIndex.setValue(i, ParcoRepart<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, distIndices->local2global(i), recursionDepth ,minCoords, maxCoords) );
  }

  hilbertIndex.sort(perm, 1);

/*
  for(int i=0; i<n; i++){
    std::cout<<i <<": "<<  perm.getValue(i);// << std::endl;
    std::cout<<" , index= "<< hilbertIndex.getValue(i) << std::endl;
  }

  ValueType max_dist = sqrt(dimensions)/pow(2,recursionDepth);
  Scalar actual_max_dist=0;
  Scalar d;
*/
  std::ofstream f;
  f.open ("hilbert3D.plt");

  int cnt=0;

  f	<< coordinates.getValue(perm.getValue(0).getValue<IndexType>()*dimensions).getValue<ValueType>()<<" "\
	<< coordinates.getValue(perm.getValue(0).getValue<IndexType>()*dimensions+1).getValue<ValueType>()<<" "\
	<< coordinates.getValue(perm.getValue(0).getValue<IndexType>()*dimensions+2).getValue<ValueType>()<<std::endl;

  for(int i=1; i<localN; i++){
//std::cout<< i <<": "<< points_sorted[i].index <<"\t| "<< points_sorted[i].p.getValue(0).getValue<ValueType>()<<", "<<points_sorted[i].p.getValue(1).getValue<ValueType>() \
					<<", "<<points_sorted[i].p.getValue(2).getValue<ValueType>()<<std::endl;

//std::cout<< "\t\t| " 	<< coordinates.getValue(perm.getValue(i).getValue<IndexType>()*dimensions).getValue<ValueType>()<<", "\
	  		<< coordinates.getValue(perm.getValue(i).getValue<IndexType>()*dimensions+1).getValue<ValueType>()<<", "\
			<< coordinates.getValue(perm.getValue(i).getValue<IndexType>()*dimensions+2).getValue<ValueType>()\
			<<"   perm(i)= "<< perm.getValue(i).getValue<IndexType>()<< std::endl;

    f	<< coordinates.getValue(perm.getValue(i).getValue<IndexType>()*dimensions).getValue<ValueType>()<<" "\
	<< coordinates.getValue(perm.getValue(i).getValue<IndexType>()*dimensions+1).getValue<ValueType>()<<" "\
	<< coordinates.getValue(perm.getValue(i).getValue<IndexType>()*dimensions+2).getValue<ValueType>()<<std::endl;
 }
  f.close();
}
//-------------------------------------------------------------------------------------------------

	
} //namespace ITI



