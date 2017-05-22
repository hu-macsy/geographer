#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <algorithm>
#include <memory>
#include <cstdlib>
#include <numeric>

#include "gtest/gtest.h"

#include "MeshGenerator.h"
#include "FileIO.h"
//#include "ParcoRepart.h"
#include "AuxiliaryFunctions.h"
#include "MultiSection.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class MultiSectionTest : public ::testing::Test {

};

TEST_F(MultiSectionTest, testGetPartition){
 
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen= 100;
    IndexType dim = 2;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();  
    
    //create local (random) weights
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            //localPart[i] = rand()%10*comm->getRank()+2;
            localPart[i] = 1;
        }
    }
    
    Settings settings;
    settings.dimensions = dim;
    settings.numBlocks = 6;
    
    std::queue<rectangle> rectangles= MultiSection<IndexType, ValueType>::getPartition( nodeWeights, sideLen, settings);
    
    // assertions - prints
    
    SCAI_ASSERT( rectangles.size()==settings.numBlocks , "Returned number of rectangles is wrong. Should be "<< settings.numBlocks<< " but it is "<< rectangles.size() );
    
    ValueType totalWeight = 0;
    ValueType minWeight = LONG_MAX, maxWeight = 0;
    
    for(int r=0; r<settings.numBlocks; r++){
        struct rectangle thisRectangle = rectangles.front();
        rectangles.pop();
        if( comm->getRank()==0 ){
            thisRectangle.print();
        }
        
        //this only works when dim=2 and nodeWeights=1
        ValueType thisArea = (thisRectangle.top[0]-thisRectangle.bottom[0])*(thisRectangle.top[1]-thisRectangle.bottom[1]);
        ValueType thisWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, thisRectangle, sideLen, settings);
        SCAI_ASSERT( thisArea==thisWeight , "This rectangle's area= "<< thisArea << " and should be equal to its weight that is= "<< thisWeight);
        
        if( thisWeight<minWeight ){
            minWeight = thisWeight;
        }
        if( thisWeight>maxWeight ){
            maxWeight = thisWeight;
        }
        
        totalWeight += thisWeight;
    }
    
    PRINT0( "minWeight= "<< minWeight << " , maxWeight= "<< maxWeight );
    
    //all points are covered by a rectangle
    SCAI_ASSERT( totalWeight==N , "total weight= "<< totalWeight << " and should be equal the number of points= "<< N);
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testCompareExtentDiff){
 
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen= 30;
    IndexType dim = 3;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();  
    
    //create local random weights
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = rand()%10*comm->getRank()+2;
        }
    }
    
    Settings settings;
    settings.dimensions = dim;
    settings.numBlocks = std::pow(2, 4);
    settings.useExtent = true;
    
    std::queue<rectangle> rectanglesExtent = MultiSection<IndexType, ValueType>::getPartition( nodeWeights, sideLen, settings);
    
    // get a second partition using the minimum max-min difference
    settings.useExtent = false;
    
    std::queue<rectangle> rectanglesDiff = MultiSection<IndexType, ValueType>::getPartition( nodeWeights, sideLen, settings);
    
    
    // assertions - prints
    
    SCAI_ASSERT( rectanglesExtent.size()==settings.numBlocks , "Returned number of rectangles is wrong. Should be "<< settings.numBlocks<< " but it is "<< rectanglesExtent.size() );
    
    ValueType minWeightExt = LONG_MAX, maxWeightExt = 0;
    ValueType minWeightDiff = LONG_MAX, maxWeightDiff = 0;
    ValueType totalWeight = 0;
    
    for(int r=0; r<settings.numBlocks; r++){
        struct rectangle thisRectangleExt = rectanglesExtent.front();
        struct rectangle thisRectangleDiff = rectanglesDiff.front();
        rectanglesExtent.pop();
        rectanglesDiff.pop();
        
        ValueType thisWeightExt = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, thisRectangleExt, sideLen, settings);
        
        ValueType thisWeightDiff = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, thisRectangleDiff, sideLen, settings);
        
        if( thisWeightExt<minWeightExt ){
            minWeightExt = thisWeightExt;
        }
        if( thisWeightExt>maxWeightExt ){
            maxWeightExt = thisWeightExt;
        }
        if( thisWeightDiff<minWeightDiff ){
            minWeightDiff = thisWeightDiff;
        }
        if( thisWeightDiff>maxWeightDiff ){
            maxWeightDiff = thisWeightDiff;
        }
        totalWeight += thisWeightExt;
    }
    
    ValueType averageWeight = totalWeight/settings.numBlocks;
    
    PRINT0("averagr weight= "<< averageWeight );
    PRINT0( "useExtent: minWeight= "<< minWeightExt << " , maxWeight= "<< maxWeightExt << ", max/min= "<< maxWeightExt/minWeightExt );
    PRINT0( "  useDiff: minWeight= "<< minWeightDiff << " , maxWeight= "<< maxWeightDiff << ", max/min= "<< maxWeightDiff/minWeightDiff );
        
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, test1DPartition){
 
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen= 10;
    IndexType dim = 2;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    //create local (random) weights
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = rand()%10*comm->getRank()+2;
            //localPart[i] = 1;
        }
    }
    IndexType k1= 5;
    IndexType dimensionToPartition = 0;
    Settings settings;
    settings.dimensions = dim;
    
    rectangle bBox;
    bBox.bottom = {0,0};
    bBox.top = {(ValueType) sideLen, (ValueType) sideLen};
    
    // the 1D partition
    
    // get the projection in one dimension
    std::vector<ValueType> projection = MultiSection<IndexType, ValueType>::projection( nodeWeights, bBox, dimensionToPartition, sideLen, settings);
    std::vector<ValueType> part1D, weightPerPart;
    std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1D( projection,  k1, settings);
    
    //assertions - checks - prints
    
    SCAI_ASSERT( !std::is_sorted(part1D.end(), part1D.begin()) , "part1D is not sorted" );
    
    ValueType minWeight=LONG_MAX, maxWeight=0;
    for(int i=0; i<weightPerPart.size(); i++){
        if( weightPerPart[i]<minWeight ){
            minWeight = weightPerPart[i];
        }
        if( weightPerPart[i]>maxWeight ){
            maxWeight = weightPerPart[i];
        }
    }
    
    //ValueType maxWeightDiff = maxWeight - minWeight;
    ValueType maxOverMin = maxWeight/minWeight;
    PRINT0("max weight part / min weight part = "<< maxOverMin);
    
    ValueType totalWeight = std::accumulate(weightPerPart.begin(), weightPerPart.end(), 0);
    ValueType averageWeight = totalWeight/k1;
    PRINT0("max weight / average =" << maxWeight/averageWeight<< " , min / average =" << minWeight/averageWeight);
   
    for(int i=0; i<weightPerPart.size(); i++){
        PRINT0("part "<< i <<" weight: "<< weightPerPart[i]);
    }    
   
    if(comm->getRank() ==0){
        for(int i=0; i<part1D.size(); i++){
            std::cout<< *comm <<": "<< part1D[i] << std::endl;
        }
    }
    
    
    // TODO: add proper tests
    
    std::vector<ValueType> part1DWeights(k1,0);
    for(int h=0; h<part1D.size(); h++){
        for(int i=0; i<localN; i++){
            
        }
    }
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, test1DProjection){
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen =8;
    IndexType dim = 2;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    //create local (random) weights
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            //localPart[i] = rand()%10+2;
            localPart[i] = 1;
        }
    }
    
    IndexType dim2proj = 0;
    //MultiSection<IndexType,ValueType>::rectangle bBox;
    rectangle bBox;
    // for all dimensions i: first[i]<second[i] 
    bBox.bottom = {1,1};
    bBox.top = {7,7};
    Settings settings;
    settings.dimensions = dim;
    
    std::vector<ValueType> projection = MultiSection<IndexType, ValueType>::projection( nodeWeights, bBox, dim2proj, sideLen, settings);
    
    //assertions
    const IndexType projLength = bBox.top[dim2proj]-bBox.bottom[dim2proj];
    SCAI_ASSERT( projLength==projection.size(), "Length of projection is not correct");
    
    for(int i=0; i<projection.size(); i++){
        //this only works when dim=2 and nodeWeights=1
        //PRINT0("proj["<< i<< "]= "<< projection[i]);
        SCAI_ASSERT( projection[i]== bBox.top[1-dim2proj]-bBox.bottom[1-dim2proj], "projection["<<i<<"]= "<< projection[i] << " should be equal to "<< bBox.top[1-dim2proj]-bBox.bottom[1-dim2proj] );
    }
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testInbBox){
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen =10;
    IndexType dim = 3;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    // for all dimensions i: first[i]<second[i] 
    //MultiSection<IndexType,ValueType>::rectangle bBox;
    rectangle bBox;
    bBox.bottom = {2,3,1};
    bBox.top = {6,8,6};
    
    //bBox area = number of points in the box
    IndexType bBoxArea = 1;
    for(int i=0; i<dim; i++){
        bBoxArea *= (bBox.top[i]-bBox.bottom[i]);
    }
    
    IndexType numPoints=0;
    for(int i=0; i<N; i++){
        std::vector<IndexType> coords = MultiSection<IndexType,ValueType>::indexToCoords(i, sideLen, dim);
        if( MultiSection<IndexType, ValueType>::inBBox(coords, bBox, sideLen) == true){
            ++numPoints;
        }
    }
    PRINT0("bBox area= "<< bBoxArea << " , num of points in bBox= "<< numPoints);
    SCAI_ASSERT( numPoints==bBoxArea, "numPoints= " << numPoints << " should equal to bBoxArea= "<< bBoxArea);
    
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testIndexTo){
    
    IndexType sideLen=4;
    IndexType gridSize;
    IndexType dim;
    
    // 2D case
    dim = 2;
    gridSize= sideLen*sideLen;
    for(int i=0; i<gridSize; i++){
        std::vector<IndexType> p = MultiSection<IndexType, ValueType>::indexToCoords( i , sideLen, dim );
        //PRINT(i<< ": " << p[0] << " , "<< p[1]);
        EXPECT_LE( p[0], sideLen);
        EXPECT_LE( p[1], sideLen);
        EXPECT_GE( p[0], 0);
        EXPECT_GE( p[1], 0);
    }
    
    // 3D case
    dim = 3;
    gridSize= sideLen*sideLen*sideLen;
    for(int i=0; i<gridSize; i++){
        std::vector<IndexType> t = MultiSection<IndexType, ValueType>::indexToCoords( i , sideLen, dim );
        //PRINT(i<< ": " << t[0] << " , "<< t[1] << " , "<< t[2] );
        EXPECT_LE( t[0], sideLen);
        EXPECT_LE( t[1], sideLen);
        EXPECT_LE( t[2], sideLen);
        EXPECT_GE( t[0], 0);
        EXPECT_GE( t[1], 0);
        EXPECT_GE( t[2], 0);
    }
}

}
