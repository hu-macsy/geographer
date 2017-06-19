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
    IndexType sideLen= 150;
    IndexType dim = 3;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();  
    
    //create weights locally
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1;
            //localPart[i] = rand()%7*comm->getRank()+2;
        }
    }

    Settings settings;
    settings.dimensions = dim;
    settings.numBlocks = 64;
    
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> rectangles= MultiSection<IndexType, ValueType>::getPartition( nodeWeights, sideLen, settings);
    
    std::chrono::duration<double> partitionTime = std::chrono::system_clock::now() - startTime;
    
    if (comm->getRank() == 0) {
        std::cout<< "Time to partition: "<< partitionTime.count() << std::endl;
    }
    // assertions - prints
    
    SCAI_ASSERT( rectangles.size()==settings.numBlocks , "Returned number of rectangles is wrong. Should be "<< settings.numBlocks<< " but it is "<< rectangles.size() );
    
    ValueType totalWeight = 0;
    IndexType totalVolume = 0;
    ValueType minWeight = LONG_MAX, maxWeight = 0;
    /*
    for(int r=0; r<settings.numBlocks; r++){
        struct rectangle thisRectangle = rectangles[r]->getRect();
        
        if( comm->getRank()==0 and settings.numBlocks<20){
            thisRectangle.print();
        }
        
        // the sum of all the volumes must be equal the volume of the grid: sideLen^dim
        ValueType thisVolume = 1;
        for(int d=0; d<dim; d++){
            thisVolume = thisVolume * (thisRectangle.top[d]-thisRectangle.bottom[d]);
        }
        totalVolume += thisVolume;
        
        ValueType thisWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, thisRectangle, sideLen, settings);
        SCAI_ASSERT( thisWeight==thisRectangle.weight, "wrong weight calculation");
        SCAI_ASSERT( thisVolume==thisWeight , "This rectangle's area= "<< thisVolume << " and should be equal to its weight that is= "<< thisWeight);
        
        if( thisWeight<minWeight ){
            minWeight = thisWeight;
        }
        if( thisWeight>maxWeight ){
            maxWeight = thisWeight;
        }
        
        totalWeight += thisWeight;
        
        // check if this rectangle overlaps with some other rectangle
        for(int r2=0; r2<settings.numBlocks; r2++){
            if( r2==r) continue; // do not self-check
            
            struct rectangle otherRectangle = rectangles[r2]->getRect();
            bool overlap = true;
            for(int d=0; d<dim; d++){
                if( thisRectangle.bottom[d]>=otherRectangle.top[d] or thisRectangle.top[d]<=otherRectangle.bottom[d] ){
                    // if they do not overlap in dimension then they cannot overlap 
                    overlap = false;
                    break;
                }
            }
            
            if( comm->getRank()==0 and overlap ){
                //PRINT0("Found overlapping rectangles:");
                thisRectangle.print();
                otherRectangle.print();
                throw std::runtime_error("The rectangles above overlap.");
            }
        }
    }
    PRINT0( "averageWeight= "<< N/settings.numBlocks );
    PRINT0( "minWeight= "<< minWeight << " , maxWeight= "<< maxWeight );
    
    //all points are covered by a rectangle
    ValueType sumWeight = nodeWeights.sum().Scalar::getValue<ValueType>();
    SCAI_ASSERT( totalWeight==sumWeight , "sum of all rectangles weight= "<< totalWeight << " and should be equal the sum of weights= "<< sumWeight);
    // this works even when weights are not 1
    SCAI_ASSERT( totalVolume==N , "total volume= "<< totalVolume << " and should be equal the number of points= "<< N);
    */
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, test1DPartition){
 
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen= 20;
    IndexType dim = 2;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    ValueType origTotalWeight = 0;
    
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            //localPart[i] = 1;
            localPart[i] = rand()%4*comm->getRank()+2;
            origTotalWeight += localPart[i];
        }
    }
    
    origTotalWeight = comm->sum(origTotalWeight);
    
    IndexType k1= 5;
    
    Settings settings;
    settings.dimensions = dim;
    
    rectangle bBox;
    bBox.bottom = {0,0};
    bBox.top = { (ValueType) sideLen, (ValueType) sideLen};
    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );

    // the 1D partition for all dimensions
    for( int dimensionToPartition=0; dimensionToPartition<dim; dimensionToPartition++){
        std::vector<IndexType> dim2proj = {dimensionToPartition};
        
        // get the projection in one dimension
        std::vector<std::vector<ValueType>> projection = MultiSection<IndexType, ValueType>::projection( nodeWeights, root, dim2proj, sideLen, settings);

        for( int proj=0; proj<projection.size(); proj++){
            std::vector<ValueType> part1D, weightPerPart;
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1D( projection[proj],  k1, settings);
            
            //assertions - checks - prints
            
            SCAI_ASSERT( !std::is_sorted(part1D.end(), part1D.begin()) , "part1D is not sorted" );
            for(int i=0; i<part1D.size(); i++){
                SCAI_ASSERT( part1D[i]>=0, "Wrong partition index " << part1D[i] << " in position "<< i );
                SCAI_ASSERT( part1D[i]<sideLen, "Wrong partition index " << part1D[i] << " in position "<< i );
                if(i+1<part1D.size()){
                    SCAI_ASSERT( part1D[i]!=part1D[i+1], "part1D[i]== part1D[i+1]== " << part1D[i] << ". Maybe this should not happen");
                }
            }
            
            // vectors are of expected size
            SCAI_ASSERT( part1D.size()==k1-1, "part1D.size()= "<< part1D.size() << " and is should be = " << k1 -1);
            SCAI_ASSERT( weightPerPart.size()==k1, "weightPerPart.size()= "<< weightPerPart.size() << " and is should be = " << k1 );
            
            // calculate min and max weights
            ValueType minWeight=LONG_MAX, maxWeight=0;
            for(int i=0; i<weightPerPart.size(); i++){
                if( weightPerPart[i]<minWeight ){
                    minWeight = weightPerPart[i];
                }
                if( weightPerPart[i]>maxWeight ){
                    maxWeight = weightPerPart[i];
                }
            }
            
            ValueType maxOverMin = maxWeight/minWeight;
            PRINT0("max weight / min weight = "<< maxOverMin);
            
            ValueType totalWeight = std::accumulate(weightPerPart.begin(), weightPerPart.end(), 0);
            ValueType averageWeight = totalWeight/k1;
        
            SCAI_ASSERT( totalWeight==origTotalWeight, "totalWeight= "<< totalWeight << " should be= "<< origTotalWeight );
            
            PRINT0("max weight / average =" << maxWeight/averageWeight<< " , min / average =" << minWeight/averageWeight);
            PRINT0("Average weight= "<< averageWeight);
            
            // print weigths for each part
            for(int i=0; i<weightPerPart.size(); i++){
                PRINT0("part "<< i <<" weight: "<< weightPerPart[i]);
            }    
            /*
            if(comm->getRank() ==0){
                for(int i=0; i<part1D.size(); i++){
                    std::cout<< *comm <<": "<< part1D[i] << std::endl;
                }
            }
            */
        }
        // TODO: add more tests
    }
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testRectTree){
    
    rectangle initialRect;
    initialRect.bottom = { 0, 0};
    initialRect.top = { 100, 100};
    
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(initialRect) );

    rectangle r1;
    r1.bottom = { 0,0};
    r1.top = { 50, 50};
    root->insert( r1 );
    
    rectangle r2;
    r2.bottom = { 70, 90};
    r2.top = {100, 100};
    root->insert( r2 );
    
    rectangle r3;
    r3.bottom = {10,20};
    r3.top = {20, 40};
    root->insert( r3 );
    
    rectangle r4;
    r4.bottom = {10, 70};
    r4.top = {40, 80};
    root->insert( r4 );
    
    rectangle r5;
    r5.bottom = {12, 20};
    r5.top = {18, 40};
    root->insert( r5 );
    
    rectangle r6;
    r6.bottom = {30, 10};
    r6.top = {40, 35};
    root->insert( r6 );
    
    /*              root
     *             / |  \
     *           r1  r2  r4
     *          / \
     *         r3  r6
     *          |  
     *         r5   
     */
    
    //checks
    SCAI_ASSERT( root->getSubtreeSize()==7, "Wrong subtree size.");
    SCAI_ASSERT( root->getNumLeaves()==4, "Wrong number of leaves.");
    SCAI_ASSERT( root->getNumLeaves()==root->indexLeaves(0), "Wrong leaf indexing");
    
    std::vector<std::vector<ValueType>> points = {  
                                                    {60.0, 31},     // in r0
                                                    { 3.4, 8.5},    // in r1
                                                    {75.0, 91.0},   // in r2
                                                    {10.2, 20},     // in r1 and r3
                                                    {12.0, 71.5},   // in r4                                                    
                                                    {15.0, 30},     // in r1, r3 and r5
                                                    {33.1, 33.1}    // in r1, r3 and r6 
                                                };
                                                    
    std::shared_ptr<rectCell<IndexType,ValueType>> retRect; 
    
    for(int p=0; p<points.size(); p++){
        retRect = root->contains( points[p] );

        if( retRect!=NULL){
            SCAI_ASSERT( retRect->getRect().bottom[0]<=points[p][0] and retRect->getRect().top[0]>=points[p][0] , "Wrong rectangle");
            SCAI_ASSERT( retRect->getRect().bottom[1]<=points[p][1] and retRect->getRect().top[1]>=points[p][1] , "Wrong rectangle");
        }
    }
    
    std::vector<ValueType> point(2); 
    
    //random points
    srand(time(NULL));
    for(int r=0; r<10; r++){
        point[0] = rand()%100;
        point[1] = rand()%100;
        
        retRect = root->contains( point );
        if( retRect!=NULL){
            SCAI_ASSERT( retRect->getRect().bottom[0]<=point[0] and retRect->getRect().top[0]>=point[0] , "Wrong rectangle");
            SCAI_ASSERT( retRect->getRect().bottom[1]<=point[1] and retRect->getRect().top[1]>=point[1] , "Wrong rectangle");
        }
        /*
        if( retRect!=NULL){
            PRINT("point ("<< point[0]<<", "<< point[1] <<") found in rectangle:");
            retRect->getRect().print();
        }else{
            PRINT("point ("<< point[0]<<", "<< point[1] <<") was not found in any rectangle.");
        }
        */
    }
    
    //all the points
    int numOfPointsIn_r2 =0;
    int numOfPointsIn_r4 =0;
    int numOfPointsIn_r6 =0;
    for( int x=0; x<100; x++){
        for(int y=0; y<100; y++){
            point[0]=x;
            point[1]=y;
            
            retRect = root->contains( point );
            
            if(retRect->getRect()==r2){
                numOfPointsIn_r2++;
            }
            if(retRect->getRect()==r4){
                numOfPointsIn_r4++;
            }
            if(retRect->getRect()==r6){
                numOfPointsIn_r6++;
            }
        }
    }
    int r2Vol, r4Vol, r6Vol;
    r2Vol = (r2.top[0]-r2.bottom[0])*(r2.top[1]-r2.bottom[1]);
    r4Vol = (r4.top[0]-r4.bottom[0])*(r4.top[1]-r4.bottom[1]);
    r6Vol = (r6.top[0]-r6.bottom[0])*(r6.top[1]-r6.bottom[1]);
    SCAI_ASSERT( numOfPointsIn_r2==r2Vol, "Wrong number of points in rectangle r2.");
    SCAI_ASSERT( numOfPointsIn_r4==r4Vol, "Wrong number of points in rectangle r4.");
    SCAI_ASSERT( numOfPointsIn_r6==r6Vol, "Wrong number of points in rectangle r6.");
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testGetRectangleWeight){

    IndexType N= 16;
    IndexType sideLen = 4;
    ValueType nodeW[N] = {  1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 0, 1, 5,
                            3, 4, 5, 6
    };
    scai::lama::DenseVector<ValueType> nodeWeights(N, nodeW);
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    
    nodeWeights.redistribute( blockDist );
    
    Settings settings;
    
    rectangle bBox;        // the sub-rectangles of the initial grid
    
    bBox.bottom = {0, 0};       //  1, 2
    bBox.top = {2, 2};          //  5, 6
    
    ValueType bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, bBox, sideLen, settings);
    SCAI_ASSERT( bBoxWeight==14, "Weight of the bounding box not correct, should be 14 but it is "<< bBoxWeight);
    
    bBox.bottom = {2, 0};       //  9, 0, 1, 5
    bBox.top = {4, 4};          //  3, 4, 5, 6  
    
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, bBox, sideLen, settings);
    SCAI_ASSERT( bBoxWeight==33, "Weight of the bounding box not correct, should be 33 but it is "<< bBoxWeight);
    
                                //  3
    bBox.bottom = {0, 2};       //  7
    bBox.top = {4, 3};          //  1
                                //  5
    
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, bBox, sideLen, settings);
    SCAI_ASSERT( bBoxWeight==16, "Weight of the bounding box not correct, should be 16 but it is "<< bBoxWeight);
}
//---------------------------------------------------------------------------------------    
    
TEST_F(MultiSectionTest, test1DProjection){
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen = 10;
    IndexType dim = 3;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    //create local weights
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1;
        }
    }
    
    // test projection in all dimensions
    
    for(int d=0; d<dim; d++){
        rectangle bBox;
        // for all dimensions i: bottom[i]<top[i] 
        bBox.bottom = {1, 1, 2};
        bBox.top =    {8, 10, 9};
        // create the root of the tree that contains the whole grid
        std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
        
        ValueType bBoxVolume = 1;
        for(int d=0; d<dim; d++){
            bBoxVolume = bBoxVolume * (bBox.top[d]-bBox.bottom[d]);
        }
        
        Settings settings;
        settings.dimensions = dim;
        std::vector<IndexType> dim2proj = {d};
        
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projection( nodeWeights, root, dim2proj, sideLen, settings);
        std::vector<ValueType> projection = projections[0];
        
        //assertions
        const IndexType projLength = bBox.top[dim2proj[0]]-bBox.bottom[dim2proj[0]];
        SCAI_ASSERT( projLength==projection.size(), "Length of projection is not correct");
        
        ValueType projSum = std::accumulate( projection.begin(), projection.end(), 0);
        SCAI_ASSERT( bBoxVolume==projSum, "Volume of bounding box= "<< bBoxVolume<< " and should be equal to the sum of the projection which is "<< projSum );
        
        ValueType bBoxSlice = bBoxVolume/(bBox.top[dim2proj[0]]-bBox.bottom[dim2proj[0]]);
        for(int i=0; i<projection.size(); i++){
            //this only works when dim=2 and nodeWeights=1
            SCAI_ASSERT( projection[i]==bBoxSlice , "projection["<< i<<"]= "<< projection[i] << " should be equal to "<< bBoxSlice );
        }
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
