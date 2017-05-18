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
            //localPart[i] = rand()%10+2;
            localPart[i] = 1;
        }
    }
    IndexType k1= 5;
    IndexType dimensionToPartition = 0;
    Settings settings;
    settings.dimensions = dim;
    
    std::pair<std::vector<ValueType>,std::vector<ValueType>> bBox;
    bBox.first = {0,0};
    bBox.second = {(ValueType) sideLen, (ValueType) sideLen};
    
    // the 1D partition
    std::vector<ValueType> part1D = MultiSection<IndexType, ValueType>::partition1D( nodeWeights, bBox, k1, dimensionToPartition, sideLen, settings);
    
    //assertions
        
    //if(comm->getRank() ==0){
        for(int i=0; i<part1D.size(); i++){
            std::cout<< *comm <<": "<< part1D[i] << std::endl;
        }
    //}
    
    SCAI_ASSERT( !std::is_sorted(part1D.end(), part1D.begin()) , "part1D is not sorted" )
    
    
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
    std::pair<std::vector<ValueType>, std::vector<ValueType>> bBox;
    // for all dimensions i: first[i]<second[i] 
    bBox.first = {1,1};
    bBox.second = {7,7};
    Settings settings;
    settings.dimensions = dim;
    
    std::vector<ValueType> projection = MultiSection<IndexType, ValueType>::projection1D( nodeWeights, bBox, dim2proj, sideLen, settings);
    
    //assertions
    const IndexType projLength = bBox.second[dim2proj]-bBox.first[dim2proj];
    SCAI_ASSERT( projLength==projection.size(), "Length of projection is not correct");
    
    for(int i=0; i<projection.size(); i++){
        //this only works when dim=2 and nodeWeights=1
        //PRINT0("proj["<< i<< "]= "<< projection[i]);
        SCAI_ASSERT( projection[i]== bBox.second[1-dim2proj]-bBox.first[1-dim2proj], "projection["<<i<<"]= "<< projection[i] << " should be equal to "<< bBox.second[1-dim2proj]-bBox.first[1-dim2proj] );
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
    std::pair<std::vector<ValueType>, std::vector<ValueType>> bBox;
    bBox.first = {2,3,1};
    bBox.second = {6,8,6};
    
    //bBox area = number of points in the box
    IndexType bBoxArea = 1;
    for(int i=0; i<dim; i++){
        bBoxArea *= (bBox.second[i]-bBox.first[i]);
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
