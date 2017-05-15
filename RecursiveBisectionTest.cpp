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
#include "RecursiveBisection.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class RecursiveBisectionTest : public ::testing::Test {

};


TEST_F(RecursiveBisectionTest, testPrefixSum){
 
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen= comm->getSize()*2;
    int dim = 2;
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
            localPart[i] = i;
        }
    }
    IndexType k1= std::pow(comm->getSize(), 0.5);
    IndexType dimensionToPartition = 0;
    Settings settings;
    settings.dimensions = dim;
    
    scai::lama::DenseVector<ValueType> part1D = RecursiveBisection<IndexType, ValueType>::partition1D( nodeWeights, k1, dimensionToPartition, sideLen, settings);
    
    if(comm->getRank() ==0){
        for(int i=0; i<part1D.getLocalValues().size(); i++){
            std::cout<< *comm <<": "<< part1D.getLocalValues()[i] << std::endl;
        }
    }
    
    // TODO: add proper tests
    /*
    std::vector<ValueType> part1DWeights(k1,0);
    for(int h=0; h<part1D.size(); h++){
        for(int i=0; i<localN; i++){
            
        }
    }
    */
}
//---------------------------------------------------------------------------------------

TEST_F(RecursiveBisectionTest, testIndexTo){
    
    IndexType sideLen=4;
    IndexType gridSize;
    IndexType dim;
    
    // 2D case
    dim = 2;
    gridSize= sideLen*sideLen;
    for(int i=0; i<gridSize; i++){
        std::vector<IndexType> p = RecursiveBisection<IndexType, ValueType>::indexToCoords( i , sideLen, dim );
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
        std::vector<IndexType> t = RecursiveBisection<IndexType, ValueType>::indexToCoords( i , sideLen, dim );
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
