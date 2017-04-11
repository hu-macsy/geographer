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
