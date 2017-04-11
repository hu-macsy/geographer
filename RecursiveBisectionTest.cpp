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

#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "gtest/gtest.h"
#include "AuxiliaryFunctions.h"
#include "RecursiveBisection.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class RecursiveBisectionTest : public ::testing::Test {

};

TEST_F(RecursiveBisectionTest, testPartition){

}

}