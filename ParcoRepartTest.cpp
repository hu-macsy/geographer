#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>

#include "ParcoRepart.h"
#include "gtest/gtest.h"

typedef double ValueType;
typedef int IndexType;

namespace ITI {

class ParcoRepartTest : public ::testing::Test {

};

TEST_F(ParcoRepartTest, partitionerInterface) {
	IndexType nroot = 100;
	IndexType n = nroot * nroot;
  IndexType k = 10;
  scai::lama::CSRSparseMatrix<ValueType>a(n,n);
  scai::lama::MatrixCreator::fillRandom(a, 0.01);
  IndexType dim = 2;

	scai::lama::DenseVector<ValueType> coordinates(dim*n, 0);
	for (IndexType i = 0; i < nroot; i++) {
		for (IndexType j = 0; j < nroot; j++) {
 			coordinates.setValue(2*(i*nroot + j), i);
 			coordinates.setValue(2*(i*nroot + j)+1, j);
 		}
 	}

 	scai::lama::DenseVector<ValueType> partition = ParcoRepart<ValueType>::partitionGraph(a, coordinates, dim,	k);

  EXPECT_EQ(partition.size(), n);
}
} //namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
/**
   if (partition.max() >= k) {
    std::cout << "Highest index is " << partition.max() << " instead of " << k-1 << std::endl;
    return 11;
   }
   */