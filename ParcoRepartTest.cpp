#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>

typedef double ValueType;
typedef int IndexType;

#include "ParcoRepart.h"

int callPartitionerLocally() {
	IndexType nroot = 100;
	IndexType n = nroot * nroot;
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

  	scai::lama::DenseVector<ValueType> partition = ITI::ParcoRepart<ValueType>::partitionGraph(a, coordinates, dim,	10);

  	if (partition.size() != n) std::cout << "Partition has " << partition.size() << " elements instead of " << n << std::endl;
  	return 0;
}

int main(int argn, char ** args) {
	int returnCode = callPartitionerLocally();
	
	return 0;
}