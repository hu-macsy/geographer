#include "MyAlgo.h"

namespace ITI {
	
template<typename IndexType, typename ValueType>
DenseVector<IndexType> MyAlgo<IndexType, ValueType>::partitionGraph(
    CSRSparseMatrix<ValueType> &input,
    std::vector<DenseVector<ValueType>> &coordinates,
    std::vector<DenseVector<ValueType>> &nodeWeights,
    Settings settings,
    struct Metrics& metrics) {
	
	//implementation of the algorithm
	std::cout<< "Starting new algorithm" << std::endl;
	
	//...
	
}//partitionGraph

//other function implementations


//to force instantiation
template class MyAlgo<IndexType, ValueType>;

}//ITI

