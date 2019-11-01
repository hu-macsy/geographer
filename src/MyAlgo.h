#pragma once

#include <scai/lama.hpp>
#include <scai/dmemo/Communicator.hpp>

#include "Settings.h"


namespace ITI {
	
using namespace scai::lama;

/** @brief Main class to partition a graph using new algorithm MyAlgo.
*/

template <typename IndexType, typename ValueType>
class MyAlgo {
public:
    /**
	 * Documentation
     */
    static DenseVector<IndexType> partitionGraph(
        CSRSparseMatrix<ValueType> &input,
        std::vector<DenseVector<ValueType>> &coordinates,
        std::vector<DenseVector<ValueType>> &nodeWeights,
        struct Settings settings,
        struct Metrics& metrics);
	
	// other functions in the class
	
}; //MyAlgo
}  //ITI
