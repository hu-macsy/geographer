#pragma once

//#include <scai/lama.hpp>

#include "Settings.h"
#include "Metrics.h"

namespace ITI {

template <typename IndexType, typename ValueType>
class Mapping{
public:

	/**Implementation of the Hoefler, Snir mapping algorithm.

	@param[in] blockGraph The graph to be mapped. Typically, it is created for a 
	partitioned input/application graph calling GraphUtils::getBlockGraph
	@param[in] PEGraph The graph of the psysical network, ie. the processor
	graph. The two graph must have the same number of nodes n.
	#return A vector of size n indicating which block should be mapped to 
	which processor. Example, if ret[4]=10, then block 4 will be mapped to
	processor 10.
	*/
	std::vector<IndexType> torstenMapping_local( 
		scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
		scai::lama::CSRSparseMatrix<ValueType>& PEGraph);

};//class Mapping

}//namespace ITI