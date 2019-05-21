#pragma once

//#include <scai/lama.hpp>

#include "Settings.h"
#include "Metrics.h"

namespace ITI {

template <typename IndexType, typename ValueType>
class Mapping{

public:
	
	/**Implementation of the Hoefler, Snir mapping algorithm copied from Roland Glantz
	code as found in TiMEr.

	@param[in] blockGraph The graph to be mapped. Typically, it is created for a 
	partitioned input/application graph calling GraphUtils::getBlockGraph
	@param[in] PEGraph The graph of the psysical network, ie. the processor
	graph. The two graph must have the same number of nodes n.
	@return A vector of size n indicating which block should be mapped to 
	which processor. Example, if ret[4]=10, then block 4 will be mapped to
	processor 10.
	**/
	std::vector<IndexType> rolandMapping_local( 
		scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
		scai::lama::CSRSparseMatrix<ValueType>& PEGraph);

	/**Implementation of the Hoefler, Snir mapping algorithm copied from libTopoMap
	library

	input and outpust as above
	**/
	
	// copy and convert/reimplement code from libTopoMap,
	// http://htor.inf.ethz.ch/research/mpitopo/libtopomap/,
	// function TPM_Map_greedy found in file libtopomap.cpp around line 580

	static std::vector<IndexType> torstenMapping_local( 
		const scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
		const scai::lama::CSRSparseMatrix<ValueType>& PEGraph);

	/**Check if a given mapping is valid.
	*/
	static bool isValid( 
		const scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
		const scai::lama::CSRSparseMatrix<ValueType>& PEGraph,
		std::vector<IndexType> mapping);

private:
	class max_compare_func {
		public:
	  	bool operator()(std::pair<double,int> x, std::pair<double,int> y) {
			if(x.first < y.first) return true;
			return false;
		}
	};
};//class Mapping

}//namespace ITI