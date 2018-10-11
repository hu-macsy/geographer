/*
 * Mapping.cpp
 *
 *  Created on: 11.10.2018
 *      Author: tzovas
 */

#include "Mapping.h"

namespace ITI {


template <typename IndexType, typename ValueType>
std::vector<IndexType> Mapping<IndexType, ValueType>::torstenMapping_local( 
	scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
	scai::lama::CSRSparseMatrix<ValueType>& PEGraph){

}

//to force instantiation
template class Mapping<IndexType, ValueType>;
}