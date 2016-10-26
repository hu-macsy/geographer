/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include "ParcoRepart.h"

namespace ITI {

template<typename IndexType>
scai::lama::DenseVector<IndexType> ParcoRepart<IndexType>::partitionGraph(scai::lama::Matrix &input, scai::lama::Vector &coordinates,
					IndexType dimensions,	IndexType k,  double epsilon) 
{

	/**
	* check input arguments for sanity
	*/
	IndexType n = input.getNumRows();
	if (n != (coordinates.size()/dimensions)) {
		throw std::runtime_error("Matrix has " + std::to_string(n) + " rows, but " + std::to_string(coordinates.size())
		 + " coordinates are given.");
	}
	if (n != input.getNumColumns()) {
		throw std::runtime_error("Matrix must be quadratic.");
	}
	if (k > n) {
		throw std::runtime_error("Creating " + std::to_string(k) + " blocks from " + std::to_string(n) + " elements is impossible.");
	}
	if (epsilon < 0) {
		throw std::runtime_error("Epsilon" + std::to_string(epsilon) + " invalid.");
	}

	/**
	* create space filling curve indices
	*/

	DenseVector<IndexType> result(input.getNumRows(),0);
	return result;
}

//to force instantiation
template scai::lama::DenseVector<double> ParcoRepart<double>::partitionGraph(scai::lama::Matrix &input, scai::lama::Vector &coordinates,
					double dimensions,	double k,  double epsilon);

}
