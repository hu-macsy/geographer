/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include "ParcoRepart.h"

namespace ITI {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(Matrix &input, DenseVector<ValueType> &coordinates,
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
	*	gather information for space-filling curves
	*/
	//std::vector<


	/**
	*	create space filling curve indices
	*/

	//sort them

	/**
	*	check for uniqueness
	*/

	DenseVector<IndexType> result(input.getNumRows(),0);
	return result;
}

//to force instantiation
template DenseVector<double> ParcoRepart<double, double>::partitionGraph(Matrix &input, DenseVector<double> &coordinates,
					double dimensions,	double k,  double epsilon);

}
