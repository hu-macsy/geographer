/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include <scai/dmemo/Distribution.hpp>
#include <assert.h>

#include "ParcoRepart.h"

namespace ITI {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, DenseVector<ValueType> &coordinates,
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

	if (!input.isConsistent()) {
		throw std::runtime_error("Input matrix inconsistent");
	}

	if (k > n) {
		throw std::runtime_error("Creating " + std::to_string(k) + " blocks from " + std::to_string(n) + " elements is impossible.");
	}

	if (epsilon < 0) {
		throw std::runtime_error("Epsilon " + std::to_string(epsilon) + " is invalid.");
	}

	const scai::dmemo::DistributionPtr coordDist = coordinates.getDistributionPtr();
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType localN = inputDist->getLocalSize();

	if (coordDist->getLocalSize() % int(dimensions) != 0) {
		throw std::runtime_error("Coordinate was split during distribution, redistribute first.");
	}

	if (coordDist->getLocalSize() != dimensions*localN) {
		throw std::runtime_error(std::to_string(coordDist->getLocalSize() / dimensions) + " point coordinates, "
		 + std::to_string(localN) + " rows present.");
	}

	/**
	*	gather information for space-filling curves
	*/
	std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());

	const scai::utilskernel::LArray<ValueType> localPartOfCoords = coordinates.getLocalValues();

	//Get extent of coordinates. Can probably speed this up with OpenMP by having thread-local min/max-Arrays and reducing them in the end
	for (IndexType i = 0; i < (localPartOfCoords.size() / dimensions); i++) {
		for (IndexType dim = 0; dim < dimensions; dim++) {
			ValueType coord = localPartOfCoords[i*dimensions + dim];
			if (coord < minCoords[dim]) minCoords[dim] = coord;
			if (coord > maxCoords[dim]) maxCoords[dim] = coord;
		}
	}

	// iterate through matrix to find closest neighbours, implying necessary recursion depth for space-filling curve
	// here it can happen that the closest neighbor is not stored on this processor.
	const CSRStorage<ValueType> localStorage = input.getLocalStorage();

	const scai::utilskernel::LArray<IndexType> ia = localStorage.getIA();
    const scai::utilskernel::LArray<IndexType> ja = localStorage.getJA();
    assert(ia.size() == localN+1);

    ValueType minDistance = std::numeric_limits<ValueType>::max();
	for (IndexType i = 0; i < localN; i++) {
		scai::utilskernel::LArray<ValueType> thisCoords(dimensions);
		for (IndexType dim = 0; dim < dimensions; dim++) {
			thisCoords[dim] = localPartOfCoords[i*dimensions + dim];
		}

		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];//assuming replicated columns
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			if (coordDist->isLocal(neighbor)) {
				scai::utilskernel::LArray<ValueType> neighborCoords(dimensions);
				for (IndexType dim = 0; dim < dimensions; dim++) {
					neighborCoords[dim] = localPartOfCoords[neighbor*dimensions + dim];
				}
				neighborCoords -= thisCoords;
				const ValueType distance = neighborCoords.l2Norm();
				if (distance < minDistance) minDistance = distance;
			}
		}
	}


	/**
	*	create space filling curve indices
	*/

	//sort them

	/**
	*	check for uniqueness. If not unique, level of detail was insufficient.
	*/

	/**
	* initial partitioning. Upgrade to chains-on-chains-partitioning later
	*/

	/**
	*
	*/

	//dummy result
	DenseVector<IndexType> result(input.getNumRows(),0);
	return result;
}

//to force instantiation
template DenseVector<double> ParcoRepart<double, double>::partitionGraph(CSRSparseMatrix<double> &input, DenseVector<double> &coordinates,
					double dimensions,	double k,  double epsilon);

}
