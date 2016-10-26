/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include <scai/dmemo/Distribution.hpp>

#include <assert.h>
#include <cmath>
#include <climits>

#include "ParcoRepart.h"

namespace ITI {

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::getMinimumNeighbourDistance(const CSRSparseMatrix<ValueType> &input, const DenseVector<ValueType> &coordinates,
 IndexType dimensions) {
	// iterate through matrix to find closest neighbours, implying necessary recursion depth for space-filling curve
	// here it can happen that the closest neighbor is not stored on this processor.

	const scai::dmemo::DistributionPtr coordDist = coordinates.getDistributionPtr();
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType localN = inputDist->getLocalSize();

	const CSRStorage<ValueType> localStorage = input.getLocalStorage();
	const scai::utilskernel::LArray<ValueType> localPartOfCoords = coordinates.getLocalValues();

	const scai::utilskernel::LArray<IndexType> ia = localStorage.getIA();
    const scai::utilskernel::LArray<IndexType> ja = localStorage.getJA();
    assert(ia.size() == localN+1);

    ValueType minDistanceSquared = std::numeric_limits<ValueType>::max();
	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];//assuming replicated columns
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			if (coordDist->isLocal(neighbor)) {
				ValueType distanceSquared = 0;
				for (IndexType dim = 0; dim < dimensions; dim++) {
					ValueType diff = localPartOfCoords[i*dimensions + dim] - localPartOfCoords[neighbor*dimensions + dim];
					distanceSquared += diff*diff;
				}
				if (distanceSquared < minDistanceSquared) minDistanceSquared = distanceSquared;
			}
		}
	}

	const ValueType minDistance = std::sqrt(minDistanceSquared);
	return minDistance;
}

template<typename IndexType, typename ValueType>
static ValueType getHilbertIndex(const DenseVector<ValueType> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {

	if (dimensions != 2) {
		throw std::logic_error("Space filling curve currently only implemented for two dimensions");
	}

	scai::dmemo::DistributionPtr coordDist = coordinates.getDistributionPtr();

	if (coordDist->getLocalSize() % int(dimensions) != 0) {
		throw std::runtime_error("Size of coordinate vector no multiple of dimension. Maybe it was split in the distribution?");
	}

	size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
	if (recursionDepth > bitsInValueType/dimensions) {
		throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
	}

	if (!coordDist->isLocal(index*dimensions)) {
		throw std::runtime_error("Coordinate with index" + std::to_string(index) + " is not present on this process.");
	}

	std::vector<ValueType> scaledCoord(dimensions);

	for (IndexType dim = 0; dim < dimensions; dim++) {
		scaledCoord[dim] = (coordinates[index*dimensions + dim] - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
	}

	ValueType result = 0;
	for (IndexType i = 0; i < recursionDepth; i++) {
		
	}


}


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
		throw std::runtime_error("Size of coordinate vector no multiple of dimension. Maybe it was split in the distribution?");
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

	//the following is ~5% faster if manually inlined, probably because localPartOfCoords doesn't have to be computed twice
	const ValueType minDistance = getMinimumNeighbourDistance(input, coordinates, dimensions);

	/**
	*	create space filling curve indices
	*/


	//sort them

	/**
	*	check for uniqueness. If not unique, level of detail was insufficient.
	*/
	std::vector<ValueType> hilbertIndices(localN);//in the distributed version, this vector needs to be distributed as well


	/**
	* initial partitioning. Upgrade to chains-on-chains-partitioning later
	*/



	/**
	* local refinement, use Fiduccia-Mattheyses
	*/



	//dummy result
	DenseVector<IndexType> result(input.getNumRows(),0);
	return result;
}

//to force instantiation
template DenseVector<double> ParcoRepart<double, double>::partitionGraph(CSRSparseMatrix<double> &input, DenseVector<double> &coordinates,
					double dimensions,	double k,  double epsilon);

}
