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
			if (neighbor != i && coordDist->isLocal(neighbor)) {
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
ValueType ParcoRepart<IndexType, ValueType>::getHilbertIndex(const DenseVector<ValueType> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
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
		const Scalar coord = coordinates.getValue(index*dimensions + dim);
		scaledCoord[dim] = (coord.getValue<ValueType>() - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
		if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
			throw std::runtime_error("Coordinate " + std::to_string(coord.getValue<ValueType>()) + " does not agree with bounds "
				+ std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
		}
	}

	long integerIndex = 0;//TODO: also check whether this data type is long enough
	for (IndexType i = 0; i < recursionDepth; i++) {
		int subSquare;
		//two dimensions only, for now
		if (scaledCoord[0] < 0.5) {
			if (scaledCoord[1] < 0.5) {
				subSquare = 0;
				//apply inverse hilbert operator
				scaledCoord[0] *= 2;
				scaledCoord[1] *= 2;
			} else {
				subSquare = 1;
				//apply inverse hilbert operator
				scaledCoord[0] *= 2;
				scaledCoord[1] = scaledCoord[1] * 2-1;
			}
		} else {
			if (scaledCoord[1] < 0.5) {
				subSquare = 3;
				//apply inverse hilbert operator
				scaledCoord[0] = -2*scaledCoord[0]+1;
				scaledCoord[1] = -2*scaledCoord[1]*2+2;

			} else {
				subSquare = 2;
				//apply inverse hilbert operator
				scaledCoord[0] = 2*scaledCoord[0]-1;
				scaledCoord[1] = 2*scaledCoord[1]-1;
			}
		}
		integerIndex = (integerIndex << 2) | subSquare;
	}

	long divisor = 1 << (2*int(recursionDepth)+1);
	return double(integerIndex) / double(divisor);
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

	ValueType maxExtent = 0;
	for (IndexType dim = 0; dim < dimensions; dim++) {
		if (maxCoords[dim] - minCoords[dim] > maxExtent) {
			maxExtent = maxCoords[dim] - minCoords[dim];
		}
	}

	//the following is ~5% faster if manually inlined, probably because localPartOfCoords doesn't have to be computed twice
	const ValueType minDistance = getMinimumNeighbourDistance(input, coordinates, dimensions);
	const IndexType recursionDepth = std::ceil(std::log2(maxExtent / minDistance) / 2);

	/**
	*	create space filling curve indices. Since lama sorting is not yet available, we first gather the vector locally, then sort it, then redistribute it
	*/

	scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n*dimensions));
	coordinates.redistribute(noDistPointer);
	assert(coordinates.getDistributionPtr()->getLocalSize() == n*dimensions);
	std::cout << "Redistributed coordinates." << std::endl;

	std::vector<IndexType> allGlobalIndices(n);
	IndexType p = 0;
	std::generate(allGlobalIndices.begin(), allGlobalIndices.end(), [&p](){return p++;});
	
	std::vector<ValueType> hilbertIndices(n);
	for (IndexType i = 0; i < n; i++) {
		IndexType globalIndex = inputDist->local2global(i);
		ValueType globalHilbertIndex = ParcoRepart<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, globalIndex, recursionDepth, minCoords, maxCoords);
		hilbertIndices[globalIndex] = globalHilbertIndex;
	}

	std::sort(allGlobalIndices.begin(), allGlobalIndices.end(), [&hilbertIndices](IndexType i, IndexType j){return hilbertIndices[i] < hilbertIndices[j];});


	/**
	* check for uniqueness. If not unique, level of detail was insufficient.
	*/


	/**
	* initial partitioning. Upgrade to chains-on-chains-partitioning later
	*/
	DenseVector<IndexType> result(n,0);//not distributed right now
	for (IndexType i = 0; i < n; i++) {
		result.setValue(allGlobalIndices[i], int(k*i / n));
	}


	/**
	* local refinement, use Fiduccia-Mattheyses
	*/



	//dummy result
	return result;
}

//to force instantiation
template DenseVector<double> ParcoRepart<double, double>::partitionGraph(CSRSparseMatrix<double> &input, DenseVector<double> &coordinates,
					double dimensions,	double k,  double epsilon);

template double ParcoRepart<int, double>::getHilbertIndex(const DenseVector<double> &coordinates, int dimensions, int index, int recursionDepth,
	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

}
