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

using namespace std;

namespace ITI {

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::getMinimumNeighbourDistance(const CSRSparseMatrix<ValueType> &input, const DenseVector<ValueType> &coordinates,
 IndexType dimensions) {
	// iterate through matrix to find closest neighbours, implying necessary recursion depth for space-filling curve
	// here it can happen that the closest neighbor is not stored on this processor.

	const scai::dmemo::DistributionPtr coordDist = coordinates.getDistributionPtr();
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType localN = inputDist->getLocalSize();

	if (coordDist->getLocalSize() % int(dimensions) != 0) {
		throw std::runtime_error("Size of coordinate vector no multiple of dimension. Maybe it was split in the distribution?");
	}

	if (!input.getColDistributionPtr()->isReplicated()) {
		throw std::runtime_error("Columns must be replicated.");
	}

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates.getLocalValues();

	const scai::utilskernel::LArray<IndexType>& ia = localStorage.getIA();
    const scai::utilskernel::LArray<IndexType>& ja = localStorage.getJA();
    assert(ia.size() == localN+1);

    ValueType minDistanceSquared = std::numeric_limits<ValueType>::max();
	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];//assuming replicated columns
		assert(ja.size() >= endCols);
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];//big question: does ja give local or global indices?
			const IndexType globalI = inputDist->local2global(i);
			if (neighbor != globalI && coordDist->isLocal(neighbor*dimensions)) {
				const IndexType localNeighbor = coordDist->global2local(neighbor*dimensions);
				ValueType distanceSquared = 0;
				for (IndexType dim = 0; dim < dimensions; dim++) {
					ValueType diff = localPartOfCoords[i*dimensions + dim] - localPartOfCoords[localNeighbor + dim];
					distanceSquared += diff*diff;
				}
				if (distanceSquared < minDistanceSquared) minDistanceSquared = distanceSquared;
			}
		}
	}

	const ValueType minDistance = std::sqrt(minDistanceSquared);
	return minDistance;
}

/**
* possible optimization: check whether all local points lie in the same region and thus have a common prefix
*/

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
		throw std::runtime_error("Coordinate with index " + std::to_string(index) + " is not present on this process.");
	}

	const scai::utilskernel::LArray<ValueType>& myCoords = coordinates.getLocalValues();
	std::vector<ValueType> scaledCoord(dimensions);

	for (IndexType dim = 0; dim < dimensions; dim++) {
		assert(coordDist->isLocal(index*dimensions+dim));
		const Scalar coord = myCoords[coordDist->global2local(index*dimensions+dim)];
		scaledCoord[dim] = (coord.getValue<ValueType>() - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
		if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
			throw std::runtime_error("Coordinate " + std::to_string(coord.getValue<ValueType>()) + " at position " 
				+ std::to_string(index*dimensions + dim) + " does not agree with bounds "
				+ std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
		}
	}
	//cout<<"point: ["<<scaledCoord[0]<<","<<scaledCoord[1]<<"]\n";
	double temp=1;
	long integerIndex = 0;//TODO: also check whether this data type is long enough
	for (IndexType i = 0; i < recursionDepth; i++) {
		int subSquare;
		//two dimensions only, for now
		if (scaledCoord[0] < 0.5) {
			if (scaledCoord[1] < 0.5) {
				subSquare = 0;
				//apply inverse hilbert operator
				temp = scaledCoord[0];
				scaledCoord[0] = 2*scaledCoord[1];
				scaledCoord[1] = 2*temp;
			} else {
				subSquare = 1;
				//apply inverse hilbert operator
				scaledCoord[0] *= 2;
				scaledCoord[1] = 2*scaledCoord[1] -1;
			}
		} else {
			if (scaledCoord[1] < 0.5) {
				subSquare = 3;
				//apply inverse hilbert operator
				temp = scaledCoord[0];
				scaledCoord[0] = -2*scaledCoord[1]+1;
				scaledCoord[1] = -2*temp+2;

			} else {
				subSquare = 2;
				//apply inverse hilbert operator
				scaledCoord[0] = 2*scaledCoord[0]-1;
				scaledCoord[1] = 2*scaledCoord[1]-1;
			}
		}
		integerIndex = (integerIndex << 2) | subSquare;	
	}

	long divisor = 1 << (2*int(recursionDepth));
	double ret = double(integerIndex) / double(divisor);
	return ret; 
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
DenseVector<ValueType> ParcoRepart<IndexType, ValueType>::Hilbert2DIndex2Point(ValueType index, IndexType level){
	//DenseVector<ValueType> p(2,1), ret(2,0);
	DenseVector<ValueType>  p(2,0), ret(2,0);
	ValueType r;
	IndexType q;
	//std::cout<< p(0)+ret(1)<< std::endl;

	if(level==0)
		return ret;
	else{
		q=int(4*index);
    		r= 4*index-q;
		p = ParcoRepart<IndexType, ValueType>::Hilbert2DIndex2Point(r, level-1);
		switch(q){
			case 0: ret.setValue(0, p(1)/2);	ret.setValue(1, p(0)/2);	return ret;
			case 1: ret.setValue(0, p(0)/2);	ret.setValue(1, p(1)/2 +0.5);	return ret;
			case 2: ret.setValue(0, p(0)/2 +0.5);	ret.setValue(1, p(1)/2 +0.5);	return ret;
			case 3: ret.setValue(0, 1-p(1)/2);	ret.setValue(1, 0.5-p(0)/2);	return ret;
		}
	}
	return ret;
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::getHilbertIndex3D(const DenseVector<ValueType> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {

	if (dimensions != 3) {
		throw std::logic_error("Space filling curve for 3 dimensions");
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
		throw std::runtime_error("Coordinate with index " + std::to_string(index) + " is not present on this process.");
	}

	const scai::utilskernel::LArray<ValueType>& myCoords = coordinates.getLocalValues();
	std::vector<ValueType> scaledCoord(dimensions);

	for (IndexType dim = 0; dim < dimensions; dim++) {
		assert(coordDist->isLocal(index*dimensions+dim));
		const Scalar coord = myCoords[coordDist->global2local(index*dimensions+dim)];
//cout<<"### "<< coord.getValue<ValueType>() << endl; 
		scaledCoord[dim] = (coord.getValue<ValueType>() - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
//cout<<"$$$ "<< scaledCoord[dim]<<endl;
		if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
			throw std::runtime_error("Coordinate " + std::to_string(coord.getValue<ValueType>()) + " at position " 
				+ std::to_string(index*dimensions + dim) + " does not agree with bounds "
				+ std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
		}
	}

//	for(int i=0;i<coordinates.size();++i)
//		std::cout<< scaledCoord[i] << std::endl;
	
//	cout<<endl<<__LINE__<<": point"<<" ("<<index<<") = ["<< scaledCoord[0]<<","<<scaledCoord[1]<<","<<scaledCoord[2]<<"]" <<endl;
	ValueType tmpX, tmpY, tmpZ;
	ValueType x ,y ,z; 	//the coordinates each of the three dimensions
	x= scaledCoord[0];
	y= scaledCoord[1];
	z= scaledCoord[2];
	long integerIndex = 0;//TODO: also check whether this data type is long enough
	for (IndexType i = 0; i < recursionDepth; i++) {
		int subSquare;
		if (z < 0.5) {
			if (y < 0.5) {
				if (x <0.5){		//x,y,z <0.5
					subSquare= 0;
					//apply inverse hilbert operator
					tmpX= x;
					x= 2*z;
					z= 2*y;
					y= 2*tmpX;
				} else{			//y,z<0.5 and x>0.5
					subSquare= 1;
					tmpX= x;
					x= 2*y-1;
					y= 2*z;
					z= 2*tmpX;
				}
			} else if (x>0.5){		//z<0.5, y,x>0,5
					subSquare= 2;
					//apply inverse hilbert operator
					tmpX= x;					
					x= 2*y-1;
					y= 2*z;
					z= 2*tmpX-1;
				}else{			//z<0.5, y>0.5, x<0.5
					subSquare= 3;
					x= -2*x+2;
					y= -2*y+1;
					z= -2*z;
				}
		} else if(y>0.5){
				if(x<0.5){ 		//z>0.5, y>0.5, x<0.5
					subSquare= 4;
					x= -2*x+2;
					y= -2*y+1;
					z= 2*z-1;
				} else{			//z>0.5, y>0.5, x>0.5
					subSquare= 5;
					tmpX= x;
					x= 2*y-1;
					y= -2*z+2;
					z= -2*tmpX+2;
				}
			}else if(x<0.5){		//z>0.5, y<0.5, x<0.5
					subSquare= 7;	//care, this is 7, not 6	
					tmpX= x;
					x= -2*z+2;
					z= -2*y+1;
					y= 2*tmpX;
				}else{			//z>0.5, y<0.5, x>0.5
					subSquare= 6;	//this is case 6
					tmpX= x;
					x= 2*y-1;
					y= -2*z +2;
					z= -2*tmpX+1;				
				}		
			
		integerIndex = (integerIndex << 2) | subSquare;
		//cout<<__LINE__<<": ["<<  scaledCoord[0]<<","<<scaledCoord[1]<<","<<scaledCoord[2]<<"]" <<endl;
		//cout<<__LINE__<<": "<< subSquare <<" _ "<< integerIndex<<endl;
		//cout<<"\t["<< scaledCoord[0]<<","<<scaledCoord[1]<<"]"<<endl<<endl;
		
	}

	long divisor = 1 << (2*int(recursionDepth)+1);
	double ret = double(integerIndex) / double(divisor);
//	cout<<"divisor= "<< divisor << " ret value= "<< ret<<endl;
	return ret; 
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, DenseVector<ValueType> &coordinates,
					IndexType dimensions,	IndexType k,  double epsilon) 
{
	/**
	* check input arguments for sanity
	*/
	IndexType n = input.getNumRows();
	if (n*dimensions != coordinates.size()) {
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
	//const ValueType minDistance = getMinimumNeighbourDistance(input, coordinates, dimensions);
	const IndexType recursionDepth = std::log2(n);// std::ceil(std::log2(maxExtent / minDistance) / 2);

	/**
	*	create space filling curve indices.
	*/
	
	scai::lama::DenseVector<ValueType> hilbertIndices(inputDist);
	for (IndexType i = 0; i < localN; i++) {
		IndexType globalIndex = inputDist->local2global(i);
		ValueType globalHilbertIndex = ParcoRepart<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, globalIndex, recursionDepth, minCoords, maxCoords);
		hilbertIndices.setValue(globalIndex, globalHilbertIndex);
	}

	/**
	* now sort the global indices by where they are on the space-filling curve. Since distributed sorting is not yet available, we gather them all and sort them locally
	*/
	
	scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));
	hilbertIndices.redistribute(noDistPointer);

	assert(hilbertIndices.getDistributionPtr()->getLocalSize() == n);
	
	const scai::utilskernel::LArray<ValueType> allHilbertIndices = hilbertIndices.getLocalValues();

	std::vector<IndexType> allGlobalIndices(n);
	IndexType p = 0;
	std::generate(allGlobalIndices.begin(), allGlobalIndices.end(), [&p](){return p++;});

	std::sort(allGlobalIndices.begin(), allGlobalIndices.end(), [&allHilbertIndices](IndexType i, IndexType j){return allHilbertIndices[i] < allHilbertIndices[j];});


	/**
	* check for uniqueness. If not unique, level of detail was insufficient.
	*/


	/**
	* initial partitioning with sfc. Upgrade to chains-on-chains-partitioning later
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

template double ParcoRepart<int, double>::getHilbertIndex3D(const DenseVector<double> &coordinates, int dimensions, int index, int recursionDepth,
	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template double ParcoRepart<int, double>::getMinimumNeighbourDistance(const CSRSparseMatrix<double> &input, const DenseVector<double> &coordinates,
 int dimensions);

template DenseVector<double> ParcoRepart<int, double>::Hilbert2DIndex2Point(double index, int level);
			     
//template struct point ParcoRepart<int, double>::hilbert(double index, int level);

}
