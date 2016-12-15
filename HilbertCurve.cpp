/*
 * ParcoReportHilbert.cpp
 *
 *  Created on: 15.11.2016
 *      Author: tzovas
 */

#include "ParcoRepart.h"
#include "HilbertCurve.h"

namespace ITI{
    
template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex(const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords){
    
    if (dimensions > 3 || dimensions < 2) {
        throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
    }
    
    if(dimensions==2) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2D( coordinates, dimensions, index, recursionDepth,
            minCoords, maxCoords);
        
    if(dimensions==3) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3D( coordinates, dimensions, index, recursionDepth,
            minCoords, maxCoords);
    //Something is wrong,should not reach this point
    return -1;
}
//-------------------------------------------------------------------------------------------------

/**
* possible optimization: check whether all local points lie in the same region and thus have a common prefix
*/

template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex2D(const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {

        scai::dmemo::DistributionPtr coordDistX = coordinates[0].getDistributionPtr();
        scai::dmemo::DistributionPtr coordDistY = coordinates[1].getDistributionPtr();
        
        std::vector<scai::dmemo::DistributionPtr> coordDist({coordinates[0].getDistributionPtr(), coordinates[1].getDistributionPtr() });

	/*if (dimensions > 3 || dimensions < 2) {
		throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
	}
        */
	/*if (coordDist->getLocalSize() % int(dimensions) != 0) {
		throw std::runtime_error("Size of coordinate vector no multiple of dimension. Maybe it was split in the distribution?");
	}*/

	size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
	if (recursionDepth > bitsInValueType/dimensions) {
		throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
	}

	if (!coordDistX->isLocal(index)) {
		throw std::runtime_error("Coordinate with index " + std::to_string(index) + " is not present on this process.");
	}

	//const scai::utilskernel::LArray<ValueType>& myCoordsX = coordinates[0].getLocalValues();
        //const scai::utilskernel::LArray<ValueType>& myCoordsY = coordinates[1].getLocalValues();
        
        const std::vector<scai::utilskernel::LArray<ValueType>>& myCoords= {coordinates[0].getLocalValues(), coordinates[1].getLocalValues() } ;
        
	std::vector<ValueType> scaledCoord(dimensions);

	for (IndexType dim = 0; dim < dimensions; dim++) {
		assert( coordDist[dim]->isLocal(index) );
		const Scalar coord = myCoords[dim][coordDist[dim]->global2local(index)];
		scaledCoord[dim] = (coord.getValue<ValueType>() - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
		if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
			throw std::runtime_error("Coordinate " + std::to_string(coord.getValue<ValueType>()) + " at position " 
				+ std::to_string(index) + " does not agree with bounds "
				+ std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
		}
	}
    // assert(scaledCoord[0]>=0 && scaledCoord[0]<=1);
    // assert(scaledCoord[1]>=0 && scaledCoord[1]<=1);

    //std::cout<<__LINE__<<":"<< scaledCoord[0]<< ", "<< scaledCoord[1]<< std::endl;
    
    //if(dimensions==2){
	long integerIndex = 0;//TODO: also check whether this data type is long enough
	for (IndexType i = 0; i < recursionDepth; i++) {
		int subSquare;
		//two dimensions only, for now
		if (scaledCoord[0] < 0.5) {
			if (scaledCoord[1] < 0.5) {
				subSquare = 0;
				//apply inverse hilbert operator
				double temp = scaledCoord[0];
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
				double temp = scaledCoord[0];
				scaledCoord[0] = -2*scaledCoord[1]+1;
				scaledCoord[1] = -2*temp+2;
			} else {
				subSquare = 2;
				//apply inverse hilbert operator
				scaledCoord[0] = 2*scaledCoord[0]-1;
				scaledCoord[1] = 2*scaledCoord[1]-1;
			}
		}
                //std::cout<< subSquare<<std::endl;
		integerIndex = (integerIndex << 2) | subSquare;	
	}
	long divisor = 1 << (2*int(recursionDepth));
        return double(integerIndex) / double(divisor);
    //}
        /*else //if dimensions==3
	return HilbertCurve<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, index, recursionDepth ,minCoords, maxCoords);
    */
}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex_noScaling(const DenseVector<ValueType> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {

        scai::dmemo::DistributionPtr coordDist = coordinates.getDistributionPtr();

	/*if (dimensions !=3) {
		throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
	}

	if (coordDist->getLocalSize() % int(dimensions) != 0) {
		throw std::runtime_error("Size of coordinate vector no multiple of dimension. Maybe it was split in the distribution?");
	}
        */
	size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
	if (recursionDepth > bitsInValueType/dimensions) {
		throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
	}

	if (!coordDist->isLocal(index*dimensions)) {
		throw std::runtime_error("Coordinate with index " + std::to_string(index) + " is not present on this process.");
	}
	
        std::vector<ValueType> tmpCoord(dimensions);
        for(IndexType i=0; i<dimensions; i++)
            tmpCoord[i]= coordinates.getValue(index*dimensions +i).Scalar::getValue<ValueType>();

        assert(tmpCoord[0]>=0 && tmpCoord[0]<=maxCoords[0]);
        assert(tmpCoord[1]>=0 && tmpCoord[1]<=maxCoords[1]);

	unsigned long integerIndex = 0;//TODO: also check whether this data type is long enough
	for (IndexType i = 0; i < recursionDepth; i++) {
		int subSquare;
		//two dimensions only, for now
		if (tmpCoord[0] < 0.5*maxCoords[0]) {
			if (tmpCoord[1] < 0.5*maxCoords[1]) {
				subSquare = 0;
				//apply inverse hilbert operator
				double temp = tmpCoord[0];
				tmpCoord[0] = 2*tmpCoord[1];
				tmpCoord[1] = 2*temp;
			} else {
				subSquare = 1;
				//apply inverse hilbert operator
				tmpCoord[0] *= 2;
				tmpCoord[1] = 2*tmpCoord[1] -1*maxCoords[1];
			}
		} else {
			if (tmpCoord[1] < 0.5*maxCoords[1]) {
				subSquare = 3;
				//apply inverse hilbert operator
				double temp = tmpCoord[0];
				tmpCoord[0] = -2*tmpCoord[1]+1*maxCoords[0];
				tmpCoord[1] = -2*temp+2*maxCoords[1];
			} else {
				subSquare = 2;
				//apply inverse hilbert operator
				tmpCoord[0] = 2*tmpCoord[0]-1*maxCoords[0];
				tmpCoord[1] = 2*tmpCoord[1]-1*maxCoords[1];
			}
		}
                //std::cout<< subSquare<<std::endl;
		integerIndex = (integerIndex << 2) | subSquare;	
	}
	unsigned long divisor = 1 << (2*int(recursionDepth));
        return double(integerIndex) / double(divisor);
    //}else //if dimensions==3
//	return HilbertCurve<IndexType, ValueType>::getHilbertIndex3D(coordinates, dimensions, index, recursionDepth ,minCoords, maxCoords);
}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
DenseVector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(ValueType index, IndexType level){
	DenseVector<ValueType>  p(2,0), ret(2,0);
	ValueType r;
	IndexType q;

	if(level==0)
		return ret;
	else{
		q=int(4*index);
    		r= 4*index-q;
		p = HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(r, level-1);
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
/**
* Given a point in 3D it returns its hilbert index, a value in [0,1]. 
**/
template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex3D(const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {

	if (dimensions != 3) {
		throw std::logic_error("Space filling curve for 3 dimensions.");
	}

	//scai::dmemo::DistributionPtr coordDist = coordinates.getDistributionPtr();
	scai::dmemo::DistributionPtr coordDistX = coordinates[0].getDistributionPtr();
        scai::dmemo::DistributionPtr coordDistY = coordinates[1].getDistributionPtr();
        scai::dmemo::DistributionPtr coordDistZ = coordinates[2].getDistributionPtr();
        
        std::vector<scai::dmemo::DistributionPtr> coordDist({coordinates[0].getDistributionPtr(), coordinates[1].getDistributionPtr(), coordinates[2].getDistributionPtr() });

	/*if (coordDist->getLocalSize() % int(dimensions) != 0) {
		throw std::runtime_error("Size of coordinate vector no multiple of dimension. Maybe it was split in the distribution?");
	}*/

	size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
	if ((unsigned int) recursionDepth > bitsInValueType/dimensions) {
		throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
	}

	if (!coordDistX->isLocal(index)) {
		throw std::runtime_error("Coordinate with index " + std::to_string(index) + " is not present on this process.");
	}

	//const scai::utilskernel::LArray<ValueType>& myCoords = coordinates.getLocalValues();
	//const scai::utilskernel::LArray<ValueType>& myCoordsX = coordinates[0].getLocalValues();
        //const scai::utilskernel::LArray<ValueType>& myCoordsY = coordinates[1].getLocalValues();
        //const scai::utilskernel::LArray<ValueType>& myCoordsZ = coordinates[2].getLocalValues();
        const std::vector<scai::utilskernel::LArray<ValueType>>& myCoords = {coordinates[0].getLocalValues(), coordinates[1].getLocalValues(), coordinates[2].getLocalValues() };
	
	std::vector<ValueType> scaledCoord(dimensions);

	for (IndexType dim = 0; dim < dimensions; dim++) {
		assert( coordDist[dim]->isLocal(index) );
		const Scalar coord = myCoords[dim][ coordDist[dim]->global2local(index) ];
		scaledCoord[dim] = (coord.getValue<ValueType>() - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
		if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
			throw std::runtime_error("Coordinate " + std::to_string(coord.getValue<ValueType>()) + " at position " 
				+ std::to_string(index) + " does not agree with bounds "
				+ std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
		}
	}
	
	ValueType x ,y ,z; 	//the coordinates each of the three dimensions
	x= scaledCoord[0];
	y= scaledCoord[1];
	z= scaledCoord[2];
        //std::cout<< x <<"__" << y<< "__"<<z<<"\t";
        assert(x>=0 && x<=1);
        assert(y>=0 && y<=1);
        assert(z>=0 && z<=1);
	long integerIndex = 0;	//TODO: also check whether this data type is long enough

	for (IndexType i = 0; i < recursionDepth; i++) {
		int subSquare;
		if (z < 0.5) {
			if (x < 0.5) {
				if (y <0.5){		//x,y,z <0.5
					subSquare= 0;
					//apply inverse hilbert operator
					ValueType tmpX= x;
					x= 2*z;
					z= 2*y;
					y= 2*tmpX;
				} else{			//z<0.5, y>0.5, x<0.5
					subSquare= 1;
					ValueType tmpX= x;
					x= 2*y-1;
					y= 2*z;
					z= 2*tmpX;
				}
			} else if (y>=0.5){		//z<0.5, y,x>0,5
					subSquare= 2;
					//apply inverse hilbert operator
					ValueType tmpX= x;					
					x= 2*y-1;
					y= 2*z;
					z= 2*tmpX-1;
				}else{			//z<0.5, y<0.5, x>0.5
					subSquare= 3;
					x= -2*x+2;
					y= -2*y+1;
					z= 2*z;
				}
		} else if(x>=0.5){
				if(y<0.5){ 		//z>0.5, y<0.5, x>0.5
					subSquare= 4;
					x= -2*x+2;
					y= -2*y+1;
					z= 2*z-1;
				} else{			//z>0.5, y>0.5, x>0.5
					subSquare= 5;
					ValueType tmpX= x;
					x= 2*y-1;
					y= -2*z+2;
					z= -2*tmpX+2;				
				}
			}else if(y<0.5){		//z>0.5, y<0.5, x<0.5
					subSquare= 7;	//care, this is 7, not 6	
					ValueType tmpX= x;
					x= -2*z+2;
					z= -2*y+1;
					y= 2*tmpX;				
				}else{			//z>0.5, y>0.5, x<0.5
					subSquare= 6;	//this is case 6
					ValueType tmpX= x;
					x= 2*y-1;
					y= -2*z +2;
					z= -2*tmpX+1;				
				}
		integerIndex = (integerIndex << 3) | subSquare;		
	}
	long divisor = 1 << (3*int(recursionDepth));
        return double(integerIndex) / double(divisor);

}
//-------------------------------------------------------------------------------------------------
/*
* Given a 3D point it returns its index in [0,1] on the hilbert curve based on the level depth.
*/

template<typename IndexType, typename ValueType>
DenseVector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(ValueType index, IndexType level){
	DenseVector<ValueType>  p(3,0), ret(3,0);
	ValueType r;
	IndexType q;
	
	if(level==0)
		return ret;
	else{		
		q=int(8*index); 
    		r= 8*index-q;
		if( (q==0) && r==0 ) return ret;
		p = HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(r, level-1);

		switch(q){
			case 0: ret.setValue(0, p(1)/2);	ret.setValue(1, p(2)/2);	ret.setValue(2, p(0)/2);	return ret;
			case 1: ret.setValue(0, p(2)/2);	ret.setValue(1, 0.5+p(0)/2);	ret.setValue(2, p(1)/2);	return ret;
			case 2: ret.setValue(0, 0.5+p(2)/2);	ret.setValue(1, 0.5+p(0)/2);	ret.setValue(2, p(1)/2);	return ret;
			case 3: ret.setValue(0, 1-p(0)/2);	ret.setValue(1, 0.5-p(1)/2);	ret.setValue(2, -p(2)/2);	return ret;
			case 4: ret.setValue(0, 1-p(0)/2);	ret.setValue(1, 0.5-p(1)/2);	ret.setValue(2, 0.5+p(2)/2);	return ret;
			case 5: ret.setValue(0, 1-p(2)/2);	ret.setValue(1, 0.5+p(0)/2);	ret.setValue(2, 1-p(1)/2);	return ret;
			case 6: ret.setValue(0, 0.5-p(2)/2);	ret.setValue(1, 0.5+p(0)/2);	ret.setValue(2, 1-p(1)/2);	return ret;
			case 7: ret.setValue(0, p(1)/2);	ret.setValue(1, 0.5-p(2)/2);	ret.setValue(2, 1-p(0)/2);	return ret;			
		}
	}
	return ret;
}

//-------------------------------------------------------------------------------------------------

template double HilbertCurve<int, double>::getHilbertIndex(const std::vector<DenseVector<double>> &coordinates, int dimensions, int index, int recursionDepth,	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template double HilbertCurve<int, double>::getHilbertIndex2D(const std::vector<DenseVector<double>> &coordinates, int dimensions, int index, int recursionDepth,	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template double HilbertCurve<int, double>::getHilbertIndex_noScaling(const DenseVector<double> &coordinates, int dimensions, int index, int recursionDepth, const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template double HilbertCurve<int, double>::getHilbertIndex3D(const std::vector<DenseVector<double>> &coordinates, int dimensions, int index, int recursionDepth,
	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template DenseVector<double> HilbertCurve<int, double>::Hilbert2DIndex2Point(double index, int level);

template DenseVector<double> HilbertCurve<int, double>::Hilbert3DIndex2Point(double index, int level);

} //namespace ITI
