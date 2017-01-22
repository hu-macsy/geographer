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
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex(ValueType* point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords){

     
    //        for debugging
    //
    //for(IndexType dim=0; dim<dimensions; dim++){
    //    PRINT(point[dim]);
    //}
    //
    //
    
    if (dimensions > 3 || dimensions < 2) {
        throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
    }
        
    if(dimensions==3) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3D( point, dimensions, recursionDepth,
            minCoords, maxCoords);
    
    if(dimensions==2) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2D( point, dimensions, recursionDepth,
            minCoords, maxCoords);
    
    PRINT("Something went wrong");
    return -1; //Something is wrong,should not reach this point
    
}

//-------------------------------------------------------------------------------------------------

/**
* possible optimization: check whether all local points lie in the same region and thus have a common prefix
*/

template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex2D(ValueType* point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {
    SCAI_REGION("HilbertCurve.getHilbertIndex2D_new")
    size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
    if (recursionDepth > bitsInValueType/dimensions) {
        throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
    }
    
    std::vector<ValueType> scaledCoord(dimensions);

    for (IndexType dim = 0; dim < dimensions; dim++) {
        scaledCoord[dim] = (point[dim] - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
        if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
            throw std::runtime_error("Coordinate " + std::to_string(point[dim]) +" does not agree with bounds "
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
    
}

//-------------------------------------------------------------------------------------------------
/**
* Given a point in 3D it returns its hilbert index, a value in [0,1]. 
**/
template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex3D(ValueType* point, IndexType dimensions, IndexType recursionDepth,
	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {

        SCAI_REGION("HilbertCurve.getHilbertIndex3D_new")
	if (dimensions != 3) {
		throw std::logic_error("Space filling curve for 3 dimensions.");
	}

	size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
	if ((unsigned int) recursionDepth > bitsInValueType/dimensions) {
		throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
	}
	
	std::vector<ValueType> scaledCoord(dimensions);

	for (IndexType dim = 0; dim < dimensions; dim++) {
		scaledCoord[dim] = (point[dim] - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
		if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
			throw std::runtime_error("Coordinate " + std::to_string(point[dim])+" does not agree with bounds "
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

template double HilbertCurve<int, double>::getHilbertIndex(double* point, int dimensions, int recursionDepth, const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template double HilbertCurve<int, double>::getHilbertIndex2D(double* point, int dimensions, int recursionDepth,	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template double HilbertCurve<int, double>::getHilbertIndex3D(double* point, int dimensions, int recursionDepth,
	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);


} //namespace ITI
