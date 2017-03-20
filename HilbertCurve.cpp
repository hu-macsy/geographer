/*
 * ParcoReportHilbert.cpp
 *
 *  Created on: 15.11.2016
 *      Author: tzovas
 */

#include "HilbertCurve.h"


/* 22.01.17
 * adding new version of getHilbertIndex.
 * keep the older versions for a while mainly for testing speed
 */


namespace ITI{

template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex(ValueType const * point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords){
    SCAI_REGION( "HilbertCurve.getHilbertIndex_newVersion")

      /*
    //probably wrong
    if (sizeof(point) != sizeof( ValueType)*dimensions){
        PRINT( sizeof(point) << " <> " <<sizeof( ValueType)*dimensions);
        throw  std::runtime_error("Input point not correct. In file " +std::string( __FILE__) + ", line "+ std::to_string(__LINE__) );
    }
    */
    if(dimensions==3) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3D( point, dimensions, recursionDepth,
            minCoords, maxCoords);
    
    if(dimensions==2) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2D( point, dimensions, recursionDepth,
            minCoords, maxCoords);
    
    throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex2D(ValueType const* point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {
    SCAI_REGION("HilbertCurve.getHilbertIndex2D")
   
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
    
    unsigned long integerIndex = 0;//TODO: also check whether this data type is long enough
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
    unsigned long divisor = size_t(1) << size_t(2*int(recursionDepth));
    return double(integerIndex) / double(divisor);
    
}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
DenseVector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(ValueType index, IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert2DIndex2Point" )
	DenseVector<ValueType>  p(2,0), ret(2,0);
	ValueType r;
	IndexType q;
        if(index>1){
            throw std::runtime_error("Index: " + std::to_string(index) +" for hilbert curve must be >0 and <1");
        }
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

template<typename IndexType, typename ValueType>
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex3D(ValueType const* point, IndexType dimensions, IndexType recursionDepth,	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {
        SCAI_REGION("HilbertCurve.getHilbertIndex3D")
        
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
        //PRINT( x <<"__" << y<< "__"<<z );
        SCAI_ASSERT(x>=0 && x<=1, x);
        SCAI_ASSERT(y>=0 && y<=1, y);
        SCAI_ASSERT(z>=0 && z<=1, z);
	unsigned long long integerIndex = 0;	//TODO: also check whether this data type is long enough

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
	unsigned long long divisor = size_t(1) << size_t(3*int(recursionDepth));
        double ret = double(integerIndex) / double(divisor);
        SCAI_ASSERT(ret<1, ret << " , divisor= "<< divisor << " , integerIndex=" << integerIndex <<" , recursionDepth= " << recursionDepth << ", sizeof(unsigned long long)="<< sizeof(unsigned long long));
        return ret;

}

//-------------------------------------------------------------------------------------------------

/*
* Given a 3D point it returns its index in [0,1] on the hilbert curve based on the level depth.
*/

template<typename IndexType, typename ValueType>
DenseVector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(ValueType index, IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert3DIndex2Point" )
	
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

template double HilbertCurve<int, double>::getHilbertIndex(double const * point, int dimensions, int recursionDepth, const std::vector<double> &minCoords, const std::vector<double> &maxCoords);


template double HilbertCurve<int, double>::getHilbertIndex2D(double const * point, int dimensions, int recursionDepth,	const std::vector<double> &minCoords, const std::vector<double> &maxCoords);

template double HilbertCurve<int, double>::getHilbertIndex3D(double const * point, int dimensions, int recursionDepth, const std::vector<double> &minCoords, const std::vector<double> &maxCoords);


template DenseVector<double> HilbertCurve<int, double>::Hilbert2DIndex2Point(double index, int level);

template DenseVector<double> HilbertCurve<int, double>::Hilbert3DIndex2Point(double index, int level);

} //namespace ITI
