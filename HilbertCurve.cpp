/*
 * ParcoReportHilbert.cpp
 *
 *  Created on: 15.11.2016
 *      Author: tzovas
 */

#include "HilbertCurve.h"

#include "RBC/Sort/SQuick.hpp"

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
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::getHilbertIndexVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth, const IndexType dimensions) {

	if(dimensions==3) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3DVector( coordinates, recursionDepth);
    
    if(dimensions==2) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2DVector( coordinates, recursionDepth);
    
    throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::getHilbertIndex2DVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth) {
    SCAI_REGION("HilbertCurve.getHilbertIndex2DVector")
	
	using scai::lama::Scalar;
	
	const IndexType dimensions = coordinates.size();
	
    size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
    if (recursionDepth > bitsInValueType/dimensions) {
        throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
    }    
    
	if( dimensions!=2 ){
		PRINT("In HilbertCurve.getHilbertIndex2DVector but dimensions is " << dimensions << " and not 2");
		throw std::runtime_error("Wrong dimensions given");
	}
	
	/**
     * get minimum / maximum of coordinates
     */
	ValueType minCoords[2];
	ValueType maxCoords[2];

	{
		SCAI_REGION( "HilbertCurve.getHilbertIndex2DVector.minMax" )
		for (IndexType dim = 0; dim < 2; dim++) {
			minCoords[dim] = coordinates[dim].min().Scalar::getValue<ValueType>();
			maxCoords[dim] = coordinates[dim].max().Scalar::getValue<ValueType>();
			assert(std::isfinite(minCoords[dim]));
			assert(std::isfinite(maxCoords[dim]));
			SCAI_ASSERT(maxCoords[dim] > minCoords[dim], "Wrong coordinates.");
		}
    }
    
    ValueType dim0Extent = maxCoords[0] - minCoords[0];
    ValueType dim1Extent = maxCoords[1] - minCoords[1];
    
    ValueType scaledPoint[2];
	unsigned long integerIndex = 0;//TODO: also check whether this data type is long enough
	const IndexType localN = coordinates[0].getLocalValues().size();
	
	// the DV to be returned
	std::vector<ValueType> hilbertIndices(localN,-1);
	
	{
		SCAI_REGION( "HilbertCurve.getHilbertIndex2DVector.indicesCalculation" )
		
		scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
		//scai::hmemo::WriteOnlyAccess<ValueType> hilbertIndices(hilbertIndices.getLocalValues());
		
		for (IndexType i = 0; i < localN; i++) {
			scaledPoint[0] = (coordAccess0[i]-minCoords[0])/dim0Extent;
			scaledPoint[1] = (coordAccess1[i]-minCoords[1])/dim1Extent;
			/*
			if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
				throw std::runtime_error("Coordinate " + std::to_string(point[dim]) +" does not agree with bounds "
            + std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
			}
			*/
			integerIndex = 0;
			for (IndexType i = 0; i < recursionDepth; i++) {
				int subSquare;
				//two dimensions only, for now
				if (scaledPoint[0] < 0.5) {
					if (scaledPoint[1] < 0.5) {
						subSquare = 0;
						//apply inverse hilbert operator
						double temp = scaledPoint[0];
						scaledPoint[0] = 2*scaledPoint[1];
						scaledPoint[1] = 2*temp;
					} else {
						subSquare = 1;
						//apply inverse hilbert operator
						scaledPoint[0] *= 2;
						scaledPoint[1] = 2*scaledPoint[1] -1;
					}
				} else {
					if (scaledPoint[1] < 0.5) {
						subSquare = 3;
						//apply inverse hilbert operator
						double temp = scaledPoint[0];
						scaledPoint[0] = -2*scaledPoint[1]+1;
						scaledPoint[1] = -2*temp+2;
					} else {
						subSquare = 2;
						//apply inverse hilbert operator
						scaledPoint[0] = 2*scaledPoint[0]-1;
						scaledPoint[1] = 2*scaledPoint[1]-1;
					}
				}
				//std::cout<< subSquare<<std::endl;
				integerIndex = (integerIndex << 2) | subSquare;	
			}
			unsigned long divisor = size_t(1) << size_t(2*int(recursionDepth));
			hilbertIndices[i] = double(integerIndex) / double(divisor);
		}
	}
	
    return hilbertIndices;   
    
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::getHilbertIndex3DVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth) {
    SCAI_REGION("HilbertCurve.getHilbertIndex2DVector")
	
	using scai::lama::Scalar;
	
	const IndexType dimensions = coordinates.size();
	
    size_t bitsInValueType = sizeof(ValueType) * CHAR_BIT;
    if (recursionDepth > bitsInValueType/dimensions) {
        throw std::runtime_error("A space-filling curve that precise won't fit into the return datatype.");
    }    
    
	if( dimensions!=3 ){
		PRINT("In HilbertCurve.getHilbertIndex2DVector but dimensions is " << dimensions << " and not 3");
		throw std::runtime_error("Wrong dimensions given");
	}
	
	/**
     * get minimum / maximum of coordinates
     */
	ValueType minCoords[3];
	ValueType maxCoords[3];

	{
		SCAI_REGION( "HilbertCurve.getHilbertIndex3DVector.minMax" )
		for (IndexType dim = 0; dim < 3; dim++) {
			minCoords[dim] = coordinates[dim].min().Scalar::getValue<ValueType>();
			maxCoords[dim] = coordinates[dim].max().Scalar::getValue<ValueType>();
			assert(std::isfinite(minCoords[dim]));
			assert(std::isfinite(maxCoords[dim]));
			SCAI_ASSERT(maxCoords[dim] > minCoords[dim], "Wrong coordinates.");
		}
    }
    
    ValueType dim0Extent = maxCoords[0] - minCoords[0];
    ValueType dim1Extent = maxCoords[1] - minCoords[1];
	ValueType dim2Extent = maxCoords[2] - minCoords[2];
    
    ValueType x ,y ,z;
	unsigned long integerIndex = 0;	//TODO: also check whether this data type is long enough
	const IndexType localN = coordinates[0].getLocalValues().size();
	
	// the DV to be returned
	std::vector<ValueType> hilbertIndices(localN,-1);
	
	{
		SCAI_REGION( "HilbertCurve.getHilbertIndex3DVector.indicesCalculation" )
		
		scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
		scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );
		//scai::hmemo::WriteOnlyAccess<ValueType> hilbertIndices(hilbertIndices.getLocalValues());
		
		for (IndexType i = 0; i < localN; i++) {
			x = (coordAccess0[i]-minCoords[0])/dim0Extent;
			y = (coordAccess1[i]-minCoords[1])/dim1Extent;
			z = (coordAccess2[i]-minCoords[2])/dim2Extent;
			/*
			if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
				throw std::runtime_error("Coordinate " + std::to_string(point[dim]) +" does not agree with bounds "
            + std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
			}
			*/
			integerIndex = 0;
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
			hilbertIndices[i] = double(integerIndex) / double(divisor);
		}
	}
	
    return hilbertIndices;   
    
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
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2PointVec(ValueType index, IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert2DIndex2Point" )
	std::vector<ValueType>  p(2,0), ret(2,0);
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
		p = HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2PointVec(r, level-1);
		switch(q){
			case 0: ret[0]=p[1]/2;		ret[1]=p[0]/2;	return ret;
			case 1: ret[0]=p[0]/2;		ret[1]=p[1]/2 +0.5;	return ret;
			case 2: ret[0]=p[0]/2 +0.5;	ret[1]=p[1]/2 +0.5;	return ret;
			case 3: ret[0]=1-p[1]/2;	ret[1]=0.5-p[0]/2;	return ret;
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

/*
* Given a 3D point it returns its index in [0,1] on the hilbert curve based on the level depth.
*/

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2PointVec(ValueType index, IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert3DIndex2Point" )
	
	std::vector<ValueType>  p(3,0), ret(3,0);
	ValueType r;
	IndexType q;
	
	if(level==0)
		return ret;
	else{		
		q=int(8*index); 
		r= 8*index-q;
		if( (q==0) && r==0 ) return ret;
		p = HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2PointVec(r, level-1);

		switch(q){
			case 0: ret[0]=p[1]/2;		ret[1]=p[2]/2;		ret[2]=p[0]/2;	return ret;
			case 1: ret[0]=p[2]/2;		ret[1]=0.5+p[0]/2;	ret[2]=p[1]/2;	return ret;
			case 2: ret[0]=0.5+p[2]/2;	ret[1]=0.5+p[0]/2;	ret[2]=p[1]/2;	return ret;
			case 3: ret[0]=1-p[0]/2;	ret[1]=0.5-p[1]/2;	ret[2]=-p[2]/2;	return ret;
			case 4: ret[0]=1-p[0]/2;	ret[1]=0.5-p[1]/2;	ret[2]=0.5+p[2]/2;	return ret;
			case 5: ret[0]=1-p[2]/2;	ret[1]=0.5+p[0]/2;	ret[2]=1-p[1]/2;	return ret;
			case 6: ret[0]=0.5-p[2]/2;	ret[1]=0.5+p[0]/2;	ret[2]=1-p[1]/2;	return ret;
			case 7: ret[0]=p[1]/2;		ret[1]=0.5-p[2]/2;	ret[2]=1-p[0]/2;	return ret;			
		}
	}
	return ret;
}


//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<sort_pair> HilbertCurve<IndexType, ValueType>::getSortedHilbertIndices( const std::vector<DenseVector<ValueType>> &coordinates){
	
	using scai::lama::Scalar;
	
	const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
	
	const IndexType dimensions = coordinates.size();
	const IndexType localN = coordDist->getLocalSize();
    const IndexType globalN = coordDist->getGlobalSize();
	
	/**
     * get minimum / maximum of coordinates
     */
	std::vector<ValueType> minCoords(dimensions);
    std::vector<ValueType> maxCoords(dimensions);
    {
		SCAI_REGION( "ParcoRepart.getSortedHilbertIndices.minMax" )
		for (IndexType dim = 0; dim < dimensions; dim++) {
			minCoords[dim] = coordinates[dim].min().Scalar::getValue<ValueType>();
			maxCoords[dim] = coordinates[dim].max().Scalar::getValue<ValueType>();
			assert(std::isfinite(minCoords[dim]));
			assert(std::isfinite(maxCoords[dim]));
			SCAI_ASSERT(maxCoords[dim] > minCoords[dim], "Wrong coordinates.");
		}
    }
    
    //const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(globalN), double(21));
    const IndexType recursionDepth = std::min(std::log2(globalN), double(21));
	
	 /**
     *	create space filling curve indices.
     */
    
	std::vector<sort_pair> localPairs(localN);
	
    {
        SCAI_REGION("ParcoRepart.getSortedHilbertIndices.spaceFillingCurve");
        // get local part of hilbert indices
        // get read access to the local part of the coordinates
        // TODO: should be coordAccess[dimension] but I don't know how ... maybe HArray::acquireReadAccess? (harry)
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        // this is faulty, if dimensions=2 coordAccess2 is equal to coordAccess1
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[dimensions-1].getLocalValues() );
        
        ValueType point[dimensions];
        for (IndexType i = 0; i < localN; i++) {
            coordAccess0.getValue(point[0], i);
            coordAccess1.getValue(point[1], i);
            // TODO change how I treat different dimensions
            if(dimensions == 3){
                coordAccess2.getValue(point[2], i);
            }
            
            ValueType globalHilbertIndex = HilbertCurve<IndexType, ValueType>::getHilbertIndex( point, dimensions, recursionDepth, minCoords, maxCoords);
			localPairs[i].value = globalHilbertIndex;
			//localPairs[i].index = 0; // we do need the index now
        	localPairs[i].index = coordDist->local2global(i);
        }
    }
    
     /**
     * now sort the global indices by where they are on the space-filling curve.
     */
	 
	 {
        SCAI_REGION( "ParcoRepart.getSortedHilbertIndices.sorting" );
        
        int typesize;
        MPI_Type_size(SortingDatatype<sort_pair>::getMPIDatatype(), &typesize);
        assert(typesize == sizeof(sort_pair));
        
		
		//call distributed sort
        //MPI_Comm mpi_comm, std::vector<value_type> &data, long long global_elements = -1, Compare comp = Compare()
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        SQuick::sort<sort_pair>(mpi_comm, localPairs, -1);

        //copy hilbert indices into array
        //IndexType newLocalN = localPairs.size();

        //check size and sanity
        //SCAI_ASSERT_EQ_ERROR( newLocalN, localPairs.size(), "New local indices mismatch.");
        //SCAI_ASSERT_LT_ERROR( *std::max_element(newLocalIndices.begin(), newLocalIndices.end()) , globalN, "Too large index (possible IndexType overflow?).");
        SCAI_ASSERT_EQ_ERROR( comm->sum(localPairs.size()), globalN, "Global index mismatch.");

		/*
        //check checksum
        long indexSumAfter = 0;
        for (IndexType i = 0; i < newLocalN; i++) {
        	indexSumAfter += newLocalIndices[i];
        }

        const long newCheckSum = comm->sum(indexSumAfter);
        SCAI_ASSERT( newCheckSum == checkSum, "Old checksum: " << checkSum << ", new checksum: " << newCheckSum );
		*/
    }
	 
	 return localPairs;
}

//-------------------------------------------------------------------------------------------------


template class HilbertCurve<long int, double>;
template class HilbertCurve<int, double>;

} //namespace ITI
