/*
 * ParcoReportHilbert.cpp
 *
 *  Created on: 15.11.2016
 *      Author: tzovas
 */

#include "HilbertCurve.h"

namespace ITI{

template<typename IndexType, typename ValueType>//TODO: template this to help branch prediction
ValueType HilbertCurve<IndexType, ValueType>::getHilbertIndex(ValueType const * point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords){
    SCAI_REGION( "HilbertCurve.getHilbertIndex_newVersion")

    if(dimensions==2) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2D( point, dimensions, recursionDepth,
            minCoords, maxCoords);
		
    if(dimensions==3) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3D( point, dimensions, recursionDepth,
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

//
// versions that take as input all the coordinates and return a vector with the indices
//

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::getHilbertIndexVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth, const IndexType dimensions) {
	
    if(dimensions==2) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2DVector( coordinates, recursionDepth);
	
	if(dimensions==3) 
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3DVector( coordinates, recursionDepth);
        
    throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::getHilbertIndex2DVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth) {
    SCAI_REGION("HilbertCurve.getHilbertIndex2DVector")
	
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
			minCoords[dim] = coordinates[dim].min();
			maxCoords[dim] = coordinates[dim].max();
			assert(std::isfinite(minCoords[dim]));
			assert(std::isfinite(maxCoords[dim]));
			SCAI_ASSERT_GT_ERROR(maxCoords[dim], minCoords[dim], "Wrong coordinates.");
		}
    }
    
    ValueType dim0Extent = maxCoords[0] - minCoords[0];
    ValueType dim1Extent = maxCoords[1] - minCoords[1];
    
    ValueType scaledPoint[2];
	unsigned long integerIndex = 0;//TODO: also check whether this data type is long enough
	const IndexType localN = coordinates[0].getLocalValues().size();
	
	// the vector to be returned
	std::vector<ValueType> hilbertIndices(localN,-1);
	
	{
		SCAI_REGION( "HilbertCurve.getHilbertIndex2DVector.indicesCalculation" )
		
		scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
		//scai::hmemo::WriteOnlyAccess<ValueType> hilbertIndices(hilbertIndices.getLocalValues());
		
		for (IndexType i = 0; i < localN; i++) {
			scaledPoint[0] = (coordAccess0[i]-minCoords[0])/dim0Extent;
			scaledPoint[1] = (coordAccess1[i]-minCoords[1])/dim1Extent;
			
			integerIndex = 0;
			for (IndexType j = 0; j < recursionDepth; j++) {
				int subSquare;
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
    SCAI_REGION("HilbertCurve.getHilbertIndex3DVector")
	
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
			minCoords[dim] = coordinates[dim].min();
			maxCoords[dim] = coordinates[dim].max();
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
			
			integerIndex = 0;
			for (IndexType j = 0; j < recursionDepth; j++) {
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

//
// reverse:  from hilbert index to 2D/3D point
//

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::HilbertIndex2Point(const ValueType index, const IndexType level, const IndexType dimensions){
	
	if (dimensions==2)
		return HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(index, level);
	
	if (dimensions==3)
		return HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(index, level);
	
	throw std::logic_error("Hilbert space filling curve only implemented for two or three dimensions");	
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(const ValueType index, const IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert2DIndex2Point" )
    std::vector<ValueType>  p(2,0), ret(2,0);
	ValueType r;
	IndexType q;
    if(index>1 || index < 0){
        throw std::runtime_error("Index: " + std::to_string(index) +" for hilbert curve must be >0 and <1");
    }

	if (level > 0) {
		q=int(4*index);
    		r= 4*index-q;
		p = HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(r, level-1);
		switch(q){
			case 0: ret = {p[1]/2,      p[0]/2}; break;
			case 1: ret = {p[0]/2,      p[1]/2 + 0.5}; break;
			case 2: ret = {p[0]/2+0.5,  p[1]/2 + 0.5}; break;
			case 3: ret = {-p[1]/2+1,   -p[0]/2 + 0.5}; break;
		}
	}
	return ret;
}

//-------------------------------------------------------------------------------------------------

/*
* Given a 3D point it returns its index in [0,1] on the hilbert curve based on the level depth.
*/

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(const ValueType index, const IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert3DIndex2Point" )
	
    std::vector<ValueType>  p(3,0), ret(3,0);
	ValueType r;
	IndexType q;
	
	if (level > 0) {
		q=int(8*index); 
    		r= 8*index-q;
		if( (q==0) && r==0 ) return ret;
		p = HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(r, level-1);

        switch(q){
            case 0: ret = {p[1]/2,   p[2]/2,     p[0]/2};  break;
            case 1: ret = {p[2]/2,   0.5+p[0]/2,     p[1]/2};  break;
            case 2: ret = {0.5+p[2]/2,   0.5+p[0]/2,     p[1]/2};  break;
            case 3: ret = {1-p[0]/2,     0.5-p[1]/2,     p[2]/2};  break;
            case 4: ret = {1-p[0]/2,     0.5-p[1]/2,     0.5+p[2]/2};  break;
            case 5: ret = {1-p[2]/2,     0.5+p[0]/2,     1-p[1]/2};    break;
            case 6: ret = {0.5-p[2]/2,   0.5+p[0]/2,     1-p[1]/2};    break;
            case 7: ret = {p[1]/2,   0.5-p[2]/2,     1-p[0]/2};    break;
		}
	}
	return ret;
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> HilbertCurve<IndexType, ValueType>::HilbertIndex2PointVec(const std::vector<ValueType> indices, const IndexType level, const IndexType dimensions){
	
	if( dimensions==2 )
		return HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2PointVec(indices, level);
	
	if( dimensions==3 )
		return HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2PointVec(indices, level);
	
	throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2PointVec(const std::vector<ValueType> indices, const IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert2DIndex2PointVec" )
	std::vector <std::vector<ValueType>> res( indices.size());
	
	for( int i=0; i<indices.size(); i++){
		res[i] = Hilbert2DIndex2Point( indices[i], level);
	}
	SCAI_ASSERT_EQ_ERROR( res[0].size(), 2 , "The points should have size 2");
	
	return res;
}
//-------------------------------------------------------------------------------------------------

/*
* Given a 3D point it returns its index in [0,1] on the hilbert curve based on the level depth.
*/

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2PointVec(const std::vector<ValueType> indices, const IndexType level){
	SCAI_REGION( "HilbertCurve.Hilbert3DIndex2PointVec" )
	std::vector <std::vector<ValueType>> res( indices.size());
	
	for( int i=0; i<indices.size(); i++){
		res[i] = Hilbert3DIndex2Point( indices[i], level);
	}
	SCAI_ASSERT_EQ_ERROR( res[0].size(), 3 , "The points should have size 3");
	
	return res;
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<sort_pair> HilbertCurve<IndexType, ValueType>::getSortedHilbertIndices( const std::vector<DenseVector<ValueType>> &coordinates, Settings settings){
	
	const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
	
	const IndexType dimensions = coordinates.size();
	const IndexType localN = coordDist->getLocalSize();
    const IndexType globalN = coordDist->getGlobalSize();

    const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(globalN), double(21));
    //const IndexType recursionDepth = std::min(std::log2(globalN), double(21));
	
	 /**
     *	create space filling curve indices.
     */
    
	std::vector<sort_pair> localPairs(localN);
	
    {
        SCAI_REGION("ParcoRepart.getSortedHilbertIndices.spaceFillingCurve");
        
        //get hilbert indices for all the points
        std::vector<ValueType> localHilbertInd = HilbertCurve<IndexType,ValueType>::getHilbertIndexVector(coordinates, recursionDepth, dimensions);
        SCAI_ASSERT_EQ_ERROR(localHilbertInd.size(), localN, "Size mismatch");

        for (IndexType i = 0; i < localN; i++) {
			localPairs[i].value = localHilbertInd[i];
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
        //assert(typesize == sizeof(sort_pair)); does not have to be true anymore due to padding
        		
		//call distributed sort
        //MPI_Comm mpi_comm, std::vector<value_type> &data, long long global_elements = -1, Compare comp = Compare()
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        SQuick::sort<sort_pair>(mpi_comm, localPairs, -1);

        //check size and sanity, TODO: move also to debugMode
        SCAI_ASSERT_EQ_ERROR( comm->sum(localPairs.size()), globalN, "Global index mismatch.");

        //check checksum
        if( settings.debugMode){
        	PRINT0("******** in debug mode");
	        long indexSumAfter = 0;
	        unsigned int newLocalN = localPairs.size();
	        for (IndexType i=0; i<newLocalN; i++) {
	        	indexSumAfter += localPairs[i].index;
	        }
	        unsigned long checkSum = globalN*(globalN-1)/2;
	        const long newCheckSum = comm->sum(indexSumAfter);
	        SCAI_ASSERT_EQ_ERROR( newCheckSum, checkSum, "Old checksum: " << checkSum << ", new checksum: " << newCheckSum );
		}
		
    }
	 
	return localPairs;
}

//-------------------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
void HilbertCurve<IndexType, ValueType>::hilbertRedistribution(std::vector<DenseVector<ValueType> >& coordinates, DenseVector<ValueType>& nodeWeights, Settings settings, struct Metrics& metrics) {
    SCAI_REGION_START("ParcoRepart.hilbertRedistribution.sfc")
    scai::dmemo::DistributionPtr inputDist = coordinates[0].getDistributionPtr();
    scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    const IndexType rank = comm->getRank();

    std::chrono::time_point<std::chrono::system_clock> beforeInitPart =  std::chrono::system_clock::now();

    bool nodesUnweighted = (nodeWeights.max() == nodeWeights.min());

    std::chrono::duration<double> migrationCalculation, migrationTime;

    std::vector<ValueType> hilbertIndices = HilbertCurve<IndexType, ValueType>::getHilbertIndexVector(coordinates, settings.sfcResolution, settings.dimensions);
    SCAI_REGION_END("ParcoRepart.hilbertRedistribution.sfc")
    SCAI_REGION_START("ParcoRepart.hilbertRedistribution.sort")
    /**
     * fill sort pair
     */

    scai::hmemo::HArray<IndexType> myGlobalIndices(localN, IndexType(0) );
    inputDist->getOwnedIndexes(myGlobalIndices);
    std::vector<sort_pair> localPairs(localN);
    {
        scai::hmemo::ReadAccess<IndexType> rIndices(myGlobalIndices);
        for (IndexType i = 0; i < localN; i++) {
            localPairs[i].value = hilbertIndices[i];
            localPairs[i].index = rIndices[i];
        }
    }

    MPI_Comm mpi_comm = MPI_COMM_WORLD; //maybe cast the communicator ptr to a MPI communicator and get getMPIComm()?
    SQuick::sort<sort_pair>(mpi_comm, localPairs, -1); //could also do this with just the hilbert index - as a valueType
    //IndexType newLocalN = localPairs.size();
    migrationCalculation = std::chrono::system_clock::now() - beforeInitPart;
    metrics.timeMigrationAlgo[rank] = migrationCalculation.count();
    std::chrono::time_point < std::chrono::system_clock > beforeMigration = std::chrono::system_clock::now();
    assert(localPairs.size() > 0);
    SCAI_REGION_END("ParcoRepart.hilbertRedistribution.sort")

    sort_pair minLocalIndex = localPairs[0];
    std::vector<ValueType> sendThresholds(comm->getSize(), minLocalIndex.value);
    std::vector<ValueType> recvThresholds(comm->getSize());

    MPI_Datatype MPI_ValueType = MPI_DOUBLE; //TODO: properly template this
    MPI_Alltoall(sendThresholds.data(), 1, MPI_ValueType, recvThresholds.data(),
            1, MPI_ValueType, mpi_comm); //TODO: replace this monstrosity with a proper call to LAMA
    //comm->all2all(recvThresholds.data(), sendTresholds.data());//TODO: maybe speed up with hypercube
    SCAI_ASSERT_LT_ERROR(recvThresholds[comm->getSize() - 1], 1, "invalid hilbert index");
    // merge to get quantities //Problem: nodes are not sorted according to their hilbert indices, so accesses are not aligned.
    // Need to sort before and after communication
    assert(std::is_sorted(recvThresholds.begin(), recvThresholds.end()));
    std::vector<IndexType> permutation(localN);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](IndexType i, IndexType j){return hilbertIndices[i] < hilbertIndices[j];});

    //now sorting hilbert indices themselves
    std::sort(hilbertIndices.begin(), hilbertIndices.end());
    std::vector<IndexType> quantities(comm->getSize(), 0);
    {
        IndexType p = 0;
        for (IndexType i = 0; i < localN; i++) {
            //increase target block counter if threshold is reached. Skip empty blocks if necessary.
            while (p + 1 < comm->getSize()
                    && recvThresholds[p + 1] <= hilbertIndices[i]) {
                p++;
            }
            assert(p < comm->getSize());

            quantities[p]++;
        }
    }

    SCAI_REGION_START("ParcoRepart.hilbertRedistribution.communicationPlan")
    // allocate sendPlan
    scai::dmemo::CommunicationPlan sendPlan(quantities.data(), comm->getSize());
    SCAI_ASSERT_EQ_ERROR(sendPlan.totalQuantity(), localN,
            "wrong size of send plan")
    // allocate recvPlan - either with allocateTranspose, or directly
    scai::dmemo::CommunicationPlan recvPlan;
    recvPlan.allocateTranspose(sendPlan, *comm);
    IndexType newLocalN = recvPlan.totalQuantity();
    SCAI_REGION_END("ParcoRepart.hilbertRedistribution.communicationPlan")

    if (settings.verbose) {
        PRINT0(std::to_string(localN) + " old local values "
                        + std::to_string(newLocalN) + " new ones.");
    }
    //transmit indices, allowing for resorting of the received values
    std::vector<IndexType> sendIndices(localN);
    {
        SCAI_REGION("ParcoRepart.hilbertRedistribution.permute");
        scai::hmemo::ReadAccess<IndexType> rIndices(myGlobalIndices);
        for (IndexType i = 0; i < localN; i++) {
            assert(permutation[i] < localN);
            assert(permutation[i] >= 0);
            sendIndices[i] = rIndices[permutation[i]];
        }
    }
    std::vector<IndexType> recvIndices(newLocalN);
    comm->exchangeByPlan(recvIndices.data(), recvPlan, sendIndices.data(),
            sendPlan);
    //get new distribution
    scai::hmemo::HArray<IndexType> indexTransport(newLocalN,
            recvIndices.data());
    scai::dmemo::DistributionPtr newDist(
            new scai::dmemo::GeneralDistribution(globalN, indexTransport,
                    comm));
    SCAI_ASSERT_EQUAL(newDist->getLocalSize(), newLocalN,
            "wrong size of new distribution");
    for (IndexType i = 0; i < newLocalN; i++) {
        SCAI_ASSERT_VALID_INDEX_DEBUG(recvIndices[i], globalN, "invalid index");
    }

    {
        SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute");
        // for each dimension: define DenseVector with new distribution, get write access to local values, call exchangeByPlan
        std::vector<ValueType> sendBuffer(localN);
        std::vector<ValueType> recvBuffer(newLocalN);

        for (IndexType d = 0; d < settings.dimensions; d++) {
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());
                for (IndexType i = 0; i < localN; i++) { //TODO:maybe extract into lambda?
                    sendBuffer[i] = rCoords[permutation[i]]; //TODO: how to make this more cache-friendly? (Probably by using pairs and sorting them.)
                }
            }

            comm->exchangeByPlan(recvBuffer.data(), recvPlan, sendBuffer.data(), sendPlan);
            coordinates[d] = DenseVector<ValueType>(newDist, 0);
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::WriteAccess<ValueType> wCoords(coordinates[d].getLocalValues());
                assert(wCoords.size() == newLocalN);
                for (IndexType i = 0; i < newLocalN; i++) {
                    wCoords[newDist->global2local(recvIndices[i])] =
                            recvBuffer[i];
                }
            }
        }
        // same for node weights
        if (nodesUnweighted) {
            nodeWeights = DenseVector<ValueType>(newDist, 1);
        }
        else {
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
                for (IndexType i = 0; i < localN; i++) {
                    sendBuffer[i] = rWeights[permutation[i]]; //TODO: how to make this more cache-friendly? (Probably by using pairs and sorting them.)
                }
            }
            comm->exchangeByPlan(recvBuffer.data(), recvPlan, sendBuffer.data(), sendPlan);
            nodeWeights = DenseVector<ValueType>(newDist, 0);
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::WriteAccess<ValueType> wWeights(nodeWeights.getLocalValues());
                for (IndexType i = 0; i < newLocalN; i++) {
                    wWeights[newDist->global2local(recvIndices[i])] = recvBuffer[i];
                }
            }
        }
    }
    migrationTime = std::chrono::system_clock::now() - beforeMigration;
    metrics.timeFirstDistribution[rank] = migrationTime.count();
}
//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
bool HilbertCurve<IndexType, ValueType>::confirmHilbertDistribution(
	//const scai::lama::CSRSparseMatrix<ValueType> &graph,
	const std::vector<DenseVector<ValueType>> &coordinates,
	const DenseVector<ValueType> &nodeWeights,
	Settings settings
	){

	//get distributions of the input
	const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
	//const scai::dmemo::DistributionPtr graphDist = graph.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr weightDist = nodeWeights.getDistributionPtr();

	if( not coordDist->isEqual( *weightDist) ){
		throw std::runtime_error( "Distributions should be equal.");
	}

	//get sfc indices in every PE
	std::vector<ValueType> localSFCInd = getHilbertIndexVector ( coordinates,  settings.sfcResolution, settings.dimensions);

	//sort local indices
	std::sort( localSFCInd.begin(), localSFCInd.end() );

	//the min and max local sfc value
	ValueType sfcMinMax[2] = { localSFCInd.front(), localSFCInd.back() };

const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();	
PRINT(*comm <<": sending "<< sfcMinMax[0] << ", " << sfcMinMax[1] )	;

	const IndexType p = comm->getSize();
	const IndexType root = 0; //set PE 0 as root 
	IndexType arraySize = 1;
	if( comm->getRank()==root ){
		arraySize = 2*p;
	}
	//so only the root PE allocates the array
	ValueType allMinMax[arraySize];

	comm->gather(allMinMax, 2, root, sfcMinMax );

PRINT0("gathered: ");
if( comm->getRank()==root )
for(unsigned int i=0; i<arraySize; i++){
	std::cout<< ", " << allMinMax[i];
}
std::cout<< std::endl;
	
	//check if array is sorted. For all PEs except the root, this is trivially
	// true since their array has only one element
	std::vector<ValueType> gatheredInd( allMinMax, allMinMax+arraySize  );
	bool isSorted = std::is_sorted( gatheredInd.begin(), gatheredInd.end() );

	return comm->all( isSorted );
}


template class HilbertCurve<IndexType, ValueType>;
//this instantiation does not work
//template class HilbertCurve<int, double>;

} //namespace ITI
