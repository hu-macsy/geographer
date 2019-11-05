/*
 * ParcoReportHilbert.cpp
 *
 *  Created on: 15.11.2016
 *      Author: tzovas
 */

#include "HilbertCurve.h"


namespace ITI {


//TODO: take node weights into account
template<typename IndexType, typename ValueType>
DenseVector<IndexType> HilbertCurve<IndexType, ValueType>::computePartition(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<ValueType> &nodeWeights, Settings settings) {

    //auto uniformWeights = fill<DenseVector<ValueType>>(coordinates[0].getDistributionPtr(), 1);
    return computePartition( coordinates, settings);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
DenseVector<IndexType> HilbertCurve<IndexType, ValueType>::computePartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings) {
    SCAI_REGION( "HilbertCurve.computePartition" )

    std::chrono::time_point<std::chrono::steady_clock> start, afterSFC;
    start = std::chrono::steady_clock::now();

    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();

    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    assert(dimensions == settings.dimensions);
    const IndexType globalN = coordDist->getGlobalSize();

    if (k != comm->getSize() && comm->getRank() == 0) {
        throw std::logic_error("Hilbert curve partition only implemented for same number of blocks and processes.");
    }

    if (comm->getSize() == 1) {
        return scai::lama::DenseVector<IndexType>(globalN, 0);
    }

    //
    // vector of size k, each element represents the size of each block
    //
    //TODO: either adapt hilbert partition to consider node weights and block
    // sizes or add checks when used with nodeweights outside the function

    /*
        std::vector<ValueType> blockSizes;
    	//TODO: for now assume uniform nodeweights
        IndexType weightSum = globalN;// = nodeWeights.sum();
        if( settings.blockSizes.empty() ){
            blockSizes.assign( settings.numBlocks, weightSum/settings.numBlocks );
        }else{
        	if (settings.blockSizes.size() > 1) {
        		throw std::logic_error("Hilbert partition not implemented for node weights or multiple block sizes.");
        	}
            blockSizes = settings.blockSizes[0];
        }
        SCAI_ASSERT( blockSizes.size()==settings.numBlocks , "Wrong size of blockSizes vector: " << blockSizes.size() );

    */

    /*
     * Several possibilities exist for choosing the recursion depth.
     * Either by user choice, or by the maximum fitting into the datatype, or by the minimum distance between adjacent points.
     */
    const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(ValueType(std::log2(globalN)), ValueType(21));

    /*
     *	create space filling curve indices.
     */

    scai::lama::DenseVector<double> hilbertIndices(coordDist, 0);
    std::vector<double> localHilberIndices = HilbertCurve<IndexType,ValueType>::getHilbertIndexVector(coordinates, recursionDepth, dimensions);
    hilbertIndices.assign( scai::hmemo::HArray<double>( localHilberIndices.size(), localHilberIndices.data()), coordDist);

    //TODO: use the blockSizes vector
    //TODO: take into account node weights: just sorting will create imbalanced blocks, not in number of node but in the total weight of each block

    /*
     * now sort the global indices by where they are on the space-filling curve.
     */

    std::vector<sort_pair> localPairs= getSortedHilbertIndices( coordinates, settings );

    //copy indices into array
    const IndexType newLocalN = localPairs.size();
    std::vector<IndexType> newLocalIndices(newLocalN);

    for (IndexType i = 0; i < newLocalN; i++) {
        newLocalIndices[i] = localPairs[i].index;
    }

    //sort local indices for general distribution
    std::sort(newLocalIndices.begin(), newLocalIndices.end());

    //check size and sanity
    SCAI_ASSERT_LT_ERROR( *std::max_element(newLocalIndices.begin(), newLocalIndices.end()), globalN, "Too large index (possible IndexType overflow?).");
    SCAI_ASSERT_EQ_ERROR( comm->sum(newLocalIndices.size()), globalN, "distribution mismatch");
    SCAI_ASSERT_EQ_ERROR( comm->sum(newLocalIndices.size()), globalN, "distribution mismatch");      


    //possible optimization: remove dummy values during first copy, then directly copy into HArray and sort with pointers. Would save one copy.
    

    DenseVector<IndexType> result;

    {
        assert(!coordDist->isReplicated() && comm->getSize() == k);
        SCAI_REGION( "HilbertCurve.computePartition.createDistribution" );

        scai::hmemo::HArray<IndexType> indexTransport(newLocalIndices.size(), newLocalIndices.data());
        assert(comm->sum(indexTransport.size()) == globalN);
        scai::dmemo::DistributionPtr newDistribution( new scai::dmemo::GeneralDistribution ( globalN, std::move(indexTransport), true) );

        if (comm->getRank() == 0) std::cout << "Created distribution." << std::endl;
        result = scai::lama::fill<DenseVector<IndexType>>(newDistribution, comm->getRank());
        if (comm->getRank() == 0) std::cout << "Created initial partition." << std::endl;
    }

    return result;
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>//TODO: template this to help branch prediction
double HilbertCurve<IndexType, ValueType>::getHilbertIndex(ValueType const * point, const IndexType dimensions, const IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {
    SCAI_REGION( "HilbertCurve.getHilbertIndex_newVersion")

    IndexType newRecursionDepth = recursionDepth;

    size_t bitsInValueType = sizeof(double) * CHAR_BIT;
    if (recursionDepth > bitsInValueType/dimensions) {
        newRecursionDepth = IndexType(bitsInValueType/dimensions);
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        PRINT0("*** Warning: Requested space-filling curve with precision " << recursionDepth << " but return datatype is double and only holds " <<bitsInValueType/dimensions << ". Setting recursion depth to " << newRecursionDepth);
    }

    if(dimensions==2)
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2D( point, dimensions, newRecursionDepth, minCoords, maxCoords);

    if(dimensions==3)
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3D( point, dimensions, newRecursionDepth, minCoords, maxCoords);

    throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
double HilbertCurve<IndexType, ValueType>::getHilbertIndex2D(ValueType const* point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {
    SCAI_REGION("HilbertCurve.getHilbertIndex2D")

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
                ValueType temp = scaledCoord[0];
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
                ValueType temp = scaledCoord[0];
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
double HilbertCurve<IndexType, ValueType>::getHilbertIndex3D(ValueType const* point, IndexType dimensions, IndexType recursionDepth,	const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords) {
    SCAI_REGION("HilbertCurve.getHilbertIndex3D")

    if (dimensions != 3) {
        throw std::logic_error("Space filling curve for 3 dimensions.");
    }

    std::vector<ValueType> scaledCoord(dimensions);

    for (IndexType dim = 0; dim < dimensions; dim++) {
        scaledCoord[dim] = (point[dim] - minCoords[dim]) / (maxCoords[dim] - minCoords[dim]);
        if (scaledCoord[dim] < 0 || scaledCoord[dim] > 1) {
            throw std::runtime_error("Coordinate " + std::to_string(point[dim])+" does not agree with bounds "
                                     + std::to_string(minCoords[dim]) + " and " + std::to_string(maxCoords[dim]));
        }
    }

    ValueType x,y,z; 	//the coordinates each of the three dimensions
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
                if (y <0.5) {		//x,y,z <0.5
                    subSquare= 0;
                    //apply inverse hilbert operator
                    ValueType tmpX= x;
                    x= 2*z;
                    z= 2*y;
                    y= 2*tmpX;
                } else {			//z<0.5, y>0.5, x<0.5
                    subSquare= 1;
                    ValueType tmpX= x;
                    x= 2*y-1;
                    y= 2*z;
                    z= 2*tmpX;
                }
            } else if (y>=0.5) {		//z<0.5, y,x>0,5
                subSquare= 2;
                //apply inverse hilbert operator
                ValueType tmpX= x;
                x= 2*y-1;
                y= 2*z;
                z= 2*tmpX-1;
            } else {			//z<0.5, y<0.5, x>0.5
                subSquare= 3;
                x= -2*x+2;
                y= -2*y+1;
                z= 2*z;
            }
        } else if(x>=0.5) {
            if(y<0.5) { 		//z>0.5, y<0.5, x>0.5
                subSquare= 4;
                x= -2*x+2;
                y= -2*y+1;
                z= 2*z-1;
            } else {			//z>0.5, y>0.5, x>0.5
                subSquare= 5;
                ValueType tmpX= x;
                x= 2*y-1;
                y= -2*z+2;
                z= -2*tmpX+2;
            }
        } else if(y<0.5) {		//z>0.5, y<0.5, x<0.5
            subSquare= 7;	//care, this is 7, not 6
            ValueType tmpX= x;
            x= -2*z+2;
            z= -2*y+1;
            y= 2*tmpX;
        } else {			//z>0.5, y>0.5, x<0.5
            subSquare= 6;	//this is case 6
            ValueType tmpX= x;
            x= 2*y-1;
            y= -2*z +2;
            z= -2*tmpX+1;
        }
        integerIndex = (integerIndex << 3) | subSquare;
    }
    unsigned long long divisor = size_t(1) << size_t(3*int(recursionDepth));
    ValueType ret = double(integerIndex) / double(divisor);
    SCAI_ASSERT(ret<1, ret << " , divisor= "<< divisor << " , integerIndex= " << integerIndex <<" , recursionDepth= " << recursionDepth << ", sizeof(unsigned long long)= "<< sizeof(unsigned long long) << ". Consider using a smaller recursion depth (see Settings.h).");
    return ret;

}
//-------------------------------------------------------------------------------------------------

//
// versions that take as input all the coordinates and return a vector with the indices
//

template<typename IndexType, typename ValueType>
std::vector<double> HilbertCurve<IndexType, ValueType>::getHilbertIndexVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth, const IndexType dimensions) {

    IndexType newRecursionDepth = recursionDepth;

    size_t bitsInValueType = sizeof(double) * CHAR_BIT;
    
    if (recursionDepth > bitsInValueType/dimensions) {
        newRecursionDepth = IndexType(bitsInValueType/dimensions);
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        PRINT0("Requested space-filling curve with precision " << recursionDepth << " but return datatype is double and only holds " << bitsInValueType/dimensions << ". Setting recursion depth to " << newRecursionDepth);
    }

    if(dimensions==2) {
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex2DVector( coordinates, newRecursionDepth);
    }

    if(dimensions==3) {
        return HilbertCurve<IndexType, ValueType>::getHilbertIndex3DVector( coordinates, newRecursionDepth);
    }

    throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<double> HilbertCurve<IndexType, ValueType>::getHilbertIndex2DVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth) {
    SCAI_REGION("HilbertCurve.getHilbertIndex2DVector")

    const IndexType dimensions = coordinates.size();

    if( dimensions!=2 ) {
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        PRINT0("In HilbertCurve.getHilbertIndex2DVector but dimensions is " << dimensions << " and not 2");
        throw std::runtime_error("Wrong dimensions given");
    }

    /*
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
            SCAI_ASSERT_GE_ERROR(maxCoords[dim], minCoords[dim], "Wrong coordinates for dimension " << dim);
            if( maxCoords[dim]==minCoords[dim] ) {
                std::cout << "WARNING: min and max coords are equal: all points are collinear" << std::endl;
            }
        }
    }

    ValueType dim0Extent = maxCoords[0] - minCoords[0];
    ValueType dim1Extent = maxCoords[1] - minCoords[1];

    ValueType scaledPoint[2];
    unsigned long integerIndex = 0;//TODO: also check whether this data type is long enough
    const IndexType localN = coordinates[0].getLocalValues().size();

    // the vector to be returned
    std::vector<double> hilbertIndices(localN,-1);

    {
        SCAI_REGION( "HilbertCurve.getHilbertIndex2DVector.indicesCalculation" )

        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );

        const unsigned long divisor = size_t(1) << size_t(2*int(recursionDepth));
        const double dDivisor = double(divisor);

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
                        ValueType temp = scaledPoint[0];
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
                        ValueType temp = scaledPoint[0];
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
            hilbertIndices[i] = integerIndex / dDivisor;
        }
    }

    return hilbertIndices;

}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<double> HilbertCurve<IndexType, ValueType>::getHilbertIndex3DVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth) {
    SCAI_REGION("HilbertCurve.getHilbertIndex3DVector")

    const IndexType dimensions = coordinates.size();

    if( dimensions!=3 ) {
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        PRINT0("In HilbertCurve.getHilbertIndex2DVector but dimensions is " << dimensions << " and not 3");
        throw std::runtime_error("Wrong dimensions given");
    }

    /*
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

    ValueType x,y,z;
    unsigned long integerIndex = 0;	//TODO: also check whether this data type is long enough
    const IndexType localN = coordinates[0].getLocalValues().size();

    // the DV to be returned
    std::vector<double> hilbertIndices(localN,-1);

    {
        SCAI_REGION( "HilbertCurve.getHilbertIndex3DVector.indicesCalculation" )

        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );
        
        const unsigned long long divisor = size_t(1) << size_t(3*int(recursionDepth));

        for (IndexType i = 0; i < localN; i++) {
            x = (coordAccess0[i]-minCoords[0])/dim0Extent;
            y = (coordAccess1[i]-minCoords[1])/dim1Extent;
            z = (coordAccess2[i]-minCoords[2])/dim2Extent;

            integerIndex = 0;
            for (IndexType j = 0; j < recursionDepth; j++) {
                int subSquare;
                if (z < 0.5) {
                    if (x < 0.5) {
                        if (y <0.5) {		//x,y,z <0.5
                            subSquare= 0;
                            //apply inverse hilbert operator
                            ValueType tmpX= x;
                            x= 2*z;
                            z= 2*y;
                            y= 2*tmpX;
                        } else {			//z<0.5, y>0.5, x<0.5
                            subSquare= 1;
                            ValueType tmpX= x;
                            x= 2*y-1;
                            y= 2*z;
                            z= 2*tmpX;
                        }
                    } else if (y>=0.5) {		//z<0.5, y,x>0,5
                        subSquare= 2;
                        //apply inverse hilbert operator
                        ValueType tmpX= x;
                        x= 2*y-1;
                        y= 2*z;
                        z= 2*tmpX-1;
                    } else {			//z<0.5, y<0.5, x>0.5
                        subSquare= 3;
                        x= -2*x+2;
                        y= -2*y+1;
                        z= 2*z;
                    }
                } else if(x>=0.5) {
                    if(y<0.5) { 		//z>0.5, y<0.5, x>0.5
                        subSquare= 4;
                        x= -2*x+2;
                        y= -2*y+1;
                        z= 2*z-1;
                    } else {			//z>0.5, y>0.5, x>0.5
                        subSquare= 5;
                        ValueType tmpX= x;
                        x= 2*y-1;
                        y= -2*z+2;
                        z= -2*tmpX+2;
                    }
                } else if(y<0.5) {		//z>0.5, y<0.5, x<0.5
                    subSquare= 7;	//care, this is 7, not 6
                    ValueType tmpX= x;
                    x= -2*z+2;
                    z= -2*y+1;
                    y= 2*tmpX;
                } else {			//z>0.5, y>0.5, x<0.5
                    subSquare= 6;	//this is case 6
                    ValueType tmpX= x;
                    x= 2*y-1;
                    y= -2*z +2;
                    z= -2*tmpX+1;
                }
                integerIndex = (integerIndex << 3) | subSquare;
            }
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
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::HilbertIndex2Point(const ValueType index, const IndexType level, const IndexType dimensions) {

    if (dimensions==2)
        return HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(index, level);

    if (dimensions==3)
        return HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(index, level);

    throw std::logic_error("Hilbert space filling curve only implemented for two or three dimensions");
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(const ValueType index, const IndexType level) {
    SCAI_REGION( "HilbertCurve.Hilbert2DIndex2Point" )
    std::vector<ValueType>  p(2,0), ret(2,0);
    ValueType r;
    IndexType q;
    if(index>1 || index < 0) {
        throw std::runtime_error("Index: " + std::to_string(index) +" for hilbert curve must be >0 and <1");
    }

    if (level > 0) {
        q=int(4*index);
        r= 4*index-q;
        p = HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2Point(r, level-1);
        switch(q) {
        case 0:
            ret = {ValueType(p[1]/2.0), ValueType(p[0]/2.0)};
            break;
        case 1:
            ret = {ValueType(p[0]/2.0), ValueType(p[1]/2.0 + 0.5)};
            break;
        case 2:
            ret = {ValueType(p[0]/2.0+0.5), ValueType(p[1]/2.0 + 0.5)};
            break;
        case 3:
            ret = {ValueType(-p[1]/2.0+1.0), ValueType(-p[0]/2.0 + 0.5)};
            break;
        }
    }
    return ret;
}

//-------------------------------------------------------------------------------------------------

/*
* Given a 3D point it returns its index in [0,1] on the hilbert curve based on the level depth.
*/

template<typename IndexType, typename ValueType>
std::vector<ValueType> HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(const ValueType index, const IndexType level) {
    SCAI_REGION( "HilbertCurve.Hilbert3DIndex2Point" )

    std::vector<ValueType>  p(3,0), ret(3,0);
    ValueType r;
    IndexType q;

    if (level > 0) {
        q=int(8*index);
        r= 8*index-q;
        if( (q==0) && r==0 ) return ret;
        p = HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2Point(r, level-1);

        switch(q) {
        case 0:
            ret = { ValueType(p[1]/2.0), ValueType(p[2]/2.0), ValueType(p[0]/2.0)};
            break;
        case 1:
            ret = { ValueType(p[2]/2.0), ValueType(0.5+p[0]/2.0), ValueType(p[1]/2.0)};
            break;
        case 2:
            ret = { ValueType(0.5+p[2]/2.0), ValueType(0.5+p[0]/2.0), ValueType(p[1]/2.0)};
            break;
        case 3:
            ret = { ValueType(1.0-p[0]/2.0), ValueType(0.5-p[1]/2.0), ValueType(p[2]/2.0)};
            break;
        case 4:
            ret = { ValueType(1.0-p[0]/2.0), ValueType(0.5-p[1]/2.0), ValueType(0.5+p[2]/2.0)};
            break;
        case 5:
            ret = { ValueType(1.0-p[2]/2.0), ValueType(0.5+p[0]/2.0), ValueType(1.0-p[1]/2.0)};
            break;
        case 6:
            ret = { ValueType(0.5-p[2]/2.0), ValueType(0.5+p[0]/2.0), ValueType(1.0-p[1]/2.0)};
            break;
        case 7:
            ret = { ValueType(p[1]/2.0), ValueType(0.5-p[2]/2.0), ValueType(1.0-p[0]/2.0)};
            break;
        }
    }
    return ret;
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> HilbertCurve<IndexType, ValueType>::HilbertIndex2PointVec(const std::vector<ValueType> indices, const IndexType level, const IndexType dimensions) {

    if( dimensions==2 )
        return HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2PointVec(indices, level);

    if( dimensions==3 )
        return HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2PointVec(indices, level);

    throw std::logic_error("Space filling curve currently only implemented for two or three dimensions");
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> HilbertCurve<IndexType, ValueType>::Hilbert2DIndex2PointVec(const std::vector<ValueType> indices, const IndexType level) {
    SCAI_REGION( "HilbertCurve.Hilbert2DIndex2PointVec" )
    std::vector <std::vector<ValueType>> res( indices.size());

    for( int i=0; i<indices.size(); i++) {
        res[i] = Hilbert2DIndex2Point( indices[i], level);
    }
    SCAI_ASSERT_EQ_ERROR( res[0].size(), 2, "The points should have size 2");

    return res;
}
//-------------------------------------------------------------------------------------------------

/*
* Given a 3D point it returns its index in [0,1] on the hilbert curve based on the level depth.
*/

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> HilbertCurve<IndexType, ValueType>::Hilbert3DIndex2PointVec(const std::vector<ValueType> indices, const IndexType level) {
    SCAI_REGION( "HilbertCurve.Hilbert3DIndex2PointVec" )
    std::vector <std::vector<ValueType>> res( indices.size());

    for( int i=0; i<indices.size(); i++) {
        res[i] = Hilbert3DIndex2Point( indices[i], level);
    }
    SCAI_ASSERT_EQ_ERROR( res[0].size(), 3, "The points should have size 3");

    return res;
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<sort_pair> HilbertCurve<IndexType, ValueType>::getSortedHilbertIndices( const std::vector<DenseVector<ValueType>> &coordinates, Settings settings) {

    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();

    const IndexType dimensions = coordinates.size();
    const IndexType localN = coordDist->getLocalSize();
    const IndexType globalN = coordDist->getGlobalSize();

    const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(ValueType(std::log2(globalN)), ValueType(21));
    
    /*
    *	create space filling curve indices.
    */

    std::vector<sort_pair> localPairs(localN);

    {
        SCAI_REGION("HilbertCurve.getSortedHilbertIndices.spaceFillingCurve");

        //get hilbert indices for all the points
        std::vector<double> localHilbertInd = HilbertCurve<IndexType,ValueType>::getHilbertIndexVector(coordinates, recursionDepth, dimensions);
        SCAI_ASSERT_EQ_ERROR(localHilbertInd.size(), localN, "Size mismatch");

        for (IndexType i = 0; i < localN; i++) {
            localPairs[i].value = localHilbertInd[i];
            localPairs[i].index = coordDist->local2Global(i);
        }
    }

    /*
    * now sort the global indices by where they are on the space-filling curve.
    */

    {
        SCAI_REGION( "HilbertCurve.getSortedHilbertIndices.sorting" );

        int typesize;
        MPI_Type_size(MPI_DOUBLE_INT, &typesize);
        //MPI_Type_size(getMPITypePair<double,IndexType>(), &typesize);
        //assert(typesize == sizeof(sort_pair)); does not have to be true anymore due to padding

        //call distributed sort
        //sfc index is hardcoded to double to allow better precision

        //MPI_Comm mpi_comm, std::vector<value_type> &data, long long global_elements = -1, Compare comp = Compare()
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        JanusSort::sort(mpi_comm, localPairs, MPI_DOUBLE_INT);
        //JanusSort::sort(mpi_comm, localPairs, getMPITypePair<ValueType,IndexType>());

        //copy hilbert indices into array

        //check size and sanity
        SCAI_ASSERT_EQ_ERROR( comm->sum(localPairs.size()), globalN, "Global index mismatch.");

        //check checksum
        if( settings.debugMode) {
            PRINT0("******** in debug mode");
            unsigned long indexSumAfter = 0;
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
void HilbertCurve<IndexType, ValueType>::redistribute(std::vector<DenseVector<ValueType> >& coordinates, std::vector<DenseVector<ValueType>>& nodeWeights, Settings settings, Metrics<ValueType>& metrics) {
    SCAI_REGION_START("HilbertCurve.redistribute.sfc")
    scai::dmemo::DistributionPtr inputDist = coordinates[0].getDistributionPtr();
    scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();

    if (comm->getSize() == 1) {
        return;
    }

    std::chrono::time_point<std::chrono::steady_clock> beforeInitPart =  std::chrono::steady_clock::now();
    const IndexType numNodeWeights = nodeWeights.size();

    bool nodesUnweighted = true;
    for (IndexType w = 0; w < numNodeWeights; w++) {
        if (nodeWeights[w].max() != nodeWeights[w].min()) nodesUnweighted = false;
    }

    std::chrono::duration<double> migrationCalculation, migrationTime;

    std::vector<double> hilbertIndices = HilbertCurve<IndexType, ValueType>::getHilbertIndexVector(coordinates, settings.sfcResolution, settings.dimensions);
    SCAI_REGION_END("HilbertCurve.redistribute.sfc")
    SCAI_REGION_START("HilbertCurve.redistribute.sort")
    /*
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

    MPI_Comm mpi_comm = MPI_COMM_WORLD; //TODO: cast the communicator ptr to a MPI communicator and get getMPIComm()?
    JanusSort::sort(mpi_comm, localPairs, MPI_DOUBLE_INT);
    //JanusSort::sort(mpi_comm, localPairs, getMPITypePair<ValueType,IndexType>() );
    
    migrationCalculation = std::chrono::steady_clock::now() - beforeInitPart;
    metrics.MM["timeMigrationAlgo"] = migrationCalculation.count();
    std::chrono::time_point < std::chrono::steady_clock > beforeMigration = std::chrono::steady_clock::now();
    assert(localPairs.size() > 0);

    SCAI_REGION_END("HilbertCurve.redistribute.sort")

    sort_pair minLocalIndex = localPairs[0];
    std::vector<double> sendThresholds(comm->getSize(), minLocalIndex.value);
    std::vector<double> recvThresholds(comm->getSize());

    comm->all2all(recvThresholds.data(), sendThresholds.data());//TODO: maybe speed up with hypercube
    SCAI_ASSERT_LT_ERROR(recvThresholds[comm->getSize() - 1], 1, "invalid hilbert index");
    // merge to get quantities //Problem: nodes are not sorted according to their hilbert indices, so accesses are not aligned.
    // Need to sort before and after communication
    assert(std::is_sorted(recvThresholds.begin(), recvThresholds.end()));
    std::vector<IndexType> permutation(localN);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](IndexType i, IndexType j) {
        return hilbertIndices[i] < hilbertIndices[j];
    });

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

    SCAI_REGION_START("HilbertCurve.redistribute.communicationPlan")

    // allocate sendPlan
    scai::dmemo::CommunicationPlan sendPlan(quantities.data(), comm->getSize());
    SCAI_ASSERT_EQ_ERROR(sendPlan.totalQuantity(), localN, "wrong size of send plan")

    // allocate recvPlan - either with allocateTranspose, or directly
    scai::dmemo::CommunicationPlan recvPlan = comm->transpose( sendPlan );
    const IndexType newLocalN = recvPlan.totalQuantity();

    //in some rare cases it can happen that some PE(s) do not get
    //any new local points; TODO: debug/investigate

    SCAI_REGION_END("HilbertCurve.redistribute.communicationPlan")

    if (settings.verbose) {
        PRINT0(std::to_string(localN) + " old local values "
               + std::to_string(newLocalN) + " new ones.");
    }
    //transmit indices, allowing for resorting of the received values
    std::vector<IndexType> sendIndices(localN);
    {
        SCAI_REGION("HilbertCurve.redistribute.permute");
        scai::hmemo::ReadAccess<IndexType> rIndices(myGlobalIndices);
        for (IndexType i = 0; i < localN; i++) {
            assert(permutation[i] < localN);
            assert(permutation[i] >= 0);
            sendIndices[i] = rIndices[permutation[i]];
        }
    }
    std::vector<IndexType> recvIndices(newLocalN);
    comm->exchangeByPlan(recvIndices.data(), recvPlan, sendIndices.data(), sendPlan);
    //get new distribution
    scai::hmemo::HArray<IndexType> indexTransport(newLocalN, recvIndices.data());
    
    auto newDist = scai::dmemo::generalDistributionUnchecked(globalN, std::move(indexTransport), comm);

    SCAI_ASSERT_EQUAL(newDist->getLocalSize(), newLocalN,
                      "wrong size of new distribution");
    for (IndexType i = 0; i < newLocalN; i++) {
        SCAI_ASSERT_VALID_INDEX_DEBUG(recvIndices[i], globalN, "invalid index");
    }

    {
        SCAI_REGION("HilbertCurve.redistribute.redistribute");
        // for each dimension: define DenseVector with new distribution, get write access to local values, call exchangeByPlan
        std::vector<ValueType> sendBuffer(localN);
        std::vector<ValueType> recvBuffer(newLocalN);

        for (IndexType d = 0; d < settings.dimensions; d++) {
            {
                SCAI_REGION("HilbertCurve.redistribute.redistribute.permute");
                scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());
                for (IndexType i = 0; i < localN; i++) { //TODO:maybe extract into lambda?
                    sendBuffer[i] = rCoords[permutation[i]]; //TODO: how to make this more cache-friendly? (Probably by using pairs and sorting them.)
                }
            }

            comm->exchangeByPlan(recvBuffer.data(), recvPlan, sendBuffer.data(), sendPlan);
            coordinates[d] = DenseVector<ValueType>(newDist, 0);
            {
                SCAI_REGION("HilbertCurve.redistribute.redistribute.permute");
                scai::hmemo::WriteAccess<ValueType> wCoords(coordinates[d].getLocalValues());
                assert(wCoords.size() == newLocalN);
                for (IndexType i = 0; i < newLocalN; i++) {
                    wCoords[newDist->global2Local(recvIndices[i])] =
                        recvBuffer[i];
                }
            }
        }
        // same for node weights
        for (IndexType w = 0; w < numNodeWeights; w++) {
            if (nodesUnweighted) {
                nodeWeights[w] = DenseVector<ValueType>(newDist, nodeWeights[w].getLocalValues()[0]);
            }
            else
            {
                {
                    SCAI_REGION("HilbertCurve.redistribute.redistribute.permute");
                    scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights[w].getLocalValues());
                    for (IndexType i = 0; i < localN; i++) {
                        sendBuffer[i] = rWeights[permutation[i]]; //TODO: how to make this more cache-friendly? (Probably by using pairs and sorting them.)
                    }
                }
                comm->exchangeByPlan(recvBuffer.data(), recvPlan, sendBuffer.data(), sendPlan);
                nodeWeights[w] = DenseVector<ValueType>(newDist, 0);
                {
                    SCAI_REGION("HilbertCurve.redistribute.redistribute.permute");
                    scai::hmemo::WriteAccess<ValueType> wWeights(nodeWeights[w].getLocalValues());
                    for (IndexType i = 0; i < newLocalN; i++) {
                        wWeights[newDist->global2Local(recvIndices[i])] = recvBuffer[i];
                    }
                }
            }
        }
    }
    migrationTime = std::chrono::steady_clock::now() - beforeMigration;
    metrics.MM["timeFirstDistribution"] = migrationTime.count();
    assert( confirmHilbertDistribution(coordinates, nodeWeights[0], settings) );
}
//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
bool HilbertCurve<IndexType, ValueType>::confirmHilbertDistribution(
    //const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<DenseVector<ValueType>> &coordinates,
    const DenseVector<ValueType> &nodeWeights,
    Settings settings
) {

    //get distributions of the input
    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    //const scai::dmemo::DistributionPtr graphDist = graph.getRowDistributionPtr();
    const scai::dmemo::DistributionPtr weightDist = nodeWeights.getDistributionPtr();

    if( not coordDist->isEqual( *weightDist) ) {
        throw std::runtime_error( "Distributions should be equal.");
    }

    //get sfc indices in every PE
    std::vector<double> localSFCInd = getHilbertIndexVector ( coordinates,  settings.sfcResolution, settings.dimensions);

    //sort local indices
    std::sort( localSFCInd.begin(), localSFCInd.end() );
    double sfcMinMax[2]= {0, 0};

    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();

    //if( localSFCInd.size()==0) {
    if( coordinates[0].getLocalValues().size()==0) {
        PRINT("\n***\tWarning: PE " << comm->getRank() << " has no local points. This probably will cause problems later. Maybe input is too small for this number of PEs." );
    }else{
        //the min and max local sfc value
        sfcMinMax[0] = localSFCInd.front();
        sfcMinMax[1] = localSFCInd.back();
    }

    if( settings.debugMode ) {
        PRINT(*comm <<": sending "<< sfcMinMax[0] << ", " << sfcMinMax[1] )	;
    }

    const IndexType p = comm->getSize();
    const IndexType root = 0; //set PE 0 as root
    IndexType arraySize = 1;
    if( comm->getRank()==root ) {
        arraySize = 2*p;
    }
    //so only the root PE allocates the array
    double allMinMax[arraySize];

    //every PE sends its local min and max to root
    comm->gather(allMinMax, 2, root, sfcMinMax );

    if( settings.debugMode and comm->getRank()==root ) {
        PRINT0("gathered: ");
        for(unsigned int i=0; i<arraySize; i++) {
            std::cout<< ", " << allMinMax[i];
        }
        std::cout<< std::endl;
    }

    //check if array is sorted. For all PEs except the root, this is trivially
    // true since their array has only one element
    std::vector<ValueType> gatheredInd( allMinMax, allMinMax+arraySize  );
    bool isSorted = std::is_sorted( gatheredInd.begin(), gatheredInd.end() );

    return comm->all( isSorted );
}
//-------------------------------------------------------------------------------------------------

//template function to get a MPI datatype. These are on purpose outside
// the class because we cannot specialize them without specializing
// the whole class. Maybe doing so it not a problem...

template<>
MPI_Datatype getMPIType<float>(){
 return MPI_FLOAT;
}

template<>
MPI_Datatype getMPIType<double>(){
    return MPI_DOUBLE ;
}

template<>
MPI_Datatype getMPITypePair<double,IndexType>(){
    return MPI_DOUBLE_INT;
}

template<>
MPI_Datatype getMPITypePair<float,IndexType>(){
    std::cout << __FILE__ << ", MPI_FLOAT_INT" << std::endl;
    return MPI_FLOAT_INT;
}

//-------------------------------------------------------------------------------------------------

template class HilbertCurve<IndexType, double>;
template class HilbertCurve<IndexType, float>;

} //namespace ITI
