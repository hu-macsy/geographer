/*
 * Repartition.cpp
 *
 *  Created on: 25.02.18
 *      Author: harry
 */

#include <assert.h>
#include <cmath>
#include <climits>
#include <queue>
#include <string>
#include <unordered_set>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <set>
#include <iostream>
#include <iomanip> 

#include "Repartition.h"
#include "HilbertCurve.h"
#include "KMeans.h"


//#include "RBC/Sort/SQuick.hpp"

using scai::lama::Scalar;

namespace ITI {

//template <typename IndexType, typename ValueType>
//class Repartition{
	
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<ValueType> Repartition<IndexType,ValueType>::sNW( const std::vector<DenseVector<ValueType> >& coordinates, const IndexType seed, const ValueType diverg, const IndexType dimensions){
	
	const scai::dmemo::DistributionPtr distPtr = coordinates[0].getDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = distPtr->getCommunicatorPtr();
	const IndexType localN = distPtr->getLocalSize();
	//const IndexType dimensions = settings.dimensions;
	
	
	//1- create objects based on some input param
	
	std::vector<ValueType> center( dimensions, 0);	//one center
	
	//WARNING: does this always produces the same sequence of numbers for all PEs?
	std::default_random_engine generator( seed );
	
	std::vector<ValueType> maxCoords(dimensions);
	
	// set the coordinates of the center and get max
	for( IndexType d=0; d<dimensions; d++){
		maxCoords[d] = coordinates[d].max().Scalar::getValue<ValueType>();
		//std::uniform_real_distribution<ValueType> dist(minCoord[d], maxCoord[d]);
		std::uniform_real_distribution<ValueType> dist( 0, maxCoords[d]);
		center[d] = dist( generator );
		//center[d] = maxCoords[d]/2.0;
//PRINT(*comm << ": cent["<< d <<"]= " << center[d]);
		
	}

	
	//2- set local node weights that respect the objects
	
	// copy coordinates to a std::vector<std::vector>
	std::vector< std::vector<ValueType> > localPoints( localN, std::vector<ValueType>(dimensions,0) );
	{
		for (IndexType d = 0; d < dimensions; d++) {
			scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[d].getLocalValues() );
			for (IndexType i = 0; i < localN; i++) {
				localPoints[i][d] = localPartOfCoords[i];
			}
		}
	}
	
	
	scai::lama::DenseVector<ValueType> nodeWeights( distPtr, 0 );
	
	{
		scai::hmemo::WriteAccess<ValueType> wWeights(nodeWeights.getLocalValues());
		std::vector<ValueType> point(dimensions);
		//ValueType minMax = *std::min_element(maxCoords.begin(), maxCoords.end() );
		ValueType maxMax = *std::max_element(maxCoords.begin(), maxCoords.end() );
		ValueType maxThres = maxMax*  std::pow(dimensions,1.0/dimensions);
//PRINT0("maxThres= " << maxThres);		
		//ValueType thresholdTop = minMax/3;
		//ValueType thresholdBot = minMax/10;
		for(IndexType i=0; i<localN; i++){
			point = localPoints[i];
			ValueType distance = aux<IndexType,ValueType>::pointDistanceL2(center, point);
			ValueType normDist = distance/maxThres;
			//reverse distance and set as weight but crop so weights is >1 and <2
			/*
			if( distance>thresholdTop ){
				wWeights[i]= 1;
			}else if( distance<thresholdBot){
				wWeights[i]= 2;
			}else{
				wWeights[i]= distance;
			}
			*/
			wWeights[i]= std::pow(2.0/(1+normDist), diverg );
//PRINT(*comm << ": " << wWeights[i] );
		}
	}
	
	return nodeWeights;
}
//-----------------------------------------------------------------------------------------------------



//} //namespace Repartition

//to force instantiation
template class Repartition<IndexType, ValueType>;

} //namespace ITI
