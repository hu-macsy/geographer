/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
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

namespace Repartition{
	
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<ValueType> sNW( const scai::dmemo::DistributionPtr distPtr, const IndexType seed, const Settings settings){
	
	const IndexType dimensions = settings.dimensions;
	const IndexType localN = distPtr->getLocalSize();
	const scai::dmemo::CommunicatorPtr comm = distPtr->getCommunicatorPtr();
	
	
	//1- create objects based on some input param
	
	std::vector<IndexType> center( dimensions, 0);	//one center
	
	std::default_random_engine generator( seed );
	
	
	// set the coordinates of the center
	for( IndexType d=0; d<dimensions; d++){
		std::uniform_real_distribution<ValueType> dist(minCoord[d], maxCoord[d]);
		center[d] = dist( generator );
PRINT(*comm << ": cent["<< d <<"]= " << center[d]);
	}
	
	
	//2- set local node weights that respect the objects
	
	scai::lama::DenseVector<ValueType> nodeWeights( distPtr, 0 );
	
	{
		scai::hmemo::WriteAccess<ValueType> wWeights(nodeWeights.getLocalValues());
		for(IndexType i=0; i<localN; i++){
			
		}
	}
}
//-----------------------------------------------------------------------------------------------------



} //namespace Repartition

template class Repartition<IndexType, ValueType>;

} //namespace ITI
