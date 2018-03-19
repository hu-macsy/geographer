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

//TODO: get rid of dependacy with wrappers
//#include "Wrappers.h"

//#include "RBC/Sort/SQuick.hpp"

using scai::lama::Scalar;

namespace ITI {
	
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
		//center[d] = dist( generator );
		center[d] = maxCoords[d]/2.0;
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

/*
//TODO: adapt to also support initial node weights

template<typename IndexType, typename ValueType>
void Repartition<IndexType,ValueType>::getImbalancedDistribution(
	scai::lama::CSRSparseMatrix<ValueType> &graph,
	std::vector<scai::lama::DenseVector<ValueType>> &coords, 
	scai::lama::DenseVector<ValueType> &nodeWeights,
	ITI::Tool tool,
	struct Settings &settings,
	struct Metrics &metrics){
	
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	
	// nodeWeights have the same distribution as the coordinates
	const IndexType seed = 0;
	ValueType diverg = 1;	//with 0=> unit weigths, larger numbers create more divergence for the weigths
	IndexType dimensions = settings.dimensions;
	
	scai::lama::DenseVector<IndexType> firstPartition;
	scai::lama::DenseVector<ValueType> imbaNodeWeights;
	
	ValueType imbalance = 0;
	
	do{
		// set node weights to create artificial imbalance
		imbaNodeWeights = ITI::Repartition<IndexType,ValueType>::sNW( coords, seed, diverg, dimensions);
		bool nodeWeightsFlag = true;
		
		if( tool==ITI::Tool::geoKmeans or tool==ITI::Tool::geographer){
			//TODO: assuming uniform block sizes
			const IndexType globalN = graph.getNumRows();
			const std::vector<IndexType> blockSizes(settings.numBlocks, globalN/settings.numBlocks);
			firstPartition = ITI::KMeans::computePartition ( coords, settings.numBlocks, imbaNodeWeights, blockSizes, settings);
		}
		else{
			firstPartition = ITI::Wrappers<IndexType,ValueType>::partition ( graph, coords, imbaNodeWeights, nodeWeightsFlag, tool, settings, metrics);
		}
		
		imbalance = ITI::GraphUtils::computeImbalance(firstPartition, settings.numBlocks, nodeWeights);
		PRINT0("diverg= " << diverg<< " , first partition imbalance= " << imbalance);
		diverg += 0.1;
	}while( imbalance<0.2 and diverg<4);
	//TODO: check that these are OK
	
	{
		//get the distribution from the partition
		scai::dmemo::DistributionPtr firstDist = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( firstPartition.getDistribution(), firstPartition.getLocalValues() ) );
		
		SCAI_ASSERT_ERROR( imbaNodeWeights.getDistributionPtr()->isEqual( graph.getRowDistribution() ), "nodeWeights->distribution and graph.getRowDistribution do not agree.");
		SCAI_ASSERT_ERROR( imbaNodeWeights.getDistributionPtr()->isEqual( coords[0].getDistribution() ), "nodeWeights->distribution and coords[0].getDistribution do not agree.");
		
		scai::dmemo::DistributionPtr defaultDist = imbaNodeWeights.getDistributionPtr();
		scai::dmemo::Redistributor prepareRedist( firstDist, defaultDist );
		
		//distribute graph, node weights and coordinates to mimic an imbalanced simulation
		//TODO: what about nodeweights? now, just use uniform nodeweights for the repartition
		scai::dmemo::DistributionPtr columnDist = graph.getColDistributionPtr();
		graph.redistribute( prepareRedist, columnDist );
		
		for(IndexType i=0; i<dimensions; i++){
			coords[i].redistribute( prepareRedist );
		}
		
		nodeWeights.redistribute( prepareRedist );
	}
	
	}
*/	
//-----------------------------------------------------------------------------------------------------


//to force instantiation
template class Repartition<IndexType, ValueType>;

} //namespace ITI
