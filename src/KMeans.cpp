/*
 * KMeans.cpp
 *
 *  Created on: 19.07.2017
 *      Author: moritz
 */

#include <set>
#include <cmath>
#include <assert.h>
#include <algorithm>

#include <scai/dmemo/NoDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>

#include "KMeans.h"
#include "HilbertCurve.h"
#include "MultiLevel.h"
//temporary, for debugging
#include "FileIO.h"

namespace ITI{
namespace KMeans{

//base implementation
template<typename IndexType, typename ValueType>
std::vector<std::vector<point>> findInitialCentersSFC(
		const std::vector<DenseVector<ValueType> >& coordinates, 
		const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords,
		const scai::lama::DenseVector<IndexType> &partition,
		const std::vector<cNode> hierLevel,

		Settings settings) {

	SCAI_REGION( "KMeans.findInitialCentersSFC" );
	const IndexType localN = coordinates[0].getLocalValues().size();
	const IndexType globalN = coordinates[0].size();
	const IndexType dimensions = settings.dimensions;
	const IndexType k = settings.numBlocks;
	//global communicator
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	//the input is already partitioned into numOldBlocks number of blocks
	//for every old block we must find a number of new centers/blocks

	const std::vector<unsigned int> numNewBlocksPerOldBlock = CommTree<IndexType,ValueType>::getGrouping( hierLevel );
	const unsigned int numOldBlocks = numNewBlocksPerOldBlock.size();

	//convert coordinates, switch inner and outer order
	std::vector<std::vector<ValueType> > convertedCoords( localN, std::vector<ValueType> (dimensions,0.0) );

	for (IndexType d = 0; d < dimensions; d++) {
		scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
		assert(rAccess.size() == localN);
		for (IndexType i = 0; i < localN; i++) {
			convertedCoords[i][d] = rAccess[i];
		}
	}

	//TODO: In the hierarchical case, we need to compute new centers
	//many times, not just once. Take the hilbert indices once
	//outside the function and not every time is called

	//the local points but sorted according to the SFC
	//needed to find the correct(based on the sfc ordering) center index
	std::vector<IndexType> sortedLocalIndices(localN);
	{
		//get local hilbert indices
		std::vector<ValueType> sfcIndices = HilbertCurve<IndexType, ValueType>::getHilbertIndexVector( coordinates, settings.sfcResolution, settings.dimensions);
		SCAI_ASSERT_EQ_ERROR( sfcIndices.size(), localN, "wrong local number of indices (?) ");

		//prepare indices for sorting
		std::iota( sortedLocalIndices.begin(), sortedLocalIndices.end(), 0);

		//sort local indices according to SFC
		std::sort( sortedLocalIndices.begin(), sortedLocalIndices.end(), [&sfcIndices](IndexType a, IndexType b){return sfcIndices[a] < sfcIndices[b];});
	}

	//get prefix sum for every known block
	//TODO?: use a DenseVector in order to use the already implemented
	//function MultiLevel::computeGlobalPrefixSum to get a global
	//prefix sum.

	const unsigned int numPEs = comm->getSize();
	const IndexType rootPE = 0; // set PE 0 as root

	//the global number of points of each old block
	std::vector<IndexType> globalBlockSizes( numOldBlocks );
	//global prefix sum vector of size (p+1)*numOldBlocks
	//ATTENTION: this a a concatenation of prefix sum arrays
	//the real prefix sums are [prefixSumArray[0]:prefixSumArray[numPEs]],
	// prefixSumArray[numPEs+1]:prefixSumArray[2*numPEs]], ... ,
	//example: [0,4,10,15, 0,7,15,22, 0,12,20,30, 0, ... ]
	//           block 1    block 2     block 3 ... block numOldBlocks
	//every "subarray" has size numPEs+1 (in this example numPEs=3)
	std::vector<IndexType> concatPrefixSumArray;

	//TODO?: convert vector above to a vector<vector> of size numPEs and every
	// inner vector of size numOldBlocks?

	{
		std::vector<IndexType> oldBlockSizes( numOldBlocks, 0);
		scai::hmemo::ReadAccess<IndexType> localPart = partition.getLocalValues();
		SCAI_ASSERT_EQ_ERROR( localPart.size(), localN, "Partition size mismatch");

		//count the size (the number of points) of every block locally
		for( unsigned int i=0; i<localN; i++){
			IndexType thisPointBlock = localPart[i];
			oldBlockSizes[ thisPointBlock ]++;
		}

		//gather all block sizes to root
		IndexType arraySize=1;
		if( comm->getRank()==rootPE ){
			arraySize = numPEs*numOldBlocks;
		}
		IndexType allOldBlockSizes[arraySize];
		comm->gather( allOldBlockSizes, numOldBlocks, rootPE, oldBlockSizes.data() );
		std::vector<IndexType> allOldSizesVec( allOldBlockSizes, allOldBlockSizes + arraySize );
		if( comm->getRank()==rootPE ){
			SCAI_ASSERT_EQ_ERROR( globalN, std::accumulate(allOldSizesVec.begin(), allOldSizesVec.end(), 0), "Mismatch in gathered array for sizes of all blocks for PE " << *comm);
		}

		// only root PE calculates the prefixSum
		if( comm->getRank()==rootPE ){
			for( unsigned int blockId=0; blockId<numOldBlocks; blockId++){
				//prefix sum for every block starts with 0
				concatPrefixSumArray.push_back(0);
				for( unsigned int pe=0; pe<numPEs; pe++){
					concatPrefixSumArray.push_back( concatPrefixSumArray.back() + allOldBlockSizes[pe*numOldBlocks+blockId] );
				}
			}
			SCAI_ASSERT_EQ_ERROR( concatPrefixSumArray.size(), (numPEs+1)*numOldBlocks, "Prefix sum array has wrong size" );
		}else{
			concatPrefixSumArray.resize( (numPEs+1)*numOldBlocks, 0 );
		}

		comm->bcast( concatPrefixSumArray.data() ,(numPEs+1)*numOldBlocks, rootPE);		


		for( unsigned int b=0; b<numOldBlocks; b++){
			globalBlockSizes[b] = concatPrefixSumArray[(b+1)*numPEs+b];
			SCAI_ASSERT_EQ_ERROR( concatPrefixSumArray[b*(numPEs+1)] , 0, "Wrong concat prefix sum array, values at indices b*(numPEs+1) must be zero, Failed for b=" << b);
		}
		IndexType prefixSumCheckSum = std::accumulate( globalBlockSizes.begin(), globalBlockSizes.end(), 0 );
		SCAI_ASSERT_EQ_ERROR( prefixSumCheckSum, globalN, "Global sizes mismatch. Wrong calculation of prefix sum?");
	}

	//compute wanted indices for initial centers
	//newCenterIndWithinBLock[i] = a vector with the indices of the 
	//centers for block i
	//newCenterIndWithinBLock[i].size() = numNewBlocksPerOldBlock[b], i.e., the 
	// new number of blocks to partition previous block i
	//ATTENTION: newCenterIndWithinBLock[i][j] = x: is the index of the 
	//center within block i. If x is 30, then we want the 30-th point
	//of block i.

	std::vector<std::vector<IndexType>> newCenterIndWithinBLock(numOldBlocks);

	//for all old blocks
	for( IndexType b=0; b<numOldBlocks; b++){
		//the number of centers for block b
		IndexType k_b = numNewBlocksPerOldBlock[b]; 
		newCenterIndWithinBLock[b].resize( k_b );
		for( IndexType i = 0; i < k_b; i++) {
			//wantedIndices[i] = i * (globalN / k) + (globalN / k)/2;
			newCenterIndWithinBLock[b][i] = i*(globalBlockSizes[b]/k_b) + (globalBlockSizes[b]/k_b)/2;
		}
	}

	const IndexType thisPE = comm->getRank();

	//the centers to be returned, each PE fills only with owned centers
	std::vector<std::vector<point>> centersPerNewBlock( numOldBlocks);
	for( IndexType b=0; b<numOldBlocks; b++){
		centersPerNewBlock[b].resize( numNewBlocksPerOldBlock[b] , point(dimensions, 0.0) );
	}

	//for debugging
	IndexType sumOfRanges = 0;
	IndexType numOwnedCenters = 0;

	for( IndexType b=0; b<numOldBlocks; b++){
		IndexType fromInd = b*(numPEs+1)+thisPE;
		assert( fromInd+1<concatPrefixSumArray.size() );
		
		//the range of the indices for block b for this PE		
		IndexType rangeStart = concatPrefixSumArray[ fromInd ];
		IndexType rangeEnd = concatPrefixSumArray[ fromInd+1];
		sumOfRanges += rangeEnd-rangeStart;
		//keep a counter that indicates the index of a point within
		//a block in this PE.
		IndexType counter = rangeStart;

		//center indices for block b, pass by reference so not to copy
		const std::vector<IndexType>& centersForThisBlock = newCenterIndWithinBLock[b];

		//TODO: optimize? Now, complexity is localN*number of owned centers
		//can we do it with one linear scan?

		//if some center indexes are local in this PE, store them.
		//Later, we will scan the local points for their coordinates
		for( unsigned int j=0; j<centersForThisBlock.size(); j++ ){
			IndexType centerInd = centersForThisBlock[j];
			counter = rangeStart;//reset counter for next center
			//if center index for block b is owned by thisPE
			if( centerInd>=rangeStart and centerInd<=rangeEnd){
//PRINT(*comm << ": owns center with index " << centerInd << " for block " << b);				
				//since we own a center, go over all local points
				//and calculate their within-block index for the block 
				//they belong to
				scai::hmemo::ReadAccess<IndexType> localPart = partition.getLocalValues();
				for(unsigned int i=0; i<localN; i++ ){
					//consider points based on their sorted sfc index
					IndexType sortedIndex = sortedLocalIndices[i];
					IndexType thisPointBlock = localPart[ sortedIndex ];
					//TODO: remove assertion?
					assert( thisPointBlock<numOldBlocks );
					if( thisPointBlock!=b ){
						continue;//not in desired block
					}
					
					IndexType withinBlockIndex = counter;
					//desired center found
					if( withinBlockIndex==centerInd ){
						//store center coords
						centersPerNewBlock[b][j] = convertedCoords[ sortedIndex ];
						numOwnedCenters++;
						//PRINT(*comm <<": adding center "<< centerInd << " with coordinates " << convertedCoords[sortedIndex][0] << ", " << convertedCoords[sortedIndex][1] );			
						break;
					}
					counter++;
				}//for i<localN
				SCAI_ASSERT_LE_ERROR( counter, rangeEnd, "Within-block index out of bounds");
			}//if center is local
		}//for j<centersForThisBlock.size()
	}//for b<numOldBlocks

	SCAI_ASSERT_EQ_ERROR(sumOfRanges, localN, thisPE << ": Sum of owned number of points per block should be equal the total number of local points");

	if( settings.debugMode ){
		PRINT( *comm << ": owns " << numOwnedCenters << " centers");
		unsigned int numNewTotalBlocks = std::accumulate(numNewBlocksPerOldBlock.begin(), numNewBlocksPerOldBlock.end(), 0);
		SCAI_ASSERT_EQ_ERROR( comm->sum(numOwnedCenters), numNewTotalBlocks , "Not all centers were found");
	}

	//
	//global sum operation. Doing it in a separate loop on purpose
	//since different PEs own centers from different blocks and for most
	//blocks they own no centers at all
	//

	for( IndexType b=0; b<numOldBlocks; b++){

		SCAI_ASSERT_EQ_ERROR( centersPerNewBlock[b][0].size(), dimensions, "Dimension mismatch for center" );
		IndexType numCenters = centersPerNewBlock[b].size();

		//pack in a raw array
		std::vector<ValueType> allCenters( numCenters*dimensions );

		for( unsigned int c=0; c<numCenters; c++ ){
			//this copies the point, this is unnecessary, TODO: fix
			point thisCenter = centersPerNewBlock[b][c];
			//copy this center
			std::copy( thisCenter.begin(), thisCenter.end(), allCenters.begin() +c*dimensions );
		}

		//global sum
		comm->sumImpl( allCenters.data(), allCenters.data(), numCenters*dimensions, scai::common::TypeTraits<ValueType>::stype  );

		//unpack back to vector<point>
		for( unsigned int c=0; c<numCenters; c++ ){
			for (IndexType d=0; d<dimensions; d++) {
				//center c, for block b
				centersPerNewBlock[b][c][d] = allCenters[ c*dimensions+d ];
			}
		}
	}

	return centersPerNewBlock;
}


//overloaded function for non-hierarchical version. 
//Set partition to 0 for all points
//A "flat" communication tree
//and return only the first (there is only one) group of centers
template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>>  findInitialCentersSFC(
	const std::vector<DenseVector<ValueType>>& coordinates,
	const std::vector<ValueType> &minCoords,
	const std::vector<ValueType> &maxCoords,
	Settings settings){

	//TODO: probably must also change the settings.numBlocks

	//homogeneous case, all PEs have the same memory and speed
	//there is only hierarchy level
	//TODO: probably this needs some further adaptation for the hierarchical version
	IndexType mem = 1;
	IndexType speed = 1;
	IndexType cores = 1;

	std::vector<cNode> leaves( settings.numBlocks );
	for(int i=0; i<settings.numBlocks; i++){ 
		leaves[i] = cNode(std::vector<unsigned int>{0}, cores, mem, speed);
	}

	//every point belongs to one block in the beginning
	scai::lama::DenseVector<IndexType> partition( coordinates[0].getDistributionPtr(), 0);

	//return a vector of size 1 with
	std::vector<std::vector<point>> initialCenters = findInitialCentersSFC( coordinates, minCoords, maxCoords, partition, leaves, settings );

	SCAI_ASSERT_EQ_ERROR( initialCenters.size(), 1, "Wrong vector size");
	SCAI_ASSERT_EQ_ERROR( initialCenters[0].size(), settings.numBlocks, "Wrong vector size" );

	//TODO: must change convert centers to a vector of size=dimensions
	//where initialCenters[0][d][i] is the d-th coordinate of the i-th center
	IndexType dimensions = settings.dimensions;

	SCAI_ASSERT_EQ_ERROR( minCoords.size(), settings.dimensions, "Wrong center dimensions");

//TODO: check/verify if we do not need to revert the vector order
	//reverse vector order here
	std::vector<std::vector<ValueType>> reversedCenters( dimensions, std::vector<ValueType>(settings.numBlocks, 0.0) );
	for( unsigned int c=0; c<settings.numBlocks; c++){
		for( unsigned int d=0; d<dimensions; d++){
			reversedCenters[d][c] = initialCenters[0][c][d];
		}
	}
	//return reversedCenters;

	return initialCenters[0];
}


template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCentersFromSFCOnly(const std::vector<ValueType> &maxCoords, Settings settings){
	//This assumes that minCoords is 0!
    //TODO: change or remove
	const IndexType dimensions = settings.dimensions;
	const IndexType k = settings.numBlocks;
		
	//set local values in vector, leave non-local values with zero
	std::vector<std::vector<ValueType> > result(dimensions);
	for (IndexType d = 0; d < dimensions; d++) {
		result[d].resize(k);
	}
	
	ValueType offset = 1.0/(ValueType(k)*2.0);
	std::vector<ValueType> centerCoords(dimensions,0);
	for (IndexType i = 0; i < k; i++) {
		ValueType centerHilbInd = i/ValueType(k) + offset;

		centerCoords = HilbertCurve<IndexType,ValueType>::HilbertIndex2Point( centerHilbInd, settings.sfcResolution, settings.dimensions);
		SCAI_ASSERT_EQ_ERROR( centerCoords.size(), dimensions, "Wrong dimensions for center.");
		
		for (IndexType d = 0; d < dimensions; d++) {
			result[d][i] = centerCoords[d]*maxCoords[d];
		}
	}
	return result;
}


template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findLocalCenters(	const std::vector<DenseVector<ValueType> >& coordinates, const DenseVector<ValueType> &nodeWeights) {
	
	const IndexType dim = coordinates.size();
	const IndexType localN = coordinates[0].getLocalValues().size();
	
	// get sum of local weights 
		
	scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
	SCAI_ASSERT_EQ_ERROR( rWeights.size(), localN, "Mismatch of nodeWeights and coordinates size. Check distributions.");
	
	ValueType localWeightSum = 0;
	for (IndexType i=0; i<localN; i++) {
		localWeightSum += rWeights[i];
	}
		
	std::vector<ValueType> localCenter( dim , 0 );
	
	for (IndexType d = 0; d < dim; d++) {
		scai::hmemo::ReadAccess<ValueType> rCoords( coordinates[d].getLocalValues() );
		for (IndexType i=0; i<localN; i++) {
			//this is more expensive than summing first and dividing later, but avoids overflows
			localCenter[d] += rWeights[i]*rCoords[i]/localWeightSum;
		}
	}
	
	// vector of size k for every center. Each PE only stores its local center and sum center
	// so all centers are replicated in all PEs
	
	const scai::dmemo::CommunicatorPtr comm = coordinates[0].getDistribution().getCommunicatorPtr();
	const IndexType numPEs = comm->getSize();
	const IndexType thisPE = comm->getRank();
	std::vector<std::vector<ValueType> > result( dim, std::vector<ValueType>(numPEs,0) );
	for (IndexType d=0; d<dim; d++){
		result[d][thisPE] = localCenter[d];
	}
	
	for (IndexType d=0; d<dim; d++){
		comm->sumImpl(result[d].data(), result[d].data(), numPEs, scai::common::TypeTraits<ValueType>::stype);
	}
	return result;
}


template<typename IndexType, typename ValueType, typename Iterator>
std::vector<point> findCenters(
		const std::vector<DenseVector<ValueType> >& coordinates,
		const DenseVector<IndexType>& partition,
		const IndexType k,
		const Iterator firstIndex,
		const Iterator lastIndex,
		const DenseVector<ValueType>& nodeWeights) {
	SCAI_REGION( "KMeans.findCenters" );

	const IndexType dim = coordinates.size();
	const scai::dmemo::DistributionPtr resultDist(new scai::dmemo::NoDistribution(k));
	const scai::dmemo::CommunicatorPtr comm = partition.getDistribution().getCommunicatorPtr();

	//TODO: check that distributions align

	std::vector<std::vector<ValueType> > result(dim);
	std::vector<ValueType> weightSum(k, 0);

	scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
	scai::hmemo::ReadAccess<IndexType> rPartition(partition.getLocalValues());

	//compute weight sums
	for (Iterator it = firstIndex; it != lastIndex; it++) {
		const IndexType i = *it;
		const IndexType part = rPartition[i];
		const ValueType weight = rWeights[i];
		weightSum[part] += weight;
		//the lines above are equivalent to: weightSum[rPartition[*it]] += rWeights[*it];
	}

	//find local centers
	for (IndexType d = 0; d < dim; d++) {
		result[d].resize(k);
		scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());

		for (Iterator it = firstIndex; it != lastIndex; it++) {
			const IndexType i = *it;
			const IndexType part = rPartition[i];
			result[d][part] += rCoords[i]*rWeights[i] / weightSum[part];//this is more expensive than summing first and dividing later, but avoids overflows
		}
	}

	//communicate local centers and weight sums
	std::vector<ValueType> totalWeight(k, 0);
	comm->sumImpl(totalWeight.data(), weightSum.data(), k, scai::common::TypeTraits<ValueType>::stype);

	//compute updated centers as weighted average
	for (IndexType d = 0; d < dim; d++) {
		for (IndexType j = 0; j < k; j++) {
			ValueType weightRatio = (ValueType(weightSum[j]) / totalWeight[j]);

			ValueType weightedCoord = weightSum[j] == 0 ? 0 : result[d][j] * weightRatio;
			result[d][j] = weightedCoord;
			assert(std::isfinite(result[d][j]));

			// make empty clusters explicit
			if (totalWeight[j] == 0) {
			    result[d][j] = NAN;
			}
		}
		comm->sumImpl(result[d].data(), result[d].data(), k, scai::common::TypeTraits<ValueType>::stype);
	}

	return result;
}


template<typename IndexType, typename ValueType>
std::vector<point> vectorTranspose( const std::vector<std::vector<ValueType>>& points){
	const IndexType dim = points.size();
	SCAI_ASSERT_GT_ERROR( dim, 0, "Dimension of points cannot be 0" );

	const IndexType numPoints = points[0].size();
	SCAI_ASSERT_GT_ERROR( numPoints, 0, "Empty vector of points" );

	std::vector<point> retPoints( numPoints, point(dim) );
	
	for( unsigned int d=0; d<dim; d++ ){
		for( unsigned int i=0; i<numPoints; i++ ){
			retPoints[i][d] = points[d][i];
		}
	}

	return retPoints;
}


template<typename IndexType, typename ValueType, typename Iterator>
DenseVector<IndexType> assignBlocks(
		const std::vector<std::vector<ValueType>>& coordinates,
		const std::vector<point>& centers,
		const std::vector<IndexType>& blockSizesPrefixSum,
		const Iterator firstIndex,
		const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights,
		const DenseVector<IndexType> &previousAssignment,
		const DenseVector<IndexType> &oldBlock,
		const std::vector<ValueType> &optWeightAllBlocks,
		const SpatialCell &boundingBox,
		std::vector<ValueType> &upperBoundOwnCenter,
		std::vector<ValueType> &lowerBoundNextCenter,
		std::vector<ValueType> &influence,
		ValueType &imbalance,
		std::vector<ValueType> &timePerPE,
		Settings settings,
		Metrics &metrics) {
	SCAI_REGION( "KMeans.assignBlocks" );

//
std::chrono::time_point<std::chrono::high_resolution_clock> assignStart = std::chrono::high_resolution_clock::now();
//

	const IndexType dim = coordinates.size();
	const scai::dmemo::DistributionPtr dist = nodeWeights.getDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType localN = nodeWeights.getLocalValues().size();

	//number of blocks from the previous hierarchy
	const IndexType numOldBlocks= blockSizesPrefixSum.size()-1;
	
	//this check is done before. TODO: remove?
	if( settings.debugMode ){
		const IndexType maxPart = oldBlock.max(); //global operation
		SCAI_ASSERT_EQ_ERROR( numOldBlocks-1, maxPart, "The provided old assignment must have equal number of blocks as the length of the vector with the new number of blocks per part");
	}

	// numNewBlocks is equivalent to 'k' in the classic version
	IndexType numNewBlocks = centers.size(); 

	SCAI_ASSERT_EQ_ERROR( blockSizesPrefixSum.back(), numNewBlocks, "Total number of new blocks mismatch" );

	//centers are given as a 1D vector alongside with a prefix sum vector
	const std::vector<point>& centers1DVector = centers;

	SCAI_ASSERT_EQ_ERROR( centers1DVector.size(), numNewBlocks, "Vector size mismatch" );
	SCAI_ASSERT_EQ_ERROR( centers1DVector[0].size(), dim, "Center dimensions mismatch" );
	SCAI_ASSERT_EQ_ERROR( influence.size(), numNewBlocks, "Vector size mismatch" );

	//pre-filter possible closest blocks
	std::vector<ValueType> minDistanceAllBlocks( numNewBlocks );
	std::vector<ValueType> effectMinDistAllBlocks( numNewBlocks );

	//for all new blocks
	for( IndexType newB=0; newB<numNewBlocks; newB++ ){
		SCAI_REGION( "KMeans.assignBlocks.filterCenters" );

		point center = centers1DVector[newB];
		minDistanceAllBlocks[newB] = boundingBox.distances(center).first;
		assert( std::isfinite(minDistanceAllBlocks[newB]) );
		effectMinDistAllBlocks[newB] = minDistanceAllBlocks[newB]\
			*minDistanceAllBlocks[newB]\
			*influence[newB];
		assert( std::isfinite(effectMinDistAllBlocks[newB]) );	
	}

	//sort centers according to their distance from the bounding box of this PE
	std::vector<IndexType> clusterIndicesAllBlocks( numNewBlocks );
	//cluster indices are "global": from 0 to numNewBlocks
	std::iota( clusterIndicesAllBlocks.begin(), clusterIndicesAllBlocks.end(), 0);

	for( IndexType oldB=0; oldB<numOldBlocks; oldB++ ){
		const unsigned int rangeStart = blockSizesPrefixSum[oldB];
		const unsigned int rangeEnd = blockSizesPrefixSum[oldB+1];
		typename std::vector<IndexType>::iterator startIt = clusterIndicesAllBlocks.begin()+rangeStart;
		typename std::vector<IndexType>::iterator endIt = clusterIndicesAllBlocks.begin()+rangeEnd;
		//TODO: remove not needed assertions
		SCAI_ASSERT_LT_ERROR( rangeStart, rangeEnd, "Prefix sum vectors is wrong");
		SCAI_ASSERT_LE_ERROR( rangeEnd, numNewBlocks, "Range out of bounds" );

		//sort the part of the indices that belong to this old block
		std::sort( startIt, endIt, 
			[&](IndexType a, IndexType b){
				return effectMinDistAllBlocks[a] < effectMinDistAllBlocks[b] \
				|| (effectMinDistAllBlocks[a] == effectMinDistAllBlocks[b] && a < b);
			}
		);

		//sort also this part of the distances
		//TODO: is this sorting needed?
		//TODO: is it correct to sort? if we sort, effectMinDistAllBlocks[clusterInd[i]] will be wrong, I think effectMinDistAllBlocks[i] will be correct
		//TODO: either not sort and get minDistanceAllBlocks[c] or sort and get minDistanceAllBlocks[i]
		//update 15/02: yes, sorting is needed and we access it as in the for loop below: effectMinDistAllBlocks[i]
		std::sort( effectMinDistAllBlocks.begin()+rangeStart, effectMinDistAllBlocks.begin()+rangeEnd);

		//just for checking
		for (IndexType i=rangeStart; i<rangeEnd; i++) {
			IndexType c = clusterIndicesAllBlocks[i];
			ValueType effectiveDist = minDistanceAllBlocks[c]*minDistanceAllBlocks[c]*influence[c];
			SCAI_ASSERT_LT_ERROR( std::abs(effectMinDistAllBlocks[i] - effectiveDist), 1e-5, "effectiveMinDistance[" << i << "] = " << effectMinDistAllBlocks[i] << " != " << effectiveDist << " = effectiveDist");
		}
	}

	//ValueType imbalance;
	IndexType iter = 0;
	IndexType skippedLoops = 0;
	ValueType time = 0;	// for timing/profiling
	std::vector<bool> influenceGrew( numNewBlocks );
	std::vector<ValueType> influenceChangeUpperBound(numNewBlocks, 1+settings.influenceChangeCap);
	std::vector<ValueType> influenceChangeLowerBound(numNewBlocks, 1-settings.influenceChangeCap);

	//compute assignment and balance
	DenseVector<IndexType> assignment = previousAssignment;

	//iterate if necessary to achieve balance
	do
	{
		std::chrono::time_point<std::chrono::high_resolution_clock> balanceStart = std::chrono::high_resolution_clock::now();
		SCAI_REGION( "KMeans.assignBlocks.balanceLoop" );

		//the block weight for all new blocks
		std::vector<ValueType> blockWeights( numNewBlocks, 0.0 );

		IndexType totalComps = 0;		
		skippedLoops = 0;
		IndexType balancedBlocks = 0;
		scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
		scai::hmemo::ReadAccess<IndexType> rOldBlock( oldBlock.getLocalValues());
		scai::hmemo::WriteAccess<IndexType> wAssignment(assignment.getLocalValues());
		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.assign" );
			IndexType forLoopCnt = 0;
			//for the sampled range
			for (Iterator it = firstIndex; it != lastIndex; it++) {
				++forLoopCnt;
				const IndexType i = *it;
				const IndexType oldCluster = wAssignment[i];
				const IndexType fatherBlock = rOldBlock[i];

				//TODO: not needed assertion, this is checked in the beginning				
				SCAI_ASSERT_LT_ERROR( fatherBlock, numOldBlocks, "Wrong father block index");

				if (lowerBoundNextCenter[i] > upperBoundOwnCenter[i]) {
					//cluster assignment cannot have changed.
					//wAssignment[i] = wAssignment[i];
					skippedLoops++;
				} else {
					ValueType sqDistToOwn = 0;
					point myCenter = centers1DVector[oldCluster];
					for (IndexType d = 0; d < dim; d++) {		
						sqDistToOwn += std::pow(myCenter[d]-coordinates[d][i], 2);
					}
					ValueType newEffectiveDistance = sqDistToOwn*influence[oldCluster];
					assert(upperBoundOwnCenter[i] >= newEffectiveDistance);
					upperBoundOwnCenter[i] = newEffectiveDistance;
					if (lowerBoundNextCenter[i] > upperBoundOwnCenter[i]) {
						//cluster assignment cannot have changed.
						//wAssignment[i] = wAssignment[i];
						skippedLoops++;
					} else {
						//check the centers of this old block to find the closest one
						int bestBlock = 0;
						ValueType bestValue = std::numeric_limits<ValueType>::max();
						IndexType secondBest = 0;
						ValueType secondBestValue = std::numeric_limits<ValueType>::max();
						
						//where the range of indices starts for the father block
						const IndexType rangeStart = blockSizesPrefixSum[fatherBlock];
						const IndexType rangeEnd = blockSizesPrefixSum[fatherBlock+1];
						SCAI_ASSERT_LE_ERROR( rangeEnd, clusterIndicesAllBlocks.size(), "Range out of bounds");

						//start with the first center index
						IndexType c = rangeStart;

						//check all centers belonging to the father block to find the closest
						while(c < rangeEnd && secondBestValue > effectMinDistAllBlocks[c]) {
							totalComps++;
							//remember: cluster centers are sorted according to their distance from the bounding box of this PE	
							//also, the cluster indices go from 0 till numNewBlocks
							IndexType j = clusterIndicesAllBlocks[c];//maybe it would be useful to sort the whole centers array, aligning memory accesses.

							//squared distance from previous assigned center
							ValueType sqDist = 0;
							point myCenter = centers1DVector[j];
							//TODO: restructure arrays to align memory accesses better in inner loop
							for (IndexType d = 0; d < dim; d++) {
								sqDist += std::pow(myCenter[d]-coordinates[d][i], 2);
							}

							const ValueType effectiveDistance = sqDist*influence[j];

							//update best and second-best centers
							if (effectiveDistance < bestValue) {
								secondBest = bestBlock;
								secondBestValue = bestValue;
								bestBlock = j;
								bestValue = effectiveDistance;
							} else if (effectiveDistance < secondBestValue) {
								secondBest = j;
								secondBestValue = effectiveDistance;
							}
							c++;
						} //while

						//TODO: this is wrong when k=1
						//SCAI_ASSERT_NE_ERROR( bestBlock, secondBest, "Best and second best should be different" );
						assert(secondBestValue >= bestValue);

						//this point has a new center
						if (bestBlock != oldCluster) {
							//assert(bestValue >= lowerBoundNextCenter[i]);
							SCAI_ASSERT_GE_ERROR( bestValue , lowerBoundNextCenter[i], \
								"PE " << comm->getRank() << ": difference " << std::abs(bestValue - lowerBoundNextCenter[i]) << \
								" for i= " << i << ", oldCluster: " << oldCluster << ", newCluster: " << bestBlock);
						}

						upperBoundOwnCenter[i] = bestValue;
						lowerBoundNextCenter[i] = secondBestValue;
						wAssignment[i] = bestBlock;	
					}
				}
				//we found the best block for this point; increase the weight of this block
				//TODO: adapt for multiple weights
				blockWeights[wAssignment[i]] += rWeights[i];		
			}//for sampled indices
			
			if (settings.verbose) {
				std::chrono::duration<ValueType,std::ratio<1>> balanceTime = std::chrono::high_resolution_clock::now() - balanceStart;			
				ValueType time = balanceTime.count() ;
				std::cout<< comm->getRank()<< ": time " << time << std::endl;
				PRINT(comm->getRank() << ": in assignBlocks, balanceIter time: " << time << ", for loops: " << forLoopCnt );
				ValueType maxTime = comm->max( time );
				PRINT0( "max time: " << maxTime << ", for loops: " << forLoopCnt );

				SCAI_ASSERT_LT_ERROR( comm->getRank(), timePerPE.size(), "vector size mismatch" );
				timePerPE[comm->getRank()] += time;
			}

			comm->synchronize();
		}// assignment block

		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.blockWeightSum" );
			comm->sumImpl( blockWeights.data(), blockWeights.data(), numNewBlocks, scai::common::TypeTraits<ValueType>::stype);
		}
		
		//calculate imbalance for every new block
		std::vector<ValueType> imbalances( numNewBlocks );
		for( int newB=0; newB<numNewBlocks; newB++ ){
			ValueType optWeight = optWeightAllBlocks[newB];
			imbalances[newB] = (ValueType(blockWeights[newB] - optWeight)/optWeight);
		}

		//imbalance in the maximum imbalance of all new blocks
		imbalance = *std::max_element(imbalances.begin(), imbalances.end() );
		//TODO: adapt for multiple node weights

		SCAI_ASSERT_GE_ERROR( imbalance, 0, "Imbalance cannot be negative");

		std::vector<ValueType> oldInfluence = influence;//size=numNewBlocks
		assert( oldInfluence.size()== numNewBlocks );

		double minRatio = std::numeric_limits<double>::max();
		double maxRatio = -std::numeric_limits<double>::min();

		for (IndexType j=0; j<numNewBlocks; j++) {
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.influence" );

			double ratio = ValueType(blockWeights[j])/optWeightAllBlocks[j];
			if (std::abs(ratio - 1) < settings.epsilon) {
				balancedBlocks++;
				if (settings.freezeBalancedInfluence) {
					if (1 < minRatio) minRatio = 1;
					if (1 > maxRatio) maxRatio = 1;
					continue;
				}
			}

			influence[j] = 	std::max( influence[j]*influenceChangeLowerBound[j],
								std::min( influence[j] * std::pow(ratio, settings.influenceExponent),
									influence[j]*influenceChangeUpperBound[j]	
								) 
				);
			assert(influence[j] > 0);

			double influenceRatio = influence[j] / oldInfluence[j];

			assert(influenceRatio <= influenceChangeUpperBound[j] + 1e-10);
			assert(influenceRatio >= influenceChangeLowerBound[j] - 1e-10);
			if (influenceRatio < minRatio) minRatio = influenceRatio;
			if (influenceRatio > maxRatio) maxRatio = influenceRatio;

			if (settings.tightenBounds && iter > 0 && (bool(ratio > 1) != influenceGrew[j])) {
				//influence change switched direction
				influenceChangeUpperBound[j] = 0.1 + 0.9*influenceChangeUpperBound[j];
				influenceChangeLowerBound[j] = 0.1 + 0.9*influenceChangeLowerBound[j];
				assert(influenceChangeUpperBound[j] > 1);
				assert(influenceChangeLowerBound[j] < 1);
			}
			influenceGrew[j] = bool(ratio > 1);
		}//for

		//update bounds
		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.updateBounds" );
			for (Iterator it = firstIndex; it != lastIndex; it++) {
				const IndexType i = *it;
				const IndexType cluster = wAssignment[i];
				upperBoundOwnCenter[i] *= (influence[cluster] / oldInfluence[cluster]) + 1e-12;
				lowerBoundNextCenter[i] *= minRatio - 1e-12;//TODO: compute separate min ratio with respect to bounding box, only update that.		
			}
		}

		//update possible closest centers
		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.filterCenters" );
			for (IndexType newB=0; newB<numNewBlocks; newB++) {
				effectMinDistAllBlocks[newB] = minDistanceAllBlocks[newB]*minDistanceAllBlocks[newB]*influence[newB];
			}
			//TODO: duplicated code as in the beginning of the assignBlocks
			for( IndexType oldB=0; oldB<numOldBlocks; oldB++ ){
				const unsigned int rangeStart = blockSizesPrefixSum[oldB];
				const unsigned int rangeEnd = blockSizesPrefixSum[oldB+1];
				typename std::vector<IndexType>::iterator startIt = clusterIndicesAllBlocks.begin()+rangeStart;
				typename std::vector<IndexType>::iterator endIt = clusterIndicesAllBlocks.begin()+rangeEnd;
				//TODO: remove not needed assertions
				SCAI_ASSERT_LT_ERROR( rangeStart, rangeEnd, "Prefix sum vectos is wrong");
				SCAI_ASSERT_LE_ERROR( rangeEnd, numNewBlocks, "Range out of bounds" );
				//SCAI_ASSERT_ERROR( endIt!=clusterIndicesAllBlocks.end(), "Iterator out of bounds");

				//sort the part of the indices that belong to this old block
				std::sort( startIt, endIt, 
					[&](IndexType a, IndexType b){
						return effectMinDistAllBlocks[a] < effectMinDistAllBlocks[b] \
						|| (effectMinDistAllBlocks[a] == effectMinDistAllBlocks[b] && a < b);
					}
				);
				//sort also this part of the distances
				//TODO: is this sorting needed?
				//TODO: is it correct to sort? if we sort, effectMinDistAllBlocks[clusterInd[i]] will be wrong, I think effectMinDistAllBlocks[i] will be correct
				//update 15/02: yes, sorting is needed (see also above)
				std::sort( effectMinDistAllBlocks.begin()+rangeStart, effectMinDistAllBlocks.begin()+rangeEnd);
			}
		}

		iter++;

		if ( settings.verbose ) {
			const IndexType currentLocalN = std::distance(firstIndex, lastIndex);
			const IndexType takenLoops = currentLocalN - skippedLoops;
			const ValueType averageComps = ValueType(totalComps) / currentLocalN;
			//double minInfluence, maxInfluence;
			auto pair = std::minmax_element(influence.begin(), influence.end());
			const ValueType influenceSpread = *pair.second / *pair.first;
			std::chrono::duration<ValueType,std::ratio<1>> balanceTime = std::chrono::high_resolution_clock::now() - balanceStart;			
			time += balanceTime.count() ;

			auto oldprecision = std::cout.precision(3);
			if (comm->getRank() == 0) std::cout << "Iter " << iter << ", loop: " << 100*ValueType(takenLoops) / currentLocalN << "%, average comparisons: "
					<< averageComps << ", balanced blocks: " << 100*ValueType(balancedBlocks) / numNewBlocks << "%, influence spread: " << influenceSpread
					<< ", imbalance : " << imbalance << ", time elapsed: " << time << std::endl;
			std::cout.precision(oldprecision);
		}

	} while (imbalance > settings.epsilon - 1e-12 && iter < settings.balanceIterations);
	
	if( settings.verbose )
		std::cout << "Process " << comm->getRank() << " skipped " << ValueType(skippedLoops*100) / (iter*localN) << "% of inner loops." << std::endl;
	//aux<IndexType,ValueType>::timeMeasurement(assignStart);
	
	//for kmeans profiling
	metrics.numBalanceIter.push_back(iter);

	return assignment;
}//assignBlocks

/**
 */
//WARNING: we do not use k as repartition assumes k=comm->getSize() and neither blockSizes and we assume
// 			that every block has the same size

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeRepartition(
	const std::vector<DenseVector<ValueType>>& coordinates,
	const DenseVector<ValueType>& nodeWeights,
	const Settings settings,
	struct Metrics& metrics) {
	
	const IndexType localN = nodeWeights.getLocalValues().size();
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType k = comm->getSize();
	SCAI_ASSERT_EQ_ERROR(k, settings.numBlocks, "Deriving the previous partition from the distribution cannot work for p == k");
	
	// calculate the global weight sum to set the block sizes
	//TODO: the local weight sums are already calculated in findLocalCenters, maybe extract info from there
	ValueType globalWeightSum;
	
	{
		ValueType localWeightSum = 0;
		scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
		SCAI_ASSERT_EQ_ERROR( rWeights.size(), localN, "Mismatch of nodeWeights and coordinates size. Chech distributions.");
		
		for (IndexType i=0; i<localN; i++) {
			localWeightSum += rWeights[i];
		}
	
		globalWeightSum = comm->sum(localWeightSum);
	}
	
	const std::vector<IndexType> blockSizes(settings.numBlocks, globalWeightSum/settings.numBlocks);

	std::chrono::time_point<std::chrono::high_resolution_clock> startCents = std::chrono::high_resolution_clock::now();
	//
	//TODO: change to findCenters
	//
	std::vector<std::vector<ValueType> > initialCenters = findLocalCenters<IndexType,ValueType>(coordinates, nodeWeights);
	std::chrono::duration<ValueType,std::ratio<1>> centTime = std::chrono::high_resolution_clock::now() - startCents;	
	ValueType time = centTime.count();
	std::cout<< comm->getRank()<< ": time " << time << std::endl;

	//aux<IndexType,ValueType>::timeMeasurement( startCents );

	//WARNING: this was in the initial version. The problem is that each PE find one center.
	// This can lead to bad solutions since dense areas may require more centers
	//std::vector<std::vector<ValueType> > initialCenters = findLocalCenters<IndexType,ValueType>(coordinates, nodeWeights);
	//std::vector<std::vector<ValueType> > initialCenters = findLocalCenters(coordinates, nodeWeights);
	
	return computePartition(coordinates, nodeWeights, blockSizes, initialCenters, settings, metrics);
}



template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeRepartition(
	const std::vector<DenseVector<ValueType>>& coordinates,
	const DenseVector<ValueType>& nodeWeights,
	const std::vector<IndexType>& blockSizes,
	const DenseVector<IndexType>& previous,
	const Settings settings) {

	const IndexType localN = nodeWeights.getLocalValues().size();
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	std::vector<std::vector<ValueType> > initialCenters;

	if (settings.numBlocks == comm->getSize()
	        && comm->all(scai::utilskernel::HArrayUtils::max(previous.getLocalValues()) == comm->getRank())
	        && comm->all(scai::utilskernel::HArrayUtils::min(previous.getLocalValues()) == comm->getRank())) {
	    //partition is equal to distribution
	    //TODO:: change with findCenters
	    initialCenters = findLocalCenters<IndexType,ValueType>(coordinates, nodeWeights);
	} else {
	    std::vector<IndexType> indices(localN);
	    std::iota(indices.begin(), indices.end(), 0);
	    initialCenters = findCenters(coordinates, previous, settings.numBlocks, indices.begin(), indices.end(), nodeWeights);
	}

	//just one group with all the centers; needed in the hierarchical version
	std::vector<std::vector<point>> groupOfCenters = { initialCenters };

	//must convert the block sizes to precentages,
	std::vector<ValueType> blockSizesPerCent( blockSizes.size() );
	//const totalWeight = std::accumulate( blockSizes.begin(), blockSizes.end(), 0);
	const IndexType maxWeight = *std::max_element( blockSizes.begin(), blockSizes.end() );
	for( IndexType i=0; i<blockSizes.size(); i++ ){
		//blockSizesPerCent[i] = blockSizes[i]/totalWeight;
		//use maxWeight instead of total to resemble the modelling from TEEC
		blockSizesPerCent[i] = blockSizes[i]/maxWeight;
	}

	Metrics metrics(settings);
//
//TODO: added previous here, not sure at all about it
//

return computePartition(coordinates, nodeWeights, blockSizesPerCent, /**/ previous /**/, groupOfCenters, settings, metrics);
}

//usual call that does not take the garph as input
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition( \
	const std::vector<DenseVector<ValueType>> &coordinates, \
	const DenseVector<ValueType> &nodeWeights, \
	const std::vector<ValueType> &blockSizesPerCent, \
	const DenseVector<IndexType> &partition, \
	std::vector<std::vector<point>> centers, \
	const Settings settings, \
	struct Metrics &metrics ) {

	const IndexType N = nodeWeights.size();
	scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));

	scai::dmemo::DistributionPtr dist = coordinates[0].getDistributionPtr();
	auto graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDistPtr);

	return computePartition( graph, coordinates, nodeWeights, blockSizesPerCent,
		partition, centers, settings, metrics );
}

//TODO: graph is not needed, this is only for debugging
//core implementation 
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition( \
	const CSRSparseMatrix<ValueType> &graph, \
	const std::vector<DenseVector<ValueType>> &coordinates, \
	const DenseVector<ValueType> &nodeWeights, \
	const std::vector<ValueType> &blockSizesPerCent, \
	const DenseVector<IndexType> &partition, \
	std::vector<std::vector<point>> centers, \
	const Settings settings, \
	struct Metrics &metrics ) {

	SCAI_REGION( "KMeans.computePartition" );
	std::chrono::time_point<std::chrono::high_resolution_clock> KMeansStart = std::chrono::high_resolution_clock::now();

	//the number of blocks from the previous hierarchy level
	const IndexType numOldBlocks = centers.size();
	if( settings.debugMode ){
		const IndexType maxPart = partition.max(); //global operation
		SCAI_ASSERT_EQ_ERROR( numOldBlocks-1, maxPart, "The provided partition must have equal number of blocks as the length of the vector with the new number of blocks per part");
	}
	
	//the number of new blocks per old block and the total number of new blocks
	std::vector<IndexType> blockSizesPrefixSum( numOldBlocks+1, 0 );
	//in a sense, this is the new k = settings.numBlocks
	IndexType totalNumNewBlocks = 0;

	for( int b=0; b<numOldBlocks; b++ ){
		blockSizesPrefixSum[b+1] += blockSizesPrefixSum[b]+centers[b].size();
		totalNumNewBlocks += centers[b].size();
	}

	//convert to a 1D vector
	std::vector<point> centers1DVector;
	for(int b=0; b<numOldBlocks; b++){
		const unsigned int k = blockSizesPrefixSum[b+1]-blockSizesPrefixSum[b];
		assert( k==centers[b].size() ); //not really needed, TODO: remove?
		for (IndexType i=0; i<k; i++) {
			centers1DVector.push_back( centers[b][i] );
		}
	}

	SCAI_ASSERT_EQ_ERROR( centers1DVector.size(), totalNumNewBlocks, "Vector size mismatch" );

	std::vector<ValueType> influence(totalNumNewBlocks, 1);
	
	const IndexType dim = coordinates.size();
	assert(dim > 0);
	const IndexType localN = nodeWeights.getLocalValues().size();
	const IndexType globalN = nodeWeights.size();
	assert(nodeWeights.getLocalValues().size() == coordinates[0].getLocalValues().size());
	SCAI_ASSERT_EQ_ERROR( centers[0][0].size(), dim, "Center dimensions mismatch" );
	SCAI_ASSERT_EQ_ERROR( centers1DVector[0].size(), dim, "Center dimensions mismatch" );

	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	const IndexType p = comm->getSize();
	
	//min and max for local part of the coordinates
	std::vector<ValueType> minCoords(dim);
	std::vector<ValueType> maxCoords(dim);
	std::vector<std::vector<ValueType> > convertedCoords(dim);

	// copy and sort coordinates according to their hilbert index
	{
		// a vector of the indices
		std::vector<IndexType> permIndices(localN);
		std::iota(permIndices.begin(), permIndices.end(), 0);

		// sfc indices of all local points
		const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(globalN), double(21));
		std::vector<ValueType> localHilbertInd = HilbertCurve<IndexType,ValueType>::getHilbertIndexVector(coordinates, recursionDepth, settings.dimensions);

		SCAI_ASSERT_EQ_ERROR(localN, localHilbertInd.size() , "vector size mismatch");

		//WARNING: the next sorting is wrong as it is now. It messes up the point indices and coordinates:
		// point i has different coordinates than in the beginning. Either leave it out or maybe also sort
		// the point indices. I am leaving it here for future reference
		// sort the point/vertex indices based on their hilbert index
		//std::sort(permIndices.begin(), permIndices.end(), [&](IndexType i, IndexType j){return localHilbertInd[i] < localHilbertInd[j];});

		for (IndexType d = 0; d < dim; d++) {
			scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
			//convertedCoords[d] = std::vector<ValueType>(rAccess.get(), rAccess.get()+localN);
			assert(rAccess.size() == localN);				
			
			// copy coordinates sorted by their hilbert index
			IndexType checkSum = 0;
			ValueType sumCoord = 0.0;
			convertedCoords[d].resize(localN);
			for(IndexType i=0; i<localN; i++){
				//TODO: the permIndices vectors is not needed
				IndexType ind = permIndices[i];
				ValueType coord = rAccess[ ind ];
				convertedCoords[d][i] = coord;

				//meant for debugging reasons, remove or add debug macros
				checkSum += ind;
				sumCoord += coord;
			}

			SCAI_ASSERT_EQ_ERROR(checkSum, (localN*(localN-1)/2), "Checksum error");
			ValueType sumCoord2 = std::accumulate( convertedCoords[d].begin(), convertedCoords[d].end(), 0.0);
			//added std::abs since some coordinates can be negative and have a negative sum
			SCAI_ASSERT_GE_ERROR( std::abs(sumCoord), std::abs(0.999*sumCoord2), "Error in sorting local coordinates");

			minCoords[d] = *std::min_element(convertedCoords[d].begin(), convertedCoords[d].end());
			maxCoords[d] = *std::max_element(convertedCoords[d].begin(), convertedCoords[d].end());

			//or, do not take the hilbert index, do not sort and just copy coordinates
			//TODO: test improvement, in some small inputs, the sorted version did less 
			// iterations

			assert(convertedCoords[d].size() == localN);
		}
	}

	std::vector<ValueType> globalMinCoords(dim);
	std::vector<ValueType> globalMaxCoords(dim);
	comm->minImpl(globalMinCoords.data(), minCoords.data(), dim, scai::common::TypeTraits<ValueType>::stype);
	comm->maxImpl(globalMaxCoords.data(), maxCoords.data(), dim, scai::common::TypeTraits<ValueType>::stype);

	ValueType diagonalLength = 0;
	ValueType volume = 1;
	ValueType localVolume = 1;
	for (IndexType d = 0; d < dim; d++) {
		const ValueType diff = globalMaxCoords[d] - globalMinCoords[d];
		const ValueType localDiff = maxCoords[d] - minCoords[d];
		diagonalLength += diff*diff;
		volume *= diff;
		localVolume *= localDiff;
	}

	//the bounding box is per PE. no need to change for the hierarchical version
	QuadNodeCartesianEuclid boundingBox(minCoords, maxCoords);
    if (settings.verbose) {
		std::cout << "(PE id, localN) = (" << comm->getRank() << ", "<< localN << ")" << std::endl;
		comm->synchronize();
		std::cout << "(PE id, localVolume/(volume/p) = (" << comm->getRank() << ", "<< localVolume / (volume / p) << ")" << std::endl;
    }

	diagonalLength = std::sqrt(diagonalLength);
	const ValueType expectedBlockDiameter = pow(volume /totalNumNewBlocks, 1.0/dim);

	std::vector<ValueType> upperBoundOwnCenter(localN, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> lowerBoundNextCenter(localN, 0);

	//
	//prepare sampling
	//

	std::vector<IndexType> localIndices(localN);
	std::iota(localIndices.begin(), localIndices.end(), 0);

	//hierar: in the heterogenous and hierarchical case, minSamplingNodes
	//makes more sense to be a percentage of the nodes, not a number. Or not?
	//hierar: number of sampling nodes is calculated per PE, right? not per block.

	const ValueType avgBlocksPerPE = ValueType(totalNumNewBlocks)/p;
	//We can calculate precisely the number of local old blocks using "partition"
	//but we will have to go over all local points
	IndexType minNodes = settings.minSamplingNodes*avgBlocksPerPE;
	if( settings.minSamplingNodes==-1 ){
		minNodes = localN;
	}

	assert(minNodes > 0);
	IndexType samplingRounds = 0;	//number of rounds needed to see all points
	std::vector<IndexType> samples;

	const bool randomInitialization = comm->all(localN > minNodes);

	//perform sampling
	{
		if (randomInitialization) {
			ITI::GraphUtils<IndexType, ValueType>::FisherYatesShuffle(localIndices.begin(), localIndices.end(), localN);
			//TODO: the cantor shuffle is more stable; random suffling can yield better
			// results occasionally but has higher fluctuation/variance
			//localIndices = GraphUtils<IndexType,ValueType>::indexReorderCantor( localN );

			SCAI_ASSERT_EQ_ERROR(*std::max_element(localIndices.begin(), localIndices.end()), localN -1, "Error in index reordering");
			SCAI_ASSERT_EQ_ERROR(*std::min_element(localIndices.begin(), localIndices.end()), 0, "Error in index reordering");

			samplingRounds = std::ceil(std::log2( globalN / ValueType(settings.minSamplingNodes*totalNumNewBlocks)))+1;

			samples.resize(samplingRounds);
			samples[0] = std::min(minNodes, localN);
		}

		if(settings.verbose){
			PRINT(*comm << ": localN= "<< localN << ", minNodes= " << minNodes << ", samplingRounds= " << samplingRounds << ", lastIndex: " << *localIndices.end() );
		}
		if (samplingRounds > 0 && settings.verbose) {
			if (comm->getRank() == 0) std::cout << "Starting with " << samplingRounds << " sampling rounds." << std::endl;
		}
		//double the number of samples per round
		for (IndexType i = 1; i < samplingRounds; i++) {
			samples[i] = std::min(IndexType(samples[i-1]*2), localN);
		}
		if (samplingRounds > 0) {
		    samples[samplingRounds-1] = localN;
		}
	}
	//
	//aux<IndexType,ValueType>::timeMeasurement(KMeansStart);
	//

	scai::hmemo::ReadAccess<ValueType> rWeight(nodeWeights.getLocalValues());
	IndexType iter = 0;
	ValueType delta = 0;
	bool balanced = false;
	const ValueType threshold = 0.002*diagonalLength;//TODO: take global point density into account
	const IndexType maxIterations = settings.maxKMeansIterations;
	const typename std::vector<IndexType>::iterator firstIndex = localIndices.begin();
	typename std::vector<IndexType>::iterator lastIndex = localIndices.end();
	ValueType imbalance = 1;


	// result[i]=b, means that point i belongs to cluster/block b
	DenseVector<IndexType> result(coordinates[0].getDistributionPtr(), 0);


	do {
		std::chrono::time_point<std::chrono::high_resolution_clock> iterStart = std::chrono::high_resolution_clock::now();
		if (iter < samplingRounds) {
		    SCAI_ASSERT_LE_ERROR(samples[iter], localN, "invalid number of samples");
			lastIndex = localIndices.begin() + samples[iter];
			std::sort(localIndices.begin(), lastIndex);//sorting not really necessary, but increases locality
			ValueType ratio = ValueType(comm->sum(samples[iter])) / globalN;
			assert(ratio <= 1);

		} else {
		    SCAI_ASSERT_EQ_ERROR(lastIndex - firstIndex, localN, "invalid iterators");
			assert(lastIndex == localIndices.end());
		}

		//moving here the sum of weights of the sampled points from assignBlock
		ValueType localSampleWeightSum = 0;
		{
			scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
			for (auto it = firstIndex; it != lastIndex; it++) {
				localSampleWeightSum += rWeights[*it];
			}
		}

		const ValueType totalSampledWeightSum = comm->sum(localSampleWeightSum);

		//TODO: adapt for multiple node weights
		//WARNING: this is related with how we store and add the relative speed
		//see also the ComMTree::createLevelAbove() function

		//the "optimum" weight every new block should have
		std::vector<ValueType> optWeightAllBlocks( totalNumNewBlocks );
		for( IndexType newB=0; newB<totalNumNewBlocks; newB++ ){
			optWeightAllBlocks[newB] = blockSizesPerCent[newB]*totalSampledWeightSum;
		}

		std::vector<ValueType> timePerPE( comm->getSize(), 0.0);

		result = assignBlocks(convertedCoords, centers1DVector, blockSizesPrefixSum, firstIndex, lastIndex, nodeWeights, result, partition, optWeightAllBlocks, boundingBox, upperBoundOwnCenter, lowerBoundNextCenter, influence, imbalance, timePerPE, settings, metrics);

		scai::hmemo::ReadAccess<IndexType> rResult(result.getLocalValues());

		if(settings.verbose){
			comm->sumImpl( timePerPE.data(), timePerPE.data(), comm->getSize(), scai::common::TypeTraits<ValueType>::stype);
			if(comm->getRank()==0 ){
				vector<IndexType> indices( timePerPE.size() );
				std::iota(indices.begin(), indices.end(), 0);
				std::sort( indices.begin(), indices.end(), 
					[&timePerPE](int i, int j){ return timePerPE[i]<timePerPE[j]; } );

				for(int i=0; i<comm->getSize(); i++){
					//std::cout << indices[i]<< ": time for PE: " << timePerPE[indices[i]] << std::endl;
					std::cout << "(" << indices[i] << "," << timePerPE[indices[i]] << ")" << std::endl;
				}
			}
		}

		std::vector<std::vector<ValueType>> newCenters = findCenters(coordinates, result, totalNumNewBlocks, firstIndex, lastIndex, nodeWeights);

		//newCenters have reversed order of the vectors
		//maybe turn centers to a 1D vector already in computePartition?

		//TODO: why does vectorTranspose needs <IndexType> ??
		std::vector<point> transCenters = vectorTranspose<IndexType>( newCenters );
		assert( transCenters.size()==totalNumNewBlocks );
		assert( transCenters[0].size()==dim );

		//keep centroids of empty blocks at their last known position
		for (IndexType j = 0; j <totalNumNewBlocks; j++) {
			//center for block j is empty
			if (std::isnan(transCenters[j][0])) {				
				transCenters[j] = centers1DVector[j];
			}
		}
		std::vector<ValueType> squaredDeltas(totalNumNewBlocks,0);
		std::vector<ValueType> deltas(totalNumNewBlocks,0);
		std::vector<ValueType> oldInfluence = influence; 
		ValueType minRatio = std::numeric_limits<double>::max();

		for (IndexType j = 0; j < totalNumNewBlocks; j++) {
			for (int d = 0; d < dim; d++) {
				//TODO: copied from the Dev branch, commit 94e40203248c9e981af98c80fb47ba60e4c77ec2
				// the same code does not exist in this version so I added the assertion here
				SCAI_ASSERT_LE_ERROR( transCenters[j][d], globalMaxCoords[d], "New center coordinate out of bounds" );
		    	SCAI_ASSERT_GE_ERROR( transCenters[j][d], globalMinCoords[d], "New center coordinate out of bounds" );
				ValueType diff = (centers1DVector[j][d] - transCenters[j][d]);
				squaredDeltas[j] += diff*diff;
			}
			deltas[j] = std::sqrt(squaredDeltas[j]);
			if (settings.erodeInfluence) {
				std::cout<< "WARNING: erodeInfluence setting is not supported in the heterogeneous or hierarchical version" <<std::endl;
				if( false ){
					const ValueType erosionFactor = 2/(1+exp(-std::max(deltas[j]/expectedBlockDiameter-0.1, 0.0))) - 1;
					influence[j] = exp((1-erosionFactor)*log(influence[j]));//TODO: will only work for uniform target block sizes
					if (oldInfluence[j] / influence[j] < minRatio) minRatio = oldInfluence[j] / influence[j];
				}
			}
		}
		centers1DVector = transCenters;

		delta = *std::max_element(deltas.begin(), deltas.end());
		assert(delta >= 0);
		const double deltaSq = delta*delta;
		const double maxInfluence = *std::max_element(influence.begin(), influence.end());
		//const double minInfluence = *std::min_element(influence.begin(), influence.end());
		{
			SCAI_REGION( "KMeans.computePartition.updateBounds" );
			for (auto it = firstIndex; it != lastIndex; it++) {
				const IndexType i = *it;
				IndexType cluster = rResult[i];
				assert( cluster<totalNumNewBlocks );

				if (settings.erodeInfluence) {
					//WARNING: erodeInfluence not supported for hierarchical version
					//TODO: or it is?? or it should be??

					//update due to erosion
					upperBoundOwnCenter[i] *= (influence[cluster] / oldInfluence[cluster]) + 1e-12;
					lowerBoundNextCenter[i] *= minRatio - 1e-12;
				}

				//update due to delta
				upperBoundOwnCenter[i] += (2*deltas[cluster]*std::sqrt(upperBoundOwnCenter[i]/influence[cluster]) + squaredDeltas[cluster])*(influence[cluster] + 1e-10);
				ValueType pureSqrt(std::sqrt(lowerBoundNextCenter[i]/maxInfluence));
				if (pureSqrt < delta) {
					lowerBoundNextCenter[i] = 0;
				} else {
					ValueType diff = (-2*delta*pureSqrt + deltaSq)*(maxInfluence + 1e-10);
					assert(diff <= 0);
					lowerBoundNextCenter[i] += diff;
					if (!(lowerBoundNextCenter[i] > 0)) lowerBoundNextCenter[i] = 0;
				}
				assert(std::isfinite(lowerBoundNextCenter[i]));
			}
		}
		
		//find local weight of each block
		std::vector<ValueType> blockWeights(totalNumNewBlocks,0.0);
		for (auto it = firstIndex; it != lastIndex; it++) {
			const IndexType i = *it;
			IndexType cluster = rResult[i];
			blockWeights[cluster] += rWeight[i];
		}

		// print times before global reduce step
		//aux<IndexType,ValueType>::timeMeasurement(iterStart);
		std::chrono::duration<ValueType,std::ratio<1>> balanceTime = std::chrono::high_resolution_clock::now() - iterStart;			
		ValueType time = balanceTime.count() ;

		if(settings.verbose){
			PRINT(*comm <<": in computePartition, iteration time: " << time );
		}

		{
			SCAI_REGION( "KMeans.computePartition.blockWeightSum" );
			comm->sumImpl(blockWeights.data(), blockWeights.data(), totalNumNewBlocks, scai::common::TypeTraits<ValueType>::stype);
		}

		//TODO: adapt for heterogeneous weights
		//TODO: adapt for multiple node weights

		//check if all blocks are balanced
		balanced = true;
		for (IndexType j=0; j<totalNumNewBlocks; j++) {
			if (blockWeights[j] > optWeightAllBlocks[j]*(1+settings.epsilon)) {
				balanced = false;
			}
		}

		ValueType maxTime=0;
		if(settings.verbose){
			balanceTime = std::chrono::high_resolution_clock::now() - iterStart;
			maxTime = comm->max( balanceTime.count() );
		}

		
		//WARNING: when sampling, a (big!) part of the result vector is not changed
		// and keeps its initial value which is 0. So, the computeImbalance finds,
		// falsely, block 0 to be over weighted. We use the returned imbalance
		// from assign centers when sampling is used and we compute a new imbalance
		// only when there is no sampling

/*		//TODO: what imbalance should we compute for multiple weights?
		if( !randomInitialization ){
			imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( result, settings.numBlocks, nodeWeights );
		}
*/

		//this does not seem to work. Even if it works, in the first iterations, almost all
		//nodes belong to block 0.
		//TODO: either fix or remove
		ValueType cut=-1;
		if( settings.debugMode ){
			cut = ITI::GraphUtils<IndexType, ValueType>::computeCut( graph, result, false );			
		}

		if (comm->getRank() == 0) {
			std::cout << "i: " << iter<< ", delta: " << delta << ", time : "<< maxTime << ", imbalance= " << imbalance<< ", cut= " << cut << std::endl;
		}

		metrics.kmeansProfiling.push_back( std::make_tuple(delta, maxTime, imbalance) );

		iter++;

		// WARNING-TODO: this (the "if() break; code" ) stops the iterations prematurely,
		// when the wanted balance is reached.
		// It is possible that if we allow more iterations, the solution
		// will converge to some optima regarding the cut/shape. Investigate that

		//WARNING2: this is also needed to ensure that the required number of sampling
		//	rounds will be performed so at the end, all nodes are accounted for

		//if(imbalance<settings.epsilon)
		//	break;

		//aux<IndexType,ValueType>::print2DGrid(graph, result);

	} while (iter < samplingRounds or (iter < maxIterations && (delta > threshold || !balanced)) ); // or (imbalance>settings.epsilon) );


	std::chrono::duration<ValueType,std::ratio<1>> KMeansTime = std::chrono::high_resolution_clock::now() - KMeansStart;
	ValueType time = comm->max( KMeansTime.count() );
	
	PRINT0("total KMeans time: " << time << " , number of iterations: " << iter );

	return result;
}//computePartition

//moved (7.11.18) from KMeans.h


//wrapper 1 - called initially with no centers parameter
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(
	const std::vector<DenseVector<ValueType>> &coordinates,
	const DenseVector<ValueType> &nodeWeights,
	const std::vector<IndexType> &blockSizes,
	const Settings settings,
	struct Metrics &metrics) {

    std::vector<ValueType> minCoords(settings.dimensions);
    std::vector<ValueType> maxCoords(settings.dimensions);
    std::tie(minCoords, maxCoords) = getLocalMinMaxCoords( coordinates );

	std::vector<point> centers = findInitialCentersSFC<IndexType,ValueType>(coordinates, minCoords, maxCoords, settings);
	SCAI_ASSERT_EQ_ERROR( centers.size(), settings.numBlocks, "Number of centers is not correct" );
	SCAI_ASSERT_EQ_ERROR( centers[0].size(), settings.dimensions, "Dimension of centers is not correct" );

	//just one group with all the centers; needed in the hierarchical version
	std::vector<std::vector<point>> groupOfCenters = { centers };
	
	//every point belongs to one block in the beginning
	scai::lama::DenseVector<IndexType> partition( coordinates[0].getDistributionPtr(), 0);

	//must convert the block sizes to precentages,
	std::vector<ValueType> blockSizesPerCent( blockSizes.size() );

	//WARNING: obviously, for uniform block sizes, all sizes are tthe same, so size/max gives 1 for all,
	//		which is not correct. 
	//use maxWeight instead of totalWeight to resemble the modelling from TEEC
	//const IndexType maxWeight = *std::max_element( blockSizes.begin(), blockSizes.end() );

	//15/02: change is back to sum of weights instead of max weight
	const IndexType sumWeight = std::accumulate( blockSizes.begin(), blockSizes.end(), 0 );
	for( IndexType i=0; i<blockSizes.size(); i++ ){
		blockSizesPerCent[i] = ValueType(blockSizes[i])/sumWeight;
	}

	return computePartition(coordinates, nodeWeights, blockSizesPerCent, partition, groupOfCenters, settings, metrics);
}


//---------------------------------------
//wrapper 2 - with CommTree
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeHierarchicalPartition(
	CSRSparseMatrix<ValueType> &graph,	//TODO: graph is not needed
	std::vector<DenseVector<ValueType>> &coordinates,
	DenseVector<ValueType> &nodeWeights,
	const CommTree<IndexType,ValueType> &commTree,
	Settings settings,
	struct Metrics& metrics){

	//check although numBlocks is not needed or used
	SCAI_ASSERT_EQ_ERROR(settings.numBlocks, commTree.numLeaves, "The number of leaves and number of blocks must agree");

	//get global communicator
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	if(settings.erodeInfluence){
		if(comm->getRank()==0){
			std::cout << "WARNING: erode influence is not supported for the hierarchical version. Will set it to false and continue." << std::endl;
		}
		settings.erodeInfluence = false;
	}

	//redistribute points based on their hilbert curve index
	//warning: this functions redistributes the coordinates and
	//the node weights. 
	//TODO: is this supposed to be here? it is also in the
	// ParcoRepart::partitionGraph
	HilbertCurve<IndexType,ValueType>::hilbertRedistribution(
		coordinates, nodeWeights, settings, metrics);

	//added check to verify that the points are indeed distributed 
	//based on the hilbert curve. Otherwise, the prefix sum needed to 
	//calculate the centers, does not have the desired meaning.
	bool hasHilbertDist = HilbertCurve<IndexType, ValueType>::confirmHilbertDistribution( coordinates, nodeWeights, settings);
	SCAI_ASSERT_EQ_ERROR( hasHilbertDist, true, "Input must be distributed according to a hilbert curve distribution");
	graph.redistribute( coordinates[0].getDistributionPtr(), graph.getColDistributionPtr() );

	std::vector<ValueType> minCoords(settings.dimensions);
    std::vector<ValueType> maxCoords(settings.dimensions);
    std::tie(minCoords, maxCoords) = getLocalMinMaxCoords( coordinates );

    //used later for debugging and calculating imbalance
    ValueType totalWeightSum;
    {
    	scai::hmemo::ReadAccess<ValueType> rW( nodeWeights.getLocalValues() );
    	ValueType localW = 0;
    	for(int i=0; i<nodeWeights.getLocalValues().size(); i++ ){
    		localW += rW[i];
    	}
    	totalWeightSum = comm->sum(localW);
    }

    //typedef of commNode from CommTree.h, see also KMeans.h
	cNode root = commTree.getRoot();
	if( settings.debugMode ){
		PRINT("Starting hierarchical KMeans.\nRoot node: ");
		root.print();
	}

	//every point belongs to one block in the beginning
	scai::lama::DenseVector<IndexType> partition( coordinates[0].getDistributionPtr(), 0);

	//skip root. If we start from the root, we will know the number
	//of blocks but not the memory and speed per block
	for(unsigned int h=1; h<commTree.hierarchyLevels; h++ ){

		/*
		There are already as many blocks as the number of leaves
		of the previous hierarchy level. The new number of blocks per
		old block is prevLeaf.numChildren. Example, if previous level
		had 3 leaves with 4, 6 and 10 children respectively, then
		the number of new blocks that we will partition in this step is 
		4 for block 0, 6 for block 1 and 10 for block 2, in total 20.
		*/
	
		//in how many blocks each known block (part) will be partitioned
		//numNewBlocksPerOldBlock[i]=k means that, current partition i should be 
		//partitioned into k new blocks

		std::vector<cNode> thisLevel = commTree.getHierLevel(h);	

		PRINT0("-- Hierarchy level " << h << " with " << thisLevel.size() << " nodes");
		if( settings.debugMode ){
			PRINT0("******* in debug mode");
			for( cNode c: thisLevel){ //print all nodes of this level
				c.print();
			}
		}
		
		//
		//1- find initial centers for this hierarchy level
		//
		//Only the new level is passed and the previous level is 
		//reconstructed internally
		
		std::vector<std::vector<point>> groupOfCenters = findInitialCentersSFC( coordinates, minCoords, maxCoords, partition, thisLevel, settings );

		SCAI_ASSERT_EQ_ERROR( groupOfCenters.size(), commTree.getHierLevel(h-1).size(), "Wrong number of blocks calculated" );
		if( settings.debugMode ){
			PRINT0("******* in debug mode");
			IndexType sumNumCenters = 0;
			for(int g=0; g<groupOfCenters.size(); g++){
				sumNumCenters += groupOfCenters[g].size();
			}
			SCAI_ASSERT_EQ_ERROR( sumNumCenters, thisLevel.size(), "Mismatch in number of new centers and hierarchy nodes")
		}

		//number of old, known blocks == previous level size
		IndexType numOldBlocks = groupOfCenters.size();

		//number of new blocks each old blocks must be partitioned to
		std::vector<unsigned int> numNewBlocks = CommTree<IndexType, ValueType>::getGrouping( thisLevel );
		SCAI_ASSERT_EQ_ERROR( numOldBlocks, numNewBlocks.size(), "Hierarchy level size mismatch" );
		const IndexType totalNumNewBlocks = std::accumulate( numNewBlocks.begin(), numNewBlocks.end(), 0 );

		if( settings.debugMode ){
			const IndexType maxPart = partition.max(); //global operation
			SCAI_ASSERT_EQ_ERROR( numOldBlocks-1, maxPart, "The provided partition must have equal number of blocks as the length of the vector with the new number of blocks per part");
		}

		//
		//2- main k-means loop
		//

		//TODO: probably blockSizes is not needed

		/*for speed weight: the optimum block weight for a PE i is 
		w = this.speed * totalWeightSum/(sum of all speed factors)
		Because we use sampling, totalWeightSum is the sum of the weight
		of the sampled nodes so it is different in every iteration.
		*/
		ValueType speedSum = 0;
		for( cNode c: thisLevel){
			speedSum += c.relatSpeed;
		}

		//TODO: do the equivalent for the memory constraint
		//later, optBlockWeight[i] = blockSpeedPercent[i]*samplePointsWeight

		//TODO: replaced speedPercent with optBlockWeight. percentages make more sense?	
		std::vector<ValueType> optBlockWeight( totalNumNewBlocks );
		std::vector<ValueType> blockSpeedPercent( totalNumNewBlocks );
		for( int i=0; i<totalNumNewBlocks; i++){
			blockSpeedPercent[i] = thisLevel[i].relatSpeed/speedSum;
			optBlockWeight[i] = blockSpeedPercent[i]*totalWeightSum;
		}

		//TODO: inside computePartition, settings.numBlocks is not
		//used. We infer the number of new blocks from the groupOfCenters
		//maybe, set also numBlocks for clarity??
		partition = computePartition( graph, coordinates, nodeWeights, blockSpeedPercent, partition, groupOfCenters, settings, metrics );


		//this check is done before. TODO: remove?
		if( settings.debugMode ){
			const IndexType maxPart = partition.max(); //global operation
			SCAI_ASSERT_EQ_ERROR( totalNumNewBlocks-1, maxPart, "The provided old assignment must have equal number of blocks as the length of the vector with the new number of blocks per part");
			if( settings.storeInfo){
				ITI::FileIO<IndexType,ValueType>::writeDenseVectorCentral( partition, "//home/harry/geographer-dev/tmp/hkmLvl"+h );
				if( comm->getRank()==0)
					std::cout << "Partition for hierarchy level " << h << 
				" stored in /home/harry/geographer-dev/tmp/hkmLvl"<<h <<std::endl;
				//std::cout<< "Press key to continue" << std::endl; int tmpInt; std::cin >> tmpInt;
			}
		}

//TODO: this is an attempt to do local refinement after every step. But local refinement demands k=p,
//	so this cannot be done inbetween steps.
/*
if( true or not settings.noRefinement ){
	bool useRedistributor = true;
	scai::dmemo::DistributionPtr distFromPartition = aux<IndexType, ValueType>::redistributeFromPartition( partition, graph, coordinates, nodeWeights, settings, useRedistributor);
	scai::dmemo::HaloExchangePlan halo = GraphUtils<IndexType, ValueType>::buildNeighborHalo( graph );
	ITI::MultiLevel<IndexType, ValueType>::multiLevelStep( graph, partition, nodeWeights, coordinates, halo, settings, metrics);
}
*/
		//TODO?: remove?
		ValueType imbalance =  ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, totalNumNewBlocks, nodeWeights, optBlockWeight );
		//IndexType cut = ITI::GraphUtils<IndexType, ValueType>::computeCut(graph, partition, true);

		PRINT0("\nFinished hierarchy level " << h <<", partitioned into " << totalNumNewBlocks << " blocks and imbalance is " << imbalance <<std::endl );
	}

	return partition;
}

//---------------------------------------


/**
 * @brief Get local minimum and maximum coordinates
 * TODO: This isn't used any more! Remove?
 */
template<typename ValueType>
std::pair<std::vector<ValueType>, std::vector<ValueType> > getLocalMinMaxCoords(const std::vector<DenseVector<ValueType>> &coordinates) {
	const int dim = coordinates.size();
	std::vector<ValueType> minCoords(dim);
	std::vector<ValueType> maxCoords(dim);
	for (int d = 0; d < dim; d++) {
		minCoords[d] = coordinates[d].min();
        maxCoords[d] = coordinates[d].max();
		SCAI_ASSERT_NE_ERROR( minCoords[d], maxCoords[d], "min=max for dimension "<< d << ", this will cause problems to the hilbert index. local= " << coordinates[0].getLocalValues().size() );
	}
	return {minCoords, maxCoords};
}


/*
template std::pair<std::vector<ValueType>, std::vector<ValueType> > getLocalMinMaxCoords(const std::vector<DenseVector<ValueType>> &coordinates);

template std::vector<std::vector<ValueType> > findInitialCentersSFC<IndexType, ValueType>( const std::vector<DenseVector<ValueType> >& coordinates, const std::vector<ValueType> &minCoords,    const std::vector<ValueType> &maxCoords, Settings settings);

template std::vector<std::vector<ValueType> > findLocalCenters<IndexType,ValueType>(const std::vector<DenseVector<ValueType> >& coordinates, const DenseVector<ValueType> &nodeWeights);

template std::vector<std::vector<ValueType> > findInitialCentersFromSFCOnly<IndexType,ValueType>(const std::vector<ValueType> &maxCoords, Settings settings);

template std::vector<std::vector<ValueType> > findCenters(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const IndexType k, std::vector<IndexType>::iterator firstIndex, std::vector<IndexType>::iterator lastIndex, const DenseVector<ValueType> &nodeWeights);

template DenseVector<IndexType> assignBlocks(
		const std::vector<std::vector<ValueType>> &coordinates,
		const std::vector<std::vector<ValueType> > &centers,
        std::vector<IndexType>::iterator firstIndex, std::vector<IndexType>::iterator lastIndex,
        const DenseVector<ValueType> &nodeWeights, const DenseVector<IndexType> &previousAssignment, const std::vector<IndexType> &blockSizes, const SpatialCell &boundingBox,
        std::vector<ValueType> &upperBoundOwnCenter, std::vector<ValueType> &lowerBoundNextCenter, std::vector<ValueType> &influence, ValueType &imbalance, std::vector<ValueType> &timePerPE, Settings settings, Metrics &metrics);


template DenseVector<IndexType> computeRepartition(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<ValueType> &nodeWeights, const Settings settings, struct Metrics& metrics);
*/

}; // namespace KMeans

template std::vector<std::vector<ValueType> > KMeans::findInitialCentersFromSFCOnly<IndexType,ValueType>(const std::vector<ValueType> &maxCoords, Settings settings);

//template std::vector<KMeans::point> KMeans::vectorTranspose( const std::vector<std::vector<ValueType>>& points);

//instantiations needed otherwise there is a undefined reference 
template DenseVector<IndexType> KMeans::computePartition(
	const std::vector<DenseVector<ValueType>> &coordinates,
	const scai::lama::DenseVector<ValueType> &nodeWeights,
	const std::vector<IndexType> &blockSizes,
	const Settings settings,
	struct Metrics& metrics);

/*
using point = std::vector<ValueType>;
//TODO: graph is not needed, this is only for debugging
 template DenseVector<IndexType> KMeans::computePartition(
 	const CSRSparseMatrix<ValueType> &graph, \
 	const std::vector<DenseVector<ValueType>> &coordinates, \
 	const DenseVector<ValueType> &nodeWeights, \
 	const std::vector<ValueType> &blockSizes, \
 	const DenseVector<IndexType>& prevPartition,\
 	std::vector<std::vector<point>> centers, \
 	const Settings settings, \
 	struct Metrics &metrics);
*/

/*
//instantiations needed or there is a undefined reference otherwise
template DenseVector<IndexType> KMeans::computePartition(
	const CSRSparseMatrix<ValueType> &graph,
	const std::vector<DenseVector<ValueType>> &coordinates,
	const scai::lama::DenseVector<ValueType> &nodeWeights,
	const std::vector<IndexType> &blockSizes,
	const Settings settings,
	struct Metrics& metrics);
*/
template DenseVector<IndexType> KMeans::computeRepartition(
	const std::vector<DenseVector<ValueType>>& coordinates,
	const DenseVector<ValueType>& nodeWeights,
	const std::vector<IndexType>& blockSizes,
	const DenseVector<IndexType>& previous,
	const Settings settings);


template DenseVector<IndexType> KMeans::computeHierarchicalPartition(
	CSRSparseMatrix<ValueType> &graph,
	std::vector<DenseVector<ValueType>> &coordinates,
	DenseVector<ValueType> &nodeWeights,
	const CommTree<IndexType,ValueType> &commTree,
	const Settings settings,
	struct Metrics& metrics);


} /* namespace ITI */
