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

namespace ITI {
namespace KMeans {


template<typename IndexType, typename ValueType>
std::vector<std::vector<point>> findInitialCentersSFC(
		const std::vector<DenseVector<ValueType> >& coordinates, 
		const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords,
		const scai::lama::DenseVector<IndexType> &partition,
		Settings settings) {

	SCAI_REGION( "KMeans.findInitialCentersSFC" );
	const IndexType localN = coordinates[0].getLocalValues().size();
	const IndexType globalN = coordinates[0].size();
	const IndexType dimensions = settings.dimensions;
	const IndexType k = settings.numBlocks;
	
	//hrr: maybe not needed
	const IndexType maxPart = partition.max();

	//hrr: convention: graph is already partitioned. We call the already 
	// known blocks as "parts". 
	// So, numXperPart means how many Xs the already knows blocks have.
	// numXperBlocks means how many Xs the new, to-be-found blocks must have
/*
	//hrr
	//in how many blocks each known block (part) will be partitioned
	//numBlocksPerPart[i]=k means than current partition i should be partitioned
	//into k blocks
	std:vector<unsigned int> numBlocksPerPart = settings.XXX;
	SCAI_ASSERT_EQ_ERROR( numBlocksPerPart.size(), maxPart, "The partition given must have equal number of blocks with the vector of block per part");
	
	std::vector<unsigned int> coresPerBlock = CCC;
	std::vector<unsigned int> memPerBlock = MMM;
	std::vector<ValueType> speedPerBlock = SSS;

	const IndexType totalNumOfBlocks = std::accumulate(numBlocksPerPart.begin(), numBlocksPerPart.end(), 0);

	//hrr: a property per new block; sizes must match
	SCAI_ASSERT_EQ_ERROR( coresPerBlock.size(), totalNumOfBlocks, "Must be provided a number of cores per new block. The size of the vector mustbe equal the total number on new blocks");
	SCAI_ASSERT_EQ_ERROR( memPerBlock.size(), totalNumOfBlocks, "Must be provided a memory capacity per new block. The size of the vector mustbe equal the total number on new blocks");
	SCAI_ASSERT_EQ_ERROR( speedPerBlock.size(), totalNumOfBlocks, "Must be provided a cpu speed per new block. The size of the vector mustbe equal the total number on new blocks");
*/
	//convert coordinates, switch inner and outer order
	std::vector<std::vector<ValueType> > convertedCoords(localN);
	for (IndexType i = 0; i < localN; i++) {
		convertedCoords[i].resize(dimensions);
	}

	for (IndexType d = 0; d < dimensions; d++) {
		scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
		assert(rAccess.size() == localN);
		for (IndexType i = 0; i < localN; i++) {
			convertedCoords[i][d] = rAccess[i];
		}
	}

	//get local hilbert indices
	std::vector<ValueType> sfcIndices = HilbertCurve<IndexType, ValueType>::getHilbertIndexVector( coordinates, settings.sfcResolution, settings.dimensions);
	SCAI_ASSERT_EQ_ERROR( sfcIndices.size(), localN, "wrong local number of indices (?) ");

	//prepare indices for sorting
	std::vector<IndexType> localIndices(localN);
	const typename std::vector<IndexType>::iterator firstIndex = localIndices.begin();
	typename std::vector<IndexType>::iterator lastIndex = localIndices.end();;
	std::iota(firstIndex, lastIndex, 0);

	//sort local indices according to SFC
	std::sort(localIndices.begin(), localIndices.end(), [&sfcIndices](IndexType a, IndexType b){return sfcIndices[a] < sfcIndices[b];});

	//compute wanted indices for initial centers
	std::vector<IndexType> wantedIndices(k);

	for (IndexType i = 0; i < k; i++) {
		wantedIndices[i] = i * (globalN / k) + (globalN / k)/2;
	}

	//setup general block distribution to model the space-filling curve
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	scai::dmemo::DistributionPtr blockDist(new scai::dmemo::GenBlockDistribution(globalN, localN, comm));

	//set local values in vector, leave non-local values with zero
	std::vector<std::vector<point>> result(dimensions);
	for (IndexType d = 0; d < dimensions; d++) {
		result[d].resize(k);
	}

	//check for all centers: if the index of a center is in this PE,
	//add it to the results vector.
	for (IndexType j = 0; j < k; j++) {
		IndexType localIndex = blockDist->global2local(wantedIndices[j]);
		if (localIndex != scai::invalidIndex) {
			assert(localIndex < localN);
			IndexType permutedIndex = localIndices[localIndex];
			assert(permutedIndex < localN);
			assert(permutedIndex >= 0);
			for (IndexType d = 0; d < dimensions; d++) {
//TODO: this is wrong, added [0] just to compile
// fix it appropiatelly				
				result[0][d][j] = convertedCoords[permutedIndex][d];
//
//
			}
		}
	}

	//global sum operation
	for (IndexType d = 0; d < dimensions; d++) {
		comm->sumImpl(result[d].data(), result[d].data(), k, scai::common::TypeTraits<ValueType>::stype);
	}

	return result;
}


//overloaded function for non-hierarchical version. Set partition to 0 for all points
//and return only the first (there is only one) group of centers
template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>>  findInitialCentersSFC(
		const std::vector<DenseVector<ValueType>>& coordinates,
		const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords,
		Settings settings){

		//TODO: probably must also change the seetings.numBlocks
	
		scai::lama::DenseVector<IndexType> partition( coordinates[0].getDistributionPtr(), 0);

		return findInitialCentersSFC( coordinates, minCoords, maxCoords, partition, settings)[0];
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
//PRINT( centerHilbInd );		
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
	//const IndexType n = coordinates[0].size();
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
std::vector<std::vector<ValueType> > findCenters(
		const std::vector<DenseVector<ValueType> >& coordinates,
		const DenseVector<IndexType>& partition,
		const IndexType k,
		const Iterator firstIndex,
		const Iterator lastIndex,
		const DenseVector<ValueType>& nodeWeights) {
	SCAI_REGION( "KMeans.findCenters" );

	const IndexType dim = coordinates.size();
	//const IndexType n = partition.size();
	//const IndexType localN = partition.getLocalValues().size();
	const scai::dmemo::DistributionPtr resultDist(new scai::dmemo::NoDistribution(k));
	const scai::dmemo::CommunicatorPtr comm = partition.getDistribution().getCommunicatorPtr();

	//TODO: check that distributions align

	std::vector<std::vector<ValueType> > result(dim);
	std::vector<IndexType> weightSum(k, 0);

	scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
	scai::hmemo::ReadAccess<IndexType> rPartition(partition.getLocalValues());

	//compute weight sums
	for (Iterator it = firstIndex; it != lastIndex; it++) {
		const IndexType i = *it;
		const IndexType part = rPartition[i];
		const IndexType weight = rWeights[i];
		weightSum[part] += weight;
		//weightSum[rPartition[*it]] += rWeights[*it];
	}

	//find local centers
	for (IndexType d = 0; d < dim; d++) {
		result[d].resize(k);
		scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());

		for (Iterator it = firstIndex; it != lastIndex; it++) {
			const IndexType i = *it;
			const IndexType part = rPartition[i];
			//const IndexType weight = rWeights[i];
			result[d][part] += rCoords[i]*rWeights[i] / weightSum[part];//this is more expensive than summing first and dividing later, but avoids overflows
		}
	}

	//communicate local centers and weight sums
	std::vector<IndexType> totalWeight(k, 0);
	comm->sumImpl(totalWeight.data(), weightSum.data(), k, scai::common::TypeTraits<IndexType>::stype);

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

template<typename IndexType, typename ValueType, typename Iterator>
DenseVector<IndexType> assignBlocks(
		const std::vector<std::vector<ValueType>>& coordinates,
		const std::vector<std::vector<ValueType>>& centers,
		const Iterator firstIndex,
		const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights,
		const DenseVector<IndexType> &previousAssignment,
		const std::vector<IndexType> &targetBlockSizes,
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
//std::chrono::time_point<std::chrono::high_resolution_clock> assignStart = std::chrono::high_resolution_clock::now();
//

	const IndexType dim = coordinates.size();
	const scai::dmemo::DistributionPtr dist = nodeWeights.getDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
//	const IndexType localN = nodeWeights.getLocalValues().size();
	const IndexType k = targetBlockSizes.size();

	assert(influence.size() == k);

	//compute assignment and balance
	DenseVector<IndexType> assignment = previousAssignment;

	//pre-filter possible closest blocks
	std::vector<ValueType> effectiveMinDistance(k);
	std::vector<ValueType> minDistance(k);
	{
		SCAI_REGION( "KMeans.assignBlocks.filterCenters" );
		for (IndexType j = 0; j < k; j++) {
			std::vector<ValueType> center(dim);
			//TODO: this conversion into points is annoying. Maybe change coordinate format and use n in the outer dimension and d in the inner?
			//Can even use points as data structure. Update: Tried it, gave no performance benefit.
			for (IndexType d = 0; d < dim; d++) {
				center[d] = centers[d][j];
			}
			minDistance[j] = boundingBox.distances(center).first;
			assert(std::isfinite(minDistance[j]));
			effectiveMinDistance[j] = minDistance[j]*minDistance[j]*influence[j];
			assert(std::isfinite(effectiveMinDistance[j]));
		}
	}

	std::vector<IndexType> clusterIndices(k);
	std::iota(clusterIndices.begin(), clusterIndices.end(), 0);
	std::sort(clusterIndices.begin(), clusterIndices.end(),
			[&effectiveMinDistance](IndexType a, IndexType b){return effectiveMinDistance[a] < effectiveMinDistance[b] || (effectiveMinDistance[a] == effectiveMinDistance[b] && a < b);});
	std::sort(effectiveMinDistance.begin(), effectiveMinDistance.end());

	for (IndexType i = 0; i < k; i++) {
		IndexType c = clusterIndices[i];
		ValueType effectiveDist = minDistance[c]*minDistance[c]*influence[c];
		SCAI_ASSERT_LT_ERROR( std::abs(effectiveMinDistance[i] - effectiveDist), 1e-5, "effectiveMinDistance[" << i << "] = " << effectiveMinDistance[i] << " != " << effectiveDist << " = effectiveDist");
	}

	ValueType localSampleWeightSum = 0;
	{
		scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());

		for (Iterator it = firstIndex; it != lastIndex; it++) {
			localSampleWeightSum += rWeights[*it];
		}
	}
	const ValueType totalWeightSum = comm->sum(localSampleWeightSum);
	ValueType optSize = std::ceil(totalWeightSum / k );

	//ValueType imbalance;
	IndexType iter = 0;
	IndexType skippedLoops = 0;
	ValueType time = 0;	// for timing/profiling
	std::vector<bool> influenceGrew(k);
	std::vector<ValueType> influenceChangeUpperBound(k,1+settings.influenceChangeCap);
	std::vector<ValueType> influenceChangeLowerBound(k,1-settings.influenceChangeCap);

	//iterate if necessary to achieve balance
	do
	{
		std::chrono::time_point<std::chrono::high_resolution_clock> balanceStart = std::chrono::high_resolution_clock::now();
		SCAI_REGION( "KMeans.assignBlocks.balanceLoop" );
		std::vector<ValueType> blockWeights(k,0.0);
		IndexType totalComps = 0;		
		skippedLoops = 0;
		IndexType balancedBlocks = 0;
		scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
		scai::hmemo::WriteAccess<IndexType> wAssignment(assignment.getLocalValues());
		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.assign" );
			IndexType forLoopCnt = 0;
			for (Iterator it = firstIndex; it != lastIndex; it++) {
				++forLoopCnt;
				const IndexType i = *it;
				const IndexType oldCluster = wAssignment[i];
				if (lowerBoundNextCenter[i] > upperBoundOwnCenter[i]) {
					//std::cout << upperBoundOwnCenter[i] << " " << lowerBoundNextCenter[i] << " " << distThreshold[oldCluster] << std::endl;
					//cluster assignment cannot have changed.
					//wAssignment[i] = wAssignment[i];
					skippedLoops++;
				} else {
					ValueType sqDistToOwn = 0;
					for (IndexType d = 0; d < dim; d++) {
						sqDistToOwn += std::pow(centers[d][oldCluster] - coordinates[d][i], 2);
					}
					ValueType newEffectiveDistance = sqDistToOwn*influence[oldCluster];
					assert(upperBoundOwnCenter[i] >= newEffectiveDistance);
					upperBoundOwnCenter[i] = newEffectiveDistance;
					if (lowerBoundNextCenter[i] > upperBoundOwnCenter[i]) {
						//cluster assignment cannot have changed.
						//wAssignment[i] = wAssignment[i];
						skippedLoops++;
					} else {
						int bestBlock = 0;
						ValueType bestValue = std::numeric_limits<ValueType>::max();
						IndexType secondBest = 0;
						ValueType secondBestValue = std::numeric_limits<ValueType>::max();

						IndexType c = 0;
						while(c < k && secondBestValue > effectiveMinDistance[c]) {
							totalComps++;
							IndexType j = clusterIndices[c];//maybe it would be useful to sort the whole centers array, aligning memory accesses.
							ValueType sqDist = 0;
							//TODO: restructure arrays to align memory accesses better in inner loop
							for (IndexType d = 0; d < dim; d++) {
								ValueType dist = centers[d][j] - coordinates[d][i];
								sqDist += dist*dist;
							}

							const ValueType effectiveDistance = sqDist*influence[j];
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
						}
						assert(bestBlock != secondBest);
						assert(secondBestValue >= bestValue);
						if (bestBlock != oldCluster) {
							if (bestValue < lowerBoundNextCenter[i]) {
								std::cout << "bestValue: " << bestValue << " lowerBoundNextCenter[" << i << "]: "<< lowerBoundNextCenter[i];
								std::cout << " difference " << std::abs(bestValue - lowerBoundNextCenter[i]) << std::endl;
							}
							assert(bestValue >= lowerBoundNextCenter[i]);
						}

						upperBoundOwnCenter[i] = bestValue;
						lowerBoundNextCenter[i] = secondBestValue;
						wAssignment[i] = bestBlock;					
					}
				}
				blockWeights[wAssignment[i]] += rWeights[i];
			}//for
			
			if (settings.verbose) {
				std::chrono::duration<ValueType,std::ratio<1>> balanceTime = std::chrono::high_resolution_clock::now() - balanceStart;			
				ValueType time = balanceTime.count() ;
				//std::cout<< comm->getRank()<< ": time " << time << std::endl;
				//PRINT(comm->getRank() << ": in assignBlocks, balanceIter time: " << time << ", for loops: " << forLoopCnt );
				ValueType maxTime = comm->max( time );
				//PRINT0( "max time: " << maxTime << ", for loops: " << forLoopCnt );

				SCAI_ASSERT_LT_ERROR( comm->getRank(), timePerPE.size(), "vector size mismatch" );
				timePerPE[comm->getRank()] += time;
			}

			//aux<IndexType,ValueType>::timeMeasurement(balanceStart);

			comm->synchronize();
		}

		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.blockWeightSum" );
			comm->sumImpl(blockWeights.data(), blockWeights.data(), k, scai::common::TypeTraits<ValueType>::stype);
		}
		
		ValueType maxBlockWeight = *std::max_element(blockWeights.begin(), blockWeights.end());
	
		imbalance = (ValueType(maxBlockWeight - optSize)/ optSize);//TODO: adapt for block sizes

		std::vector<ValueType> oldInfluence = influence;

		double minRatio = std::numeric_limits<double>::max();
		double maxRatio = -std::numeric_limits<double>::min();

		for (IndexType j = 0; j < k; j++) {
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.influence" );
			double ratio = ValueType(blockWeights[j]) / targetBlockSizes[j];

			if (std::abs(ratio - 1) < settings.epsilon) {
				balancedBlocks++;
				if (settings.freezeBalancedInfluence) {
					if (1 < minRatio) minRatio = 1;
					if (1 > maxRatio) maxRatio = 1;
					continue;
				}
			}

			influence[j] = std::max(influence[j]*influenceChangeLowerBound[j], std::min(influence[j] * std::pow(ratio, settings.influenceExponent), influence[j]*influenceChangeUpperBound[j]));
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
				//if (comm->getRank() == 0) std::cout << "Block " << j << ": reduced bounds to " << influenceChangeUpperBound[j] << " and " << influenceChangeLowerBound[j] << std::endl;
				assert(influenceChangeUpperBound[j] > 1);
				assert(influenceChangeLowerBound[j] < 1);
			}
			influenceGrew[j] = bool(ratio > 1);
		}

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
			for (IndexType j = 0; j < k; j++) {
				effectiveMinDistance[j] = minDistance[j]*minDistance[j]*influence[j];
			}

			std::sort(clusterIndices.begin(), clusterIndices.end(),
						[&effectiveMinDistance](IndexType a, IndexType b){return effectiveMinDistance[a] < effectiveMinDistance[b] || (effectiveMinDistance[a] == effectiveMinDistance[b] && a < b);});
			std::sort(effectiveMinDistance.begin(), effectiveMinDistance.end());

		}

		iter++;
		if (settings.verbose) {
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
					<< averageComps << ", balanced blocks: " << 100*ValueType(balancedBlocks) / k << "%, influence spread: " << influenceSpread
					<< ", imbalance : " << imbalance << ", time elapsed: " << time << std::endl;
			std::cout.precision(oldprecision);
		}

	} while (imbalance > settings.epsilon - 1e-12 && iter < settings.balanceIterations);
	//std::cout << "Process " << comm->getRank() << " skipped " << ValueType(skippedLoops*100) / (iter*localN) << "% of inner loops." << std::endl;
	//aux<IndexType,ValueType>::timeMeasurement(assignStart);
	
	//for kmeans profiling
	metrics.numBalanceIter.push_back(iter);

	return assignment;
}

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

	Metrics metrics;
	return computePartition(coordinates, nodeWeights, blockSizes, initialCenters, settings, metrics);
}


template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition( \
	const std::vector<DenseVector<ValueType>> &coordinates, \
	const DenseVector<ValueType> &nodeWeights, \
	const std::vector<IndexType> &blockSizes, \
	std::vector<std::vector<ValueType> > centers, \
	scai::dmemo::CommunicatorPtr comm,
	const Settings settings, \
	struct Metrics &metrics ) {

	SCAI_REGION( "KMeans.computePartition" );
	std::chrono::time_point<std::chrono::high_resolution_clock> KMeansStart = std::chrono::high_resolution_clock::now();

	const IndexType k = settings.numBlocks;
	std::vector<ValueType> influence(k,1);
	const IndexType dim = coordinates.size();
	assert(dim > 0);
	const IndexType localN = nodeWeights.getLocalValues().size();
	const IndexType globalN = nodeWeights.size();
	assert(nodeWeights.getLocalValues().size() == coordinates[0].getLocalValues().size());
	//scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	const IndexType p = comm->getSize();
	const ValueType blocksPerProcess = ValueType(k)/p;

	std::vector<ValueType> minCoords(dim);
	std::vector<ValueType> maxCoords(dim);
	std::vector<std::vector<ValueType> > convertedCoords(dim);

	// copy and sort coordinates according to their hilbert index
	{
		// a vector of the indices
		std::vector<IndexType> permIndices(localN);
		std::iota(permIndices.begin(), permIndices.end(), 0);

		// hilbert curve indices of all local points
		const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(globalN), double(21));
		std::vector<ValueType> localHilbertInd = HilbertCurve<IndexType,ValueType>::getHilbertIndexVector(coordinates, recursionDepth, settings.dimensions);

		SCAI_ASSERT_EQ_ERROR(localN, localHilbertInd.size() , "vector size mismatch");

		// sort the point/vertex indices based on their hilbert index
		std::sort(permIndices.begin(), permIndices.end(), [&](IndexType i, IndexType j){return localHilbertInd[i] < localHilbertInd[j];});

		for (IndexType d = 0; d < dim; d++) {
			scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
			assert(rAccess.size() == localN);				
			
			// copy coordinates sorted by their hilbert index
			IndexType checkSum = 0;
			ValueType sumCoord = 0.0;
			convertedCoords[d].resize(localN);
			for(IndexType i=0; i<localN; i++){
				IndexType ind = permIndices[i];
				ValueType coord = rAccess[ ind ];
				convertedCoords[d][i] = coord;

				//meant for debugging reasons, remove or add debug macros
				checkSum += ind;
				sumCoord += coord;
			}

			SCAI_ASSERT_EQ_ERROR(checkSum, (localN*(localN-1)/2), "Checksum error");
			ValueType sumCoord2 = std::accumulate( convertedCoords[d].begin(), convertedCoords[d].end(), 0.0);
			SCAI_ASSERT_GE_ERROR( sumCoord, 0.999*sumCoord2, "Error in sorting local coordinates");

			minCoords[d] = *std::min_element(convertedCoords[d].begin(), convertedCoords[d].end());
			maxCoords[d] = *std::max_element(convertedCoords[d].begin(), convertedCoords[d].end());

			//or, do not take the hilbert index, do not sort and just copy coordinates
			//TODO: test improvement, in some small inputs, the sorted version did less 
			// iterations
			//convertedCoords[d] = std::vector<ValueType>(rAccess.get(), rAccess.get()+localN);

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

	QuadNodeCartesianEuclid boundingBox(minCoords, maxCoords);
    if (settings.verbose) {
		std::cout << "(" << comm->getRank() << ", "<< localN << ")" << std::endl;
		comm->synchronize();
		std::cout << "(" << comm->getRank() << ", "<< localVolume / (volume / p) << ")" << std::endl;
    }

	diagonalLength = std::sqrt(diagonalLength);
	const ValueType expectedBlockDiameter = pow(volume / k, 1.0/dim);

	std::vector<ValueType> upperBoundOwnCenter(localN, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> lowerBoundNextCenter(localN, 0);

	//
	//prepare sampling
	//
	std::vector<IndexType> localIndices(localN);
	std::iota(localIndices.begin(), localIndices.end(), 0);

	IndexType minNodes = settings.minSamplingNodes*blocksPerProcess;
	if( settings.minSamplingNodes==-1 ){
		minNodes = localN;
	}

	assert(minNodes > 0);
	IndexType samplingRounds = 0;
	std::vector<IndexType> samples;
	std::vector<IndexType> adjustedBlockSizes(blockSizes);
	const bool randomInitialization = comm->all(localN > minNodes);

	//perform sampling
	{
		if (randomInitialization) {
			ITI::GraphUtils<IndexType, ValueType>::FisherYatesShuffle(localIndices.begin(), localIndices.end(), localN);
			//TODO: the cantor shuffle is more stable, random suffling can yield better
			// results occasionally but has higher fluxations/variance
			//localIndices = GraphUtils<IndexType,ValueType>::indexReorderCantor( localN );

			SCAI_ASSERT_EQ_ERROR(*std::max_element(localIndices.begin(), localIndices.end()), localN -1, "Error in index reordering");
			SCAI_ASSERT_EQ_ERROR(*std::min_element(localIndices.begin(), localIndices.end()), 0, "Error in index reordering");

			samplingRounds = std::ceil(std::log2( globalN / ValueType(settings.minSamplingNodes*k)))+1;
			samples.resize(samplingRounds);
			samples[0] = std::min(minNodes, localN);
		}

		if(settings.verbose){
			PRINT(*comm << ": localN= "<< localN << ", samplingRounds= " << samplingRounds << ", lastIndex: " << *localIndices.end() );
		}
		if (samplingRounds > 0 && settings.verbose) {
			if (comm->getRank() == 0) std::cout << "Starting with " << samplingRounds << " sampling rounds." << std::endl;
		}
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
			for (IndexType j = 0; j < k; j++) {
				adjustedBlockSizes[j] = ValueType(blockSizes[j]) * ratio;
			}
		} else {
		    SCAI_ASSERT_EQ_ERROR(lastIndex - firstIndex, localN, "invalid iterators");
			assert(lastIndex == localIndices.end());
		}

		std::vector<ValueType> timePerPE( comm->getSize(), 0.0);

		result = assignBlocks(convertedCoords, centers, firstIndex, lastIndex, nodeWeights, result, adjustedBlockSizes, boundingBox, upperBoundOwnCenter, lowerBoundNextCenter, influence, imbalance, timePerPE, settings, metrics);
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

		std::vector<std::vector<ValueType> > newCenters = findCenters(coordinates, result, k, firstIndex, lastIndex, nodeWeights);

		//keep centroids of empty blocks at their last known position
		for (IndexType j = 0; j < k; j++) {
		    for (int d = 0; d < dim; d++) {
		        if (std::isnan(newCenters[d][j])) {
		            newCenters[d][j] = centers[d][j];
		        }
		    }
		}
		std::vector<ValueType> squaredDeltas(k,0);
		std::vector<ValueType> deltas(k,0);
		std::vector<ValueType> oldInfluence = influence;
		ValueType minRatio = std::numeric_limits<double>::max();

		for (IndexType j = 0; j < k; j++) {
			for (int d = 0; d < dim; d++) {
				ValueType diff = (centers[d][j] - newCenters[d][j]);
				squaredDeltas[j] += diff*diff;
			}
			deltas[j] = std::sqrt(squaredDeltas[j]);
			if (settings.erodeInfluence) {
				const ValueType erosionFactor = 2/(1+exp(-std::max(deltas[j]/expectedBlockDiameter-0.1, 0.0))) - 1;
				influence[j] = exp((1-erosionFactor)*log(influence[j]));//TODO: will only work for uniform target block sizes
				if (oldInfluence[j] / influence[j] < minRatio) minRatio = oldInfluence[j] / influence[j];
			}
		}

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

				if (settings.erodeInfluence) {
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
		centers = newCenters;

		std::vector<ValueType> blockWeights(k,0.0);
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
			comm->sumImpl(blockWeights.data(), blockWeights.data(), k, scai::common::TypeTraits<ValueType>::stype);
		}

		balanced = true;
		for (IndexType j = 0; j < k; j++) {
			if (blockWeights[j] > blockSizes[j]*(1+settings.epsilon)) {
				balanced = false;
			}
		}

		ValueType maxTime=0;
		if(settings.verbose){
			balanceTime = std::chrono::high_resolution_clock::now() - iterStart;
			maxTime = comm->max( balanceTime.count() );
		}

		
		//WARNING: when sampling, a (big!) part of the result vector is not changed
		//	and keeps its initial value which is 0. So, the computeImbalance finds,
		//	falsely, block 0 to be over weighted. We use the returned imabalance
		//	from assign centers when sampling is used and we compute a new imbalance
		// only when thene is no sampling
		if( !randomInitialization ){
			imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( result, settings.numBlocks, nodeWeights );
		}

		if (comm->getRank() == 0) {
			std::cout << "i: " << iter<< ", delta: " << delta << ", time : "<< maxTime << ", imbalance= " << imbalance<< std::endl;
		}

		metrics.kmeansProfiling.push_back( std::make_tuple(delta, maxTime, imbalance) );

		iter++;

		// WARNING-TODO: this stops the iterations prematurely, when the wanted balance
		// is reached. It is possible that if we allow more iterations, the solution
		// will converge to some optima reagaridng the cut/shape. Investigate that

		//WARNING2: this is also needed to ensure that the required number of sampling
		//	rounds will be performed so at the end, all nodes are accounted for

		//if(imbalance<settings.epsilon)
		//	break;

	} while (iter < samplingRounds or (iter < maxIterations && (delta > threshold || !balanced)) ); // or (imbalance>settings.epsilon) );


	std::chrono::duration<ValueType,std::ratio<1>> KMeansTime = std::chrono::high_resolution_clock::now() - KMeansStart;
	ValueType time = comm->max( KMeansTime.count() );
	
	PRINT0("total KMeans time: " << time << " , number of iterations: " << iter );

	return result;
}

//moved (7.11.18) from KMeans.h

/**
 * Implementations
 */

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
		minCoords[d] = scai::utilskernel::HArrayUtils::min(coordinates[d].getLocalValues());
		maxCoords[d] = scai::utilskernel::HArrayUtils::max(coordinates[d].getLocalValues());
	}
	return {minCoords, maxCoords};
}

//called initially with no centers parameter
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(
	const std::vector<DenseVector<ValueType>> &coordinates,
	const DenseVector<ValueType> &nodeWeights,
	const std::vector<IndexType> &blockSizes,
	const Settings settings,
	struct Metrics& metrics) {

    std::vector<ValueType> minCoords(settings.dimensions);
    std::vector<ValueType> maxCoords(settings.dimensions);
    for (IndexType dim = 0; dim < settings.dimensions; dim++) {
        minCoords[dim] = coordinates[dim].min();
        maxCoords[dim] = coordinates[dim].max();
		SCAI_ASSERT_NE_ERROR( minCoords[dim], maxCoords[dim], "min=max for dimension "<< dim << ", this will cause problems to the hilbert index. local= " << coordinates[0].getLocalValues().size() );
    }

	std::vector<std::vector<ValueType> > centers = findInitialCentersSFC<IndexType,ValueType>(coordinates, minCoords, maxCoords, settings);
	//Metrics metrics;

	return computePartition(coordinates, nodeWeights, blockSizes, centers, settings, metrics);
}

// Wrapper function to keep compatibility for calls without a communicator
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition( \
	const std::vector<DenseVector<ValueType>> &coordinates, \
	const DenseVector<ValueType> &nodeWeights, \
	const std::vector<IndexType> &blockSizes, \
	std::vector<std::vector<ValueType> > centers, \
	const Settings settings, \
	struct Metrics &metrics ){

	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	return computePartition(coordinates, nodeWeights, blockSizes, centers, comm, settings, metrics) ;
}

//---------------------------------------

//TODO: why these two are needed since they are defined above?

template DenseVector<IndexType> computePartition(
	const std::vector<DenseVector<ValueType>>& coordinates,
	const DenseVector<ValueType>& nodeWeights,
	const std::vector<IndexType>& blockSizes,
	const Settings settings,
	struct Metrics& metrics);

template DenseVector<IndexType> computeRepartition(
	const std::vector<DenseVector<ValueType>>& coordinates,
	const DenseVector<ValueType>& nodeWeights,
	const std::vector<IndexType>& blockSizes,
	const DenseVector<IndexType>& previous,
	const Settings settings);


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

}

} /* namespace ITI */
