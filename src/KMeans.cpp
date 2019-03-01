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
std::vector<std::vector<ValueType> > findInitialCentersSFC(
		const std::vector<DenseVector<ValueType> >& coordinates, const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords, Settings settings) {

	SCAI_REGION( "KMeans.findInitialCentersSFC" );
	const IndexType localN = coordinates[0].getLocalValues().size();
	const IndexType globalN = coordinates[0].size();
	const IndexType dimensions = settings.dimensions;
	const IndexType k = settings.numBlocks;
	
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
    auto comm = scai::dmemo::Communicator::getCommunicatorPtr();
	auto blockDist = scai::dmemo::genBlockDistributionBySize(globalN, localN, comm);

	//set local values in vector, leave non-local values with zero
	std::vector<std::vector<ValueType> > result(dimensions);
	for (IndexType d = 0; d < dimensions; d++) {
		result[d].resize(k);
	}

	for (IndexType j = 0; j < k; j++) {
		IndexType localIndex = blockDist->global2Local(wantedIndices[j]);
		if (localIndex != scai::invalidIndex) {
			assert(localIndex < localN);
			IndexType permutedIndex = localIndices[localIndex];
			assert(permutedIndex < localN);
			assert(permutedIndex >= 0);
			for (IndexType d = 0; d < dimensions; d++) {
				result[d][j] = convertedCoords[permutedIndex][d];
			}
		}
	}

	//global sum operation
	for (IndexType d = 0; d < dimensions; d++) {
		comm->sumImpl(result[d].data(), result[d].data(), k, scai::common::TypeTraits<ValueType>::stype);
	}

	return result;
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
	std::vector<ValueType> weightSum(k, 0.0);

	scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
	scai::hmemo::ReadAccess<IndexType> rPartition(partition.getLocalValues());

	//compute weight sums
	for (Iterator it = firstIndex; it != lastIndex; it++) {
		const IndexType i = *it;
		const IndexType part = rPartition[i];
		const ValueType weight = rWeights[i];
		weightSum[part] += weight;
	}

	//find local centers
	for (IndexType d = 0; d < dim; d++) {
		result[d].resize(k,0);
		scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());

		for (Iterator it = firstIndex; it != lastIndex; it++) {			
			const IndexType i = *it;
			const IndexType part = rPartition[i];
			//const IndexType weight = rWeights[i];
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

template<typename IndexType, typename ValueType, typename Iterator>
DenseVector<IndexType> assignBlocks(
		const std::vector<std::vector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers,
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
std::chrono::time_point<std::chrono::high_resolution_clock> assignStart = std::chrono::high_resolution_clock::now();
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
DenseVector<IndexType> computeRepartition(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<ValueType> &nodeWeights, const Settings settings, struct Metrics& metrics) {
	
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
