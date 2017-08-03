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
#include <numeric>

#include <scai/dmemo/NoDistribution.hpp>

#include "KMeans.h"

namespace ITI {
namespace KMeans {

using scai::lama::Scalar;

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCenters(
		const std::vector<DenseVector<ValueType> >& coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights) {

	SCAI_REGION( "KMeans.findInitialCenters" );

	const IndexType dim = coordinates.size();
	const IndexType n = coordinates[0].size();
	const IndexType localN = coordinates[0].getLocalValues().size();

	std::vector<std::vector<ValueType> > result(dim);

	for (IndexType d = 0; d < dim; d++) {
		result[d].resize(k);
	}

	std::vector<IndexType> indices(k);

	for (IndexType i = 0; i < k; i++) {
		indices[i] = i * (n / k);
	}

	for (IndexType d = 0; d < dim; d++) {

		IndexType i = 0;
		for (IndexType index : indices) {
			//yes, this is very expensive, since each call triggers a global communication. However, this needs to be done anyway, if not here then later.
			result[d][i] = coordinates[d].getValue(index).Scalar::getValue<ValueType>();
			i++;
		}
	}

	//alternative:
	//sort coordinates according to x coordinate first, then y coordinate. The schizoQuicksort currently only supports one sorting key, have to extend that.

	return result;
}

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findCenters(
		const std::vector<DenseVector<ValueType> >& coordinates,
		const DenseVector<IndexType>& partition,
		const IndexType k,
		const DenseVector<IndexType>& nodeWeights) {
	SCAI_REGION( "KMeans.findCenters" );

	const IndexType dim = coordinates.size();
	const IndexType n = partition.size();
	const IndexType localN = partition.getLocalValues().size();
	const scai::dmemo::DistributionPtr resultDist(new scai::dmemo::NoDistribution(k));
	const scai::dmemo::CommunicatorPtr comm = partition.getDistribution().getCommunicatorPtr();

	//TODO: check that distributions align

	std::vector<std::vector<ValueType> > result(dim);
	std::vector<IndexType> weightSum(k, 0);

	scai::hmemo::ReadAccess<IndexType> rWeights(nodeWeights.getLocalValues());
	scai::hmemo::ReadAccess<IndexType> rPartition(partition.getLocalValues());

	//compute weight sums
	for (IndexType i = 0; i < localN; i++) {
		const IndexType part = rPartition[i];
		const IndexType weight = rWeights[i];
		weightSum[part] += weight;
	}

	//find local centers
	for (IndexType d = 0; d < dim; d++) {
		result[d].resize(k);
		scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());

		for (IndexType i = 0; i < localN; i++) {
			const IndexType part = rPartition[i];
			const IndexType weight = rWeights[i];
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
			//if no points in cluster, set it to left upper corner. TODO: find better way to seed new cluster.
			ValueType weightedCoord = weightSum[j] == 0 ? 0 : result[d][j] * weightRatio;
			result[d][j] = weightedCoord;
			assert(std::isfinite(result[d][j]));
		}
		comm->sumImpl(result[d].data(), result[d].data(), k, scai::common::TypeTraits<ValueType>::stype);
	}

	return result;
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(const std::vector<DenseVector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers) {

	const IndexType dim = coordinates.size();
	assert(dim > 0);
	assert(centers.size() == dim);
	const IndexType n = coordinates[0].size();
	const IndexType localN = coordinates[0].getLocalValues().size();
	const IndexType k = centers[0].size();

	DenseVector<IndexType> assignment(coordinates[0].getDistributionPtr(), 0);

	std::vector<std::vector<ValueType>> squaredDistances(k);
	for (std::vector<ValueType>& sublist : squaredDistances) {
		sublist.resize(localN, 0);
	}

	{
		SCAI_REGION( "KMeans.assignBlocks.squaredDistances" );
		//compute squared distances. Since only the nearest neighbor is required, we could speed this up with a suitable data structure.
		for (IndexType d = 0; d < dim; d++) {
			scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());
			for (IndexType j = 0; j < k; j++) {
				ValueType centerCoord = centers[d][j];
				for (IndexType i = 0; i < localN; i++) {
					ValueType coord = rCoords[i];
					ValueType dist = std::abs(centerCoord - coord);
					squaredDistances[j][i] += dist*dist; //maybe use Manhattan distance here? Should align better with grid structure.
				}
			}
		}

		scai::hmemo::WriteAccess<IndexType> wAssignment(assignment.getLocalValues());
		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.assign" );
			for (IndexType i = 0; i < localN; i++) {
				int bestBlock = 0;
				int bestValue = squaredDistances[bestBlock][i];
				for (IndexType j = 0; j < k; j++) {
					if (squaredDistances[j][i] < bestValue) {
						bestBlock = j;
						bestValue = squaredDistances[bestBlock][i];
					}
				}
				wAssignment[i] = bestBlock;
			}
		}
	}
	return assignment;
}

//	std::vector<ValueType> upperBoundToOwnCenter(localN);
//std::vector<ValueType> lowerBoundToNextCenter(localN, 0);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(
		const std::vector<std::vector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers,
		const DenseVector<IndexType> &nodeWeights,
		const DenseVector<IndexType> &previousAssignment,
		const std::vector<IndexType> &targetBlockSizes,
		const SpatialCell &boundingBox,
		const ValueType epsilon,
		std::vector<ValueType> &upperBoundOwnCenter,
		std::vector<ValueType> &lowerBoundNextCenter,
		std::vector<ValueType> &influence) {
	SCAI_REGION( "KMeans.assignBlocks" );

	const IndexType dim = coordinates.size();
	const scai::dmemo::DistributionPtr dist = nodeWeights.getDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType localN = nodeWeights.getLocalValues().size();
	const IndexType totalWeightSum = nodeWeights.sum().Scalar::getValue<IndexType>();
	const IndexType k = targetBlockSizes.size();

	const IndexType maxIter = 20;

	const IndexType optSize = std::ceil(totalWeightSum / k );

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
			effectiveMinDistance[j] = minDistance[j]*minDistance[j]*influence[j];
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
		assert(effectiveMinDistance[i] == effectiveDist);
	}

	std::vector<ValueType> distThreshold(k);
	{
		SCAI_REGION( "KMeans.assignBlocks.pairWise" );
		for (IndexType j = 0; j < k; j++) {
			distThreshold[j] = std::numeric_limits<ValueType>::max();
			for (IndexType l = 0; l < k; l++) {
				if (j == l) continue;
				ValueType sqDist = 0;
				for (IndexType d = 0; d < dim; d++) {
					sqDist += std::pow(centers[d][j] - centers[d][l], 2);
				}
				ValueType weightedDist = sqDist * influence[j]*influence[l] / (influence[j]+influence[l]+2*std::sqrt(influence[j]*influence[l]));
				if (weightedDist < distThreshold[j]) distThreshold[j] = weightedDist;
			}

		}
	}

	ValueType imbalance;
	IndexType iter = 0;
	IndexType skippedLoops = 0;
	//iterate if necessary to achieve balance
	do
	{
		SCAI_REGION( "KMeans.assignBlocks.balanceLoop" );
		std::vector<IndexType> blockWeights(k,0);
		scai::hmemo::ReadAccess<IndexType> rWeights(nodeWeights.getLocalValues());
		scai::hmemo::WriteAccess<IndexType> wAssignment(assignment.getLocalValues());
		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.assign" );
			for (IndexType i = 0; i < localN; i++) {
				const IndexType oldCluster = wAssignment[i];
				if (lowerBoundNextCenter[i] > upperBoundOwnCenter[i] || upperBoundOwnCenter[i] < distThreshold[oldCluster]) {
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
					if (lowerBoundNextCenter[i] > upperBoundOwnCenter[i] || upperBoundOwnCenter[i] < distThreshold[oldCluster]) {
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
			}
			comm->synchronize();
		}

		{
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.blockWeightSum" );
			comm->sumImpl(blockWeights.data(), blockWeights.data(), k, scai::common::TypeTraits<IndexType>::stype);
		}
		IndexType maxBlockWeight = *std::max_element(blockWeights.begin(), blockWeights.end());
		imbalance = (ValueType(maxBlockWeight - optSize)/ optSize);

		std::vector<ValueType> oldInfluence = influence;

		double minRatio = 1.05;
		double maxRatio = 0.95;
		for (IndexType j = 0; j < k; j++) {
			SCAI_REGION( "KMeans.assignBlocks.balanceLoop.influence" );
			double ratio = ValueType(blockWeights[j]) / targetBlockSizes[j];
			influence[j] = std::max(influence[j]*0.95, std::min(influence[j] * std::pow(ratio, 0.5), influence[j]*1.05));

			double influenceRatio = influence[j] / oldInfluence[j];
			assert(influenceRatio <= 1.05 + 1e-10);
			assert(influenceRatio >= 0.95 - 1e-10);
			if (influenceRatio < minRatio) minRatio = influenceRatio;
			if (influenceRatio > maxRatio) maxRatio = influenceRatio;
		}

		//update bounds
		for (IndexType i = 0; i < localN; i++) {
			const IndexType cluster = wAssignment[i];
			upperBoundOwnCenter[i] *= (influence[cluster] / oldInfluence[cluster]) + 1e-12;
			lowerBoundNextCenter[i] *= minRatio - 1e-12;
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

		{
			SCAI_REGION( "KMeans.assignBlocks.pairWise" );
			for (IndexType j = 0; j < k; j++) {
				distThreshold[j] = std::numeric_limits<ValueType>::max();

				for (IndexType l = 0; l < k; l++) {
					if (j == l) continue;
					ValueType sqDist = 0;
					for (IndexType d = 0; d < dim; d++) {
						sqDist += std::pow(centers[d][j] - centers[d][l], 2);
					}
					ValueType weightedDist = sqDist * influence[j]*influence[l] / (influence[j]+influence[l]+2*std::sqrt(influence[j]*influence[l]));
					if (weightedDist < distThreshold[j]) distThreshold[j] = weightedDist;
				}

			}
		}

		iter++;

		if (comm->getRank() == 0) std::cout << "Iter " << iter << ", imbalance : " << imbalance << std::endl;
	} while (imbalance > epsilon && iter < maxIter);
	//std::cout << "Process " << comm->getRank() << " skipped " << ValueType(skippedLoops*100) / (iter*localN) << "% of inner loops." << std::endl;

	return assignment;
}

template<typename ValueType>
ValueType biggestDelta(const std::vector<std::vector<ValueType>> &firstCoords, const std::vector<std::vector<ValueType>> &secondCoords) {
	SCAI_REGION( "KMeans.biggestDelta" );
	assert(firstCoords.size() == secondCoords.size());
	const int dim = firstCoords.size();
	assert(dim > 0);
	const int n = firstCoords[0].size();

	ValueType result = 0;
	for (int i = 0; i < n; i++) {
		ValueType squaredDistance = 0;
		for (int d = 0; d < dim; d++) {
			ValueType diff = (firstCoords[d][i] - secondCoords[d][i]);
			squaredDistance +=  diff*diff;
		}
		result = std::max(squaredDistance, result);
	}
	return std::sqrt(result);
}

template double biggestDelta(const std::vector<std::vector<double>> &firstCoords, const std::vector<std::vector<double>> &secondCoords);
template std::vector<std::vector<double> > findInitialCenters(const std::vector<DenseVector<double>> &coordinates, int k, const DenseVector<int> &nodeWeights);
template std::vector<std::vector<double> > findCenters(const std::vector<DenseVector<double>> &coordinates, const DenseVector<int> &partition, const int k, const DenseVector<int> &nodeWeights);
template DenseVector<int> assignBlocks(const std::vector<std::vector<double>> &coordinates, const std::vector<std::vector<double> > &centers,
		const DenseVector<int> &nodeWeights, const DenseVector<int> &previousAssignment, const std::vector<int> &blockSizes, const SpatialCell &boundingBox,
		const double epsilon, std::vector<double> &upperBoundOwnCenter, std::vector<double> &lowerBoundNextCenter, std::vector<double> &influence);
template DenseVector<int> assignBlocks(const std::vector<DenseVector<double> >& coordinates, const std::vector<std::vector<double> >& centers);

}
} /* namespace ITI */
