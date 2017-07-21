/*
 * KMeans.cpp
 *
 *  Created on: 19.07.2017
 *      Author: moritz
 */

#include <set>
#include <cmath>

#include <scai/dmemo/NoDistribution.hpp>

#include "KMeans.h"

namespace ITI {
namespace KMeans {

using scai::lama::Scalar;

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCenters(
		const std::vector<DenseVector<ValueType> >& coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights) {

	const IndexType dim = coordinates.size();
	const IndexType n = coordinates[0].size();
	const IndexType localN = coordinates[0].getLocalValues().size();

	std::vector<std::vector<ValueType> > result(dim);

	for (IndexType d = 0; d < dim; d++) {
		result[d].resize(k);
	}

	//broadcast seed value from root to ensure equal pseudorandom numbers.
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
	comm->bcast( seed, 1, 0 );
	srand(seed[0]);

	std::set<IndexType> indices;
	while (indices.size() < k) {
		indices.insert(rand() % localN);
	}

	for (IndexType d = 0; d < dim; d++) {
		const scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());

		IndexType i = 0;
		for (IndexType index : indices) {
			result[d][i] = rCoords[index];
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
	for (IndexType j = 0; j < k; j++) {
		totalWeight[j] = comm->sum(weightSum[j]);
	}

	//compute updated centers as weighted average
	for (IndexType d = 0; d < dim; d++) {
		for (IndexType j = 0; j < k; j++) {
			ValueType weightRatio = (ValueType(weightSum[j]) / totalWeight[j]);
			ValueType weightedCoord = result[d][j] * weightRatio;
			result[d][j] = comm->sum(weightedCoord);
		}
	}

	return result;
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(
		const std::vector<DenseVector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers,
		const DenseVector<IndexType> &nodeWeights, const std::vector<IndexType> &targetBlockSizes,  const ValueType epsilon) {

	const IndexType dim = coordinates.size();
	const IndexType n = nodeWeights.size();
	const scai::dmemo::DistributionPtr dist = coordinates[0].getDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType localN = nodeWeights.getLocalValues().size();
	const IndexType totalWeightSum = nodeWeights.sum().Scalar::getValue<IndexType>();
	const IndexType k = targetBlockSizes.size();

	const IndexType maxIter = 20;

	const IndexType optSize = std::ceil(totalWeightSum / k );

	//prepare data structure for squared distances
	std::vector<std::vector<ValueType>> squaredDistances(k);
	for (std::vector<ValueType>& sublist : squaredDistances) {
		sublist.resize(localN, 0);
	}

	//compute squared distances. Since only the nearest neighbor is required, we could speed this up with a suitable data structure.
	for (IndexType d = 0; d < dim; d++) {
		scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());

		for (IndexType j = 0; j < k; j++) {
			ValueType centerCoord = centers[d][j];
			for (IndexType i = 0; i < localN; i++) {
				ValueType coord = rCoords[i];
				ValueType dist = (centerCoord - coord);
				squaredDistances[j][i] += dist*dist; //maybe use Manhattan distance here? Should align better with grid structure.
			}
		}
	}

	//compute assignment and balance
	std::vector<ValueType> influence(k, 1);
	DenseVector<IndexType> assignment(dist, 0);

	ValueType imbalance;
	IndexType iter = 0;
	//iterate if necessary to achieve balance
	do
	{
		std::vector<IndexType> blockWeights(k,0);
		scai::hmemo::ReadAccess<IndexType> rWeights(nodeWeights.getLocalValues());
		scai::hmemo::WriteAccess<IndexType> wAssignment(assignment.getLocalValues());
		for (IndexType i = 0; i < localN; i++) {
			int bestBlock = 0;
			for (IndexType j = 0; j < k; j++) {
				if (squaredDistances[j][i]/influence[j] < squaredDistances[bestBlock][i]/influence[bestBlock]) {
					bestBlock = j;
				}
			}
			wAssignment[i] = bestBlock;
			blockWeights[bestBlock] += rWeights[i];
		}

		std::vector<IndexType> totalWeight(k, 0);
		for (IndexType j = 0; j < k; j++) {
			totalWeight[j] = comm->sum(blockWeights[j]);
			influence[j] = std::min(influence[j] * std::pow(ValueType(optSize) / totalWeight[j], 0.7), ValueType(20));

			if (comm->getRank() == 0) {
				std::cout << "Iter " << iter << ", block " << j << " has size " << totalWeight[j] << ", setting influence to ";
				std::cout << influence[j];
				std::cout << std::endl;
			}
		}

		IndexType maxBlockWeight = *std::max_element(totalWeight.begin(), totalWeight.end());
		imbalance = (ValueType(maxBlockWeight - optSize)/ optSize);
		iter++;

		if (comm->getRank() == 0) std::cout << "Iter " << iter << ", imbalance : " << imbalance << std::endl;
	} while (imbalance > epsilon && iter < maxIter);


	return assignment;
}

template std::vector<std::vector<double> > findInitialCenters(const std::vector<DenseVector<double>> &coordinates, int k, const DenseVector<int> &nodeWeights);
template std::vector<std::vector<double> > findCenters(const std::vector<DenseVector<double>> &coordinates, const DenseVector<int> &partition, const int k, const DenseVector<int> &nodeWeights);
template DenseVector<int> assignBlocks(const std::vector<DenseVector<double>> &coordinates, const std::vector<std::vector<double> > &centers, const DenseVector<int> &nodeWeights, const std::vector<int> &blockSizes,  const double epsilon);

}
} /* namespace ITI */
