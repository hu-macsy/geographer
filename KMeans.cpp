/*
 * KMeans.cpp
 *
 *  Created on: 19.07.2017
 *      Author: moritz
 */

#include "KMeans.h"

namespace ITI {
namespace KMeans {



template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType> > ITI::KMeans::findInitialCenters(
		const std::vector<DenseVector<ValueType> >& coordinates, IndexType k, const DenseVector<IndexType> &nodeWeights) {

	const IndexType dim = coordinates.size();
	const IndexType n = coordinates[0].size();
	std::vector<DenseVector<ValueType> > result(dim);

	for (IndexType d = 0; d < dim; d++) {
		result[d] = DenseVector(k, 0);
	}

	//broadcast seed value from root to ensure equal pseudorandom numbers.
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
	comm->bcast( seed, 1, 0 );
	srand(seed[0]);

	std::set<IndexType> indices;
	while (indices.size() < k) {
		indices.insert(rand() % n);
	}

	//const scai::hmemo::ReadAccess<IndexType> rWeights(nodeWeights.getLocalValues());

	for (IndexType d = 0; d < dim; d++) {
		const scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());
		const scai::hmemo::WriteAccess<ValueType> wResult(result[d].getLocalValues());
		for (IndexType i = 0; i < k; i++) {
			wResult[i] = rCoords[indices[i]];
		}
	}

	//alternative:
	//sort coordinates according to x coordinate first, then y coordinate. The schizoQuicksort currently only supports one sorting key, have to extend that.

	return result;
}

template<typename IndexType, typename ValueType>
std::vector<DenseVector<IndexType> > ITI::KMeans::findCenters(
		const std::vector<DenseVector<ValueType> >& coordinates,
		const DenseVector<IndexType>& partition,
		const IndexType k,
		const DenseVector<IndexType>& nodeWeights) {

	const IndexType dim = coordinates.size();
	const IndexType n = partition.size();
	const IndexType localN = partition.getLocalValues().size();
	const scai::dmemo::CommunicatorPtr comm = partition.getDistribution().getCommunicatorPtr();

	//TODO: check that distributions align

	std::vector<DenseVector<ValueType> > result(dim);
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
		result[d] = DenseVector(k, 0);
		scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d]);
		scai::hmemo::WriteAccess<ValueType> wResult(result[d]);
		for (IndexType i = 0; i < localN; i++) {
			const IndexType part = rPartition[i];
			const IndexType weight = rWeights[i];
			wResult[part] += rCoords[i]*rWeights[i] / weightSum[part];//this is more expensive than summing first and dividing later, but avoids overflows
		}
	}

	//communicate local centers and weight sums
	std::vector<IndexType> totalWeight(k, 0);
	for (IndexType j = 0; j < k; j++) {
		totalWeight[j] = comm->sum(weightSum[j]);
	}

	//compute updated centers as weighted average
	for (IndexType d = 0; d < dim; d++) {
		scai::hmemo::WriteAccess<ValueType> wResult(result[d]);
		for (IndexType j = 0; j < k; j++) {
			ValueType weightRatio = (ValueType(weightSum[j]) / totalWeight[j]);
			wResult[j] = comm->sum(wResult[j] * weightRatio);
		}
	}

	return result;
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ITI::KMeans::assignBlocks(
		const std::vector<DenseVector<ValueType> >& coordinates,
		const std::vector<DenseVector<IndexType> >& centers,
		const DenseVector<IndexType> &nodeWeights, const std::vector<IndexType> &targetBlockSizes,  const ValueType epsilon = 0.05) {

	const IndexType dim = coordinates.size();
	const IndexType n = nodeWeights.size();
	const scai::dmemo::DistributionPtr dist = nodeWeights.getDistributionPtr();
	const IndexType localN = nodeWeights.getLocalValues().size();
	const IndexType k = targetBlockSizes.size();

	//prepare data structure for squared distances
	std::vector<std::vector<ValueType>> squaredDistances(k);
	for (std::vector<ValueType> sublist : squaredDistances) {
		sublist.resize(localN, 0);
	}

	//compute squared distances. Since only the nearest neighbor is required, we could speed this up with a suitable data structure.
	for (IndexType d = 0; d < dim; d++) {
		scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());
		scai::hmemo::ReadAccess<ValueType> rCenters(centers[d].getLocalValues());
		for (IndexType j = 0; j < k; j++) {
			ValueType centerCoord = rCenters[j];
			for (IndexType i = 0; i < localN; i++) {
				ValueType coord = rCoords[i];
				ValueType dist = (centerCoord - coord);
				squaredDistances[j][i] += dist*dist; //maybe use Manhattan distance here?
			}
		}
	}

	//compute assignment and balance
	DenseVector<IndexType> assignment(dist, 0);
	std::vector<IndexType> blockWeights(k,0);
	{
		scai::hmemo::ReadAccess<IndexType> rWeights(nodeWeights.getLocalValues());
		scai::hmemo::WriteAccess<IndexType> wAssignment(assignment.getLocalValues());
		for (IndexType i = 0; i < localN; i++) {
			int bestBlock = 0;
			for (IndexType j = 0; j < k; j++) {
				if (squaredDistances[j][i] < squaredDistances[bestBlock][i]) {
					bestBlock = j;
				}
			}
			wAssignment[i] = bestBlock;
			blockWeights[bestBlock] += rWeights[i];
		}


	}

	return assignment;
}

}
} /* namespace ITI */
