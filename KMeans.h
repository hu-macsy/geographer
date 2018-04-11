/*
 * KMeans.h
 *
 *  Created on: 19.07.2017
 *      Author: moritz
 */

#pragma once

#include <vector>
#include <numeric>
#include <scai/lama/DenseVector.hpp>
#include <scai/tracing.hpp>
#include <chrono>

#include "quadtree/QuadNodeCartesianEuclid.h"
#include "Settings.h"
#include "GraphUtils.h"
#include "HilbertCurve.h"

using scai::lama::DenseVector;

namespace ITI {
namespace KMeans {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &nodeWeights,
		const std::vector<IndexType> &blockSizes, const Settings settings);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &  nodeWeights,
		const std::vector<IndexType> &blockSizes, const DenseVector<IndexType>& previous, const Settings settings);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &nodeWeights,
		const std::vector<IndexType> &blockSizes, std::vector<std::vector<ValueType> > centers, const Settings settings);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> >  findInitialCentersSFC(
		const std::vector<DenseVector<ValueType> >& coordinates, IndexType k, const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords, Settings settings);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCentersFromSFCOnly( const IndexType k, const std::vector<ValueType> &maxCoords, Settings settings);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCenters(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &nodeWeights);

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> findLocalCenters(	const std::vector<DenseVector<ValueType> >& coordinates, const DenseVector<ValueType> &nodeWeights);

template<typename IndexType, typename ValueType, typename Iterator>
std::vector<std::vector<ValueType> > findCenters(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const IndexType k,
		const Iterator firstIndex, const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> assignBlocks(const std::vector<DenseVector<ValueType> >& coordinates,
		const std::vector<std::vector<ValueType> >& centers);

template<typename IndexType, typename ValueType, typename Iterator>
DenseVector<IndexType> assignBlocks(const std::vector<std::vector<ValueType>> &coordinates, const std::vector<std::vector<ValueType> > &centers,
		const Iterator firstIndex, const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights, const DenseVector<IndexType> &previousAssignment,
		const std::vector<IndexType> &blockSizes,  const SpatialCell &boundingBox,
		std::vector<ValueType> &upperBoundOwnCenter, std::vector<ValueType> &lowerBoundNextCenter,
		std::vector<ValueType> &influence,
		Settings settings);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> getPartitionWithSFCCoords(
		const scai::lama::CSRSparseMatrix<ValueType>& adjM, \
		const std::vector<DenseVector<ValueType> >& coordinates,\
		const DenseVector<ValueType> &  nodeWeights,\
		const Settings settings);
//TODO: add block weights as an input param?: const std::vector<IndexType> &blockSizes


template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeRepartition(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<ValueType> &nodeWeights, const Settings settings);

/**
 * Implementations
 */

template<typename ValueType>
std::pair<std::vector<ValueType>, std::vector<ValueType> > getLocalMinMaxCoords(const std::vector<DenseVector<ValueType>> &coordinates) {
	const int dim = coordinates.size();
	std::vector<ValueType> minCoords(dim);
	std::vector<ValueType> maxCoords(dim);
	for (int d = 0; d < dim; d++) {
		minCoords[d] = coordinates[d].getLocalValues().min();//.Scalar::getValue<ValueType>();
		maxCoords[d] = coordinates[d].getLocalValues().max();//.Scalar::getValue<ValueType>();
	}
	return {minCoords, maxCoords};
}

//wrapper for initial partitioning
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &  nodeWeights,
		const std::vector<IndexType> &blockSizes, const Settings settings) {

    std::vector<ValueType> minCoords(settings.dimensions);
    std::vector<ValueType> maxCoords(settings.dimensions);
    for (IndexType dim = 0; dim < settings.dimensions; dim++) {
        minCoords[dim] = coordinates[dim].min().scai::lama::Scalar::getValue<ValueType>();
        maxCoords[dim] = coordinates[dim].max().scai::lama::Scalar::getValue<ValueType>();
		SCAI_ASSERT_NE_ERROR( minCoords[dim], maxCoords[dim], "min=max for dimension "<< dim << ", this will cause problems to the hilbert index. local= " << coordinates[0].getLocalValues().size() );
    }

	std::vector<std::vector<ValueType> > centers = findInitialCentersSFC(coordinates, k, minCoords, maxCoords, settings);

	return computePartition(coordinates, k, nodeWeights, blockSizes, centers, settings);
}

//wrapper for repartitioning
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &  nodeWeights, const std::vector<IndexType> &blockSizes, const DenseVector<IndexType>& previous, const Settings settings) {
	const IndexType localN = nodeWeights.getLocalValues().size();
	std::vector<IndexType> indices(localN);
	const typename std::vector<IndexType>::iterator firstIndex = indices.begin();
	typename std::vector<IndexType>::iterator lastIndex = indices.end();
	std::iota(firstIndex, lastIndex, 0);
	std::vector<std::vector<ValueType> > initialCenters = findCenters(coordinates, previous, k,	indices.begin(), indices.end(), nodeWeights);
	return computePartition(coordinates, k, nodeWeights, blockSizes, initialCenters, settings);
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(const std::vector<DenseVector<ValueType>> &coordinates, IndexType k, const DenseVector<ValueType> &  nodeWeights,
		const std::vector<IndexType> &blockSizes, std::vector<std::vector<ValueType> > centers, const Settings settings) {
	SCAI_REGION( "KMeans.computePartition" );

	std::vector<ValueType> influence(k,1);
	const IndexType dim = coordinates.size();
	assert(dim > 0);
	const IndexType localN = nodeWeights.getLocalValues().size();
	const IndexType globalN = nodeWeights.size();
	assert(nodeWeights.getLocalValues().size() == coordinates[0].getLocalValues().size());
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	const IndexType p = comm->getSize();
	const ValueType blocksPerProcess = ValueType(k)/p;

	std::vector<ValueType> minCoords(dim);
	std::vector<ValueType> maxCoords(dim);
	std::vector<std::vector<ValueType> > convertedCoords(dim);
	for (IndexType d = 0; d < dim; d++) {
		scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
		assert(rAccess.size() == localN);
		convertedCoords[d] = std::vector<ValueType>(rAccess.get(), rAccess.get()+localN);
		assert(convertedCoords[d].size() == localN);
		minCoords[d] = *std::min_element(convertedCoords[d].begin(), convertedCoords[d].end());
		maxCoords[d] = *std::max_element(convertedCoords[d].begin(), convertedCoords[d].end());
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
        std::cout << "Process " << comm->getRank() << ": ( ";
        for (auto coord : minCoords) std::cout << coord << " ";
        std::cout << ") , ( ";
        for (auto coord : maxCoords) std::cout << coord << " ";
        std::cout << ")";
        std::cout << ", " << localN << " nodes, " << nodeWeights.getLocalValues().sum() << " total weight";
        std::cout << ", volume ratio " << localVolume / (volume / p);
        std::cout << std::endl;
    }

	diagonalLength = std::sqrt(diagonalLength);
	const ValueType expectedBlockDiameter = pow(volume / k, 1.0/dim);

	DenseVector<IndexType> result(coordinates[0].getDistributionPtr(), 0);
	std::vector<ValueType> upperBoundOwnCenter(localN, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> lowerBoundNextCenter(localN, 0);

	//prepare sampling
	std::vector<IndexType> localIndices(localN);
	const typename std::vector<IndexType>::iterator firstIndex = localIndices.begin();
	typename std::vector<IndexType>::iterator lastIndex = localIndices.end();;
	std::iota(localIndices.begin(), localIndices.end(), 0);

	IndexType minNodes = settings.minSamplingNodes*blocksPerProcess;
	assert(minNodes > 0);
	IndexType samplingRounds = 0;
	std::vector<IndexType> samples;
	std::vector<IndexType> adjustedBlockSizes(blockSizes);
	if (comm->all(localN > minNodes)) {
		ITI::GraphUtils::FisherYatesShuffle(localIndices.begin(), localIndices.end(), localN);

		samplingRounds = std::ceil(std::log2( globalN / ValueType(settings.minSamplingNodes*k)))+1;
		samples.resize(samplingRounds);
		samples[0] = std::min(minNodes, localN);
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


	scai::hmemo::ReadAccess<ValueType> rWeight(nodeWeights.getLocalValues());

	IndexType iter = 0;
	ValueType delta = 0;
	bool balanced = false;
	const ValueType threshold = 0.002*diagonalLength;//TODO: take global point density into account
	const IndexType maxIterations = settings.maxKMeansIterations;
	do {
		//std::chrono::time_point<std::chrono::high_resolution_clock> iterStart = std::chrono::high_resolution_clock::now();
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

		Settings balanceSettings = settings;
		//balanceEpsilon -= 0.01;
		//balanceSettings.epsilon = std::max( settings.epsilon, balanceEpsilon);
		//balanceSettings.balanceIterations = 0;//iter >= samplingRounds ? settings.balanceIterations : 0;
		//balanceSettings.balanceIterations = std::max(settings.balanceIterations, balanceIter++);
		result = assignBlocks(convertedCoords, centers, firstIndex, lastIndex, nodeWeights, result, adjustedBlockSizes, boundingBox, upperBoundOwnCenter, lowerBoundNextCenter, influence, balanceSettings);
		scai::hmemo::ReadAccess<IndexType> rResult(result.getLocalValues());

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

		std::vector<IndexType> blockWeights(k,0);
		for (auto it = firstIndex; it != lastIndex; it++) {
			const IndexType i = *it;
			IndexType cluster = rResult[i];
			blockWeights[cluster] += rWeight[i];
		}
		{
			SCAI_REGION( "KMeans.computePartition.blockWeightSum" );
			comm->sumImpl(blockWeights.data(), blockWeights.data(), k, scai::common::TypeTraits<IndexType>::stype);
		}

		balanced = true;
		for (IndexType j = 0; j < k; j++) {
			if (blockWeights[j] > blockSizes[j]*(1+settings.epsilon)) {
				balanced = false;
			}
		}

		if (comm->getRank() == 0) {
			std::cout << "i: " << iter << ", delta: " << delta << std::endl;
		}
		iter++;
/*		
		// if only cares about balance and running time, exit loop when balance is reached
		if( settings.repartition and balanced )
			break;
		
		if (settings.verbose) {
			std::chrono::duration<ValueType,std::ratio<1>> iterTime = std::chrono::high_resolution_clock::now() - iterStart;			
			ValueType time = iterTime.count() ;
			std::cout<< "\t"<<comm->getRank() <<": " << time << " , balanced: "<< balanced << ", iter= "<< iter<< " , samplingRounds: " << samplingRounds;

			std::cout << " , while cond: "<< (iter < samplingRounds || (iter < maxIterations && (delta > threshold || !balanced)) )  << std::endl;
		}
*/		
	} while (iter < samplingRounds or (iter < maxIterations && (delta > threshold || !balanced)) );

	return result;
}

}
} /* namespace ITI */
