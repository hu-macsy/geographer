/*
 * KMeans.h
 *
 *  Created on: 19.07.2017
 *      Author: Moritz von Looz
 */

#pragma once

#include <vector>
#include <numeric>
#include <scai/lama/DenseVector.hpp>
#include <scai/tracing.hpp>
#include <chrono>

#include "quadtree/QuadNodeCartesianEuclid.h"
#include "Settings.h"
#include "Metrics.h"
#include "GraphUtils.h"
#include "HilbertCurve.h"
#include "AuxiliaryFunctions.h"

using scai::lama::DenseVector;

namespace ITI {
namespace KMeans {

/**
 * @brief Partition a point set using balanced k-means
 *
 * Wrapper without initial centers. Calls computePartition with centers derived from a Hilbert Curve
 *
 * @param[in] coordinates first level index specifies dimension, second level index the point id
 * @param[in] nodeWeights
 * @param[in] blockSizes target block sizes, not maximum sizes
 * @param[in] settings Settings struct
 *
 * @return partition
 */
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(
	const std::vector<DenseVector<ValueType>> &coordinates, \
	const DenseVector<ValueType> &nodeWeights, \
	const std::vector<IndexType> &blockSizes, \
	const Settings settings);

/**
 * @brief Repartition a point set using balanced k-means.
 *
 * @param[in] coordinates first level index specifies dimension, second level index the point id
 * @param[in] nodeWeights
 * @param[in] blockSizes target block sizes, not maximum sizes
 * @param[in] previous Previous partition
 * @param[in] settings Settings struct
 *
 * @return partition
 */
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeRepartition(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<ValueType> &  nodeWeights,
		const std::vector<IndexType> &blockSizes, const DenseVector<IndexType>& previous, const Settings settings);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeRepartition(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<ValueType> &nodeWeights, const Settings settings, struct Metrics& metrics);


/**
 * @brief Partition a point set using balanced k-means.
 *
 * This is the main function, others with the same name are wrappers for this one.
 *
 * @param[in] coordinates first level index specifies dimension, second level index the point id
 * @param[in] nodeWeights
 * @param[in] blockSizes target block sizes, not maximum sizes
 * @param[in] centers initial k-means centers
 * @param[in] settings Settings struct
 *
 * @return partition
 */

 template<typename IndexType, typename ValueType>
 DenseVector<IndexType> computePartition(
 	const std::vector<DenseVector<ValueType>> &coordinates, \
 	const DenseVector<ValueType> &nodeWeights, \
 	const std::vector<IndexType> &blockSizes, \
 	std::vector<std::vector<ValueType> > centers, \
 	const Settings settings, \
 	struct Metrics& metrics);


/**
 * Find initial centers for k-means by sorting the local points along a space-filling curve.
 * Assumes that points are already divided globally according to their SFC indices, but the local order was changed to have increasing global node IDs,
 * as required by the GeneralDistribution constructor.
 *
 * @param[in] coordinates
 * @param[in] minCoords Minimum coordinate in each dimension, lower left point of bounding box (if in 2D)
 * @param[in] maxCoords Maximum coordinate in each dimension, upper right point of bounding box (if in 2D)
 * @param[in] settings
 *
 * @return coordinates of centers
 */
template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> >  findInitialCentersSFC(
		const std::vector<DenseVector<ValueType> >& coordinates, const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords, Settings settings);

/**
 * @brief Compute initial centers from space-filling curve without considering point positions
 * TODO: Currently assumes that minCoords = vector<ValueType>(settings.dimensions, 0). This is not always true! Fix or remove.
 *
 * @param[in] maxCoords Maximum coordinate in each dimension
 * @param[in] settings
 *
 * @return coordinates of centers
 */
template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType> > findInitialCentersFromSFCOnly(const std::vector<ValueType> &maxCoords, Settings settings);

/**
 * Compute centers based on the assumption that the partition is equal to the distribution.
 * Each process then picks the average of mass of its local points.
 *
 * @param[in] coordinates
 * @param[in] nodeWeights
 *
 * @return coordinates of centers
 */
template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> findLocalCenters(const std::vector<DenseVector<ValueType> >& coordinates, const DenseVector<ValueType> &nodeWeights);

/**
 * Find centers of current partition.
 * To enable random initialization of k-means with a subset of nodes, this function accepts iterators for the first and last local index that should be considered.
 *
 * @param[in] coordinates input points
 * @param[in] partition
 * @param[in] k number of blocks
 * @param[in] firstIndex begin of local node indices
 * @param[in] lastIndex end of local node indices
 * @param[in] nodeWeights node weights
 *
 * @return coordinates of centers
 */
template<typename IndexType, typename ValueType, typename Iterator>
std::vector<std::vector<ValueType> > findCenters(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<IndexType> &partition, const IndexType k,
		const Iterator firstIndex, const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights);

/**
 * Assign points to block with smallest effective distance, adjusted for influence values.
 * Repeatedly adjusts influence values to adhere to the balance constraint given by settings.epsilon
 * To enable random initialization with a subset of the points, this function accepts iterators for the first and last local index that should be considered.
 *
 * The parameters upperBoundOwnCenter, lowerBoundNextCenter and influence are updated during point assignment.
 *
 * In contrast to the paper, the influence value is multiplied with the plain distance to compute the effective distance.
 * Thus, blocks with higher influence values have larger distances to all points and will receive less points in the next iteration.
 * Blocks which too few points get a lower influence value to attract more points in the next iteration.
 *
 * The returned vector has always as many entries as local points, even if only some of them are non-zero.
 *
 * @param[in] coordinates input points
 * @param[in] centers block centers
 * @param[in] firstIndex begin of local node indices
 * @param[in] lastIndex end local node indices
 * @param[in] nodeWeights node weights
 * @param[in] previousAssignment previous assignment of points
 * @param[in] blockSizes target block sizes
 * @param[in] boundingBox min and max coordinates of local points, used to compute distance bounds
 * @param[in,out] upperBoundOwnCenter for each point, an upper bound of the effective distance to its own center
 * @param[in,out] lowerBoundNextCenter for each point, a lower bound of the effective distance to the next-closest center
 * @param[in,out] influence a multiplier for each block to compute the effective distance
 * @param[in] settings
 *
 * @return assignment of points to blocks
 */
template<typename IndexType, typename ValueType, typename Iterator>
DenseVector<IndexType> assignBlocks(const std::vector<std::vector<ValueType>> &coordinates, const std::vector<std::vector<ValueType> > &centers,
		const Iterator firstIndex, const Iterator lastIndex,
		const DenseVector<ValueType> &nodeWeights, const DenseVector<IndexType> &previousAssignment,
		const std::vector<IndexType> &blockSizes,  const SpatialCell &boundingBox,
		std::vector<ValueType> &upperBoundOwnCenter, std::vector<ValueType> &lowerBoundNextCenter,
		std::vector<ValueType> &influence, std::vector<ValueType> &timePerPE,
		Settings settings, Metrics &metrics);

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



template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeRepartition(const std::vector<DenseVector<ValueType>> &coordinates, const DenseVector<ValueType> &  nodeWeights, const std::vector<IndexType> &blockSizes, const DenseVector<IndexType>& previous, const Settings settings) {
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
		
		minCoords[d] = *std::min_element(convertedCoords[d].begin(), convertedCoords[d].end());
		maxCoords[d] = *std::max_element(convertedCoords[d].begin(), convertedCoords[d].end());

		// sort local coordinates based on their HilbertCurve index
		const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(globalN), double(21));
		std::vector<ValueType> localHilbertInd = HilbertCurve<IndexType,ValueType>::getHilbertIndexVector(coordinates, recursionDepth, settings.dimensions);

		SCAI_ASSERT_EQ_ERROR( convertedCoords[0].size(), localHilbertInd.size() , "vector mismatch");

		ValueType sumCoord = std::accumulate( convertedCoords[d].begin(), convertedCoords[d].end(), 0.0);
		
		std::sort(convertedCoords[d].begin(), convertedCoords[d].end(), \
				[&localHilbertInd](ValueType i, ValueType j){ return localHilbertInd[i]<localHilbertInd[j]; });

		ValueType sumCoord2 = std::accumulate( convertedCoords[d].begin(), convertedCoords[d].end(), 0.0);

		SCAI_ASSERT_GE_ERROR( sumCoord, 0.999*sumCoord2, "Error in sorting local coordinates");
		assert(convertedCoords[d].size() == localN);
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

	//


	QuadNodeCartesianEuclid boundingBox(minCoords, maxCoords);
    if (settings.verbose) {
    	/*
        std::cout << "Process " << comm->getRank() << ": ( ";
        for (auto coord : minCoords) std::cout << coord << " ";
        std::cout << ") , ( ";
        for (auto coord : maxCoords) std::cout << coord << " ";
        std::cout << ")";
        std::cout << ", " << localN << " nodes, " << scai::utilskernel::HArrayUtils::sum(nodeWeights.getLocalValues()) << ", " << ((ValueType)localN)/globalN *100<< "%, total weight";
        std::cout << ", volume ratio " << localVolume / (volume / p);
        std::cout << std::endl;
		*/
		std::cout << "(" << comm->getRank() << ", "<< localN << ")" << std::endl;
		comm->synchronize();
		std::cout << "(" << comm->getRank() << ", "<< localVolume / (volume / p) << ")" << std::endl;
    }

	diagonalLength = std::sqrt(diagonalLength);
	const ValueType expectedBlockDiameter = pow(volume / k, 1.0/dim);

	DenseVector<IndexType> result(coordinates[0].getDistributionPtr(), 0);
	std::vector<ValueType> upperBoundOwnCenter(localN, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> lowerBoundNextCenter(localN, 0);

	//prepare sampling
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


	if (randomInitialization) {
		ITI::GraphUtils::FisherYatesShuffle(localIndices.begin(), localIndices.end(), localN);
		//std::vector<IndexType> localIndices = GraphUtils::indexReorderCantor( localN );
/*
IndexType indexSumFY = std::accumulate( localIndices.begin(), localIndices.end(), 0);
IndexType indexSumC = std::accumulate( localIndices2.begin(), localIndices2.end(), 0);
SCAI_ASSERT_EQ_ERROR( indexSumFY, indexSumC, "Erros in index reordering");
*/
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

	//
	aux<IndexType,ValueType>::timeMeasurement(KMeansStart);
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

		result = assignBlocks(convertedCoords, centers, firstIndex, lastIndex, nodeWeights, result, adjustedBlockSizes, boundingBox, upperBoundOwnCenter, lowerBoundNextCenter, influence, timePerPE, settings, metrics);
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

		std::vector<IndexType> blockWeights(k,0);
		for (auto it = firstIndex; it != lastIndex; it++) {
			const IndexType i = *it;
			IndexType cluster = rResult[i];
			blockWeights[cluster] += rWeight[i];
		}

		// print times before global reduce step
		//aux<IndexType,ValueType>::timeMeasurement(iterStart);
		std::chrono::duration<ValueType,std::ratio<1>> balanceTime = std::chrono::high_resolution_clock::now() - iterStart;			
		ValueType time = balanceTime.count() ;
		PRINT(*comm <<": in computePartition, iteration time: " << time );
		//

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

		ValueType maxTime=0;
		if(settings.verbose){
			balanceTime = std::chrono::high_resolution_clock::now() - iterStart;
			maxTime = comm->max( balanceTime.count() );
		}

		imbalance = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( result, settings.numBlocks, nodeWeights );

		if (comm->getRank() == 0) {
			std::cout << "i: " << iter<< ", delta: " << delta << ", time : "<< maxTime << ", imbalance= " << imbalance<< std::endl;
		}

		metrics.kmeansProfiling.push_back( std::make_tuple(delta, maxTime, imbalance) );

		iter++;

		// WARNING-TODO: this stops the iterations prematurely, when the wanted balance
		// is reached. It is possible that if we allow more iterations, the solution
		// will converge to some optima reagaridng the cut/shape. Investigate that
		if(imbalance<settings.epsilon)
			break;

	} while (iter < samplingRounds or (iter < maxIterations && (delta > threshold || !balanced)) ); // or (imbalance>settings.epsilon) );


	std::chrono::duration<ValueType,std::ratio<1>> KMeansTime = std::chrono::high_resolution_clock::now() - KMeansStart;
	ValueType time = comm->max( KMeansTime.count() );
	
	PRINT0("total KMeans time: " << time << " , number of iterations: " << iter );

	return result;
}

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

}
} /* namespace ITI */
