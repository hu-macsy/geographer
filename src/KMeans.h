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
#include "CommTree.h"

using scai::lama::DenseVector;



namespace ITI {
namespace KMeans {

//TODO: any other more proper way to do this?
//typedef typename CommTree<IndexType,ValueType>::commNode cNode;

//to make it more readable
using point = std::vector<ValueType>;


/**
 * @brief Partition a point set using balanced k-means.
 *
 * This is the main function, others with the same name are wrappers for this one.
 *
 * @param[in] coordinates first level index specifies dimension, second level index the point id
 * @param[in] nodeWeights
 * @param[in] blockSizes target block sizes, not maximum sizes
 * @param[in] prevPartition This is used for the hierarchical version, it is the partition from the previous hierarchy level.
 * @param[in] centers initial k-means centers
 * @param[in] settings Settings struct
 *
 * @return partition
 */

//core implementation
 template<typename IndexType, typename ValueType>
 DenseVector<IndexType> computePartition(
 	const std::vector<DenseVector<ValueType>> &coordinates, \
 	const std::vector<DenseVector<ValueType>> &nodeWeights, \
 	const std::vector<std::vector<ValueType>> &blockSizes, \
 	const DenseVector<IndexType>& prevPartition,\
 	std::vector<std::vector<point>> centers, \
 	const Settings settings, \
 	struct Metrics &metrics);

//TODO: graph is not needed, this is only for debugging
//TODO/WARNING: here is percentage, elsewhere is block weight
 template<typename IndexType, typename ValueType>
 DenseVector<IndexType> computePartition(
 	const CSRSparseMatrix<ValueType> &graph, \
 	const std::vector<DenseVector<ValueType>> &coordinates, \
 	const std::vector<DenseVector<ValueType>> &nodeWeights, \
 	const std::vector<std::vector<ValueType>> &blockSizes, \
 	const DenseVector<IndexType>& prevPartition,\
 	std::vector<std::vector<point>> centers, \
 	const Settings settings, \
 	struct Metrics &metrics);

//minimal wrapper
template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(
	const std::vector<DenseVector<ValueType>> &coordinates,
	const Settings settings);

/**
 * @brief Partition a point set using balanced k-means
 *
 * Wrapper without initial centers. Calls computePartition with centers derived from a Hilbert Curve
 *
 * @param[in] coordinates first level index specifies dimension, second level index the point id
 * @param[in] nodeWeights
 * @param[in] blockSizes target block sizes, not maximum sizes
 * @param[in] settings Settings struct
 * @param[in] metrics Metrics struct
 *
 * @return partition
 */

//wrapper 1- no centers
//template<typename IndexType, typename ValueType>
 template<typename IndexType, typename ValueType>
DenseVector<IndexType> computePartition(
	const std::vector<DenseVector<ValueType>> &coordinates,
	const std::vector<DenseVector<ValueType>> &nodeWeights,
	const std::vector<std::vector<ValueType>> &blockSizes,
	const Settings settings,
	struct Metrics &metrics);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeHierarchicalPartition(
	CSRSparseMatrix<ValueType> &graph, //TODO: only for debugging
	std::vector<DenseVector<ValueType>> &coordinates,
	std::vector<DenseVector<ValueType>> &nodeWeights,
	const CommTree<IndexType,ValueType> &commTree,
	Settings settings,
	struct Metrics& metrics);

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
DenseVector<IndexType> computeRepartition(
	const std::vector<DenseVector<ValueType>> &coordinates,
	const std::vector<DenseVector<ValueType>> &nodeWeights,
	const std::vector<std::vector<ValueType>> &blockSizes,
	const DenseVector<IndexType> &previous,
	const Settings settings);

template<typename IndexType, typename ValueType>
DenseVector<IndexType> computeRepartition(
	const std::vector<DenseVector<ValueType>> &coordinates,
	const std::vector<DenseVector<ValueType>> &nodeWeights,
	const Settings settings,
	struct Metrics& metrics);



/**
	@brief Version for hierarchical version. The centers now are a vector of vectors,
	a set o centers for every block/center in the previous hierarchy level.
	We need two hierarchy levels to partition. If we only use the top one,
	e.g., start from the root, we know the number of children but we do not
	know the memory ans speeds. If we use the one below, e.g., start from
	level 1, we know the number of blocks and the properties for each block
	but we do not know where block belong to.

	@param [in] prevHierarLevel The previous hierarch level. 
	//@param[in] centers Centers from previous partition: centers[i] are the centers 
	for block i in the previous hierarchy level.
	@param[in] partition The block id of every point in the previous hierarchy.
	partition[i]=b means that point i was in block b in the previous hierarchy
	level. 
*/
template<typename IndexType, typename ValueType>
std::vector<std::vector<point>> findInitialCentersSFC(
		const std::vector<DenseVector<ValueType> >& coordinates, 
		const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords,
		const scai::lama::DenseVector<IndexType> &partition,
		const std::vector<cNode> hierLevel,	
		Settings settings);


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
std::vector<std::vector<ValueType>>  findInitialCentersSFC(
		const std::vector<DenseVector<ValueType> >& coordinates,
		const std::vector<ValueType> &minCoords,
		const std::vector<ValueType> &maxCoords,
		Settings settings);

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
 * @param[in] partition an already know partition of the points
 * @param[in] k number of blocks
 * @param[in] firstIndex begin of local node indices
 * @param[in] lastIndex end of local node indices
 * @param[in] nodeWeights node weights
 *
 * @return coordinates of centers
 */
template<typename IndexType, typename ValueType, typename Iterator>
std::vector<point> findCenters(
	const std::vector<DenseVector<ValueType>> &coordinates, 
	const DenseVector<IndexType> &partition, 
	const IndexType k,
	const Iterator firstIndex,
	const Iterator lastIndex,
	const DenseVector<ValueType> &nodeWeights);

/**
 * Computes the weighted distance between a vertex and a cluster, given the geometric distance and the weights and influence values.
 *
 * @param[in] distance
 * @param[in] nodeWeights
 * @param[in] influence
 * @param[in] vertex
 * @param[in] cluster
 */
template<typename IndexType, typename ValueType>
ValueType computeEffectiveDistance(
	const ValueType distance,
	const std::vector<DenseVector<ValueType>> &nodeWeights,
	const std::vector<std::vector<ValueType>> &influence,
	const IndexType vertex,
	const IndexType cluster);

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
 * @param[in] oldBlock The block from the previous hierarchy that every point
 belongs to. In case of the non-hierarchical version, this is 0 for all points. This is different from previousAssignment 
 because it does not chacge inbetween kmeans iteration while
 previousAssignement changes until it converges and the
 algorithm stops.
 * @param[in] blockSizesPerCent A value indicating a percentage per block of
 the points weight. If, W is the sum of weights of all the points, then
 for block i, its weight (sum of the weight of points in the block) must
 be at most (or near) blockSizesPerCent[i]*W.
 * @param[in] boundingBox min and max coordinates of local points, used to compute distance bounds
 * @param[in,out] upperBoundOwnCenter for each point, an upper bound of the effective distance to its own center
 * @param[in,out] lowerBoundNextCenter for each point, a lower bound of the effective distance to the next-closest center
 * @param[in,out] influence a multiplier for each block to compute the effective distance
 * @param[in] settings
 *
 * @return assignment of points to blocks
 */
template<typename IndexType, typename ValueType, typename Iterator>
DenseVector<IndexType> assignBlocks(
	const std::vector<std::vector<ValueType>> &coordinates,
	//const std::vector<std::vector<point>> &centers,
	const std::vector<point>& centers,
	const std::vector<IndexType>& blockSizesPrefixSum,
	const Iterator firstIndex,
	const Iterator lastIndex,
	const std::vector<std::vector<ValueType>> &nodeWeights, 
	const DenseVector<IndexType> &previousAssignment,
	const DenseVector<IndexType> &oldBlocks,
	const std::vector<std::vector<ValueType>> &targetBlockWeights,
	const SpatialCell &boundingBox,
	std::vector<ValueType> &upperBoundOwnCenter,
	std::vector<ValueType> &lowerBoundNextCenter,
	std::vector<std::vector<ValueType>> &influence,
	std::vector<ValueType> &imbalance,
	Settings settings,
	Metrics &metrics);


/**
 * @brief Get local minimum and maximum coordinates
 * TODO: This isn't used any more! Remove?
 * Update, 27/11/8: start reusing 
 */
template<typename ValueType>
std::pair<std::vector<ValueType>, std::vector<ValueType> > getLocalMinMaxCoords(const std::vector<DenseVector<ValueType>> &coordinates);

/** Reverse the order of the vectors: given a 2D vector of size 
dimension*numPoints, reverse it and retunr a vector of points
of size numPoints; in other words, the returned vector has size
numPoints*dimensions. In general, if the given 2D vector has size
A*B, the returned vector has size B*A.
*/
template<typename IndexType, typename ValueType>
std::vector<point> vectorTranspose( const std::vector<std::vector<ValueType>>& points);

} /* namespace KMeans */
} /* namespace ITI */
