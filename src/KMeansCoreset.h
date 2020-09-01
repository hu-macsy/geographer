#pragma once

#include <scai/lama/DenseVector.hpp>

#include "Settings.h"
#include "Metrics.h"


namespace ITI {

/** @brief Class for KMeansCoreset computation. A class is kind of unnecessary and overcomplicating stuff here, but I am trying to be consistent with all the other parts
*/

template <typename IndexType, typename ValueType>
class KMeansCorset {
public:
	/**
	 * Return type for coreset computation. Basically a simple pair of coordinates and nodeWeights,
	 * but as a separate type for better naming
	**/
	struct result_type
	{
		std::vector<scai::lama::DenseVector<ValueType>> coordinates;
		std::vector<scai::lama::DenseVector<ValueType>> nodeWeights;
	};
	
    /**
	 * Computes a coreset for use with kmeans
	 * @param coordinates Points
	 * @param nodeWeights weights (so only one is accepted at this time)
	 * @return A smaller coordinates and nodeWeights vector for use with kmeans
     */
    static result_type computeCoreset(
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        Settings settings);
	
	
	/**
	 * I could not think of a good name for this unfortuantely. Takes a coreset,
	 * the result of applying kmeans to it and the original points. Distributes
	 * the original pointset to partitions based on the coreset centers
	 * 
	 * @param coreset result of calling computeCoreset before kmeans
	 * @param kmeans_result result of computing kmeans on the coreset
	 * @param original_coordinates The coordinates on which computeCoreset was called and which are to be distributed to the coresets blocks
	 * @param original_weights The weights on which computeCoreset was called and which are to be distributed to the coresets blocks
	 * @param settings Same settings that were passed to computeCoreset
	 * 
	 * @return A partitioning of the original pointset based on the coreset results 
	**/
	static scai::lama::DenseVector<IndexType> partitionResult(
		const result_type& coreset,
		const scai::lama::DenseVector<IndexType>& kmeans_result,
		const std::vector<scai::lama::DenseVector<ValueType>>& original_coordinates,
		const std::vector<scai::lama::DenseVector<ValueType>>& original_weights,
		Settings settings
	);
    
    /** Main entry point: given the input, compute a partition by using coresets internally to get a first partition fast.
     * \sa KMeans::computePartition()
     * 
     **/
    static scai::lama::DenseVector<IndexType> computePartition(
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const std::vector<std::vector<ValueType>> &blockSizes,
        //const DenseVector<IndexType>& prevPartition,
        //std::vector<std::vector< std::vector<ValueType> >> centers,
        const Settings settings,
        Metrics<ValueType>& metrics
    );
	
}; //KMeansCoreset
}  //ITI
