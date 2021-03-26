#include "KMeansCoreset.h"

#include "KMeans.h"

#include <kmeans_coreset/coreset.h>

#include <numeric>
#include <random>
#include <vector>

namespace
{
	using namespace kmeans_coreset;
	
	template <typename distance_fun_t>
	auto kmeanspp_select(distance_fun_t distance)
	{
		return [=](std::size_t k, auto& range)
		{
			using point_type=std::decay_t<decltype(*range.begin())>;
			using distance_type=std::decay_t<decltype(distance(std::declval<point_type>(),std::declval<point_type>()))>;
			
			std::mt19937 gen{std::random_device{}()};
			std::uniform_int_distribution<std::size_t> dist(0,range.size()-1);
PRINT( "range.size() "<< range.size()  << " k " << k );
			std::swap(range[0],range[dist(gen)]);
			
			for(std::size_t i=1;i<k;++i)
			{
				std::vector<distance_type> dsquared;
PRINT( "i " << i << " range.size()-i " << range.size()-i  );
				dsquared.resize(range.size()-i);
				
				for(std::size_t j=i;j<range.size();++j)
				{
					auto min_dist=distance(range[0],range[j]);
					for(std::size_t u=1;u<i;++u)
						min_dist=std::min(min_dist,distance(range[u],range[j]));
					dsquared[j-i]=min_dist*min_dist;
				}
				
				std::discrete_distribution<std::size_t> d2dist(std::begin(dsquared),std::end(dsquared));
PRINT("will swap " << i << " with "<< range[i+d2dist(gen)] );
				std::swap(range[i],range[i+d2dist(gen)]);
			}
		};
	};
	
	template <typename distance_fun_t, typename weight_fun_t>
	auto partition_cost(distance_fun_t distance, weight_fun_t weight)
	{
		return [=](const auto& p)
		{
			using point_type=std::decay_t<decltype(*p.begin())>;
			using result_type=std::decay_t<decltype(weight(std::declval<point_type>())*distance(std::declval<point_type>(),std::declval<point_type>()))>;
		
			result_type sum{};
			for(const auto& point: p)
				sum+=weight(point)*distance(point,p.centroid);
			return sum;
		};
	}
	
	template <typename distance_fun_t>
	auto select_target_partition(distance_fun_t distance)
	{
		return [=](const auto& partitions, const auto& p)
		{
			std::size_t min=0;
			auto min_dist=distance(partitions[0].centroid,p);
			for(std::size_t i=1;i<partitions.size();++i)
			{
				auto d=distance(partitions[i].centroid,p);
				if(d<min_dist)
				{
					min_dist=d;
					min=i;
				}
			}
			return min;
		};
	};
	
	template <typename ValueType>
	auto euclidian_distance_squared(const scai::lama::DenseVector<ValueType>& point0, const scai::lama::DenseVector<ValueType>& point1)
	{
		ValueType sum{};
		
		for(std::size_t i=0;i<point0.size();++i)
			sum+=(point0[i]-point1[i])*(point0[i]-point1[i]);
		
		return sum;
	}
	
}

namespace ITI {
	
template<typename IndexType, typename ValueType>
typename KMeansCorset<IndexType, ValueType>::result_type KMeansCorset<IndexType, ValueType>::computeCoreset(
    const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    Settings settings) {
	
    const IndexType localN = coordinates[0].getLocalValues().size();
    const IndexType globalN = coordinates[0].size();
    const IndexType dimensions = coordinates.size();

	std::vector<std::size_t> indices;
	//indices.resize(coordinates.size());
    indices.resize(localN);
	std::iota(std::begin(indices),std::end(indices),0);
	
	const auto distance_fun=[&](std::size_t p0, std::size_t p1)
	{
		return euclidian_distance_squared(coordinates[p0],coordinates[p1]);
	};
	
	//only one weight for now...
	const auto weight_fun=[&](std::size_t idx)
	{
		return nodeWeights[idx][0];
	};
	
	kmeans_coreset::parameters params;
	params.k=settings.numBlocks;
	params.c=settings.coresetC;
	params.d=settings.coresetD;
	params.max_recursion_depth=settings.coresetMaxRecursionDepth;
	params.cost_factor=settings.coresetCostFactor;
	
	auto partitions_result=coreset(indices,select_target_partition(distance_fun), partition_cost(distance_fun,weight_fun),kmeanspp_select(distance_fun),params);
	
	result_type ret_val;
	for(const auto& partition: partitions_result)
	{
		ret_val.coordinates.push_back(coordinates[partition.centroid]);
		
		scai::lama::DenseVector<ValueType> node_sum(nodeWeights[0].size(),0);//I guess that should be ok? All nodes are supposed to have the same number of weights, right?
		for(auto p: partition)
		{
			for(std::size_t i=0;i<node_sum.size();++i)
				node_sum[i]=node_sum[i]+nodeWeights[p][i];
		}
		ret_val.nodeWeights.push_back(node_sum);
	}
	
	return ret_val;
}

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> KMeansCorset<IndexType, ValueType>::partitionResult(
	const result_type& coreset,
	const scai::lama::DenseVector<IndexType>& kmeans_result,
	const std::vector<scai::lama::DenseVector<ValueType>>& original_coordinates,
	const std::vector<scai::lama::DenseVector<ValueType>>& original_weights,
	Settings settings)
{
	const auto k=settings.numBlocks;
	
	std::vector<IndexType> indices;
	indices.resize(coreset.coordinates.size());
	std::iota(std::begin(indices),std::end(indices),0);
	
	const auto centers=KMeans<IndexType,ValueType>::findCenters(coreset.coordinates, kmeans_result, k, std::begin(indices), std::end(indices), coreset.nodeWeights);
	
	/*const auto result=KMeans<IndexType,ValueType>::assignBlocks(?));
		
	return result;
	*/
	
	return {};
	//return KMeans<IndexType,ValueType>::computePartition( original_coordinates, original_weights, blockSizes, prevPartition, centers, settings, metrics );
	
}


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> KMeansCorset<IndexType, ValueType>::computePartition(
    const std::vector<DenseVector<ValueType>> &coordinates,
    const std::vector<DenseVector<ValueType>> &nodeWeights,
    const std::vector<std::vector<ValueType>> &blockSizes,
    //const DenseVector<IndexType>& prevPartition,
    //std::vector<std::vector< std::vector<ValueType> >> centers,
    const Settings settings,
    Metrics<ValueType>& metrics)
{
    //get the coreset from the input data points
    auto coreset=ITI::KMeansCorset<IndexType,ValueType>::computeCoreset(coordinates, nodeWeights, settings);
    //get a kmeans partition of the coreset
    auto coresetKmeans = ITI::KMeans<IndexType,ValueType>::computePartition(coreset.coordinates, coreset.nodeWeights, blockSizes, settings, metrics);
    
    //return ITI::KMeansCorset<IndexType,ValueType>::partitionResult(coreset, coresetKmeans, coordinateCopy, nodeWeightCopy, settings);
    
    //find the centers of the partition using the coreset
    const auto k=settings.numBlocks;
	std::vector<IndexType> indices;
	indices.resize(coreset.coordinates.size());
	std::iota(std::begin(indices),std::end(indices),0);
	
	const std::vector<std::vector<ValueType>> centers = KMeans<IndexType,ValueType>::findCenters(coreset.coordinates, coresetKmeans, k, std::begin(indices), std::end(indices), coreset.nodeWeights);
    
    //use these centers to calculate and return a partition of the original points
    
    // every point belongs to one block in the beginning
    scai::lama::DenseVector<IndexType> intialPartition(coordinates[0].getDistributionPtr(), 0);
    // just one group with all the centers; needed in the hierarchical version
    std::vector<std::vector<std::vector<ValueType>>> groupOfCenters = { centers };
    
    return KMeans<IndexType,ValueType>::computePartition( coordinates, nodeWeights, blockSizes, intialPartition, groupOfCenters, settings, metrics );
}

//to force instantiation (still unconvinced this is a good idea...)
template class KMeansCorset<IndexType, double>;
template class KMeansCorset<IndexType, float>;

}//ITI

