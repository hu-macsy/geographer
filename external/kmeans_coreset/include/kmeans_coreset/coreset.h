#ifndef KMEANS_CORESET_SIMPLIFIED_CORESET_H
#define KMEANS_CORESET_SIMPLIFIED_CORESET_H

#include "partition_view.h"
#include "simplified_span.h"

#include <algorithm>
//#include <execution>
#include <vector>

#include <cmath>
#include <cstddef>

namespace kmeans_coreset
{
	/**
	 * Parameters to drive and customize the coreset algorithm
	**/
	struct parameters
	{
		unsigned k = 16; /**< The k parameter for kmeans intended for use with the algorithms result*/
		unsigned c = 10; /**< Initially, the pointset is divided into c*k partitions*/
		unsigned d = 3; /**< Each refinement potentially partitions a given partition further into d*k smaller ones*/
		unsigned max_recursion_depth = 3; /**< The maximum number of times a partition is split up further*/
		double cost_factor = 0.1; /**< A subdivision is considered better if its accumulated cost is smaller than cost_factor*cost_of_the_original_partition*/
	};

	/**
	 * Computes a coreset for use with kmeans according to the algorithm described here: https://arxiv.org/pdf/1807.04518.pdf
	 * 
	 * @param points A collection of points. The alogrithm will not add or remove anything, but the points will be reordered. It must be possible to call begin and end (yielding random access iterators) as well as size on this. The value type (i.e. the points) are assumed to be cheap to copy. If this is not the case users can provide a list of indices or pointers and access the real points in the provided select_target_partition and partition_cost callables.
	 * @param select_target_partition Callable used to determine to which of a given number of partitions a given point is assigned. Must be callable as select_target_partition(partitions,p), with p being the type of *begin(points) and partitions being a partition_view of that same type.
	 * @param partition_cost Callable used to determine the total 'cost' of a given partition. In its simplest form it will simply sum the distanced from its centroid. Must be callable als partition_cost(parition_view<point_type>).
	 * @param sample Callable used to select a sample of points which will be used as centroids for smaller partitions. Must be callable as sample(n,range) and sort the selected n points to the beginning of range.
	 * @param params Parameters used to customize the algorithm
	 * 
	 * @return A vector of partition_view's, each corresponding to one member of the coreset(that is the centroid of that partition_view)
	**/
	template <typename range_t, typename select_target_partition_fun_t, typename partition_cost_fun_t, typename sample_fun_t>
	auto coreset(range_t& points, select_target_partition_fun_t select_target_partition, partition_cost_fun_t partition_cost, sample_fun_t sample, parameters params)
	{
		using std::begin;
		using std::end;
		
		using point_type=std::decay_t<decltype(*begin(points))>;
		using cost_type=std::decay_t<decltype(partition_cost(std::declval<partition_view<point_type>>()))>;
		
		struct partition_with_depth: partition_view<point_type>
		{
			std::size_t depth=0;
		};
		
		const auto partitions_cost=[&](const auto& partitions_range) -> cost_type
		{
			auto sum=cost_type{};
			for(const auto& p:partitions_range)
				sum+=partition_cost(p);
			return sum;
		};
		
		const std::size_t maximum_number_of_partitions=
			std::min(static_cast<std::size_t>(params.k*params.c*std::pow(params.k*params.d,params.max_recursion_depth-1)),points.size());

		std::vector<partition_with_depth> partitions;
		partitions.reserve(maximum_number_of_partitions+1); //1 more to have a working copy?
		
		const auto create_new_partitions=[&](std::size_t n, partition_view<point_type> points, std::size_t depth)
		{
			partitions.resize(partitions.size()+n);
			auto parts=simplified_span<partition_with_depth>{&partitions[partitions.size()-n],n};
			
			sample(n,points);
			
			for(std::size_t i=0;i<n;++i)
			{
				parts[i].centroid=points[i];
				parts[i].start_=&points[0];
				parts[i].size_=0;
				parts[i].depth=depth;
			}
			
			distribute(parts,points,[&](const auto& p)
			{
				return select_target_partition(parts,p);
			});
			return parts;
		};
		
		create_new_partitions(params.k*params.c,{&points[0],points.size()},0);
		
		for(std::size_t i=0;i<partitions.size();++i)
		{
			if(partitions[i].size()<params.d*params.k)
				continue;
			
			if(partitions[i].depth>=params.max_recursion_depth)
				continue;
			
			auto parts=create_new_partitions(params.d*params.k,partitions[i],partitions[i].depth+1);
			
			const auto cost=partition_cost(partitions[i]);
			const auto parts_cost=partitions_cost(parts);
			
			if(parts_cost<params.cost_factor*cost)
			{
				//the order our partitions are handled in does not matter.
				//As such, to erase one it is sufficient to swap it with the last and pop_back. Saved on a lot of shifting around(O(n) vs O(1))
				std::swap(partitions[i],partitions[partitions.size()-1]);
				partitions.pop_back();
				--i;
			}
			else
				partitions.erase(partitions.begin()+partitions.size()-params.d*params.k, partitions.end());

		}
		
		return partitions;
	}
	
}

#endif
