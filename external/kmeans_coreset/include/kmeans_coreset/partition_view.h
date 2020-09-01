#ifndef KMEANS_CORESET_PARTITION_VIEW_H
#define KMEANS_CORESET_PARTITION_VIEW_H

#include <cstddef>

namespace kmeans_coreset
{
	/**
	 * A view over contiguous points considered one partition of an input set. 
	 * Basically a span + a centroid, but as a special type to convey meaning
	**/
	template <typename point_t>
	struct partition_view
	{
		constexpr point_t& operator[](std::size_t idx) noexcept { return start_[idx]; }
		constexpr const point_t& operator[](std::size_t idx) const noexcept { return start_[idx]; }
		
		constexpr auto begin() noexcept { return &start_[0]; }
		constexpr auto begin() const noexcept { return &start_[0]; }
		
		constexpr auto end() noexcept { return &start_[size_]; }
		constexpr auto end() const noexcept { return &start_[size_]; }
		
		constexpr auto size() const { return size_; }
		
		point_t* start_;
		std::size_t size_;
		
		point_t centroid;
	};
	
	/**
	 * Return type for the distribute function, indicating whether or not the partitioning was changed by redistribution
	**/
	enum class distribution_change
	{
		some,
		none
	};

	/**
	 * Distribute a number of points into a collection of partition_views. This makes a few assumptions:
	 * - partitions are valid ranges (so they may be empty) and all point to the collection points
	 * - points is a contiguous range
	 * - All points in partitions[0] appear before those of partitions[1] before those of partitions[2]...
	 * 
	 * @param partitions 
	 * @param points
	 * @param get_bucket_id
	 * @return distribution_change::none if partitions was unchanged, distribution_change::some otherwise
	**/
	template <typename partition_range_t, typename point_range_t, typename hash_fun_t>
	distribution_change distribute(partition_range_t& partitions, point_range_t& points, hash_fun_t get_bucket_id)
	{
		const auto insert=[&](auto p, std::size_t bucket_id)
		{
			auto& part=partitions[bucket_id];
			auto old_p=part[part.size()];
			part[part.size_++]=p;
			
			//correct subsequent bucktes we stole the element from:
			for(std::size_t next_bucket_id=bucket_id+1;next_bucket_id<partitions.size();++next_bucket_id)
			{
				auto& part=partitions[next_bucket_id];
				++part.start_;
				if(partitions[next_bucket_id].size()>0)
				{
					auto tmp=part[part.size()-1];
					part[part.size()-1]=old_p;
					old_p=tmp;
				}
			}
		};
		
		//we first assume our points are already perfectly ordered, so points of partition0 appear before partition1 before partition2....
		//this first loop checks how far into our range this assumption holds and avoids unneccessarily shifting around those points
		std::size_t offset=0;
		for(std::size_t i=0;i<partitions.size();++i)
		{
			std::size_t j=0;
			for(;j<partitions[i].size();++j,++offset)
			{
				if(get_bucket_id(partitions[i][j])!=i) //could(and should?) cache the result here to avoid calling get_bucket_id twice for the failing one?
					break;
			}
			if(j<partitions[i].size())
			{
				partitions[i].size_=j;
				for(std::size_t u=i+1;u<partitions.size();++u)
				{
					partitions[u].size_=0;
					partitions[u].start_=&points[offset];
				}
				break;
			}
		}
			
		//if not all points were ordered, we have to reinsert all the violating ones
		for(std::size_t i=offset;i<points.size();++i)
			insert(points[i],get_bucket_id(points[i]));
		
		//if all points were ordered correctly, no change happend
		return offset==points.size()?
			distribution_change::none:
			distribution_change::some;
	}
}

#endif
