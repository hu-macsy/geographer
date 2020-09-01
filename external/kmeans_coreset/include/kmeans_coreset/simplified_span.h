#ifndef KMEANS_CORESET_SIMPLIFIED_SPAN_H
#define KMEANS_CORESET_SIMPLIFIED_SPAN_H

#include <cstddef>

namespace kmeans_coreset
{
	/**
	 * For lack of C++20 or GSL
	**/
	template <typename element_type>
	class simplified_span
	{
		public:
		simplified_span(element_type* start, std::size_t size):
			start_{start},
			size_{size}
		{}
		
		constexpr auto& operator[](std::size_t idx) noexcept { return start_[idx]; }
		constexpr const auto& operator[](std::size_t idx) const noexcept { return start_[idx]; }
		
		constexpr auto begin() noexcept { return &start_[0]; }
		constexpr auto begin() const noexcept { return &start_[0]; }
		
		constexpr auto end() noexcept { return &start_[size_]; }
		constexpr auto end() const noexcept { return &start_[size_]; }
		
		constexpr auto size() const { return size_; }
		
		private:
		element_type* start_;
		std::size_t size_;
	};
}

#endif
