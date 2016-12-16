/*
 * ParcoReportHilbert.h
 *
 *  Created on: 15.11.2016
 *      Author: tzovas
 */

#pragma once

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <queue>

using namespace scai::lama;

namespace ITI {
	template <typename IndexType, typename ValueType>
	class HilbertCurve {
		public:
                        /* Wrapper function that calls either the 2D or 3D hilbert curve depending on dimensions.
                         * */
                        static ValueType getHilbertIndex(const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);
                        
			/**
			* Accepts a point and calculates where along the hilbert curve it lies.
			*
			* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		* @param[in] index The index of the points whose hilbert index is desired
	 		* @param[in] recursionDepth The number of refinement levels the hilbert curve should have
	 		* @param[in] minCoords A vector containing the minimal value for each dimension
	 		* @param[in] maxCoords A vector containing the maximal value for each dimension
			*
	 		* @return A value in the unit interval [0,1]
			*/
			static ValueType getHilbertIndex2D(const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);
                        

                        static ValueType getHilbertIndex_noScaling(std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);
		
			/**
			*Accepts a point in 3 dimensions and calculates where along the hilbert curve it lies.
			*
			* @param[in] coordinates Node positions. In d dimensions, coordinates of node v are at v*d ... v*d+(d-1).
	 		* @param[in] dimensions Number of dimensions of coordinates.
	 		* @param[in] index The index of the points whose hilbert index is desired
	 		* @param[in] recursionDepth The number of refinement levels the hilbert curve should have
	 		* @param[in] minCoords A vector containing the minimal value for each dimension
	 		* @param[in] maxCoords A vector containing the maximal value for each dimension
			*
	 		* @return A value in the unit interval [0,1]
			*/
			static ValueType getHilbertIndex3D(const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions, IndexType index, IndexType recursionDepth,
			 const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);

			/**
			* Given an index between 0 and 1 returns a point in 2 dimensions along the hilbert curve based on
			* the recursion depth. Mostly for test reasons.
			* @param[in] index The index in the hilbert curve, a number in [0,1].
			* @param[in] recursionDepth The number of refinement levels the hilbert curve should have
			*
			* @return A point in the unit square [0,1]^2.
			*/
			static DenseVector<ValueType> Hilbert2DIndex2Point(ValueType index, IndexType recursionDepth);

			/**
			* Given an index between 0 and 1 returns a point in 3 dimensions along the hilbert curve based on
			* the recursion depth. Mostly for test reasons.
			* @param[in] index The index in the hilbert curve, a number in [0,1].
			* @param[in] recursionDepth The number of refinement levels the hilbert curve should have
			*
			* @return A point in the unit cube [0,1]^3
			*/
			static DenseVector<ValueType> Hilbert3DIndex2Point(ValueType index, IndexType recursionDepth);

	};
}//namespace ITI
