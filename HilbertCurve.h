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

#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <queue>

#include "Settings.h"

#include "RBC/Sort/SQuick.hpp"

#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

using scai::lama::DenseVector;

namespace ITI {
	template <typename IndexType, typename ValueType>
	class HilbertCurve {
		public:
			/* Wrapper function that calls either the 2D or 3D hilbert curve depending on dimensions.
			* */
			static ValueType getHilbertIndex(ValueType const * point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);

			/* Gets a vector of coordinates (either 2D or 3D) as input and returns a vector with the
			 * hilbert indices for all coordinates.
			 */
			static std::vector<ValueType> getHilbertIndexVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth, const IndexType dimensions);
			
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
			static ValueType getHilbertIndex2D(ValueType const * point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);
		
			/* Gets a vector of coordinates (either 2D or 3D) as input and returns a vector with the
			 * hilbert indices for all coordinates.
			 */
			static std::vector<ValueType> getHilbertIndex2DVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth);
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
			static ValueType getHilbertIndex3D(ValueType const * point, IndexType dimensions, IndexType recursionDepth, const std::vector<ValueType> &minCoords, const std::vector<ValueType> &maxCoords);

			/* Gets a vector of coordinates (either 2D or 3D) as input and returns a vector with the
			 * hilbert indices for all coordinates.
			 */
			static std::vector<ValueType> getHilbertIndex3DVector (const std::vector<DenseVector<ValueType>> &coordinates, IndexType recursionDepth);
			
			/*	Wrapper to to get a hilbert index from a point as a vector based on its dimension.
			 */
			static std::vector<ValueType> HilbertIndex2PointVec(ValueType index, IndexType level, IndexType dimensions);
			
			/**
			* Given an index between 0 and 1 returns a point in 2 dimensions along the hilbert curve based on
			* the recursion depth. Mostly for test reasons.
			* @param[in] index The index in the hilbert curve, a number in [0,1].
			* @param[in] recursionDepth The number of refinement levels the hilbert curve should have
			*
			* @return A point in the unit square [0,1]^2.
			*/
			static std::vector<ValueType> Hilbert2DIndex2Point(ValueType index, IndexType recursionDepth);

			static std::vector<ValueType> Hilbert2DIndex2PointVec(ValueType index, IndexType recursionDepth);
			
			
			/**
			* Given an index between 0 and 1 returns a point in 3 dimensions along the hilbert curve based on
			* the recursion depth. Mostly for test reasons.
			* @param[in] index The index in the hilbert curve, a number in [0,1].
			* @param[in] recursionDepth The number of refinement levels the hilbert curve should have
			*
			* @return A point in the unit cube [0,1]^3
			*/
			static std::vector<ValueType> Hilbert3DIndex2Point(ValueType index, IndexType recursionDepth);
			
			static std::vector<ValueType> Hilbert3DIndex2PointVec(ValueType index, IndexType recursionDepth);
			
			/*Get the hilbert indices sorted. Every PE will own its part of the hilbert indices
			 */			
			static std::vector<sort_pair> getSortedHilbertIndices( const std::vector<DenseVector<ValueType>> &coordinates);			

	};
}//namespace ITI
