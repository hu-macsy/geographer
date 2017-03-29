/*
 * SpectralPartition
 *
 *  Created on: 15.03.17
 *      Author: tzovas
 */

#include <scai/dmemo/HaloBuilder.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>
#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>
#include <scai/tracing.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <queue>
#include <string>
#include <unordered_set>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <tuple>
#include <chrono>

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "HilbertCurve.h"
#include "MultiLevel.h"

#include "../Eigen/Dense"

namespace ITI {

    template <typename IndexType, typename ValueType>
    class SpectralPartition {
        public:
            /** Returns the degree of every node of the graph.
             * @param[in] adjM The NxN adjacency matrix of the input graph.
             * @return A distributed DenseVector of size N with the degree of every node. The DenseVector
             * has the same distribution as the the rows of adjM.
             */
            static scai::lama::DenseVector<IndexType> getDegreeVector( const scai::lama::CSRSparseMatrix<ValueType>& adjM);
            
            /** Returns a distributed laplacian matrix.
             */
            static scai::lama::CSRSparseMatrix<ValueType> getLaplacian( const scai::lama::CSRSparseMatrix<ValueType>& adjM);
            
            /** Returns a spectral partition of the input graph
             * @param[in] adjM The adjacency matrix of the input graph to partition.
             * @param[in] coordinates Node positions
             */
            static scai::lama::DenseVector<IndexType> getPartition(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
            
            /**Main program to determine the Fiedler vector for a Laplacian matrix
             * 
             *  Method: Inverse power method incorporated with Householder deflation
             *
             *  Paper: An efficient and accurate method to compute the Fiedler vector based on 
             *         Householder deflation and inverse power iteration
             *         Jian-ping Wu, Jun-qiang Song, Wei-min Zhang 
             * @param[in] adjM The adjacency matrix of the input graph to get the Fiedler eigenvector.
             * @return The Fiedler eigenvector, aka the vector corresponding to the second smallest eigenvalue of adjM.
             */
            static scai::lama::DenseVector<ValueType> getFiedlerVector(const scai::lama::CSRSparseMatrix<ValueType>& adjM );
    };
    
}