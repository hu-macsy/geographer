/*
 * SpectralPartition
 *
 *  Created on: 15.03.17
 *      Author: tzovas
 */

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
#include "MultiLevel.h"
#include "Settings.h"

namespace ITI {

	/** @brief Partition a graph using the spectral partitioning method.
	@warning This class is heavily unstable and not further developed.
	*/

    template <typename IndexType, typename ValueType>
    class SpectralPartition {
        public:
            /** Returns a spectral partition of the input graph
             * @param[in] adjM The adjacency matrix of the input graph to partition.
             * @param[in] coordinates Node positions
             */
            static scai::lama::DenseVector<IndexType> getPartition(const CSRSparseMatrix<ValueType> &adjM, const std::vector<DenseVector<ValueType>> &coordinates, Settings settings);
            
            /**Main program to determine the Fiedler vector for a Laplacian matrix
             * 
             *  Method: Inverse power method incorporated with Householder deflation
             *
             *  Paper: An efficient and accurate method to compute the Fiedler vector based on 
             *         Householder deflation and inverse power iteration
             *         Jian-ping Wu, Jun-qiang Song, Wei-min Zhang 
             * @param[in] adjM The adjacency matrix of the input graph to get the Fiedler eigenvector.
             * @param[out] eigenvalue The second smallest eigenvalue that corresponds to the fiedler vector.
             * @return The Fiedler eigenvector, aka the vector corresponding to the second smallest eigenvalue of adjM.
             */
            static scai::lama::DenseVector<ValueType> getFiedlerVector(const scai::lama::CSRSparseMatrix<ValueType>& adjM,
                ValueType& eigenvalue );
    };
    
}
