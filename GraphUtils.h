/*
 * GraphUtils.h
 *
 *  Created on: 29.06.2017
 *      Author: moritzl
 */

#ifndef GRAPHUTILS_H_
#define GRAPHUTILS_H_

#include <scai/lama/matrix/CSRSparseMatrix.hpp>

namespace ITI {

namespace GraphUtils {

template<typename IndexType, typename ValueType>
IndexType getFarthestLocalNode(const scai::lama::CSRSparseMatrix<ValueType> graph, std::vector<IndexType> seedNodes);


}

} /* namespace ITI */
#endif /* GRAPHUTILS_H_ */
