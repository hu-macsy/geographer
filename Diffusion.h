/*
 * Diffusion.h
 *
 *  Created on: 26.06.2017
 *      Author: moritzl
 */

#ifndef DIFFUSION_H_
#define DIFFUSION_H_

#include <scai/lama.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/matrix/CSRSparseMatrix.hpp>

namespace ITI {

/**
 * maybe have free functions instead of a class with static members?
 */
template<typename IndexType, typename ValueType>
class Diffusion {

public:
	Diffusion() = default;
	virtual ~Diffusion() = default;
	static scai::lama::DenseVector<ValueType> potentials(scai::lama::CSRSparseMatrix<ValueType> graph, scai::lama::DenseVector<IndexType> nodeWeights, IndexType source);
	static scai::lama::CSRSparseMatrix<ValueType> constructLaplacian(scai::lama::CSRSparseMatrix<ValueType>);
};

} /* namespace ITI */
#endif /* DIFFUSION_H_ */
