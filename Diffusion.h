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
#include <scai/lama/matrix/DenseMatrix.hpp>

namespace ITI {

/**
 * maybe have free functions instead of a class with static functions? - Not yet, as all other code still uses static functions.
 */
template<typename IndexType, typename ValueType>
class Diffusion {

public:
	Diffusion() = default;
	virtual ~Diffusion() = default;
	static scai::lama::DenseVector<ValueType> potentialsFromSource(scai::lama::CSRSparseMatrix<ValueType> laplacian, scai::lama::DenseVector<ValueType> nodeWeights, IndexType source, ValueType eps=1e-6);
	static scai::lama::DenseMatrix<ValueType> multiplePotentials(scai::lama::CSRSparseMatrix<ValueType> laplacian, scai::lama::DenseVector<ValueType> nodeWeights, std::vector<IndexType> sources, ValueType eps=1e-6);
	static scai::lama::CSRSparseMatrix<ValueType> constructLaplacian(scai::lama::CSRSparseMatrix<ValueType>);

	//taken from https://stackoverflow.com/a/9345144/494085
	template<class BidiIter >
	static BidiIter FisherYatesShuffle(BidiIter begin, BidiIter end, size_t num_random) {
	    size_t left = std::distance(begin, end);
	    for (IndexType i = 0; i < num_random; i++) {
	        BidiIter r = begin;
	        std::advance(r, rand()%left);
	        std::swap(*begin, *r);
	        ++begin;
	        --left;
	    }
	    return begin;
	}

	static scai::lama::CSRSparseMatrix<ValueType> constructFJLTMatrix(ValueType epsilon, IndexType n, IndexType origDimension);
	static scai::lama::DenseMatrix<ValueType> constructHadamardMatrix(IndexType d);
};

} /* namespace ITI */
#endif /* DIFFUSION_H_ */
