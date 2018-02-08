/*
 * Diffusion.cpp
 *
 *  Created on: 26.06.2017
 *      Author: moritzl
 */
#include <assert.h>
#include <vector>
#include <random>

#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>
#include <scai/solver.hpp>

#include "Diffusion.h"

namespace ITI {

using scai::lama::DenseVector;
using scai::lama::CSRSparseMatrix;
using scai::lama::DIASparseMatrix;
using scai::lama::DIAStorage;
using scai::lama::DenseMatrix;
using scai::lama::DenseStorage;
using scai::hmemo::ReadAccess;
using scai::hmemo::WriteAccess;

template<typename IndexType, typename ValueType>
DenseVector<ValueType> Diffusion<IndexType, ValueType>::potentialsFromSource( CSRSparseMatrix<ValueType> laplacian, DenseVector<ValueType> nodeWeights, IndexType source, ValueType eps) {
	using scai::lama::NormPtr;
	using scai::lama::L2Norm;
	using scai::lama::fill;
	using scai::lama::eval;
	using namespace scai::solver;

	const IndexType n = laplacian.getNumRows();
	if (laplacian.getNumColumns() != n) {
		throw std::runtime_error("Matrix must be symmetric to be a Laplacian");
	}

	scai::dmemo::DistributionPtr dist(laplacian.getRowDistributionPtr());
	laplacian.redistribute(dist, dist);

	//making sure that the source is the same on all processors
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	IndexType sourceSum = comm->sum(source);
	assert(sourceSum == source*comm->getSize());

	ValueType weightSum = nodeWeights.sum();

	IndexType sourceWeight;

	IndexType sourceIndex = dist->global2local(source);

	auto nullVector = fill<DenseVector<ValueType>>(dist,0);
	auto d = eval<DenseVector<ValueType>>(nullVector - nodeWeights);

	if (sourceIndex != scai::invalidIndex) {
		d.getLocalValues()[sourceIndex] = weightSum - nodeWeights.getLocalValues()[sourceIndex];
	}

	ValueType newWeightSum = d.sum();
	if (std::abs(newWeightSum) >= eps) {
		throw std::logic_error("Residual weight sum " + std::to_string(newWeightSum) + " too large!");
	}

	auto solution = fill<DenseVector<ValueType>>( dist, 0.0 );

        auto norm = std::make_shared<L2Norm<ValueType>>();

        auto rt = std::make_shared<ResidualThreshold<ValueType>>( norm, eps, ResidualCheck::Relative );

        auto logger = std::make_shared<CommonLogger>( "myLogger: ",
                                                      LogLevel::convergenceHistory,
 	                                              LoggerWriteBehaviour::toConsoleOnly );

	CG<ValueType> solver( "simpleCG" );

	//solver.setLogger( logger );

	solver.setStoppingCriterion( rt );

	solver.initialize( laplacian );
	solver.solve( solution, d );

	return solution;
}

template<typename IndexType, typename ValueType>
DenseMatrix<ValueType> Diffusion<IndexType, ValueType>::multiplePotentials(scai::lama::CSRSparseMatrix<ValueType> laplacian, scai::lama::DenseVector<ValueType> nodeWeights, std::vector<IndexType> sources, ValueType eps) {
	using scai::hmemo::HArray;

	const IndexType l = sources.size();
	const IndexType n = laplacian.getNumRows();
	const IndexType localN = laplacian.getLocalNumRows();
	HArray<ValueType> resultContainer(localN*l);
	IndexType offset = 0;

	scai::dmemo::DistributionPtr dist(laplacian.getRowDistributionPtr());
    scai::dmemo::DistributionPtr lDist(new scai::dmemo::NoDistribution(l));

	//get potentials and copy them into common vector
	for (IndexType landmark : sources) {
		DenseVector<ValueType> potentials = potentialsFromSource(laplacian, nodeWeights, landmark, eps);
		assert(potentials.size() == n);
		WriteAccess<ValueType> wResult(resultContainer);
		ReadAccess<ValueType> rPotentials(potentials.getLocalValues());
		assert(rPotentials.size() == localN);
		assert(offset < wResult.size());
		std::copy(rPotentials.get(), rPotentials.get()+localN, wResult.get()+offset);
		offset += localN;
	}
	assert(offset == localN*l);

	//the matrix is transposed, not sure if this is a problem.
	return scai::lama::distribute<DenseMatrix<ValueType>>(DenseStorage<ValueType>(l, localN, resultContainer), lDist, dist);
}


template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> Diffusion<IndexType, ValueType>::constructLaplacian(CSRSparseMatrix<ValueType> graph) {
	using scai::lama::CSRStorage;
	using scai::hmemo::HArray;
	using std::vector;

	const IndexType n = graph.getNumRows();
	const IndexType localN = graph.getLocalNumRows();

	if (graph.getNumColumns() != n) {
		throw std::runtime_error("Matrix must be square to be an adjacency matrix");
	}

	scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));

    if (dist->getBlockDistributionSize() == scai::invalidIndex) {
    	throw std::runtime_error("Only replicated or block distributions supported.");
    }

    assert(dist->getBlockDistributionSize() == localN);
    const IndexType firstIndex = dist->local2global(0);

	const CSRStorage<ValueType>& storage = graph.getLocalStorage();
	const ReadAccess<IndexType> ia(storage.getIA());
	const ReadAccess<IndexType> ja(storage.getJA());
	const ReadAccess<ValueType> values(storage.getValues());
	assert(ia.size() == localN+1);

	vector<ValueType> targetDegree(localN,0);
	for (IndexType i = 0; i < localN; i++) {
		const IndexType globalI = dist->local2global(i);
		for (IndexType j = ia[i]; j < ia[i+1]; j++) {
			if (ja[j] == globalI) {
				throw std::runtime_error("No self loops allowed.");
			}
			targetDegree[i] += values[j];
		}
	}

	DIAStorage<ValueType> dstor(localN, n, HArray<IndexType>( { firstIndex } ), HArray<ValueType>(localN, targetDegree.data()) );
	DIASparseMatrix<ValueType>D(dist, std::move( dstor ));
	auto result = scai::lama::eval<CSRSparseMatrix<ValueType>>(D-graph);
	assert(result.getNumValues() == graph.getNumValues() + n);

	return result;
}

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> Diffusion<IndexType, ValueType>::constructFJLTMatrix(ValueType epsilon, IndexType n, IndexType origDimension) {
	using scai::hmemo::HArray;

	const IndexType magicConstant = 0.1;
	const ValueType logn = std::log(n);
	const IndexType targetDimension = magicConstant * std::pow(epsilon, -2)*logn;

	if (origDimension <= targetDimension) {
		//better to just return the identity
		std::cout << "Target dimension " << targetDimension << " is higher than original dimension " << origDimension << ". Returning identity instead." << std::endl;
		DIASparseMatrix<ValueType> D(DIAStorage<ValueType>(origDimension, origDimension, HArray<IndexType>( { 0 } ), HArray<ValueType>(origDimension, ValueType(1) )));
		return CSRSparseMatrix<ValueType>(D);
	}

	const IndexType p = 2;
	ValueType q = std::min((std::pow(epsilon, p-2)*std::pow(logn,p))/origDimension, 1.0);

	std::default_random_engine generator;
	std::normal_distribution<ValueType> gauss(0,q);
	std::uniform_real_distribution<ValueType> unit_interval(0.0,1.0);

	DenseMatrix<ValueType> P(targetDimension, origDimension);
	{
		WriteAccess<ValueType> wP(P.getLocalStorage().getData());
		for (IndexType i = 0; i < targetDimension*origDimension; i++) {
			if (unit_interval(generator) < q) {
				wP[i] = gauss(generator);
			} else {
				wP[i] = 0;
			}
		}
	}

	DenseMatrix<ValueType> H = constructHadamardMatrix(origDimension);

	HArray<ValueType> randomDiagonal(origDimension);
	{
		WriteAccess<ValueType> wDiagonal(randomDiagonal);
		//the following can definitely be optimized
		for (IndexType i = 0; i < origDimension; i++) {
			wDiagonal[i] = 1-2*(rand() ^ 1);
		}
	}
	DIAStorage<ValueType> dstor(origDimension, origDimension, HArray<IndexType>( { 0 } ), randomDiagonal );
	DIASparseMatrix<ValueType>D( std::move( dstor ) );
	DenseMatrix<ValueType> Ddense(D);

	auto PH = scai::lama::eval<DenseMatrix<ValueType>>(P*H);
	auto denseTemp = scai::lama::eval<DenseMatrix<ValueType>>(PH*Ddense);
	return CSRSparseMatrix<ValueType>(denseTemp);
}

template<typename IndexType, typename ValueType>
DenseMatrix<ValueType> Diffusion<IndexType, ValueType>::constructHadamardMatrix(IndexType d) {
	const ValueType scalingFactor = 1/sqrt(d);
        scai::hmemo::HArray<ValueType> result(d * d);
	WriteAccess<ValueType> wResult( result );
	for (IndexType i = 0; i < d; i++) {
		for (IndexType j = 0; j < d; j++) {
			IndexType dotProduct = (i-1) ^ (j-1);
			IndexType entry = 1-2*(dotProduct & 1);//(-1)^{dotProduct}
			wResult[i*d+j] = scalingFactor*entry;
		}
	}
	return DenseMatrix<ValueType>( DenseStorage<ValueType>( d, d, std::move(result) ) );
}

template class Diffusion<IndexType, ValueType>;

} /* namespace ITI */
