/*
 * Diffusion.cpp
 *
 *  Created on: 26.06.2017
 *      Author: Moritz v. Looz
 */
#include <assert.h>
#include <vector>
#include <random>

#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>
#include <scai/lama/Scalar.hpp>
#include <scai/solver.hpp>

#include "Diffusion.h"

namespace ITI {

using scai::lama::DenseVector;
using scai::lama::CSRSparseMatrix;
using scai::lama::DIASparseMatrix;
using scai::lama::DIAStorage;
using scai::lama::DenseMatrix;
using scai::lama::DenseStorage;
using scai::lama::Scalar;
using scai::hmemo::ReadAccess;
using scai::hmemo::WriteAccess;

template<typename IndexType, typename ValueType>
DenseVector<ValueType> Diffusion<IndexType, ValueType>::potentialsFromSource( CSRSparseMatrix<ValueType> laplacian, DenseVector<ValueType> nodeWeights, IndexType source, ValueType eps) {
	using scai::lama::NormPtr;
	using scai::lama::L2Norm;
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

	ValueType weightSum = nodeWeights.sum().Scalar::getValue<IndexType>();

	IndexType sourceIndex = dist->global2local(source);

	DenseVector<ValueType> nullVector(dist,0);
	DenseVector<ValueType> d(nullVector - nodeWeights);

	if (sourceIndex != nIndex) {
		d.getLocalValues()[sourceIndex] = weightSum - nodeWeights.getLocalValues()[sourceIndex];
	}

	ValueType newWeightSum = d.sum().Scalar::getValue<IndexType>();
	if (std::abs(newWeightSum) >= eps) {
		throw std::logic_error("Residual weight sum " + std::to_string(newWeightSum) + " too large!");
	}

	DenseVector<ValueType> solution( dist, 0.0 );

	NormPtr norm( new L2Norm() );

	CriterionPtr rt( new ResidualThreshold( norm, eps, ResidualThreshold::Relative ) );

	LoggerPtr logger( new CommonLogger ( "myLogger: ",
	                                        LogLevel::convergenceHistory,
	                                        LoggerWriteBehaviour::toConsoleOnly ) );

	CG solver( "simpleCG" );

	//solver.setLogger( logger );

	solver.setStoppingCriterion( rt );

	solver.initialize( laplacian );
	solver.solve( solution, d );

	return solution;
}

template<typename IndexType, typename ValueType>
DenseMatrix<ValueType> Diffusion<IndexType, ValueType>::multiplePotentials(scai::lama::CSRSparseMatrix<ValueType> laplacian, scai::lama::DenseVector<ValueType> nodeWeights, std::vector<IndexType> sources, ValueType eps) {
	using scai::hmemo::HArray;

    if (!laplacian.getRowDistributionPtr()->isReplicated() or !nodeWeights.getDistributionPtr()->isReplicated()) {
        throw std::logic_error("Should only be called with replicated input.");
    }

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
	return DenseMatrix<ValueType>(DenseStorage<ValueType>(resultContainer, l, localN), lDist, dist);
}

template class Diffusion<IndexType, ValueType>;

} /* namespace ITI */
