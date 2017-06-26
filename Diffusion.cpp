/*
 * Diffusion.cpp
 *
 *  Created on: 26.06.2017
 *      Author: moritzl
 */
#include <assert.h>
#include <vector>

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
using scai::lama::Scalar;

template<typename IndexType, typename ValueType>
DenseVector<ValueType> Diffusion<IndexType, ValueType>::potentials(CSRSparseMatrix<ValueType> laplacian, DenseVector<IndexType> nodeWeights, IndexType source) {
	using scai::lama::NormPtr;
	using scai::lama::L2Norm;
	using namespace scai::solver;

	const IndexType n = laplacian.getNumRows();
	if (laplacian.getNumColumns() != n) {
		throw std::runtime_error("Matrix must be symmetric to be a Laplacian");
	}

	IndexType weightSum = nodeWeights.sum().Scalar::getValue<IndexType>();

	IndexType sourceWeight;
	if (nodeWeights.getDistributionPtr()->isReplicated()) {
		sourceWeight = nodeWeights.getLocalValues()[source];
	}
	sourceWeight = nodeWeights.getValue(source).Scalar::getValue<IndexType>();

	DenseVector<ValueType> nullVector(n,0);
	DenseVector<ValueType> d(nullVector - nodeWeights);
	d.setValue(source, weightSum - sourceWeight);
	assert(d.sum() == 0);

	DenseVector<ValueType> solution( n, 0.0 );

	const IndexType maxIter = 20;
	const ValueType eps        = 1e-5;
	LoggerPtr logger( new CommonLogger ( "myLogger: ",	LogLevel::convergenceHistory,	LoggerWriteBehaviour::toConsoleOnly ) );

	NormPtr norm( new L2Norm() );

	CriterionPtr rt( new ResidualThreshold( norm, eps, ResidualThreshold::Absolute ) );
	CriterionPtr it( new IterationCount( maxIter ) );
	CriterionPtr both( new Criterion ( it, rt, Criterion::OR ) );

	CG solver( "simpleExampleCG" );

	solver.setLogger( logger );
	solver.setStoppingCriterion( both );

	solver.initialize( laplacian );
	solver.solve( solution, d );

	return solution;
}

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> Diffusion<IndexType, ValueType>::constructLaplacian(CSRSparseMatrix<ValueType> graph) {
	using scai::lama::CSRStorage;
	using scai::hmemo::ReadAccess;
	using scai::hmemo::WriteAccess;
	using scai::hmemo::HArray;
	using std::vector;

	const IndexType n = graph.getNumRows();
	if (graph.getNumColumns() != n) {
		throw std::runtime_error("Matrix must be symmetric to be an adjacency matrix");
	}

	if (!graph.getRowDistribution().isReplicated()) {
		throw std::runtime_error("Input data must be replicated, for now.");
	}

    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));

	const CSRStorage<ValueType>& storage = graph.getLocalStorage();
	const ReadAccess<IndexType> ia(storage.getIA());
	const ReadAccess<IndexType> ja(storage.getJA());
	const ReadAccess<ValueType> values(storage.getValues());
	assert(ia.size() == n+1);

	vector<ValueType> targetDegree(n,0);
	IndexType degreeSum = 0;
	for (IndexType i = 0; i < n; i++) {
		for (IndexType j = ia[i]; j < ia[i+1]; j++) {
			if (ja[j] == i) {
				throw std::runtime_error("No self loops allowed.");
			}
			if (values[j] != 1) {
				throw std::runtime_error("Wrong edge weights.");
			}
		}

		targetDegree[i] = ia[i+1]-ia[i];
		degreeSum += targetDegree[i];
	}
	assert(degreeSum >= storage.getNumValues());
	assert(degreeSum <= storage.getNumValues()+n);

	DIASparseMatrix<ValueType> D(n,n);
	DIAStorage<ValueType> dstor(n, n, 1, HArray<IndexType>(1,0), HArray<ValueType>(n, targetDegree.data()) );
	D.swapLocalStorage(dstor);
	CSRSparseMatrix<ValueType> result(D-graph);

	assert(result.isConsistent());

	return result;
}

template CSRSparseMatrix<double> Diffusion<int, double>::constructLaplacian(CSRSparseMatrix<double> graph);
template DenseVector<double> Diffusion<int, double>::potentials(CSRSparseMatrix<double> laplacian, DenseVector<int> nodeWeights, int source);


} /* namespace ITI */
