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


template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> Diffusion<IndexType, ValueType>::constructLaplacian(CSRSparseMatrix<ValueType> graph, bool weighted) {
	using scai::lama::CSRStorage;
	using scai::hmemo::HArray;
	using std::vector;

	const IndexType globalN = graph.getNumRows();
	const IndexType localN = graph.getLocalNumRows();

	if (graph.getNumColumns() != globalN) {
		throw std::runtime_error("Matrix must be square to be an adjacency matrix");
	}

	scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

    if (dist->getBlockDistributionSize() == nIndex) {
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
			targetDegree[i] += weighted ? values[j] : 1;
		}
	}

	CSRSparseMatrix<ValueType> D(dist, noDist);
	//in the diagonal matrix, each node has one loop
	scai::hmemo::HArray<IndexType> dIA(localN+1, IndexType(0));
    scai::utilskernel::HArrayUtils::setSequence(dIA, IndexType(0), IndexType(1), dIA.size());
    //... to itself
    scai::hmemo::HArray<IndexType> dJA(localN, IndexType(0));
    scai::utilskernel::HArrayUtils::setSequence(dJA, firstIndex, IndexType(1), dJA.size());
    // with the degree as value
    scai::hmemo::HArray<ValueType> dValues(localN, targetDegree.data());

    CSRStorage<ValueType> dStorage(localN, globalN, localN, dIA, dJA, dValues );

    D.swapLocalStorage(dStorage);

	CSRSparseMatrix<ValueType> result(D-graph);
	assert(result.getNumValues() == graph.getNumValues() + globalN);

	return graph;
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
		DIASparseMatrix<ValueType> D(DIAStorage<ValueType>(origDimension, origDimension, IndexType(1), HArray<IndexType>(IndexType(1), IndexType(0) ), HArray<ValueType>(origDimension, IndexType(1) )));
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

	DIASparseMatrix<ValueType> D(origDimension,origDimension);
	HArray<ValueType> randomDiagonal(origDimension);
	{
		WriteAccess<ValueType> wDiagonal(randomDiagonal);
		//the following can definitely be optimized
		for (IndexType i = 0; i < origDimension; i++) {
			wDiagonal[i] = 1-2*(rand() ^ 1);
		}
	}
	DIAStorage<ValueType> dstor(origDimension, origDimension, IndexType(1), HArray<IndexType>(IndexType(1), IndexType(0) ), randomDiagonal );
	D.swapLocalStorage(dstor);
	DenseMatrix<ValueType> Ddense(D);

	DenseMatrix<ValueType> PH(P*H);
	DenseMatrix<ValueType> denseTemp(PH*Ddense);
	return CSRSparseMatrix<ValueType>(denseTemp);
}

template<typename IndexType, typename ValueType>
DenseMatrix<ValueType> Diffusion<IndexType, ValueType>::constructHadamardMatrix(IndexType d) {
	const ValueType scalingFactor = 1/sqrt(d);
	DenseMatrix<ValueType> result(d,d);
	WriteAccess<ValueType> wResult(result.getLocalStorage().getData());
	for (IndexType i = 0; i < d; i++) {
		for (IndexType j = 0; j < d; j++) {
			IndexType dotProduct = (i-1) ^ (j-1);
			IndexType entry = 1-2*(dotProduct & 1);//(-1)^{dotProduct}
			wResult[i*d+j] = scalingFactor*entry;
		}
	}
	return result;
}

template class Diffusion<IndexType, ValueType>;

} /* namespace ITI */
