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
DenseVector<ValueType> Diffusion<IndexType, ValueType>::computeFlow(const CSRSparseMatrix<ValueType>& laplacian, const DenseVector<ValueType>& demand, ValueType eps) {
    using namespace scai::solver;
    using scai::lama::L2Norm;
    using scai::lama::fill;

    ValueType newWeightSum = demand.sum();
    if (std::abs(newWeightSum) >= eps) {
        throw std::logic_error("Residual weight sum " + std::to_string(newWeightSum) + " too large!");
    }

    //create a  copy as we need to change the column distribution if the matrix
    //TODO: not the fastest way but we maintain the const-ness of the laplacian
    CSRSparseMatrix<ValueType> laplacianCopy(laplacian);
    scai::dmemo::DistributionPtr dist( laplacianCopy.getRowDistributionPtr() );
    laplacianCopy.redistribute(dist, dist);

    auto solution = fill<DenseVector<ValueType>>( dist, 0.0 );

    auto norm = std::make_shared<L2Norm<ValueType>>();

    auto rt = std::make_shared<ResidualThreshold<ValueType>>( norm, eps, ResidualCheck::Relative );

    auto logger = std::make_shared<CommonLogger>( "myLogger: ",
                  LogLevel::convergenceHistory,
                  LoggerWriteBehaviour::toConsoleOnly );
    CG<ValueType> solver( "simpleCG" );

    //solver.setLogger( logger );

    solver.setStoppingCriterion( rt );

    solver.initialize( laplacianCopy );
    solver.solve( solution, demand );

    return solution;

}

template<typename IndexType, typename ValueType>
DenseVector<ValueType> Diffusion<IndexType, ValueType>::potentialsFromSource(const CSRSparseMatrix<ValueType>& laplacian, const DenseVector<ValueType>& nodeWeights, IndexType source, ValueType eps) {
    using scai::lama::NormPtr;
    using scai::lama::fill;
    using scai::lama::eval;

    const IndexType n = laplacian.getNumRows();
    if (laplacian.getNumColumns() != n) {
        throw std::invalid_argument("Matrix must be symmetric to be a Laplacian");
    }

    scai::dmemo::DistributionPtr dist(laplacian.getRowDistributionPtr());

    //making sure that the source is the same on all processors
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    [[maybe_unused]] IndexType sourceSum = comm->sum(source);
    assert(sourceSum == source*comm->getSize());

    ValueType weightSum = nodeWeights.sum();

    IndexType sourceIndex = dist->global2Local(source);

    auto nullVector = fill<DenseVector<ValueType>>(dist,0);
    auto d = eval<DenseVector<ValueType>>(nullVector - nodeWeights);

    if (sourceIndex != scai::invalidIndex) {
        d.getLocalValues()[sourceIndex] = weightSum - nodeWeights.getLocalValues()[sourceIndex];
    }

    return computeFlow(laplacian, d, eps);
}

template<typename IndexType, typename ValueType>
DenseMatrix<ValueType> Diffusion<IndexType, ValueType>::multiplePotentials(const scai::lama::CSRSparseMatrix<ValueType>& laplacian, const scai::lama::DenseVector<ValueType>& nodeWeights, const std::vector<IndexType>& sources, ValueType eps) {
    using scai::hmemo::HArray;

    if (!laplacian.getRowDistributionPtr()->isReplicated() or !nodeWeights.getDistributionPtr()->isReplicated()) {
        throw std::logic_error("Should only be called with replicated input.");
    }

    const IndexType l = sources.size();
    [[maybe_unused]] const IndexType n = laplacian.getNumRows();
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


template class Diffusion<IndexType, double>;
template class Diffusion<IndexType, float>;

} /* namespace ITI */
