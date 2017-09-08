#include <numeric>
#include <string>
#include <assert.h>

#include <scai/lama.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/matrix/CSRSparseMatrix.hpp>

#include "Diffusion.h"
#include "FileIO.h"
#include "GraphUtils.h"

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::lama::DenseMatrix;

typedef double ValueType;
typedef int IndexType;

int main(int argc, char *argv[])
{
	std::string graphFile(argv[1]);
    std::string coordFile("diffusion-coords.xyz");

    IndexType numLandmarks = 2;

    if (argc >= 3) {
    	numLandmarks = std::atoi(argv[2]);
    }

	CSRSparseMatrix<ValueType> graph = ITI::FileIO<IndexType, ValueType>::readGraph(graphFile );
	const IndexType n = graph.getNumRows();
	scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
	graph.redistribute(noDist, noDist);

	CSRSparseMatrix<ValueType> L = ITI::Diffusion<IndexType, ValueType>::constructLaplacian(graph);

	DenseVector<ValueType> nodeWeights(n,1.0);

	std::vector<IndexType> nodeIndices(n);
	std::iota(nodeIndices.begin(), nodeIndices.end(), 0);

	ITI::GraphUtils::FisherYatesShuffle(nodeIndices.begin(), nodeIndices.end(), numLandmarks);

	std::vector<IndexType> landmarks(numLandmarks);
	std::copy(nodeIndices.begin(), nodeIndices.begin()+numLandmarks, landmarks.begin());

	DenseMatrix<ValueType> potentials = ITI::Diffusion<IndexType, ValueType>::multiplePotentials(L, nodeWeights, landmarks, 1e-5);
	assert(numLandmarks == potentials.getNumRows());
	assert(n == potentials.getNumColumns());

	std::vector<DenseVector<ValueType> > convertedCoords(numLandmarks);
	for (IndexType i = 0; i < numLandmarks; i++) {
		convertedCoords[i] = DenseVector<ValueType>(n,0);
		potentials.getLocalRow(convertedCoords[i].getLocalValues(), i);
	}
	ITI::FileIO<IndexType, ValueType>::writeCoords(convertedCoords, coordFile);
}
