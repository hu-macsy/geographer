#include "FileIO.h"

#include "KMeans.h"

#include "gtest/gtest.h"

typedef double ValueType;
typedef int IndexType;

namespace ITI {

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;

class KMeansTest : public ::testing::Test {

};

TEST_F(KMeansTest, testFindInitialCenters) {
	std::string path = "meshes/bubbles/";
	std::string fileName = "bubbles-00010.graph";
	std::string graphFile = path + fileName;
	std::string coordFile = graphFile + ".xyz";
	const IndexType dimensions = 2;
	const IndexType k = 8;

	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
	const IndexType n = graph.getNumRows();
	std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);
	DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType p = comm->getSize();


	std::vector<std::vector<ValueType> > centers = KMeans::findInitialCenters(coords, k, uniformWeights);

	//check for size
	EXPECT_EQ(dimensions, centers.size());
	EXPECT_EQ(k, centers[0].size());

	//check for distinctness
	bool allDistinct = true;
	for (IndexType i = 0; i < k; i++) {
		for (IndexType j = i+1; j < k; j++) {
			bool differenceFound = false;
			for (IndexType d = 0; d < dimensions; d++) {
				if (centers[d][i] != centers[d][j]) {
					differenceFound = true;
				}
			}
			if (!differenceFound) {
				allDistinct = false;
				std::cout << "Centers " << i << " and " << j << " are both at ";
				for (IndexType d = 0; d < dimensions; d++) {
					std::cout << "(" << centers[d][i] << "|" << centers[d][j] << ") ";
				}
				std::cout << std::endl;
			}
		}
	}
	EXPECT_TRUE(allDistinct);

	//check for equality across processors
	for (IndexType d = 0; d < dimensions; d++) {
		ValueType coordSum = std::accumulate(centers[d].begin(), centers[d].end(), 0);
		ValueType totalSum = comm->sum(coordSum);
		EXPECT_EQ(p*coordSum, totalSum);
	}
}

TEST_F(KMeansTest, testFindCenters) {
	std::string path = "meshes/bubbles/";
	std::string fileName = "bubbles-00010.graph";
	std::string graphFile = path + fileName;
	std::string coordFile = graphFile + ".xyz";
	const IndexType dimensions = 2;

	//load graph and coords
	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType n = graph.getNumRows();
	std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);

	const IndexType k = comm->getSize();

	//create random partition
	DenseVector<IndexType> randomValues;
	randomValues.setRandom(dist, 1);
	randomValues.sort(true);
	DenseVector<IndexType> part = DenseVector<IndexType>(randomValues.getDistributionPtr(), comm->getRank());
	part.redistribute(dist);

	DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);

	//get centers
	std::vector<IndexType> nodeIndices(uniformWeights.getLocalValues().size());
	std::iota(nodeIndices.begin(), nodeIndices.end(), 0);
	std::vector<std::vector<ValueType> > centers = KMeans::findCenters(coords, part, k,	nodeIndices.begin(), nodeIndices.end(), uniformWeights);

	//check for size
	EXPECT_EQ(dimensions, centers.size());
	EXPECT_EQ(k, centers[0].size());

	//create invalid partition
	part = DenseVector<IndexType>(dist, 0);

	//get centers
	centers = KMeans::findCenters(coords, part, k, nodeIndices.begin(), nodeIndices.end(), uniformWeights);
}

}
