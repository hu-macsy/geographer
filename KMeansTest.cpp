#include "FileIO.h"

#include "KMeans.h"
#include "AuxiliaryFunctions.h"

#include "gtest/gtest.h"


namespace ITI {

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;


class KMeansTest : public ::testing::Test {
protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";

};

TEST_F(KMeansTest, testFindInitialCenters) {
	std::string fileName = "bubbles-00010.graph";
	std::string graphFile = graphPath + fileName;
	std::string coordFile = graphFile + ".xyz";
	const IndexType dimensions = 2;
	const IndexType k = 8;

	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
	const IndexType n = graph.getNumRows();
	std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);
	DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);
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
		ValueType coordSum = std::accumulate(centers[d].begin(), centers[d].end(), 0.0);
		ValueType totalSum = comm->sum(coordSum);
		EXPECT_LT(std::abs(p*coordSum - totalSum), 1e-5);
	}
}

TEST_F(KMeansTest, testFindCenters) {
	std::string fileName = "bubbles-00010.graph";
	std::string graphFile = graphPath + fileName;
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

	DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);

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


TEST_F(KMeansTest, testPartitionWithSFCCoords) {
	//std::string fileName = "bubbles-00010.graph";
	std::string fileName = "Grid16x16";
	std::string graphFile = graphPath + fileName;
	std::string coordFile = graphFile + ".xyz";
	const IndexType dimensions = 2;
	
	//load graph and coords
	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType n = graph.getNumRows();
	std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);
	
	DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);
	
	struct Settings settings;
	settings.dimensions = dimensions;
	settings.numBlocks = comm->getSize();
	
	{ 
		DenseVector<IndexType> initialPartition( dist, -1);
		for( int i=0; i< initialPartition.getLocalValues().size(); i++){
			initialPartition.getLocalValues()[i]= comm->getRank();
		}
		ITI::aux<IndexType,ValueType>::print2DGrid(graph, initialPartition );
		
		
	}
		
	DenseVector<IndexType> partition = KMeans::getPartitionWithSFCCoords<IndexType>( graph, coords, uniformWeights, settings);
	
	partition.redistribute( dist );
	
	//checks
	
	scai::hmemo::ReadAccess<IndexType> rPart( partition.getLocalValues() );
	const IndexType localN = rPart.size();
	EXPECT_EQ( comm->sum(localN), n);
	
	for( IndexType i=0; i<localN; i++){
		EXPECT_LT( rPart[i], settings.numBlocks);
		EXPECT_GE( rPart[i], 0);
		//PRINT(*comm << ": part[ " << dist->local2global(i) << " ] = " << rPart[i]);
	}
	ITI::aux<IndexType,ValueType>::print2DGrid(graph, partition );
	
}

TEST_F(KMeansTest, testCentersOnlySfc) {
	std::string fileName = "bubbles-00010.graph";
	std::string graphFile = graphPath + fileName;
	std::string coordFile = graphFile + ".xyz";
	const IndexType dimensions = 2;

	//load graph and coords
	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType n = graph.getNumRows();
	std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);

	const IndexType k = comm->getSize();
	
	struct Settings settings;
	settings.dimensions = dimensions;
	settings.numBlocks = comm->getSize();
	
	//get min and max
	std::vector<ValueType> minCoords, maxCoords;
	std::tie(minCoords, maxCoords) = KMeans::getLocalMinMaxCoords(coords);
	
	// get centers
	std::vector<std::vector<ValueType>> centers1 = KMeans::findInitialCentersSFC(coords, k, minCoords, maxCoords, settings);
	
	settings.sfcResolution = std::log2(k);
	std::vector<std::vector<ValueType>> centers2 = KMeans::findInitialCentersFromSFCOnly<IndexType,ValueType>( k, maxCoords, settings);
	
	EXPECT_EQ( centers1.size(), centers2.size() );
	EXPECT_EQ( centers1[0].size(), centers2[0].size() );
	EXPECT_EQ( centers1[0].size(), k);
	
	if(comm->getRank()==0){
		std::cout<<"maxCoords= ";
		for(int d=0; d<dimensions; d++){
			std::cout<< maxCoords[d] <<", ";
		}
		std::cout<< std::endl;
		std::cout<<"center1" << std::endl;
		for( int c=0; c<k; c++){
			//std::cout<<"center1["<< c << "]= ";
			std::cout<<"( ";
			for(int d=0; d<dimensions; d++){
				std::cout<< centers1[d][c]<< ", ";
			}
			std::cout<< "\b\b )" << std::endl;
		}
		std::cout<< std::endl;
		
		std::cout<<"center2" << std::endl;
		for( int c=0; c<k; c++){
			//std::cout<<"center2["<< c << "]= ";
			std::cout<<"( ";
			for(int d=0; d<dimensions; d++){
				std::cout<<  centers2[d][c]*maxCoords[d]<< ", ";
			}
			std::cout<< "\b\b )" << std::endl;
		}
	}
}


}
