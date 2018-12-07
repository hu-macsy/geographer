#include "FileIO.h"
#include "KMeans.h"

#include "gtest/gtest.h"


namespace ITI {

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;


class KMeansTest : public ::testing::Test {
protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";

};

TEST_F(KMeansTest, testFindInitialCentersSFC) {
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
	Settings settings;
	settings.numBlocks = k;
	settings.dimensions = dimensions;

    std::vector<ValueType> minCoords(settings.dimensions);
    std::vector<ValueType> maxCoords(settings.dimensions);
    for (IndexType dim = 0; dim < settings.dimensions; dim++) {
        minCoords[dim] = coords[dim].min();
        maxCoords[dim] = coords[dim].max();
        SCAI_ASSERT_NE_ERROR( minCoords[dim], maxCoords[dim], "min=max for dimension "<< dim << ", this will cause problems to the hilbert index. local= " << coords[0].getLocalValues().size() );
    }

	std::vector<std::vector<ValueType>> centers = KMeans::findInitialCentersSFC<IndexType,ValueType>(coords,  minCoords, maxCoords, settings);

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

//TODO: got undefined reference for getLocalMinMaxCoords and findInitialCentersFromSFCOnly
//update: instantiation is needed


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
	std::vector<std::vector<ValueType>> centers1 = KMeans::findInitialCentersSFC<IndexType,ValueType>(coords, minCoords, maxCoords, settings);
	
	settings.sfcResolution = std::log2(k);
	std::vector<std::vector<ValueType>> centers2 = KMeans::findInitialCentersFromSFCOnly<IndexType,ValueType>( maxCoords, settings);
	
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


TEST_F(KMeansTest, testHierarchicalPartition) {
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

	//set uniform node weights
	scai::lama::DenseVector<ValueType> unitNodeWeights = scai::lama::DenseVector<ValueType>( dist, 1);

	//const IndexType k = comm->getSize();
	//using KMeans::cNode;

	//set CommTree
	std::vector<cNode> leaves = {
		// 				{hierachy ids}, numCores, mem, speed
		cNode( std::vector<unsigned int>{0,0}, 4, 8, 50),
		cNode( std::vector<unsigned int>{0,1}, 4, 8, 90),

		cNode( std::vector<unsigned int>{1,0}, 6, 10, 80),
		cNode( std::vector<unsigned int>{1,1}, 6, 10, 90),
		cNode( std::vector<unsigned int>{1,2}, 6, 10, 70),

		cNode( std::vector<unsigned int>{2,0}, 8, 12, 80),
		cNode( std::vector<unsigned int>{2,1}, 8, 12, 90),
		cNode( std::vector<unsigned int>{2,2}, 8, 12, 90),
		cNode( std::vector<unsigned int>{2,3}, 8, 12, 100),
		cNode( std::vector<unsigned int>{2,4}, 8, 12, 90)
	};

	ITI::CommTree<IndexType,ValueType> cTree( leaves );

	struct Settings settings;
	settings.dimensions = dimensions;
	settings.numBlocks = leaves.size();
	settings.debugMode = false;
	
	Metrics metrics(settings);

	scai::lama::DenseVector<IndexType> partition = KMeans::computeHierarchicalPartition( coords, unitNodeWeights, cTree, settings, metrics);

	//checks
}

/*
TEST_F(KMeansTest, testPartitionWithNodeWeights) {
	//std::string fileName = "Grid32x32";
	std::string fileName = "bubbles-00010.graph";
	std::string graphFile = graphPath + fileName;
	std::string coordFile = graphFile + ".xyz";
	const IndexType dimensions = 2;

	//load graph and coords
	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType globalN = graph.getNumRows();
	std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), globalN, dimensions);

	const IndexType k = comm->getSize();
		
	const std::vector<IndexType> blockSizes(k, globalN/k);
	
	scai::lama::DenseVector<ValueType> unitNodeWeights = scai::lama::DenseVector<ValueType>( dist, 1);

	
	struct Settings settings;
	settings.dimensions = dimensions;
	settings.numBlocks = k;
	settings.minSamplingNodes = unitNodeWeights.getLocalValues().size()/20;
	
	scai::lama::DenseVector<IndexType> firstPartition = ITI::KMeans::computePartition ( coords, settings.numBlocks, unitNodeWeights, blockSizes, settings);
	
	struct Metrics metrics(1);
	metrics.getEasyMetrics( graph, firstPartition, unitNodeWeights, settings );
	if(comm->getRank()==0){
		printMetricsShort( metrics, std::cout);
	}
	
	const IndexType seed = 0;
	ValueType diverg = 0.8;
		
	scai::lama::DenseVector<ValueType> imbaNodeWeights = ITI::Repartition<IndexType,ValueType>::sNW( coords, seed, diverg, dimensions);
	
	ValueType maxWeight = imbaNodeWeights.max();
	ValueType minWeight = imbaNodeWeights.min();
	PRINT0("maxWeight= "<< maxWeight << " , minWeight= "<< minWeight);

	const IndexType localN = graph.getLocalNumRows();
	
	//settings.verbose = true;
	settings.maxKMeansIterations = 30;
	settings.balanceIterations = 10;
	settings.minSamplingNodes = localN;
	
	scai::lama::DenseVector<IndexType> imbaPartition = ITI::KMeans::computePartition ( coords, settings.numBlocks, imbaNodeWeights, blockSizes, settings);
	
	metrics.getEasyMetrics( graph, imbaPartition, imbaNodeWeights, settings );
	if(comm->getRank()==0){
		printMetricsShort( metrics, std::cout);
	}
	
}
*/


}
