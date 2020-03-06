#include "FileIO.h"
#include "KMeans.h"

#include "gtest/gtest.h"


namespace ITI {

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;

template<typename T>
class KMeansTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(KMeansTest, testTypes);

//-----------------------------------------------

TYPED_TEST(KMeansTest, testFindInitialCentersSFC) {
    using ValueType = TypeParam;

    std::string fileName = "bubbles-00010.graph";
    std::string graphFile = KMeansTest<ValueType>::graphPath + fileName;
    std::string coordFile = graphFile + ".xyz";
    const IndexType dimensions = 2;
    const IndexType k = 8;

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile);
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

    std::vector<std::vector<ValueType>> centers = KMeans<IndexType,ValueType>::findInitialCentersSFC(coords,  minCoords, maxCoords, settings);

    //check for size
    EXPECT_EQ(k, centers.size());
    EXPECT_EQ(dimensions, centers[0].size());

    //check for distinctness
    bool allDistinct = true;
    for (IndexType i = 0; i < k; i++) {
        for (IndexType j = i+1; j < k; j++) {
            bool differenceFound = false;
            for (IndexType d = 0; d < dimensions; d++) {
                if (centers[i][d] != centers[j][d]) {
                    differenceFound = true;
                }
            }
            if (!differenceFound) {
                allDistinct = false;
                std::cout << "Centers " << i << " and " << j << " are both at ";
                for (IndexType d = 0; d < dimensions; d++) {
                    std::cout << "(" << centers[i][d] << "|" << centers[j][d] << ") ";
                }
                std::cout << std::endl;
            }
        }
    }
    EXPECT_TRUE(allDistinct);

    //check for equality across processors
    for (IndexType i=0; i<k; i++) {
        ValueType coordSum = std::accumulate(centers[i].begin(), centers[i].end(), 0.0);
        ValueType totalSum = comm->sum(coordSum);
        //WARNING: fails with p=5 or 11, for ValueType float
        EXPECT_LT(std::abs(p*coordSum - totalSum), 1e-5);
        //basically, that coordSum is equal on all PEs
        EXPECT_EQ( comm->max(coordSum), comm->min(coordSum) );
    }
}
//------------------------------------------- -----------------------

TYPED_TEST(KMeansTest, testFindCenters) {
    using ValueType = TypeParam;

    std::string fileName = "bubbles-00010.graph";
    std::string graphFile = KMeansTest<ValueType>::graphPath + fileName;
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
    std::vector<std::vector<ValueType> > centers = KMeans<IndexType,ValueType>::findCenters(coords, part, k,	nodeIndices.begin(), nodeIndices.end(), uniformWeights);

    //check for size
    EXPECT_EQ(dimensions, centers.size());
    EXPECT_EQ(k, centers[0].size());

    //create invalid partition
    part = DenseVector<IndexType>(dist, 0);

    //get centers
    centers = KMeans<IndexType,ValueType>::findCenters(coords, part, k, nodeIndices.begin(), nodeIndices.end(), uniformWeights);
}



TYPED_TEST(KMeansTest, testCentersOnlySfc) {
    using ValueType = TypeParam;

    std::string fileName = "bubbles-00010.graph";
    std::string graphFile = KMeansTest<ValueType>::graphPath + fileName;
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
    std::tie(minCoords, maxCoords) = KMeans<IndexType,ValueType>::getGlobalMinMaxCoords(coords);

    // get centers
    std::vector<std::vector<ValueType>> centers1 = KMeans<IndexType,ValueType>::findInitialCentersSFC(coords, minCoords, maxCoords, settings);
    EXPECT_EQ( centers1.size(), k );
    EXPECT_EQ( centers1[0].size(), dimensions );

    settings.sfcResolution = std::log2(k);
    std::vector<std::vector<ValueType>> centers2 = KMeans<IndexType,ValueType>::findInitialCentersFromSFCOnly( minCoords, maxCoords, settings);
    EXPECT_EQ( centers2.size(), dimensions );

//WARNING: quick fix for tests to pass
//TODO: the functions should agree on their return type: either a vector of size k*dimensions (centers2) or dimension*k (centers2)
    {
        std::vector<std::vector<ValueType>> reversedCenters( dimensions, std::vector<ValueType>(settings.numBlocks, 0.0) );
        for( unsigned int c=0; c<settings.numBlocks; c++) {
            for( unsigned int d=0; d<dimensions; d++) {
                reversedCenters[d][c] = centers1[c][d];
            }
        }

        centers1 = reversedCenters;
    }

    EXPECT_EQ( centers1.size(), centers2.size() );
    EXPECT_EQ( centers1[0].size(), centers2[0].size() );
    EXPECT_EQ( centers1[0].size(), k);

    if(comm->getRank()==0) {
        std::cout<<"minCoords= ";
        for(int d=0; d<dimensions; d++) {
            std::cout<< minCoords[d] <<", ";
        }
        std::cout<<"maxCoords= ";
        for(int d=0; d<dimensions; d++) {
            std::cout<< maxCoords[d] <<", ";
        }
        std::cout<< std::endl;
        std::cout<<"center1" << std::endl;
        for( int c=0; c<k; c++) {
            std::cout<<"( ";
            for(int d=0; d<dimensions; d++) {
                std::cout<< centers1[d][c]<< ", ";
            }
            std::cout<< "\b\b )" << std::endl;
        }
        std::cout<< std::endl;

        std::cout<<"center2" << std::endl;
        for( int c=0; c<k; c++) {
            std::cout<<"( ";
            for(int d=0; d<dimensions; d++) {
                std::cout<<  centers2[d][c]<< ", ";
            }
            std::cout<< "\b\b )" << std::endl;
        }
    }
}


TYPED_TEST(KMeansTest, testHierarchicalPartition) {
    using ValueType = TypeParam;

    //std::string fileName = "bubbles-00010.graph";
    std::string fileName = "Grid32x32";
    //std::string fileName = "Grid8x8";
    std::string graphFile = KMeansTest<ValueType>::graphPath + fileName;
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

    //or fixed weights
    scai::lama::DenseVector<ValueType> fixedNodeWeights = scai::lama::DenseVector<ValueType>( dist, 1);
    {
        scai::hmemo::WriteAccess<ValueType> localWeights( fixedNodeWeights.getLocalValues() );
        int localN = localWeights.size();
        for(int i=0; i<localN; i++) {
            localWeights[i] = dist->local2Global(i);
        }

    }

    ValueType c = 2;
    scai::lama::DenseVector<ValueType> constNodeWeights = unitNodeWeights;
    constNodeWeights *= c;

    //scai::lama::DenseVector<ValueType> nodeWeights = fixedNodeWeights;
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights = { unitNodeWeights, unitNodeWeights, constNodeWeights };

    std::cout << "Sum of node weights are: " << std::endl;
    for( unsigned int i=0; i<nodeWeights.size(); i++ ) {
        std::cout << "weight " << i << ": " << nodeWeights[i].sum() << std::endl;
    }

    typedef typename CommTree<IndexType,ValueType>::commNode cNode;
    //set CommTree
    std::vector<cNode> leaves = {
        // 				{hierachy ids}, numCores, mem, speed
        cNode( std::vector<unsigned int>{0,0}, {141, 8, 0.3} ),
        cNode( std::vector<unsigned int>{0,1}, {154, 8, 0.9} ),

        cNode( std::vector<unsigned int>{1,0}, {126, 10, 0.8} ),
        cNode( std::vector<unsigned int>{1,1}, {276, 10, 0.9} ),
        cNode( std::vector<unsigned int>{1,2}, {67, 10, 0.7} ),

        cNode( std::vector<unsigned int>{2,0}, {81, 12, 0.6} ),
        cNode( std::vector<unsigned int>{2,1}, {88, 12, 0.7} ),
        cNode( std::vector<unsigned int>{2,2}, {156, 12, 0.7} ),
        cNode( std::vector<unsigned int>{2,3}, {108, 12, 0.5} ),
        cNode( std::vector<unsigned int>{2,4}, {221, 12, 0.5} )
    };

    ITI::CommTree<IndexType,ValueType> cTree( leaves, { false, true, true } );

    cTree.adaptWeights( nodeWeights );

    struct Settings settings;
    settings.dimensions = dimensions;
    settings.numBlocks = leaves.size();
    settings.debugMode = false;
    settings.verbose = false;
    settings.storeInfo = false;
    settings.epsilon = 0.05;
    settings.balanceIterations = 5;
    settings.maxKMeansIterations = 5;
    settings.minSamplingNodes = -1;

    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition = KMeans<IndexType,ValueType>::computeHierarchicalPartition( coords, nodeWeights, cTree, settings, metrics);

    //checks - prints

    std::vector<ValueType> imbalances = cTree.computeImbalance( partition, settings.numBlocks, nodeWeights );

    if (comm->getRank() == 0) {
        std::cout << "final imbalances: ";
        for (IndexType i = 0; i < imbalances.size(); i++) {
            std::cout << " " << imbalances[i];
        }
        std::cout << std::endl;
    }
}

TYPED_TEST(KMeansTest, testComputePartitionWithMultipleWeights) {
    using ValueType = TypeParam;

    std::string fileName = "bubbles-00010.graph";
    std::string graphFile = KMeansTest<ValueType>::graphPath + fileName;
    std::string coordFile = graphFile + ".xyz";

    const IndexType numNodeWeights = 2;

    //load graph
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

    struct Settings settings;
    settings.dimensions = 2;
    settings.epsilon = 0.05;
    settings.numBlocks = comm->getSize();
    settings.verbose = true;

    //load coords
    const IndexType globalN = graph.getNumRows();
    const IndexType localN = dist->getLocalSize();
    const std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), globalN, settings.dimensions);

    //set first weight uniform, second weight random
    const scai::lama::DenseVector<ValueType> unitNodeWeights = scai::lama::DenseVector<ValueType>( dist, 1);
    scai::lama::DenseVector<ValueType> randomNodeWeights(dist, 0);
    randomNodeWeights.fillRandom(10);

    const std::vector<scai::lama::DenseVector<ValueType>> nodeWeights = {unitNodeWeights, randomNodeWeights};
    std::vector<std::vector<ValueType>> blockSizes(numNodeWeights);

    std::vector<ValueType> nodeWeightSum(numNodeWeights);
    for (IndexType i = 0; i < numNodeWeights; i++) {
        nodeWeightSum[i] = nodeWeights[i].sum();

        blockSizes[i].resize(settings.numBlocks, std::ceil(nodeWeightSum[i]/settings.numBlocks));
    }

    Metrics<ValueType> metrics(settings);
    //use hilbert redistribution before?
    scai::lama::DenseVector<IndexType> partition = KMeans<IndexType, ValueType>::computePartition( coords, nodeWeights, blockSizes, settings, metrics);

    //assert that distributions are still the same

    {
        scai::hmemo::ReadAccess<IndexType> rPartition(partition.getLocalValues());
        for (IndexType j = 0; j < numNodeWeights; j++) {
            std::vector<ValueType> blockWeights(settings.numBlocks);
            scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights[j].getLocalValues());

            for (IndexType i = 0; i < localN; i++) {
                IndexType block = rPartition[i];
                blockWeights[block] += rWeights[i];
            }

            comm->sumImpl( blockWeights.data(), blockWeights.data(), settings.numBlocks, scai::common::TypeTraits<ValueType>::stype);

            for (IndexType b = 0; b < settings.numBlocks; b++) {
                if (settings.verbose && comm->getRank() == 0) std::cout << "blockWeights[" << j << "][" << b << "] = " << blockWeights[b] << std::endl;
                EXPECT_LE(blockWeights[b], std::ceil(blockSizes[j][b])*(1+settings.epsilon));
            }
        }
    }

    //check for correct error messages: block sizes not aligned to node weights, different distributions in coordinates and weights, weights not fitting into blocks, balance
}

TYPED_TEST(KMeansTest, testGetGlobalMinMax) {
    using ValueType = TypeParam;

    std::string graphFile = "bubbles-00010.graph";
    std::string coordFile = graphFile + ".xyz";
    const IndexType dimensions = 2;
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType n;
    {
        CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
        n = graph.getNumRows();
    }
    //load coords
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);

    //get min and max
    std::vector<ValueType> minCoords, maxCoords;
    std::tie(minCoords, maxCoords) = KMeans<IndexType,ValueType>::getGlobalMinMaxCoords(coords);

    for( int d=0; d<dimensions; d++){
        ValueType minMin=comm->min(minCoords[d]);
        ValueType maxMax=comm->max(maxCoords[d]);

        EXPECT_EQ(minCoords[d], minMin);
        EXPECT_EQ(maxCoords[d], maxMax);
    }
}
/*
TYPED_TEST(KMeansTest, testPartitionWithNodeWeights) {
    using ValueType = TypeParam;

	//std::string fileName = "Grid32x32";
	std::string fileName = "bubbles-00010.graph";
	std::string graphFile = KMeansTest<ValueType>::graphPath + fileName;
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

	Metrics<ValueType> metrics(1);
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
