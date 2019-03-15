#include "gtest/gtest.h"

#include "Mapping.h"
#include "GraphUtils.h"
#include "FileIO.h"
#include "Metrics.h"
#include "Settings.h"
#include "KMeans.h"
#include "CommTree.h"
#include "ParcoRepart.h"


namespace ITI {

class benchmarkTest : public ::testing::Test {
    protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";
};


TEST_F( benchmarkTest, testMapping ){

	//std::string fileName = "Grid32x32";
	std::string fileName = "slowrot-00000.graph";
    std::string file = graphPath + fileName;
    const IndexType dimensions = 2;

    Settings settings;
    settings.dimensions = dimensions;
    settings.numBlocks = 8;
    settings.noRefinement = true;

    const IndexType k = settings.numBlocks;

    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    IndexType globalN = graph.getNumRows();

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), globalN, dimensions);

//1 - read PE graph
    std::string PEfile = "./tools/myPEgraph8_2.txt";
    CommTree<IndexType,ValueType> cTree = FileIO<IndexType, ValueType>::readPETree( PEfile );
	PRINT( cTree.getNumLeaves() << ", " );
    const scai::lama::CSRSparseMatrix<ValueType> PEGraph = cTree.exportAsGraph_local();

	SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), k , "Wrong number of rows/vertices" );
	SCAI_ASSERT_LE_ERROR( scai::utilskernel::HArrayUtils::max(PEGraph.getLocalStorage().getIA() ) , PEGraph.getNumValues(), "some ia value is too large" );
	FileIO<IndexType,ValueType>::writeGraph( PEGraph, "peFromTree"+std::to_string(k)+".graph", 1);

//2 - read and partition graph without using the PEgraph
    settings.initialPartition = InitialPartitioningMethods::KMeans;
    struct Metrics metrics(settings);

    //balances[0] is memory, balances[1] is cpu speed
    std::vector<std::vector<ValueType>> balances = cTree.getBalanceVectors();
    SCAI_ASSERT_EQ_ERROR( balances.size(), 2, "Wrong number of balance constrains");
    SCAI_ASSERT_EQ_ERROR( balances[0].size(), k, "Wrong size of balance vector");

    //cpu speed is given as the relative speed compared to the fastest cpu.
    //convert it to absolute number
    std::vector<std::vector<ValueType>> wantedBlockSizes( 1, std::vector<ValueType>(k, 0 ));
    for(IndexType i=0; i<k; i++ ){
    	wantedBlockSizes[0][i] = balances[1][i]*globalN; //we assume unit node weights
    }

    settings.blockSizes = wantedBlockSizes;

    // get partition ParcoRepart::partitionGraph to accept constrains
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);
    ASSERT_EQ(globalN, partition.size());

FileIO<IndexType,ValueType>::writePartitionParallel( partition, "./partResults/partKM"+std::to_string(settings.numBlocks)+".out");

//FileIO<IndexType,ValueType>::writeGraph( GraphUtils<IndexType,ValueType>::getBlockGraph(graph, partition, settings.numBlocks), "blockKM"+std::to_string(settings.numBlocks)+".graph", 1);



//3 - partition graph with the PEgraph
    //read graph again or redistribute because the previous partition might have change it
    scai::lama::CSRSparseMatrix<ValueType> graph2 = FileIO<IndexType, ValueType>::readGraph(file );
    std::vector<scai::lama::DenseVector<ValueType>> unitWeights(1, scai::lama::DenseVector<ValueType>(graph.getRowDistributionPtr(), 1));
    struct Metrics metrics2( settings );

    scai::lama::DenseVector<IndexType> partitionWithPE = ITI::KMeans::computeHierarchicalPartition(  \
    	graph2, coords, unitWeights, cTree, settings, metrics );

FileIO<IndexType,ValueType>::writePartitionParallel( partitionWithPE, "./partResults/partHKM"+std::to_string(settings.numBlocks)+".out");

//FileIO<IndexType,ValueType>::writeGraph( GraphUtils<IndexType,ValueType>::getBlockGraph(graph, partitionWithPE, settings.numBlocks), "blockHKM"+std::to_string(settings.numBlocks)+".graph", 1);

//4 - compare quality
    PRINT("--------- Metrics for regular partition");
    metrics.getMappingMetrics( graph, partition, PEGraph);
    metrics.getEasyMetrics( graph, partition, unitWeights[0], settings );
    metrics.print( std::cout );

    PRINT("--------- Metrics for hierarchical partition");
    metrics2.getMappingMetrics( graph2, partitionWithPE, PEGraph);
    metrics2.getEasyMetrics( graph2, partitionWithPE, unitWeights[0], settings );
    metrics2.print( std::cout );


}//TEST_F( benchmarkTest, testMapping )

}// namespace