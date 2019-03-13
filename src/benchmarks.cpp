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

//1 - read and partition graph without using the PEgraph
	//std::string fileName = "Grid32x32";
	std::string fileName = "bigtrace-00000.graph";
    std::string file = graphPath + fileName;
    const IndexType dimensions = 2;

    Settings settings;
    settings.dimensions = dimensions;
    settings.numBlocks = 10;
    settings.noRefinement = true;

    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    IndexType globalN = graph.getNumRows();

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), globalN, dimensions);

    settings.initialPartition = InitialPartitioningMethods::KMeans;
    struct Metrics metrics(settings);

    // get partition
    const scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);
    ASSERT_EQ(globalN, partition.size());

//FileIO<IndexType,ValueType>::writeGraph( GraphUtils<IndexType,ValueType>::getBlockGraph(graph, partition, settings.numBlocks), "blockKM"+std::to_string(settings.numBlocks)+".graph", 1);

//2 - read PE graph
    std::string PEfile = "./tools/myPEgraph10.txt";
    CommTree<IndexType,ValueType> cTree = FileIO<IndexType, ValueType>::readPETree( PEfile );
PRINT( cTree.getNumLeaves() << ", " );
    const scai::lama::CSRSparseMatrix<ValueType> PEGraph = cTree.exportAsGraph_local();

SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), settings.numBlocks , "Wrong number of rows/vertices" );
SCAI_ASSERT_LE_ERROR( scai::utilskernel::HArrayUtils::max(PEGraph.getLocalStorage().getIA() ) , PEGraph.getNumValues(), "some ia value is too large" );
FileIO<IndexType,ValueType>::writeGraph( PEGraph, "peFromTree"+std::to_string(settings.numBlocks)+".graph", 1);

//3 - partition graph with the PEgraph
    //read graph again or redistribute because the previous partition might have change it
    scai::lama::CSRSparseMatrix<ValueType> graph2 = FileIO<IndexType, ValueType>::readGraph(file );
    scai::lama::DenseVector<ValueType> unitWeights( graph.getRowDistributionPtr(), 1);
    struct Metrics metrics2( settings );

    scai::lama::DenseVector<IndexType> partitionWithPE = ITI::KMeans::computeHierarchicalPartition(  \
    	graph2, coords, unitWeights, cTree, settings, metrics );

FileIO<IndexType,ValueType>::writeGraph( GraphUtils<IndexType,ValueType>::getBlockGraph(graph, partitionWithPE, settings.numBlocks), "blockHKM"+std::to_string(settings.numBlocks)+".graph", 1);

//4 - compare quality
    PRINT("--------- Metrics for regular partition");
    metrics.getMappingMetrics( graph, partition, PEGraph);

    PRINT("--------- Metrics for hierarchical partition");
    metrics2.getMappingMetrics( graph2, partitionWithPE, PEGraph);


}//TEST_F( benchmarkTest, testMapping )

}// namespace