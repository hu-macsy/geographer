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


TEST_F( benchmarkTest, benchMapping ) {

    //std::string fileName = "Grid32x32";
    //std::string fileName = "slowrot-00000.graph";
    std::string fileName = "bubbles-00010.graph";
    std::string file = graphPath + fileName;
    const IndexType dimensions = 2;

    Settings settings;
    settings.dimensions = dimensions;
    settings.numBlocks = 8;
    settings.noRefinement = true;
    settings.writePEgraph = true;
    settings.writeInFile = true;
    settings.storeInfo = true;

    const IndexType k = settings.numBlocks;

    //
    // 1 - read graph, coordiantes and create unit nodeweights
    //

    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    IndexType globalN = graph.getNumRows();

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), globalN, dimensions);

    std::vector<scai::lama::DenseVector<ValueType>> unitWeights(2, scai::lama::DenseVector<ValueType>(graph.getRowDistributionPtr(), 1));

    //
    // 2 - read the PE graph
    //

    std::string PEfile = "./meshes/PEgraphs/myPEgraph8.txt";
    CommTree<IndexType,ValueType> cTree = FileIO<IndexType, ValueType>::readPETree( PEfile );
    cTree.adaptWeights( unitWeights );

    const scai::lama::CSRSparseMatrix<ValueType> PEGraph = cTree.exportAsGraph_local();

    SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), k, "Wrong number of rows/vertices" );
    SCAI_ASSERT_LE_ERROR( scai::utilskernel::HArrayUtils::max(PEGraph.getLocalStorage().getIA() ), PEGraph.getNumValues(), "some ia value is too large" );
    if( settings.writePEgraph ) {
        FileIO<IndexType,ValueType>::writeGraph( PEGraph, "peFromTree"+std::to_string(k)+".graph", 1);
    }

    //
    // 3 - read and partition graph without using the PEgraph
    //

    settings.initialPartition = InitialPartitioningMethods::KMeans;
    struct Metrics metrics(settings);

    std::vector<std::vector<ValueType>> balances = cTree.getBalanceVectors( -1 );
    SCAI_ASSERT_EQ_ERROR( balances.size(), 2, "Wrong number of balance constrains");
    SCAI_ASSERT_EQ_ERROR( balances[0].size(), k, "Wrong size of balance vector");


    // get partition ParcoRepart::partitionGraph to accept constrains
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, unitWeights, balances, settings, metrics);
    ASSERT_EQ(globalN, partition.size());

    //
    // 4 - partition graph with the PEgraph
    //

    //read graph again or redistribute because the previous partition might have change it
    scai::lama::CSRSparseMatrix<ValueType> graph2 = FileIO<IndexType, ValueType>::readGraph(file );

    struct Metrics metrics2( settings );

    scai::lama::DenseVector<IndexType> partitionWithPE = ITI::KMeans::computeHierarchicalPartition(  \
            graph2, coords, unitWeights, cTree, settings, metrics );

    if(settings.writeInFile) {
        FileIO<IndexType,ValueType>::writePartitionParallel( partition, "./partResults/partKM"+std::to_string(settings.numBlocks)+".out");
        FileIO<IndexType,ValueType>::writePartitionParallel( partitionWithPE, "./partResults/partHKM"+std::to_string(settings.numBlocks)+".out");

        //FileIO<IndexType,ValueType>::writeGraph( GraphUtils<IndexType,ValueType>::getBlockGraph(graph, partition, settings.numBlocks), "blockKM"+std::to_string(settings.numBlocks)+".graph", 1);
        //FileIO<IndexType,ValueType>::writeGraph( GraphUtils<IndexType,ValueType>::getBlockGraph(graph, partitionWithPE, settings.numBlocks), "blockHKM"+std::to_string(settings.numBlocks)+".graph", 1);
    }

    //
    // 5 - compare quality
    //

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    PRINT0("--------- Metrics for regular partition");

    //graph and partition are distributed inside partitionGraph; distributions must allign
    unitWeights[0].redistribute( partition.getDistributionPtr() );

    metrics.getMappingMetrics( graph, partition, PEGraph);
    metrics.getEasyMetrics( graph, partition, unitWeights[0], settings );
    if(comm->getRank()==0)
        metrics.print( std::cout );

    PRINT0("--------- Metrics for hierarchical partition");

    //graph and partition are distributed inside partitionGraph; distributions must allign
    unitWeights[0].redistribute( partitionWithPE.getDistributionPtr() );

    metrics2.getMappingMetrics( graph2, partitionWithPE, PEGraph);
    metrics2.getEasyMetrics( graph2, partitionWithPE, unitWeights[0], settings );
    if(comm->getRank()==0)
        metrics2.print( std::cout );


}//TEST_F( benchmarkTest, testMapping )

}// namespace