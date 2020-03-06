#include <scai/lama.hpp>

#include "gtest/gtest.h"

#include "ParcoRepart.h"
#include "FileIO.h"
#include "GraphUtils.h"
#include "MeshGenerator.h"

#include <scai/dmemo/CyclicDistribution.hpp>
#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>

//remove
#include "HilbertCurve.h"

namespace ITI {

template<typename T>
class GraphUtilsTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(GraphUtilsTest, testTypes);

//-----------------------------------------------

TYPED_TEST(GraphUtilsTest, testReindexCut) {
    using ValueType = TypeParam;

    //std::string fileName = "trace-00008.graph";
    //std::string fileName = "delaunayTest.graph";
    std::string fileName = "Grid8x8";

    std::string file = GraphUtilsTest<ValueType>::graphPath + fileName;
    std::string coordsFile = file + ".xyz";

    const IndexType dimensions= 2;
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType n = graph.getNumRows();

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    // for now local refinement requires k = P
    const IndexType k = comm->getSize();
    const IndexType localN = dist->getLocalSize();

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( coordsFile, n, dimensions);
    ASSERT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));

    ValueType l2Norm = graph.l2Norm(); 

    EXPECT_TRUE( graph.isConsistent() );
    EXPECT_TRUE( graph.checkSymmetry() );
    
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(1, scai::lama::DenseVector<ValueType>(dist, 1));

    //get sfc partition
    Settings settings;
    settings.numBlocks = k;
    settings.noRefinement = true;
    settings.dimensions = dimensions;
    settings.debugMode = true;

    //DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, nodeWeights, settings);

    //try random partition
    srand(0);
    DenseVector<IndexType> partition( dist, 0 );
    for (IndexType i = 0; i < localN; i++) {
        IndexType blockId = (rand() % k);
        partition.getLocalValues()[i] = blockId;
    }

    //redistribute based on the partition to simulate a real scenario
    aux<IndexType, ValueType>::redistributeFromPartition( partition, graph, coords, nodeWeights, settings );    
    const scai::dmemo::DistributionPtr genDist = graph.getRowDistributionPtr();

    //graph.checkSettings();
    SCAI_ASSERT_DEBUG( graph.isConsistent(), graph << ": is invalid matrix after redistribution" )
    EXPECT_TRUE( graph.isConsistent() );
    EXPECT_TRUE( graph.checkSymmetry() );
    EXPECT_NEAR( l2Norm, graph.l2Norm(), 1e-5 );

    //assert that all input data have the same distribution
    ASSERT_TRUE( coords[0].getDistributionPtr()->isEqual(*genDist) );
    ASSERT_TRUE( partition.getDistributionPtr()->isEqual(*genDist) );


    //get first cut
    ValueType initialCut = GraphUtils<IndexType, ValueType>::computeCut(graph, partition, true);
    ASSERT_GE(initialCut, 0);
    ValueType initialImbalance = GraphUtils<IndexType, ValueType>::computeImbalance( partition, k );
    std::pair<std::vector<IndexType>,std::vector<IndexType>> initialBorderInnerNodes = GraphUtils<IndexType, ValueType>::getNumBorderInnerNodes( graph, partition, settings );

    PRINT0("about to redistribute the graph");
    //now reindex and get second metrics
    const  scai::dmemo::DistributionPtr newGenBlockDist = GraphUtils<IndexType, ValueType>::genBlockRedist(graph);

    //checks

    //graph.checkSettings();
    EXPECT_TRUE( graph.isConsistent() );
    EXPECT_TRUE( graph.checkSymmetry() );
    EXPECT_NEAR( l2Norm, graph.l2Norm(), 1e-5 );

    partition.redistribute( newGenBlockDist );
    DenseVector<IndexType> reIndexedPartition = partition;

    ASSERT_TRUE(reIndexedPartition.getDistributionPtr()->isEqual(*newGenBlockDist));


    ValueType secondCut = GraphUtils<IndexType, ValueType>::computeCut(graph, reIndexedPartition, true);
    ValueType secondImbalance = GraphUtils<IndexType, ValueType>::computeImbalance( reIndexedPartition, k );
    EXPECT_EQ( initialCut, secondCut );
    EXPECT_NEAR( initialImbalance, secondImbalance, 1e-5 );

    ASSERT_TRUE( newGenBlockDist->isEqual(*graph.getRowDistributionPtr()) );
    ASSERT_TRUE( reIndexedPartition.getDistributionPtr()->isEqual(*newGenBlockDist) );

    std::pair<std::vector<IndexType>,std::vector<IndexType>> secondBorderInnerNodes = GraphUtils<IndexType, ValueType>::getNumBorderInnerNodes( graph, reIndexedPartition, settings );
    EXPECT_EQ( initialBorderInnerNodes.first, secondBorderInnerNodes.first );
    EXPECT_EQ( initialBorderInnerNodes.second, secondBorderInnerNodes.second );

}
//-----------------------------------------------------------------

TYPED_TEST(GraphUtilsTest, testConstructLaplacian) {
    using ValueType = TypeParam;

    std::string fileName = "rotation-00000.graph";
    std::string file = GraphUtilsTest<ValueType>::graphPath + fileName;
    const CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType n = graph.getNumRows();
    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();

    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(n));
    ASSERT_TRUE(graph.isConsistent());

    CSRSparseMatrix<ValueType> L = GraphUtils<IndexType, ValueType>::constructLaplacian(graph);

    ASSERT_EQ(L.getRowDistribution(), graph.getRowDistribution());
    ASSERT_TRUE(L.isConsistent());
{
//PRINT( (GraphUtils<IndexType, ValueType>::hasSelfLoops(graph)) );

// scai::lama::CSRSparseMatrix<ValueType> copyGraph( graph ); 
// copyGraph.redistribute(graph.getRowDistributionPtr(), graph.getRowDistributionPtr());
// PRINT( (GraphUtils<IndexType, ValueType>::hasSelfLoops(copyGraph)) );
// CSRSparseMatrix<ValueType> lolo = GraphUtils<IndexType, ValueType>::constructLaplacian(copyGraph);
// PRINT("\n\n");
}
    //test that L*1 = 0
    DenseVector<ValueType> x( L.getRowDistributionPtr(), 1 );
    DenseVector<ValueType> y = scai::lama::eval<DenseVector<ValueType>>( L * x );

    ValueType norm = y.maxNorm();
    EXPECT_EQ(norm,0);

    //test consistency under distributions
    const CSRSparseMatrix<ValueType> replicatedGraph = scai::lama::distribute<CSRSparseMatrix<ValueType>>(graph, noDistPtr, noDistPtr);
    CSRSparseMatrix<ValueType> LFromReplicated = GraphUtils<IndexType, ValueType>::constructLaplacian(replicatedGraph);
    LFromReplicated.redistribute(L.getRowDistributionPtr(), L.getColDistributionPtr());
    CSRSparseMatrix<ValueType> diff = scai::lama::eval<CSRSparseMatrix<ValueType>> (LFromReplicated - L);
    EXPECT_EQ(0, diff.l2Norm());
}
//-----------------------------------------------------------------

TYPED_TEST(GraphUtilsTest, benchConstructLaplacian) {
    using ValueType = TypeParam;

    std::string fileName = "bubbles-00010.graph";
    std::string file = GraphUtilsTest<ValueType>::graphPath + fileName;
    const CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );

    CSRSparseMatrix<ValueType> L = GraphUtils<IndexType, ValueType>::constructLaplacian(graph);
}

//TODO: test also with edge weights
//-----------------------------------------------------------------

TYPED_TEST (GraphUtilsTest, testLocalDijkstra) {
    using ValueType = TypeParam;

    std::string file = GraphUtilsTest<ValueType>::graphPath + "Grid4x4";
    IndexType N;
    bool executed = true;

    //TODO:probably there is a better way to do that: exit tests as failed if p>7. Regular googletest tests
    // do not exit the test. Maybe by using death test, but I could not figure it out.

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if( comm->getSize()>6 ) {
        PRINT0("\n\tWARNING:File too small to read. If p>6 this test is not executed\n" );
        executed=false;
    } else {
        //EXPECT_EXIT( /*CSRSparseMatrix<ValueType> graph = */FileIO<IndexType, ValueType>::readGraph( file ), ::testing::ExitedWithCode(1), "too few PEs" );
        // read graph
        CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
        N= graph.getNumRows();

        //replicate graph to all PEs
        scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
        graph.redistribute( noDistPointer, noDistPointer );

        std::vector<IndexType> predecessor;
        std::vector<ValueType> shortDist = GraphUtils<IndexType, ValueType>::localDijkstra( graph, 0, predecessor);

        EXPECT_EQ( shortDist.size(), N );

        //check specific distances for Grid4x4
        EXPECT_EQ( shortDist[15], 6);
        EXPECT_EQ( shortDist[7], 4);

        //PRINT0("set edge (14, 15) to 1.5");
        graph.setValue(5, 6, 1.5);
        graph.setValue(8, 12, 1.1);
        shortDist = GraphUtils<IndexType,ValueType>::localDijkstra( graph, 0, predecessor);
        EXPECT_EQ( shortDist[15], 6);

        //PRINT0("set edge (11, 15) to 0.5");
        graph.setValue(11, 15, 0.5);
        shortDist = GraphUtils<IndexType,ValueType>::localDijkstra( graph, 0, predecessor);
        EXPECT_EQ( shortDist[15], 5.5);
        EXPECT_EQ( predecessor[15], 11);

        //PRINT0("set edge (9, 10) to 0.3");
        graph.setValue(9, 10, 0.3);
        shortDist = GraphUtils<IndexType,ValueType>::localDijkstra( graph, 0, predecessor);
        EXPECT_NEAR( shortDist[15], 4.8, 1e-5);
        EXPECT_NEAR( shortDist[11], 4.3, 1e-5);
        EXPECT_EQ( predecessor[10], 9);
        EXPECT_EQ( predecessor[11], 10);
        EXPECT_EQ( predecessor[14], 10);

        graph.setValue(5, 9, 1.3);
        graph.setValue(7, 11, 1.3);
        shortDist = GraphUtils<IndexType,ValueType>::localDijkstra( graph, 0, predecessor);
        EXPECT_NEAR( shortDist[15], 4.8, 1e-5);
        EXPECT_NEAR( shortDist[11], 4.3, 1e-5);
        EXPECT_EQ( predecessor[9], 8);

        shortDist = GraphUtils<IndexType,ValueType>::localDijkstra( graph, 13, predecessor);
        EXPECT_EQ( shortDist[0], 4);
        EXPECT_EQ( shortDist[15], 2);
        EXPECT_NEAR( shortDist[7], 3.3, 1e-5);

        //for( int i=0; i<N; i++){
        //    PRINT0("dist to vertex " << i << "= " << shortDist[i]);
        //}
    }
    EXPECT_TRUE(executed ) << "too many PEs, must be <7 for this test" ;
}

//---------------------------------------------------------------------------------------

TYPED_TEST (GraphUtilsTest, testComputeCommVolumeAndBoundaryNodes) {
    using ValueType = TypeParam;

    std::string file = GraphUtilsTest<ValueType>::graphPath + "Grid32x32";
    IndexType dimensions = 2;
    IndexType N;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType k =comm->getSize();

    // read graph and coords
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    N= graph.getNumRows();
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.minGainForNextRound = 10;
    settings.storeInfo = false;

    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    std::vector<IndexType> commVolume;
    std::vector<IndexType> numBorderNodes;
    std::vector<IndexType> numInnerNodes;

    std::tie( commVolume, numBorderNodes, numInnerNodes) = \
            ITI::GraphUtils<IndexType,ValueType>::computeCommBndInner( graph, partition, settings );

    SCAI_ASSERT_EQ_ERROR( commVolume.size(), numBorderNodes.size(), "size mismatch");

    for(int i=0; i< commVolume.size(); i++) {
        if( k<10) {
            PRINT0("block " << i << ": commVol= " << commVolume[i] << " , boundaryNodes= "<< numBorderNodes[i]);
        }
        EXPECT_LE( numBorderNodes[i], commVolume[i] ) << "Communication volume must be greater or equal than boundary nodes";
    }

}

//---------------------------------------------------------------------------------------

TYPED_TEST (GraphUtilsTest, testGraphMaxDegree) {

    using ValueType = TypeParam;
    const IndexType N = 1000;

    //define distributions
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));

    //generate random complete matrix
    auto graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDistPointer);

    for( int i=0; i<10; i++) {
        scai::lama::MatrixCreator::fillRandom(graph, i/9.0);

        IndexType maxDegree;
        maxDegree = GraphUtils<IndexType,ValueType>::getGraphMaxDegree(graph);
        //PRINT0("maxDegree= " << maxDegree);

        EXPECT_LE( maxDegree, N);
        EXPECT_LE( 0, maxDegree);
        if ( i==0 ) {
            EXPECT_EQ( maxDegree, 0);
        } else if( i==9 ) {
            EXPECT_EQ( maxDegree, N);
        }
    }
}

//---------------------------------------------------------------------------------------

TYPED_TEST (GraphUtilsTest,testEdgeList2CSR) {
    using ValueType = TypeParam;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType thisPE = comm->getRank();
    const IndexType numPEs = comm->getSize();

    const IndexType localM = 40+2*thisPE;
    const IndexType N = numPEs * 7;
    std::vector< std::pair<IndexType, IndexType>> localEdgeList( localM );

    srand( std::time(NULL)*thisPE );
    int x = numPEs==1 ? 1 : thisPE*(N/(numPEs-1));

    for(int i=0; i<localM; i++) {
        //to ensure that there are no isolated vertices
        IndexType v1 = (i + x)%N; //(rand())%N;
        IndexType v2 = (v1+rand())%N;
        localEdgeList[i] = std::make_pair( v1, v2 );
        //PRINT(thisPE << ": inserting edge " << v1 << " - " << v2 );
    }

    scai::lama::CSRSparseMatrix<ValueType> graph = GraphUtils<IndexType,ValueType>::edgeList2CSR( localEdgeList, comm );

    SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");
    EXPECT_TRUE( graph.checkSymmetry() );
}

//---------------------------------------------------------------------------------------
// trancated function
/*
TYPED_TEST(GraphUtilsTest, testIndexReordering){
    using ValueType = TypeParam;

	IndexType M = 1000;
	for( IndexType maxIndex = 100; maxIndex<M; maxIndex++){
		std::vector<IndexType> indices = GraphUtils::indexReorderCantor( maxIndex);
		//std::cout <<std::endl;

		EXPECT_EQ( indices.size(), maxIndex );

		IndexType indexSum = std::accumulate( indices.begin(), indices.end(), 0);
		EXPECT_EQ( indexSum, maxIndex*(maxIndex-1)/2);
	}

}
*/
//------------------------------------------------------------------------------------

TYPED_TEST(GraphUtilsTest, testNonLocalNeighbors) {
    using ValueType = TypeParam;

    std::string file = GraphUtilsTest<ValueType>::graphPath + "trace-00008.graph";

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );

    std::vector<IndexType> nonLocalN = GraphUtils<IndexType, ValueType>::nonLocalNeighbors( graph );

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    //kind of "obviously correct" test since this the same condition checked inside nonLocalNeighbors
    for( IndexType ind : nonLocalN ) {
        EXPECT_TRUE( not dist->isLocal(ind) );
    }

}
//------------------------------------------------------------------------------------

TYPED_TEST(GraphUtilsTest, testMEColoring_local) {
    using ValueType = TypeParam;

    std::string file = GraphUtilsTest<ValueType>::graphPath + "Grid8x8";
    //std::string file = GraphUtilsTest<ValueType>::graphPath + "delaunayTest.graph";
    //std::string file = GraphUtilsTest<ValueType>::graphPath + "bigtrace-00000.graph";

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    // read graph and coords
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    IndexType N = graph.getNumRows();
    IndexType M = graph.getNumValues()/2;
    IndexType colors;

    //add random weights to edges, too slow for bif graphs
    if(N<2000) {
        CSRStorage<ValueType>& storage = graph.getLocalStorage();
        IndexType localNumRows = graph.getLocalNumRows();
        IndexType localNumCols = graph.getLocalNumColumns();
        for(int i=0; i<localNumRows; i++) {
            for(int j=0; j<localNumCols; j++) {
                ValueType val = storage.getValue(i,j);
                if( val != 0) {
                    storage.setValue(i, j, rand()%N) ;
                }
            }
        }
    }

    if (!graph.getRowDistributionPtr()->isReplicated()) {
        const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));
        graph.redistribute(noDist, noDist);
    }

    std::chrono::time_point<std::chrono::steady_clock> start= std::chrono::steady_clock::now();
    //
    std::vector<std::vector<IndexType>> coloring = GraphUtils<IndexType,ValueType>::mecGraphColoring( graph, colors);
    //
    std::chrono::duration<double> elapTime = std::chrono::steady_clock::now() - start;

    EXPECT_EQ( coloring[0].size(), M);

    IndexType maxDegree =  GraphUtils<IndexType,ValueType>::getGraphMaxDegree( graph );
    EXPECT_LE(colors, 2*maxDegree);
    PRINT0("num colors: " << colors << " , max degree: " << maxDegree);

    IndexType maxNode0 = *std::max_element( coloring[0].begin(), coloring[0].end() );
    IndexType maxNode1 = *std::max_element( coloring[1].begin(), coloring[1].end() );
    EXPECT_LE(maxNode0,N-1);
    EXPECT_LE(maxNode1,N-1);
    EXPECT_GE(maxNode0,0);
    EXPECT_GE(maxNode1,0);

    IndexType minColors = *std::min_element( coloring[2].begin(), coloring[2].end() );
    IndexType maxColors = *std::max_element( coloring[2].begin(), coloring[2].end() );
    PRINT(*comm << ": "<< minColors << " -- " << maxColors);

    std::vector<ValueType> maxEdge( colors, 0);

    //Check that it is a valid coloring
    for(int col=0; col<colors; col++) {
        std::vector<int> alreadyColored(N, 0);
        for(int i=0; i<coloring[2].size(); i++) {
            if( coloring[2][i]== col ) {
                IndexType v0 = coloring[0][i];
                IndexType v1 = coloring[1][i];
                EXPECT_LE( v0, N-1);
                EXPECT_LE( v1, N-1);
                EXPECT_TRUE( alreadyColored[v0]==0 );
                EXPECT_TRUE( alreadyColored[v1]==0 );
                alreadyColored[v0] = 1;
                alreadyColored[v1] = 1;

                if( maxEdge[col] < graph.getValue(v0,v1)) {
                    maxEdge[col] = graph.getValue(v0,v1);
                }
            }
            //PRINT0(i << ": "<< coloring[2][i]);
        }
    }

    //ValueType sumEdgeWeight = std::accumulate(maxEdge.begin(), maxEdge.end(), 0.0);
}
//------------------------------------------------------------------------------------

TYPED_TEST(GraphUtilsTest, testImbalance) {
    using ValueType = TypeParam;

    std::string file = GraphUtilsTest<ValueType>::graphPath + "Grid8x8";
    const IndexType k = 4;

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    const IndexType localN = graph.getLocalNumRows();

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    //the partition
    DenseVector<IndexType> partition(dist, 0);

    //uniform node weights
    DenseVector<ValueType> nodeWeights( dist, 1);

    ValueType imbalance = -1.0;

    //
    // --------- balanced, no node weights
    //
    {
        scai::hmemo::WriteAccess<IndexType> wPart(partition.getLocalValues() );
        for(int i=0; i<localN; i++) {
            IndexType globalID = dist->local2Global(i);
            wPart[i] = globalID%k;
        }
    }

    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k );

    EXPECT_EQ( imbalance, 0 ); //for Grid8x8 and k=4

    //
    // --------- balanced, with uniform node weights
    //
    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k, nodeWeights );

    EXPECT_EQ( imbalance, 0 ); //for Grid8x8 and k=4

    //
    // --------- imbalanced, no node weights
    //
    {
        scai::hmemo::WriteAccess<IndexType> wPart(partition.getLocalValues() );
        for(int i=0; i<localN; i++) {
            IndexType globalID = dist->local2Global(i);
            wPart[i] = globalID%k;
            //For Grid8x8, with N=64, block 1 will have 24 point and block 0, 8
            if( globalID%(2*k)==0 ) {
                wPart[i] = 1;
            }
        }
    }

    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k );

    //std::cout << "imbalance= " << imbalance << std::endl;
    EXPECT_EQ( imbalance, 0.5 ); //for Grid8x8 and k=4

    //
    // --------- balanced, with node weights
    //
    {
        scai::hmemo::WriteAccess<IndexType> wPart(partition.getLocalValues() );
        scai::hmemo::WriteAccess<ValueType> wWeights(nodeWeights.getLocalValues() );
        for(int i=0; i<localN; i++) {
            IndexType globalID = dist->local2Global(i);
            wPart[i] = globalID%k;
            wWeights[i] = globalID%k+1;
        }
    }
    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k, nodeWeights );

    std::cout << "imbalance= " << imbalance << std::endl;
    //total sum: 16+32+48+64 = 160
    //max block weight = 64
    //optimum size = total/k +(max-min) = 160/4 + (4-1) = 43
    //imbalance = (max-opt)/opt
    EXPECT_NEAR( imbalance, (64-43)/43.0, 1e-5 ); //for Grid8x8 and k=4

    //
    // --------- heterogeneous
    //

    //std::vector<ValueType> blockSizes = { 16.0/160, 32.0/160, 48.0/160, 64.0/160 };

    //balanced for the previous case
    std::vector<ValueType> blockSizes = { 16, 32, 48, 64 };
    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k, nodeWeights, blockSizes );

    //std::cout << "imbalance= " << imbalance << std::endl;
    EXPECT_EQ( imbalance, 0 );

    //imbalanced

    blockSizes = { 16.0, 32.0, 40.0, 64.0 };
    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k, nodeWeights, blockSizes );

    // (48-40)/48 = 8/40
    EXPECT_NEAR( imbalance, 8.0/40, 1e-5 );

    //imbalanced 2

    // all weights changed but the most imbalanced is the first, its actual weight is 16
    blockSizes = { 10.0, 30.0, 40.0, 60.0 };
    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k, nodeWeights, blockSizes );

    // (16-10)/10 = 6/10
    EXPECT_NEAR( imbalance, 6.0/10, 1e-5 );
}
//------------------------------------------------------------------------------

TYPED_TEST ( GraphUtilsTest, testGetPEGraph) {
    using ValueType = TypeParam;

    std::string file = GraphUtilsTest<ValueType>::graphPath + "trace-00008.graph";
    //std::string file = GraphUtilsTest<ValueType>::graphPath + "Grid8x8";
    std::ifstream f(file);
    IndexType dimensions= 2, k;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.minGainForNextRound = 100;
    settings.noRefinement = true;
    settings.initialPartition = Tool::geoSFC;
    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    //get the PE graph
    scai::lama::CSRSparseMatrix<ValueType> PEgraph =  GraphUtils<IndexType, ValueType>::getPEGraph( graph );
    EXPECT_EQ( PEgraph.getNumColumns(), comm->getSize() );
    EXPECT_EQ( PEgraph.getNumRows(), comm->getSize() );
    EXPECT_TRUE( PEgraph.checkSymmetry() );

    // if local number of columns and rows equal comm->getSize() must mean that graph is not distributed but replicated, TODO:not sure
    EXPECT_EQ( PEgraph.getLocalNumColumns(), comm->getSize() );
    EXPECT_EQ( PEgraph.getLocalNumRows(), 1 );
}
//------------------------------------------------------------------------------

TYPED_TEST ( GraphUtilsTest, testGetBlockGraph) {
    using ValueType = TypeParam;

    std::string file = GraphUtilsTest<ValueType>::graphPath + "trace-00008.graph";
    //std::string file = GraphUtilsTest<ValueType>::graphPath + "Grid8x8";
    std::ifstream f(file);
    IndexType dimensions= 2, k;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    k = comm->getSize()+2; //just something not equal p
    //

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.1;
    settings.dimensions = dimensions;
    //settings.minGainForNextRound = 100;
    settings.noRefinement = true;
    settings.initialPartition = Tool::geoKmeans;
    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    scai::lama::CSRSparseMatrix<ValueType> blockGraph1, blockGraph2;
    ValueType edgeSum = 0;

    blockGraph1 = GraphUtils<IndexType, ValueType>::getBlockGraph_dist( graph, partition, k);

    blockGraph2 = GraphUtils<IndexType, ValueType>::getBlockGraph( graph, partition, k);

    //graphs should be identical
    for(int i=0; i<k; i++ ) {
        for( int j=0; j<k; j++) {
            //PRINT0(i << ", " << j << ": " << blockGraph.getValue(i,j) );
            EXPECT_EQ( blockGraph1.getValue(i,j), blockGraph2.getValue(i,j) );
            edgeSum += blockGraph1.getValue(i,j);
        }
    }

    //checks
    EXPECT_EQ( blockGraph1.getNumRows(), k );
    EXPECT_TRUE( blockGraph1.checkSymmetry() );
    EXPECT_TRUE( blockGraph1.isConsistent() );

    //check if matrix is same in all PEs
    for(int i=0; i<k; i++) {
        for(int j=0; j<k; j++) {
            ValueType myVal = blockGraph1.getValue(i,j);
            ValueType sumVal = comm->sum(myVal);
            EXPECT_EQ( sumVal, myVal*comm->getSize() ) \
                    << " for position ["<<i <<"," << j << "]. PE " << comm->getRank() <<", myVal= " << myVal ;
        }
    }

    ValueType cut = GraphUtils<IndexType, ValueType>::computeCut(graph, partition, true);
    EXPECT_EQ( cut*2, edgeSum );
}
//------------------------------------------------------------------------------

TYPED_TEST ( GraphUtilsTest, testPEGraphBlockGraph_k_equal_p_Distributed) {
    using ValueType = TypeParam;
    
    //std::string file = GraphUtilsTest<ValueType>::graphPath + "Grid16x16";
    std::string file = GraphUtilsTest<ValueType>::graphPath + "trace-00008.graph";
    std::ifstream f(file);
    IndexType dimensions= 2, k;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed

    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.minGainForNextRound = 100;
    //settings.noRefinement = true;
    settings.initialPartition = Tool::geoSFC;
    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition(dist, -1);
    partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    //get the PE graph
    scai::lama::CSRSparseMatrix<ValueType> PEgraph =  GraphUtils<IndexType, ValueType>::getPEGraph( graph);
    EXPECT_EQ( PEgraph.getNumColumns(), comm->getSize() );
    EXPECT_EQ( PEgraph.getNumRows(), comm->getSize() );
    EXPECT_TRUE( PEgraph.checkSymmetry() );

    scai::dmemo::DistributionPtr noPEDistPtr(new scai::dmemo::NoDistribution( comm->getSize() ));
    PEgraph.redistribute(noPEDistPtr, noPEDistPtr);

    // if local number of columns and rows equal comm->getSize() must mean that graph is not distributed but replicated
    EXPECT_EQ( PEgraph.getLocalNumColumns(), comm->getSize() );
    EXPECT_EQ( PEgraph.getLocalNumRows(), comm->getSize() );
    EXPECT_EQ( comm->getSize()* PEgraph.getLocalNumValues(),  comm->sum( PEgraph.getLocalNumValues()) );
    EXPECT_TRUE( noPEDistPtr->isReplicated() );
    //test getBlockGraph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType, ValueType>::getBlockGraph( graph, partition, k);

    //when k=p block graph and PEgraph should be equal
    EXPECT_EQ( PEgraph.getNumColumns(), blockGraph.getNumColumns() );
    EXPECT_EQ( PEgraph.getNumRows(), blockGraph.getNumRows() );
    EXPECT_EQ( PEgraph.getNumRows(), k);
    EXPECT_TRUE( blockGraph.checkSymmetry() );

    // !! this check is extremly costly !!
    for(IndexType i=0; i<PEgraph.getNumRows() ; i++) {
        for(IndexType j=0; j<PEgraph.getNumColumns(); j++) {
            //PRINT0( "("<<i <<", "<< j <<") = "<< PEgraph(i,j) << " __ " << blockGraph(i,j) );
            EXPECT_EQ( PEgraph(i,j), blockGraph(i,j) ) << "i, j: " << i << ", " << j;
        }
    }

    //print
    /*
    std::cout<<"----------------------------"<< " PE graph  "<< *comm << std::endl;
    for(IndexType i=0; i<PEgraph.getNumRows(); i++){
        std::cout<< *comm<< ":";
        for(IndexType j=0; j<PEgraph.getNumColumns(); j++){
            std::cout<< PEgraph(i,j) << "-";
        }
        std::cout<< std::endl;
    }
    */
}
//------------------------------------------------------------------------------

} //namespace
