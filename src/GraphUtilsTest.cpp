#include <scai/lama.hpp>

#include "gtest/gtest.h"

#include "ParcoRepart.h"
#include "FileIO.h"
#include "GraphUtils.h"

#include <scai/dmemo/CyclicDistribution.hpp>
#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>

namespace ITI {

class GraphUtilsTest : public ::testing::Test {
    protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";
};

TEST_F(GraphUtilsTest, testReindexCut){
    std::string fileName = "trace-00008.graph";
	//std::string fileName = "delaunayTest.graph";

    std::string file = graphPath + fileName;
    
    const IndexType dimensions= 2;
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType n = graph.getNumRows();

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    // for now local refinement requires k = P
    const IndexType k = comm->getSize();
    
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), n, dimensions);
    ASSERT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));

    //get sfc partition
    Settings settings;
    settings.numBlocks = k;
    settings.noRefinement = true;
    settings.dimensions = dimensions;

    DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings);
	
	//WARNING: with the noRefinement flag the partition is not destributed
	partition.redistribute( dist);
	
	ASSERT_TRUE( coords[0].getDistributionPtr()->isEqual(*dist) );
	ASSERT_TRUE( partition.getDistributionPtr()->isEqual(*dist) );
    //scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));    

    //get first cut
    ValueType initialCut = GraphUtils<IndexType, ValueType>::computeCut(graph, partition, true);
    ASSERT_GE(initialCut, 0);
    ValueType sumNonLocalInitial = ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(graph, true);

	PRINT("about to reindex the graph");
    //now reindex and get second cut
    GraphUtils<IndexType, ValueType>::reindex(graph);
    ValueType sumNonLocalAfterReindexing = ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(graph, true);
    EXPECT_EQ(sumNonLocalInitial, sumNonLocalAfterReindexing);

    DenseVector<IndexType> reIndexedPartition = DenseVector<IndexType>(graph.getRowDistributionPtr(), partition.getLocalValues());
    ASSERT_TRUE(reIndexedPartition.getDistributionPtr()->isEqual(*graph.getRowDistributionPtr()));

    ValueType secondCut = GraphUtils<IndexType, ValueType>::computeCut(graph, reIndexedPartition, true);

    EXPECT_EQ(initialCut, secondCut);
}
//-----------------------------------------------------------------

TEST_F(GraphUtilsTest, testConstructLaplacian) {
    std::string fileName = "bubbles-00010.graph";
    std::string file = graphPath + fileName;
    const CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType n = graph.getNumRows();
    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();

    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
    scai::dmemo::DistributionPtr cyclidCist(new scai::dmemo::CyclicDistribution(n, 10, comm));
    ASSERT_TRUE(graph.isConsistent());

    CSRSparseMatrix<ValueType> L = GraphUtils<IndexType, ValueType>::constructLaplacian(graph);

    ASSERT_EQ(L.getRowDistribution(), graph.getRowDistribution());
    ASSERT_TRUE(L.isConsistent());

    //test that L*1 = 0
    DenseVector<ValueType> x( n, 1 );
    DenseVector<ValueType> y = scai::lama::eval<DenseVector<ValueType>>( L * x );

    ValueType norm = y.maxNorm();
    EXPECT_EQ(norm,0);

    //test consistency under distributions
    const CSRSparseMatrix<ValueType> replicatedGraph = scai::lama::distribute<CSRSparseMatrix<ValueType>>(graph, noDist, noDist);
    CSRSparseMatrix<ValueType> LFromReplicated = GraphUtils<IndexType, ValueType>::constructLaplacian(replicatedGraph);
    LFromReplicated.redistribute(L.getRowDistributionPtr(), L.getColDistributionPtr());
    CSRSparseMatrix<ValueType> diff = scai::lama::eval<CSRSparseMatrix<ValueType>> (LFromReplicated - L);
    EXPECT_EQ(0, diff.l2Norm());

    //sum is zero
    EXPECT_EQ(0, comm->sum(scai::utilskernel::HArrayUtils::sum(L.getLocalStorage().getValues())) );
}

TEST_F(GraphUtilsTest, benchConstructLaplacian) {
    std::string fileName = "bubbles-00010.graph";
    std::string file = graphPath + fileName;
    const CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );

    CSRSparseMatrix<ValueType> L = GraphUtils<IndexType, ValueType>::constructLaplacian(graph);
}

TEST_F(GraphUtilsTest, DISABLED_benchConstructLaplacianBig) {
    std::string fileName = "hugebubbles-00000.graph";
    std::string file = graphPath + fileName;
    const scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );

    CSRSparseMatrix<ValueType> L = GraphUtils<IndexType, ValueType>::constructLaplacian(graph);
}

//TODO: test also with edge weights

//-----------------------------------------------------------------

TEST_F (GraphUtilsTest, testLocalDijkstra){

    std::string file = graphPath + "Grid4x4";
    IndexType dimensions = 2;
    IndexType N;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
	if( comm->getSize() != 1 ){
		if( comm->getRank() == 0){
			std::cout << "\n\t\t### WARNING: this test, " << __FUNCTION__ << " will fail if run with multiple PES" << std::endl<< std::endl;
		}		
	}
	ASSERT_EQ( comm->getSize(), 1) << "Specific number of processors needed for this test: 1";

    // read graph
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    N= graph.getNumRows();

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
    EXPECT_EQ( shortDist[15], 4.8);
    EXPECT_EQ( shortDist[11], 4.3);
    EXPECT_EQ( predecessor[10], 9);
    EXPECT_EQ( predecessor[11], 10);
    EXPECT_EQ( predecessor[14], 10);

    graph.setValue(5, 9, 1.3);
    graph.setValue(7, 11, 1.3);
    shortDist = GraphUtils<IndexType,ValueType>::localDijkstra( graph, 0, predecessor);
    EXPECT_EQ( shortDist[15], 4.8);
    EXPECT_EQ( shortDist[11], 4.3);
    EXPECT_EQ( predecessor[9], 8);

    shortDist = GraphUtils<IndexType,ValueType>::localDijkstra( graph, 13, predecessor);
    EXPECT_EQ( shortDist[0], 4);
    EXPECT_EQ( shortDist[15], 2);
    EXPECT_EQ( shortDist[7], 3.3);

    //for( int i=0; i<N; i++){
    //    PRINT0("dist to vertex " << i << "= " << shortDist[i]);
    //}
}

//--------------------------------------------------------------------------------------- 

TEST_F (GraphUtilsTest, testComputeCommVolumeAndBoundaryNodes){
 
    std::string file = graphPath + "Grid32x32";
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
    
    struct Metrics metrics(settings);
    
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);
    
    std::vector<IndexType> commVolume;
    std::vector<IndexType> numBorderNodes;
    std::vector<IndexType> numInnerNodes;

	std::tie( commVolume, numBorderNodes, numInnerNodes) = \
		ITI::GraphUtils<IndexType,ValueType>::computeCommBndInner( graph, partition, settings );
    
    SCAI_ASSERT_EQ_ERROR( commVolume.size(), numBorderNodes.size(), "size mismatch");
    
    for(int i=0; i< commVolume.size(); i++){
        if( k<10){
            PRINT0("block " << i << ": commVol= " << commVolume[i] << " , boundaryNodes= "<< numBorderNodes[i]);
        }
        EXPECT_LE( numBorderNodes[i], commVolume[i] ) << "Communication volume must be greater or equal than boundary nodes";
    }
    
}

//--------------------------------------------------------------------------------------- 

TEST_F (GraphUtilsTest, testGraphMaxDegree){
    
    const IndexType N = 1000;
    
    //define distributions
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));

    //generate random complete matrix
    auto graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDistPointer);
    
    for( int i=0; i<10; i++){
        scai::lama::MatrixCreator::fillRandom(graph, i/9.0);
    
        IndexType maxDegree;
        maxDegree = GraphUtils<IndexType,ValueType>::getGraphMaxDegree(graph);
        //PRINT0("maxDegree= " << maxDegree);
        
        EXPECT_LE( maxDegree, N);
        EXPECT_LE( 0, maxDegree);
        if ( i==0 ){
            EXPECT_EQ( maxDegree, 0);
        }else if( i==9 ){
            EXPECT_EQ( maxDegree, N);
        }
    }
}

//--------------------------------------------------------------------------------------- 

TEST_F (GraphUtilsTest,testEdgeList2CSR){
	
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType thisPE = comm->getRank();
	const IndexType numPEs = comm->getSize();
	
    const IndexType localM = 10;	
	const IndexType N = numPEs * 4;
	std::vector< std::pair<IndexType, IndexType>> localEdgeList( localM );

	srand( std::time(NULL)*thisPE );
	
    for(int i=0; i<localM; i++){
		//IndexType v1 = i;
		IndexType v1 = (rand())%N;
		IndexType v2 = (v1+rand())%N;
		localEdgeList[i] = std::make_pair( v1, v2 );
		//PRINT(thisPE << ": inserting edge " << v1 << " - " << v2 );
	}
	
	scai::lama::CSRSparseMatrix<ValueType> graph = GraphUtils<IndexType,ValueType>::edgeList2CSR( localEdgeList );
	
	SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");
	EXPECT_TRUE( graph.checkSymmetry() );
}

//--------------------------------------------------------------------------------------- 
// trancated function
/*
TEST_F(GraphUtilsTest, testIndexReordering){
	
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

TEST_F(GraphUtilsTest, testNonLocalNeighbors){
	std::string file = graphPath + "trace-00008.graph";
	IndexType dimensions = 2;

	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );

	std::vector<IndexType> nonLocalN = GraphUtils<IndexType, ValueType>::nonLocalNeighbors( graph );
	
	scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

	//kind of "obviously correct" test since this the same condition checked inside nonLocalNeighbors
	for( IndexType ind : nonLocalN ){
		EXPECT_TRUE( not dist->isLocal(ind) );
	}

}
//------------------------------------------------------------------------------------ 

TEST_F(GraphUtilsTest, testMEColoring_local){
    std::string file = graphPath + "Grid8x8";
    //std::string file = graphPath + "delaunayTest.graph";
    //std::string file = graphPath + "bigtrace-00000.graph";
    IndexType dimensions = 2;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    // read graph and coords
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    IndexType N = graph.getNumRows();
    IndexType M = graph.getNumValues()/2;
    IndexType colors;

    //add random weights to edges, too slow for bif graphs
    if(N<2000){
        CSRStorage<ValueType>& storage = graph.getLocalStorage();
        IndexType localNumRows = graph.getLocalNumRows();
        IndexType localNumCols = graph.getLocalNumColumns();
        for(int i=0; i<localNumRows; i++){
            for(int j=0; j<localNumCols; j++){
                ValueType val = storage.getValue(i,j);
                if( val != 0){
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
    ValueType ourTime = elapTime.count();

    EXPECT_EQ( coloring[0].size(), M);

    IndexType maxDegree =  GraphUtils<IndexType,ValueType>::getGraphMaxDegree( graph );
    EXPECT_LE(colors, 2*maxDegree);
    PRINT0("num colors: " << colors << " , max degree: " << maxDegree);

    IndexType maxNode0 = *std::max_element( coloring[0].begin(), coloring[0].end() );
    IndexType maxNode1 = *std::max_element( coloring[1].begin(), coloring[1].end() );
    IndexType minNode0 = *std::min_element( coloring[0].begin(), coloring[0].end() );
    IndexType minNode1 = *std::min_element( coloring[1].begin(), coloring[1].end() );
    EXPECT_LE(maxNode0,N-1);
    EXPECT_LE(maxNode1,N-1);
    EXPECT_GE(maxNode0,0);
    EXPECT_GE(maxNode1,0);

    IndexType minColors = *std::min_element( coloring[2].begin(), coloring[2].end() );
    IndexType maxColors = *std::max_element( coloring[2].begin(), coloring[2].end() );
    PRINT(*comm << ": "<< minColors << " -- " << maxColors);

    std::vector<ValueType> maxEdge( colors, 0);

    //Check that it is a valid coloring
    for(int col=0; col<colors; col++){
        vector<int> alreadyColored(N, 0);
        for(int i=0; i<coloring[2].size(); i++){
            if( coloring[2][i]== col ){
                IndexType v0 = coloring[0][i];
                IndexType v1 = coloring[1][i];
                EXPECT_LE( v0, N-1);
                EXPECT_LE( v1, N-1);
                EXPECT_TRUE( alreadyColored[v0]==0 );
                EXPECT_TRUE( alreadyColored[v1]==0 );
                alreadyColored[v0] = 1;
                alreadyColored[v1] = 1;

                if( maxEdge[col] < graph.getValue(v0,v1)){
                    maxEdge[col] = graph.getValue(v0,v1);
                }
            }
        //PRINT0(i << ": "<< coloring[2][i]);
        }
    }

    ValueType sumEdgeWeight = std::accumulate(maxEdge.begin(), maxEdge.end() , 0.0);
}
//------------------------------------------------------------------------------------ 

TEST_F(GraphUtilsTest, testBetweennessCentrality){
    //std::string file = graphPath + "trace-00008.graph";
    std::string file = graphPath + "Grid8x8";

    IndexType dimensions = 2;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    // read graph and coords
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    IndexType N = graph.getNumRows();
    IndexType M = graph.getNumValues()/2;

    std::chrono::time_point<std::chrono::steady_clock> start= std::chrono::steady_clock::now();
    //
    std::vector<ValueType> betwCentr = GraphUtils<IndexType,ValueType>::getBetweennessCentrality( graph, false);
    //
    std::chrono::duration<double> elapTime = std::chrono::steady_clock::now() - start;
    ValueType ourTime = elapTime.count();
    std::cout << "time for getting betweenness " << ourTime << std::endl;

    EXPECT_EQ( betwCentr.size() , N);

    std::vector<IndexType> IDs(N,0);
    std::iota( IDs.begin(), IDs.end(), 0);
    std::sort(IDs.begin(), IDs.end(), [&](IndexType i, IndexType j){return betwCentr[i] > betwCentr[j];});

    std::cout << "Top-5 nodes" << std::endl;
    for(int i=0; i<5; i++)
        std::cout<<IDs[i] << ": " << betwCentr[IDs[i]] << std::endl;

}
//------------------------------------------------------------------------------------ 

TEST_F(GraphUtilsTest, testImbalance){

    std::string file = graphPath + "Grid8x8";
    const IndexType dimensions = 2;
    const IndexType k = 4;

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    const IndexType N = graph.getNumRows();
    const IndexType localN = graph.getLocalNumRows();

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType thisPE = comm->getRank();

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
        for(int i=0; i<localN; i++){
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
        for(int i=0; i<localN; i++){
            IndexType globalID = dist->local2Global(i);
            wPart[i] = globalID%k;
            //For Grid8x8, with N=64, block 1 will have 24 point and block 0, 8
            if( globalID%(2*k)==0 ){
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
        for(int i=0; i<localN; i++){
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
    EXPECT_EQ( imbalance, (64-43)/43.0 ); //for Grid8x8 and k=4

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
    EXPECT_EQ( imbalance, 8.0/40 );

    //imbalanced 2

    // all weights changed but the most imbalanced is the first, its actual weight is 16
    blockSizes = { 10.0, 30.0, 40.0, 60.0 };
    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, k, nodeWeights, blockSizes );

    // (16-10)/10 = 6/10
    EXPECT_EQ( imbalance, 6.0/10 );
}
//------------------------------------------------------------------------------

TEST_F ( GraphUtilsTest, testPEGraphBlockGraph_k_equal_p_Distributed) {
    //std::string file = graphPath + "Grid16x16";
    std::string file = graphPath + "trace-00008.graph";
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
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.minGainForNextRound = 100;
    //settings.noRefinement = true;
    settings.initialPartition = InitialPartitioningMethods::SFC;
    struct Metrics metrics(settings);
    
    scai::lama::DenseVector<IndexType> partition(dist, -1);
    partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    //get the PE graph
    scai::lama::CSRSparseMatrix<ValueType> PEgraph =  GraphUtils<IndexType, ValueType>::getPEGraph( graph); 
    EXPECT_EQ( PEgraph.getNumColumns(), comm->getSize() );
    EXPECT_EQ( PEgraph.getNumRows(), comm->getSize() );
    EXPECT_TRUE( PEgraph.checkSymmetry() );
    
    scai::dmemo::DistributionPtr noPEDistPtr(new scai::dmemo::NoDistribution( comm->getSize() ));
    PEgraph.redistribute(noPEDistPtr , noPEDistPtr);
    
    // if local number of columns and rows equal comm->getSize() must mean that graph is not distributed but replicated
    EXPECT_EQ( PEgraph.getLocalNumColumns() , comm->getSize() );
    EXPECT_EQ( PEgraph.getLocalNumRows() , comm->getSize() );
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
    for(IndexType i=0; i<PEgraph.getNumRows() ; i++){
        for(IndexType j=0; j<PEgraph.getNumColumns(); j++){
            EXPECT_EQ( PEgraph(i,j), blockGraph(i,j) );
//PRINT0( "("<<i <<", "<< j <<") = "<< PEgraph(i,j) << " __ " << blockGraph(i,j) );
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

} //namespace
