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
    std::string file = graphPath + fileName;
    
    IndexType dimensions= 2;
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
    DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings);
	
	//WARNING: with the noRefinement flag the partition is not destributed
	partition.redistribute( dist);
	
	ASSERT_TRUE( coords[0].getDistributionPtr()->isEqual(*dist) );
	ASSERT_TRUE( partition.getDistributionPtr()->isEqual(*dist) );
    //scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));    

    //get first cut
    ValueType initialCut = GraphUtils::computeCut<IndexType, ValueType>(graph, partition, true);
    ASSERT_GE(initialCut, 0);
    ValueType sumNonLocalInitial = ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(graph, true);

	PRINT("about to reindex the graph");
    //now reindex and get second cut
    GraphUtils::reindex<IndexType, ValueType>(graph);
    ValueType sumNonLocalAfterReindexing = ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(graph, true);
    EXPECT_EQ(sumNonLocalInitial, sumNonLocalAfterReindexing);

    DenseVector<IndexType> reIndexedPartition = DenseVector<IndexType>(graph.getRowDistributionPtr(), partition.getLocalValues());
    ASSERT_TRUE(reIndexedPartition.getDistributionPtr()->isEqual(*graph.getRowDistributionPtr()));

    ValueType secondCut = GraphUtils::computeCut<IndexType, ValueType>(graph, reIndexedPartition, true);

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

    CSRSparseMatrix<ValueType> L = GraphUtils::constructLaplacian<IndexType, ValueType>(graph);

    ASSERT_EQ(L.getRowDistribution(), graph.getRowDistribution());
    ASSERT_TRUE(L.isConsistent());

    //test that L*1 = 0
    DenseVector<ValueType> x( n, 1 );
    DenseVector<ValueType> y = scai::lama::eval<DenseVector<ValueType>>( L * x );

    ValueType norm = y.maxNorm();
    EXPECT_EQ(norm,0);

    //test consistency under distributions
    const CSRSparseMatrix<ValueType> replicatedGraph = scai::lama::distribute<CSRSparseMatrix<ValueType>>(graph, noDist, noDist);
    CSRSparseMatrix<ValueType> LFromReplicated = GraphUtils::constructLaplacian<IndexType, ValueType>(replicatedGraph);
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

    CSRSparseMatrix<ValueType> L = GraphUtils::constructLaplacian<IndexType, ValueType>(graph);
}

TEST_F(GraphUtilsTest, DISABLED_benchConstructLaplacianBig) {
    std::string fileName = "hugebubbles-00000.graph";
    std::string file = graphPath + fileName;
    const scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );

    CSRSparseMatrix<ValueType> L = GraphUtils::constructLaplacian<IndexType, ValueType>(graph);
}

//TODO: test also with edge weights

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
    
    struct Metrics metrics(settings.numBlocks);
    
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);
    
    std::vector<IndexType> commVolume;
    std::vector<IndexType> numBorderNodes;
    std::vector<IndexType> numInnerNodes;
    
	/*
	//older version, TODO:remove if these functions are no longer used	
	commVolume = ITI::GraphUtils::computeCommVolume( graph, partition, k );	
    std::tie( numBorderNodes, numInnerNodes) = ITI::GraphUtils::getNumBorderInnerNodes( graph, partition, settings);
    */
	
	std::tie( commVolume, numBorderNodes, numInnerNodes) = \
		ITI::GraphUtils::computeCommBndInner( graph, partition, settings );
    
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
        maxDegree = GraphUtils::getGraphMaxDegree<IndexType, ValueType>(graph);
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
	
	scai::lama::CSRSparseMatrix<ValueType> graph = GraphUtils::edgeList2CSR<IndexType, ValueType>( localEdgeList );
	
	SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");
	EXPECT_TRUE( graph.checkSymmetry() );
}

//--------------------------------------------------------------------------------------- 

TEST_F(GraphUtilsTest, testIndexReordering){
	
	IndexType M = 1000;
	for( IndexType maxIndex = 100; maxIndex<M; maxIndex++){
		std::vector<IndexType> indices = GraphUtils::indexReorderCantor( maxIndex);
		//std::cout <<std::endl;
		
		EXPECT_EQ( indices.size(), maxIndex );
		
		IndexType indexSum = std::accumulate( indices.begin(), indices.end(), 0);
		EXPECT_EQ( indexSum, maxIndex*(maxIndex-1)/2);
		/*
		if(maxIndex==15){
			for(int i=0; i<indices.size(); i++){
				std::cout<< i <<": " << indices[i]<<std::endl;
			}
		}
		*/
	}
	
}
//------------------------------------------------------------------------------------ 

TEST_F(GraphUtilsTest, testMEColoring_local){
    
    std::string file = graphPath + "delaunayTest.graph";
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
                //PRINT( *comm << ": "<< i <<", " <<j << " = "<< val);
            }
        }
    }

    if (!graph.getRowDistributionPtr()->isReplicated()) {
        const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));
        graph.redistribute(noDist, noDist);
    }

    std::chrono::time_point<std::chrono::steady_clock> start= std::chrono::steady_clock::now();
    //
    std::vector<std::vector<IndexType>> coloring = GraphUtils::mecGraphColoring<IndexType, ValueType>( graph, colors);
    //
    std::chrono::duration<double> elapTime = std::chrono::steady_clock::now() - start;
    ValueType ourTime = elapTime.count();

    EXPECT_EQ( coloring[0].size(), M);

    IndexType maxDegree =  GraphUtils::getGraphMaxDegree<IndexType, ValueType>( graph );
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
//                EXPECT_TRUE( alreadyColored[v0]==0 );
//                EXPECT_TRUE( alreadyColored[v1]==0 );
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

    // benchmarking
    //
    //take a coloring using Hasan code and compare the results
    //

    start= std::chrono::steady_clock::now();
    //
    IndexType colors2 = -1;
    std::vector<std::vector<IndexType>> coloring2 = ParcoRepart<IndexType, ValueType>::getGraphMEC_local( graph, colors2);
    //
    elapTime = std::chrono::steady_clock::now() - start;
    ValueType hasanTime = elapTime.count();

    std::vector<ValueType> maxEdge2( colors2, 0);
    //for( int i=0; i<3; i++ ){
        for( int j=0; j<coloring2[0].size(); j++ ){
          //  EXPECT_EQ( coloring[i][j], coloring2[i][j]);
//            PRINT0(coloring[0][j] << ", " << coloring[1][j] << ") -- " << coloring[2][j] <<  "  +=+=+=+  " << coloring2[0][j] << ", " << coloring2[1][j] << ") -- " << coloring2[2][j]);
            IndexType v0 = coloring2[0][j];
            IndexType v1 = coloring2[1][j];
            IndexType color = coloring2[2][j];

            if( maxEdge2[color] < graph.getValue(v0,v1)){
                maxEdge2[color] = graph.getValue(v0,v1);
            }
        }
        ValueType sumEdgeWeight2 = std::accumulate(maxEdge2.begin(), maxEdge2.end() , 0.0);
    //}

    PRINT0("colors, sumEdges: " << colors << ", " << sumEdgeWeight << " , in time " << ourTime);
    PRINT0("colors2, sumEdges2: " << colors2 << ", " << sumEdgeWeight2 << " , in time " << hasanTime);
}

} //namespace
