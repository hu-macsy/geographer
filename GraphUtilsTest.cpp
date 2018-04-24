#include <scai/lama.hpp>

#include "gtest/gtest.h"

#include "ParcoRepart.h"
#include "FileIO.h"
#include "GraphUtils.h"

#include <scai/dmemo/CyclicDistribution.hpp>

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
		ITI::GraphUtils::computeCommBndInner( graph, partition, k );
    
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


} //namespace
