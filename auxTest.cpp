#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <algorithm>
#include <memory>
#include <cstdlib>
#include <numeric>

#include "MeshGenerator.h"
#include "FileIO.h"
#include "MultiLevel.h"
#include "ParcoRepart.h"
#include "LocalRefinement.h"
#include "SpectralPartition.h"
#include "GraphUtils.h"
#include "AuxiliaryFunctions.h"

#include "gtest/gtest.h"

#include <boost/filesystem.hpp>



using namespace scai;

namespace ITI {

class auxTest : public ::testing::Test {
protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";
};

TEST_F (auxTest, testInitialPartitions){
    
    std::string fileName = "bigtrace-00000.graph";
    std::string file = graphPath + fileName;
    std::ifstream f(file);
    IndexType dimensions= 2;
    IndexType N, edges;
    f >> N >> edges; 
    
    std::cout<< "node= "<< N << std::endl;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    IndexType k = comm->getSize();
    //
    
    std::string destPath = "./partResults/"+fileName+"/blocks_"+std::to_string(k)+"/";
    boost::filesystem::create_directories( destPath );    
    /*
    if( !boost::filesystem::create_directory( destPath ) ){
        throw std::runtime_error("Directory "+ destPath + " could not be created");
    }
    */
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );  
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    graph.redistribute(dist, noDistPointer);
    
    std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coordinates[0].getDistributionPtr()->isEqual(*dist));
    
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ(edges, (graph.getNumValues())/2 );   
    
    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.1;
    settings.dimensions = dimensions;
    settings.pixeledSideLen = 16;        //4 for a 16x16 coarsen graph in 2D, 16x16x16 in 3D
    settings.useGeometricTieBreaking =1;
    //settings.fileName = fileName;
    settings.useGeometricTieBreaking =1;
    settings.multiLevelRounds = 6;
    settings.minGainForNextRound= 10;
    // 5% of (approximetely, if at every round you get a 60% reduction in nodes) the nodes of the coarsest graph
    settings.minBorderNodes = N*std::pow(0.6, settings.multiLevelRounds)/k * 0.05;
    settings.coarseningStepsBetweenRefinement =2;
    
    DenseVector<ValueType> uniformWeights;
    
    ValueType cut;
    ValueType imbalance;
    
    std::string logFile = destPath + "results.log";
    std::ofstream logF(logFile);
    
    if( comm->getRank()==0){
        logF<< "Results for file " << fileName << std::endl;
        logF<< "node= "<< N << " , edges= "<< edges << " , blocks= "<< k<< std::endl<< std::endl;
        settings.print( logF, comm );
        settings.print( std::cout, comm );
    }
    //logF<< std::endl<< std::endl << "Only initial partition, no MultiLevel or LocalRefinement"<< std::endl << std::endl;
    
    //------------------------------------------- pixeled
    
    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    PRINT0("Get a pixeled partition");
    // get a pixeledPartition
    scai::lama::DenseVector<IndexType> pixeledPartition = ParcoRepart<IndexType, ValueType>::pixelPartition(coordinates, settings);
    
    scai::dmemo::DistributionPtr newDist( new scai::dmemo::GeneralDistribution ( pixeledPartition.getDistribution(), pixeledPartition.getLocalValues() ) );
    pixeledPartition.redistribute(newDist);
    graph.redistribute(newDist, noDistPointer);
	for (IndexType d = 0; d < dimensions; d++) {
		coordinates[d].redistribute(newDist);
	}

    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, N, dimensions, destPath+"pixelPart");
    }
    //cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;
    cut = GraphUtils::computeCut( graph, pixeledPartition);
    imbalance = GraphUtils::computeImbalance<IndexType, ValueType>( pixeledPartition, k);
    if(comm->getRank()==0){
        logF<< "-- Initial Pixeled partition " << std::endl;
        logF<< "\tcut: " << cut << " , imbalance= "<< imbalance<< std::endl;
    }
    uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);
	scai::dmemo::Halo halo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(graph);
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, pixeledPartition, uniformWeights, coordinates, halo, settings);
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, N, dimensions, destPath+"finalWithPixel");
    }
    cut = GraphUtils::computeCut( graph, pixeledPartition);
    imbalance = GraphUtils::computeImbalance<IndexType, ValueType>( pixeledPartition, k);
    if(comm->getRank()==0){
        logF<< "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        logF  << std::endl  << std::endl; 
        std::cout << "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        std::cout  << std::endl  << std::endl; 
    }
    
    //------------------------------------------- hilbert/sfc
    
    // the partitioning may redistribute the input graph
    graph.redistribute(dist, noDistPointer);
    for(int d=0; d<dimensions; d++){
        coordinates[d].redistribute( dist );
    }    
    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    PRINT0( "Get a hilbert/sfc partition");
    // get a hilbertPartition
    scai::lama::DenseVector<IndexType> hilbertPartition = ParcoRepart<IndexType, ValueType>::hilbertPartition(coordinates, settings);
    
    newDist = scai::dmemo::DistributionPtr( new scai::dmemo::GeneralDistribution ( hilbertPartition.getDistribution(), hilbertPartition.getLocalValues() ) );
    hilbertPartition.redistribute(newDist);
    graph.redistribute(newDist, noDistPointer);
	for (IndexType d = 0; d < dimensions; d++) {
		coordinates[d].redistribute(newDist);
	}

    //aux::print2DGrid( graph, hilbertPartition );
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, N, dimensions, destPath+"hilbertPart");
    }
    cut = GraphUtils::computeCut( graph, hilbertPartition);
    imbalance = GraphUtils::computeImbalance<IndexType, ValueType>( hilbertPartition, k);
    if(comm->getRank()==0){
        logF<< "-- Initial Hilbert/sfc partition " << std::endl;
        logF<< "\tcut: " << cut << " , imbalance= "<< imbalance<< std::endl;
    }
    uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);
	halo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(graph);

    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, hilbertPartition, uniformWeights, coordinates, halo, settings);
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, N, dimensions, destPath+"finalWithHilbert");
    }
    cut = GraphUtils::computeCut( graph, hilbertPartition);
    imbalance = GraphUtils::computeImbalance<IndexType, ValueType>( hilbertPartition, k);
    if(comm->getRank()==0){
        logF<< "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        logF  << std::endl  << std::endl; 
        std::cout << "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        std::cout  << std::endl  << std::endl; 
    }
   
    if(comm->getRank()==0){
        std::cout<< "Output files written in " << destPath << std::endl;
    }

}
//-----------------------------------------------------------------

TEST_F (auxTest,testGetBorderAndInnerNodes){
 
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
    
    std::vector<IndexType> numBorderNodes;
    std::vector<IndexType> numInnerNodes;
    
    std::tie( numBorderNodes, numInnerNodes) = ITI::GraphUtils::getNumBorderInnerNodes( graph, partition, settings);
    
    //assertions - prints
    SCAI_ASSERT_EQ_ERROR( numBorderNodes.size(), k, "Size of numBorderNodes is wrong");
    SCAI_ASSERT_EQ_ERROR( numInnerNodes.size(), k, "Size of numInnerNodes is wrong");
    
    if( comm->getRank()==0 ){
        for(int i=0; i<k; i++){
            std::cout<<"Block " << i << " has " << numBorderNodes[i] << " border nodes and " << numInnerNodes[i] << " inner nodes"<< std::endl;
        }
    }
    
    IndexType totalBorderNodes = std::accumulate( numBorderNodes.begin(), numBorderNodes.end(), 0);
    IndexType totalInnerNodes = std::accumulate( numInnerNodes.begin(), numInnerNodes.end(), 0);

    SCAI_ASSERT_EQ_ERROR( totalBorderNodes+totalInnerNodes, N, "Sum of nodes not correct" );
    
}
//-----------------------------------------------------------------

TEST_F (auxTest,testComputeCommVolume){
 
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
    
    std::vector<IndexType> commVolume = ITI::GraphUtils::computeCommVolume( graph, partition, k );
        
    std::vector<IndexType> numBorderNodes;
    std::vector<IndexType> numInnerNodes;
    
    std::tie( numBorderNodes, numInnerNodes) = ITI::GraphUtils::getNumBorderInnerNodes( graph, partition, settings);
    
    SCAI_ASSERT_EQ_ERROR( commVolume.size(), numBorderNodes.size(), "size mismatch");
    
    for(int i=0; i< commVolume.size(); i++){
        if( k<20){
            PRINT0("block " << i << ": commVol= " << commVolume[i] << " , boundaryNodes= "<< numBorderNodes[i]);
        }
        SCAI_ASSERT_LE_ERROR( numBorderNodes[i], commVolume[i], "Communication volume must be less than boundary nodes")
    }
    
}
//-----------------------------------------------------------------
 
TEST_F (auxTest,testGraphMaxDegree){
    
    const IndexType N = 1000;
    const IndexType k = 10;

    //define distributions
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));

    //generate random complete matrix
    scai::lama::CSRSparseMatrix<ValueType> graph(dist, noDistPointer);
    
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
//-----------------------------------------------------------------
 
TEST_F (auxTest,testEdgeList2CSR){
	
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
//-----------------------------------------------------------------

TEST_F (auxTest, testPixelDistance) {
    
    typedef aux<IndexType,ValueType> aux;
    
    IndexType sideLen = 100;
    
    ValueType maxl2Dist = aux::pixelL2Distance2D(0,sideLen*sideLen-1, sideLen);
    PRINT( maxl2Dist );
    for(int i=0; i<sideLen*sideLen; i++){
        //std::cout << "dist1(" << 0<< ", "<< i << ")= "<< aux::pixelDistance2D( 0, i, sideLen) << std::endl;
        EXPECT_LE(aux::pixelL1Distance2D( 0, i, sideLen), sideLen+sideLen-2);
        EXPECT_LE(aux::pixelL2Distance2D( 0, i, sideLen), maxl2Dist);
    }
    
    srand(time(NULL));
    IndexType pixel;
    std::tuple<IndexType, IndexType> coords2D;
    std::vector<IndexType> maxPoints= {sideLen, sideLen};
    do{
        pixel= rand()%(sideLen*(sideLen-2)) +2*sideLen;
        coords2D = aux::index2_2DPoint( pixel, maxPoints );
    }while( (std::get<0>(coords2D)>sideLen-4 or std::get<0>(coords2D)<4) and ( std::get<1>(coords2D)>sideLen-4 or std::get<1>(coords2D)<4) );
    
    //PRINT( std::get<0>(coords2D) << " , " << std::get<1>(coords2D) );
    
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel, sideLen), 0);
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel+1, sideLen), 1);
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel+sideLen, sideLen), 1);
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel-sideLen, sideLen), 1);
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel+sideLen-3, sideLen), 4);
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel-sideLen-2, sideLen), 3);
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel-2*sideLen+3, sideLen), 5);
    EXPECT_EQ(aux::pixelL1Distance2D( pixel, pixel-2*sideLen-3, sideLen), 5);
}
//-----------------------------------------------------------------

TEST_F(auxTest, testIndex2_3DPoint){
    std::vector<IndexType> numPoints(3);
    
    srand(time(NULL));
    for(int i=0; i<3; i++){
        numPoints[i] = (IndexType) (rand()%5 + 10);
    }
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];
    
    for(IndexType i=0; i<N; i++){
        std::tuple<IndexType, IndexType, IndexType> ind = aux<IndexType,ValueType>::index2_3DPoint(i, numPoints);
        EXPECT_LE(std::get<0>(ind) , numPoints[0]-1);
        EXPECT_LE(std::get<1>(ind) , numPoints[1]-1);
        EXPECT_LE(std::get<2>(ind) , numPoints[2]-1);
        EXPECT_GE(std::get<0>(ind) , 0);
        EXPECT_GE(std::get<1>(ind) , 0);
        EXPECT_GE(std::get<2>(ind) , 0);
    }
}
//-----------------------------------------------------------------

TEST_F(auxTest, testIndex2_2DPoint){
    std::vector<IndexType> numPoints= {9, 11};
    
    srand(time(NULL));
    for(int i=0; i<2; i++){
        numPoints[i] = (IndexType) (rand()%5 + 10);
    }
    IndexType N= numPoints[0]*numPoints[1];
    
    for(IndexType i=0; i<N; i++){
        std::tuple<IndexType, IndexType> ind = aux<IndexType,ValueType>::index2_2DPoint(i, numPoints);
        EXPECT_LE(std::get<0>(ind) , numPoints[0]-1);
        EXPECT_LE(std::get<1>(ind) , numPoints[1]-1);
        EXPECT_GE(std::get<0>(ind) , 0);
        EXPECT_GE(std::get<1>(ind) , 0);
    }
}
//-----------------------------------------------------------------

TEST_F(auxTest, testInexReordering){
	
	IndexType maxIndex = 10;
	std::vector<IndexType> indices = GraphUtils::indexReorder( maxIndex, maxIndex/3 );
	
	for(int i=0; i<indices.size(); i++){
		std::cout<< i <<": " << indices[i]<<std::endl;
	}
	
	std::cout << std::endl;
}

}
