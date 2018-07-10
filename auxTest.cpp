#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/lama/Vector.hpp>

#include <algorithm>
#include <memory>
#include <cstdlib>
#include <numeric>
#include <chrono>

#include "MeshGenerator.h"
#include "FileIO.h"
#include "MultiLevel.h"
#include "ParcoRepart.h"
#include "LocalRefinement.h"
#include "SpectralPartition.h"
#include "GraphUtils.h"
#include "AuxiliaryFunctions.h"
#include "KMeans.h"
#include "Metrics.h"

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
    
    std::string fileName = "trace-00008.graph";
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
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"pixelPart");
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
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"finalWithPixel");
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
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"hilbertPart");
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
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"finalWithHilbert");
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


TEST_F(auxTest, testBenchIndexReordering){

	//const IndexType M = 51009; << overflow
    const IndexType M = 41009;
	
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();	
	std::chrono::time_point<std::chrono::high_resolution_clock> FYstart = std::chrono::high_resolution_clock::now();
	
	std::vector<IndexType> indices(M);
	const typename std::vector<IndexType>::iterator firstIndex = indices.begin();
	typename std::vector<IndexType>::iterator lastIndex = indices.end();
	std::iota(firstIndex, lastIndex, 0);
	GraphUtils::FisherYatesShuffle(firstIndex, lastIndex, M);
	
	std::chrono::duration<ValueType,std::ratio<1>> FYtime = std::chrono::high_resolution_clock::now() - FYstart;
	PRINT0("FisherYatesShuffle for " << M <<" points is " << FYtime.count() );
	
	for(int i=0; i<M; i++){
		EXPECT_LT(indices[i],M);
		EXPECT_GE(indices[i],0);
	}
	EXPECT_EQ( indices.size(), M );
	
	
	
	// cantor reordering
	std::chrono::time_point<std::chrono::high_resolution_clock> Cstart = std::chrono::high_resolution_clock::now();
	std::vector<IndexType> indicesCantor(M);
	indicesCantor = GraphUtils::indexReorderCantor( M );
		
	std::chrono::duration<ValueType,std::ratio<1>> Ctime = std::chrono::high_resolution_clock::now() - Cstart;
	PRINT0("Cantor for " << M <<" points is " << Ctime.count() );
	
	for(int i=0; i<M; i++){
		EXPECT_LT(indicesCantor[i],M);
		EXPECT_GE(indicesCantor[i],0);
	}
	EXPECT_EQ( indices.size(), M );
	
	//checks
	
	IndexType indexSumFY = std::accumulate( indices.begin(), indices.end(), 0);
	IndexType indexSumC = std::accumulate( indicesCantor.begin(), indicesCantor.end(), 0);
	// even with integer oveflow they should be the same, TODO: right?
	EXPECT_EQ( indexSumFY, indexSumC);
	
	
	//WARNING: integer overflow for bigger values
	if(M<60000){
		EXPECT_EQ( indexSumFY, M*(M-1)/2);
		EXPECT_EQ( indexSumC, M*(M-1)/2);
	}
		
}

//-----------------------------------------------------------------

//testing csr2edgelist converter and mec edge coloring
TEST_F(auxTest, testMEColoring_local){

    std::string file = graphPath+ "Grid8x8";
    std::ifstream f(file);
    IndexType dimensions= 2, k=16;
    IndexType N, edges;
    f >> N >> edges; 
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );  
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    
    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );
    
    //reading coordinates
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.initialPartition = InitialPartitioningMethods::SFC;
    struct Metrics metrics(settings.numBlocks);
    
    //get the partition
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);
    
    aux<IndexType,ValueType>::print2DGrid( graph, partition );

    //check distributions
    assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );

    scai::lama::CSRSparseMatrix<ValueType> processGraph = GraphUtils::getPEGraph<IndexType, ValueType>(graph);

    {

        const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(k) );
        if (!processGraph.getRowDistributionPtr()->isReplicated()) {
            PRINT0("Replicating process graph");
            processGraph.redistribute(noDist, noDist);
            //throw std::runtime_error("Input matrix must be replicated.");
        }

        if(comm->getRank()==0){
            for( int i=0; i<k; i++){
                for( int j=0; j<k; j++){
                    std::cout << processGraph.getLocalStorage().getValue(i,j) << ", ";
                }
                std::cout << std::endl;
            }
        }
    }

    EXPECT_EQ( processGraph.getNumRows(), k);//, "Wrong process graph num rows");

    IndexType colorsBoost, colorsMEC, maxDegree;
    
    std::vector<std::vector<IndexType>> coloringBoost = ParcoRepart<IndexType,ValueType>::getGraphEdgeColoring_local( processGraph, colorsBoost );

    std::vector<std::tuple<IndexType,IndexType,IndexType>> edgeList = GraphUtils::CSR2EdgeList_local<IndexType,ValueType>( processGraph, maxDegree );

    // undirected graph so every edge appears once in the edge list
    EXPECT_EQ( edgeList.size()*2, processGraph.getNumValues() ); 
    for( int i=0; i<edgeList.size(); i++){
        //if( std::get<0>(edgeList[i]) == std::get<1>(edgeList[i]) ){
        //    PRINT0("Edge " << i << " is a self-loop for vertex "<< std::get<0>(edgeList[i]) );
        //}
        PRINT0("edge " << i << ": (" << std::get<0>(edgeList[i]) << ", " << std::get<1>(edgeList[i]) << "), weight= " << std::get<2>(edgeList[i]));
    }

    std::vector< std::vector<IndexType>>  coloringMEC = ParcoRepart<IndexType,ValueType>::getGraphMEC_local( processGraph, colorsMEC );

}


//TODO: 11.04.18, Moritz has started a new branch for that. Remove tests and functions if not needed here
/*
TEST_F(auxTest, testBenchKMeansSFCCoords){
	//std::string fileName = "bubbles-00010.graph";
	std::string fileName = "Grid32x32";
	std::string graphFile = graphPath + fileName;
	std::string coordFile = graphFile + ".xyz";
	const IndexType dimensions = 2;
	
	//load graph and coords
	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );
	const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
	const IndexType n = graph.getNumRows();
	std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);
	
	DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);
	
	struct Settings settings;
	settings.dimensions = dimensions;
	settings.numBlocks = comm->getSize();
		
	std::chrono::time_point<std::chrono::high_resolution_clock> SFCstart = std::chrono::high_resolution_clock::now();
	DenseVector<IndexType> partitionSFC = KMeans::getPartitionWithSFCCoords<IndexType>( graph, coords, uniformWeights, settings);
	std::chrono::duration<ValueType,std::ratio<1>> SFCtime = std::chrono::high_resolution_clock::now() - SFCstart;
	ValueType time = comm->max( SFCtime.count() );
	PRINT0("time for the sfc coordinates: " << time );
	
	partitionSFC.redistribute( dist );
	struct Metrics metrics1(1);
	
	metrics1.getAllMetrics( graph, partitionSFC, uniformWeights, settings );
	printMetricsShort( metrics1, std::cout );
	
	ITI::aux<IndexType,ValueType>::print2DGrid(graph, partitionSFC  );
	
	// get k-means partition by copying and redistributing the original coords
	
	std::chrono::time_point<std::chrono::high_resolution_clock> startOrig = std::chrono::high_resolution_clock::now();
	DenseVector<IndexType> tempResult = ParcoRepart<IndexType, ValueType>::hilbertPartition(coords, settings);
	
	std::vector<DenseVector<ValueType> > coordinateCopy = coords;
	for (IndexType d = 0; d < dimensions; d++) {
		coordinateCopy[d].redistribute( tempResult.getDistributionPtr() );
	}
	
	const std::vector<IndexType> blockSizes(settings.numBlocks, n/settings.numBlocks);
	DenseVector<IndexType> partitionOrig = KMeans::computePartition(coordinateCopy, settings.numBlocks, uniformWeights, blockSizes, settings);
	
	std::chrono::duration<ValueType,std::ratio<1>> timeOrig = std::chrono::high_resolution_clock::now() - startOrig;
	time = comm->max( timeOrig.count() );
	PRINT0("time for the original coordinates: " << time );
	
	
	partitionOrig.redistribute( dist );
	
	struct Metrics metrics2(1);
	
	metrics2.getAllMetrics( graph, partitionOrig, uniformWeights, settings );
	printMetricsShort( metrics2, std::cout );
	
	ITI::aux<IndexType,ValueType>::print2DGrid(graph, partitionOrig  );
}
*/

}
