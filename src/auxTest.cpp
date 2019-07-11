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

class auxTest : public ::testing::TestWithParam<int> {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    std::string graphPath = projectRoot+"/meshes/";
};

TEST_F (auxTest, testInitialPartitions) {

    //std::string fileName = "trace-00008.graph";
    std::string fileName = "Grid64x64";
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

    if( comm->getRank()==0) {
        logF<< "Results for file " << fileName << std::endl;
        logF<< "node= "<< N << " , edges= "<< edges << " , blocks= "<< k<< std::endl<< std::endl;
        settings.print( logF, comm );
        settings.print( std::cout, comm );
    }
    //logF<< std::endl<< std::endl << "Only initial partition, no MultiLevel or LocalRefinement"<< std::endl << std::endl;

    //------------------------------------------- pixeled

    settings.noRefinement = true;

    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    PRINT0("Get a pixeled partition");
    // get a pixeledPartition
    scai::lama::DenseVector<IndexType> pixeledPartition = ParcoRepart<IndexType, ValueType>::pixelPartition(coordinates, settings);
    EXPECT_EQ( pixeledPartition.size(), N );

    //scai::dmemo::DistributionPtr newDist( new scai::dmemo::GeneralDistribution ( pixeledPartition.getDistribution(), pixeledPartition.getLocalValues() ) );
    scai::dmemo::DistributionPtr newDist = scai::dmemo::generalDistributionByNewOwners( pixeledPartition.getDistribution(), pixeledPartition.getLocalValues() );
    pixeledPartition.redistribute(newDist);
    graph.redistribute(newDist, noDistPointer);
    for (IndexType d = 0; d < dimensions; d++) {
        coordinates[d].redistribute(newDist);
    }

    if(dimensions==2) {
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"pixelPart");
    }
    //cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;
    cut = GraphUtils<IndexType, ValueType>::computeCut( graph, pixeledPartition);
    imbalance = GraphUtils<IndexType, ValueType>::computeImbalance( pixeledPartition, k);
    if(comm->getRank()==0) {
        logF<< "-- Initial Pixeled partition " << std::endl;
        logF<< "\tcut: " << cut << " , imbalance= "<< imbalance<< std::endl;
    }
    uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);
    scai::dmemo::HaloExchangePlan halo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(graph);
    Metrics metrics(settings);
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, pixeledPartition, uniformWeights, coordinates, halo, settings, metrics);
    if(dimensions==2) {
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"finalWithPixel");
    }
    cut = GraphUtils<IndexType, ValueType>::computeCut( graph, pixeledPartition);
    imbalance = GraphUtils<IndexType, ValueType>::computeImbalance( pixeledPartition, k);
    if(comm->getRank()==0) {
        logF<< "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        logF  << std::endl  << std::endl;
        std::cout << "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        std::cout  << std::endl  << std::endl;
    }

    //------------------------------------------- hilbert/sfc

    // the partitioning may redistribute the input graph
    graph.redistribute(dist, noDistPointer);
    for(int d=0; d<dimensions; d++) {
        coordinates[d].redistribute( dist );
    }
    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    PRINT0( "Get a hilbert/sfc partition");

    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(1, DenseVector<ValueType>(graph.getRowDistributionPtr(), 1));

    settings.initialPartition = ITI::Tool::geoSFC;
    settings.noRefinement = true;

    // get a hilbertPartition
    scai::lama::DenseVector<IndexType> hilbertPartition = ParcoRepart<IndexType, ValueType>::partitionGraph(coordinates, nodeWeights, settings, metrics);

    //newDist = scai::dmemo::DistributionPtr( new scai::dmemo::GeneralDistribution ( hilbertPartition.getDistribution(), hilbertPartition.getLocalValues() ) );
    /*scai::dmemo::DistributionPtr*/ newDist = scai::dmemo::generalDistributionByNewOwners( hilbertPartition.getDistribution(), hilbertPartition.getLocalValues() );
    hilbertPartition.redistribute(newDist);
    graph.redistribute(newDist, noDistPointer);
    for (IndexType d = 0; d < dimensions; d++) {
        coordinates[d].redistribute(newDist);
    }

    //aux::print2DGrid( graph, hilbertPartition );
    if(dimensions==2) {
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"hilbertPart");
    }
    cut = GraphUtils<IndexType, ValueType>::computeCut( graph, hilbertPartition);
    imbalance = GraphUtils<IndexType, ValueType>::computeImbalance( hilbertPartition, k);
    if(comm->getRank()==0) {
        logF<< "-- Initial Hilbert/sfc partition " << std::endl;
        logF<< "\tcut: " << cut << " , imbalance= "<< imbalance<< std::endl;
    }
    uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1);
    halo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(graph);

    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, hilbertPartition, uniformWeights, coordinates, halo, settings, metrics);
    if(dimensions==2) {
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, dimensions, destPath+"finalWithHilbert");
    }
    cut = GraphUtils<IndexType, ValueType>::computeCut( graph, hilbertPartition);
    imbalance = GraphUtils<IndexType, ValueType>::computeImbalance( hilbertPartition, k);
    if(comm->getRank()==0) {
        logF<< "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        logF  << std::endl  << std::endl;
        std::cout << "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
        std::cout  << std::endl  << std::endl;
    }

    if(comm->getRank()==0) {
        std::cout<< "Output files written in " << destPath << std::endl;
    }

}
//-----------------------------------------------------------------

TEST_F (auxTest, testPixelDistance) {

    typedef aux<IndexType,ValueType> aux;

    IndexType sideLen = 100;

    ValueType maxl2Dist = aux::pixelL2Distance2D(0,sideLen*sideLen-1, sideLen);
    PRINT( maxl2Dist );
    for(int i=0; i<sideLen*sideLen; i++) {
        //std::cout << "dist1(" << 0<< ", "<< i << ")= "<< aux::pixelDistance2D( 0, i, sideLen) << std::endl;
        EXPECT_LE(aux::pixelL1Distance2D( 0, i, sideLen), sideLen+sideLen-2);
        EXPECT_LE(aux::pixelL2Distance2D( 0, i, sideLen), maxl2Dist);
    }

    srand(time(NULL));
    IndexType pixel;
    std::tuple<IndexType, IndexType> coords2D;
    std::vector<IndexType> maxPoints= {sideLen, sideLen};
    do {
        pixel= rand()%(sideLen*(sideLen-2)) +2*sideLen;
        coords2D = aux::index2_2DPoint( pixel, maxPoints );
    } while( (std::get<0>(coords2D)>sideLen-4 and std::get<0>(coords2D)<4) and ( std::get<1>(coords2D)>sideLen-4 and std::get<1>(coords2D)<4) );

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

TEST_F(auxTest, testIndex2_3DPoint) {
    std::vector<IndexType> numPoints(3);

    srand(time(NULL));
    for(int i=0; i<3; i++) {
        numPoints[i] = (IndexType) (rand()%5 + 10);
    }
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];

    for(IndexType i=0; i<N; i++) {
        std::tuple<IndexType, IndexType, IndexType> ind = aux<IndexType,ValueType>::index2_3DPoint(i, numPoints);
        EXPECT_LE(std::get<0>(ind), numPoints[0]-1);
        EXPECT_LE(std::get<1>(ind), numPoints[1]-1);
        EXPECT_LE(std::get<2>(ind), numPoints[2]-1);
        EXPECT_GE(std::get<0>(ind), 0);
        EXPECT_GE(std::get<1>(ind), 0);
        EXPECT_GE(std::get<2>(ind), 0);
    }
}
//-----------------------------------------------------------------

TEST_F(auxTest, testIndex2_2DPoint) {
    std::vector<IndexType> numPoints= {9, 11};

    srand(time(NULL));
    for(int i=0; i<2; i++) {
        numPoints[i] = (IndexType) (rand()%5 + 10);
    }
    IndexType N= numPoints[0]*numPoints[1];

    for(IndexType i=0; i<N; i++) {
        std::tuple<IndexType, IndexType> ind = aux<IndexType,ValueType>::index2_2DPoint(i, numPoints);
        EXPECT_LE(std::get<0>(ind), numPoints[0]-1);
        EXPECT_LE(std::get<1>(ind), numPoints[1]-1);
        EXPECT_GE(std::get<0>(ind), 0);
        EXPECT_GE(std::get<1>(ind), 0);
    }
}
//-----------------------------------------------------------------

// trancated function
/*
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
*/

//-----------------------------------------------------------------


TEST_P(auxTest, testRedistributeFromPartition) {
    std::string fileName = "bubbles-00010.graph";
    std::string file = graphPath + fileName;

    const IndexType dimensions= 2;
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    const IndexType k = comm->getSize();
    //

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType N = graph.getNumRows();
    std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);

    const scai::dmemo::DistributionPtr inputDist  = graph.getRowDistributionPtr();
    const IndexType localN = inputDist->getLocalSize();

    DenseVector<ValueType> nodeWeights( inputDist, 1 ); //unit weights

    EXPECT_TRUE( coordinates[0].getDistributionPtr()->isEqual( *inputDist ) );

    srand( comm->getRank() ); //so not all PEs claim the same IDs
    //create a random partition
    DenseVector<IndexType> partition(inputDist, 0);
    for (IndexType i = 0; i < localN; i++) {
        //IndexType blockId = ( (rand() % k) % (comm->getRank()+1) )%k; //hevily imbalanced partition
        IndexType blockId = (rand() % k);
        partition.getLocalValues()[i] = blockId;
    }

    Settings settings;
    settings.numBlocks = comm->getSize();

    const bool useRedistributor = GetParam();
    const bool renumberPEs = not GetParam();

    //get some metrics of the current partition to verify that it does not change after renumbering
    std::pair<std::vector<IndexType>,std::vector<IndexType>> borderAndInnerNodes = GraphUtils<IndexType,ValueType>::getNumBorderInnerNodes( graph, partition, settings);
    ValueType cut = GraphUtils<IndexType,ValueType>::computeCut( graph, partition );
    ValueType imbalance = GraphUtils<IndexType,ValueType>::computeImbalance( partition, settings.numBlocks );
    PRINT0( "imbalance: " << imbalance );

    //redistribute

    scai::dmemo::DistributionPtr distFromPart = aux<IndexType,ValueType>::redistributeFromPartition(
                partition,
                graph,
                coordinates,
                nodeWeights,
                settings,
                useRedistributor,
                renumberPEs);

    //checks

    const scai::dmemo::DistributionPtr newDist = graph.getRowDistributionPtr();
    EXPECT_TRUE( nodeWeights.getDistribution().isEqual(*newDist) );//, "Distribution mismatch" );
    SCAI_ASSERT_ERROR( coordinates[0].getDistribution().isEqual(*newDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( partition.getDistribution().isEqual(*newDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( partition.getDistribution().isEqual(*distFromPart), "Distribution mismatch" );

    const IndexType newLocalN = newDist->getLocalSize();
    //PRINT( comm->getRank() <<": " << newLocalN );

    //the border nodes, inner nodes, cut and imbalance shoulb be the same
    std::pair<std::vector<IndexType>,std::vector<IndexType>> newborderAndInnerNodes = GraphUtils<IndexType,ValueType>::getNumBorderInnerNodes( graph, partition, settings);

    std::sort( borderAndInnerNodes.first.begin(), borderAndInnerNodes.first.end(), std::greater<IndexType>() ) ;
    std::sort( newborderAndInnerNodes.first.begin(), newborderAndInnerNodes.first.end(), std::greater<IndexType>() );
    EXPECT_EQ( borderAndInnerNodes.first, borderAndInnerNodes.first );

    std::sort( borderAndInnerNodes.second.begin(), borderAndInnerNodes.second.end() ) ;
    std::sort( newborderAndInnerNodes.second.begin(), newborderAndInnerNodes.second.end() );
    EXPECT_EQ( borderAndInnerNodes.second, borderAndInnerNodes.second );

    ValueType newCut = GraphUtils<IndexType,ValueType>::computeCut( graph, partition );
    EXPECT_EQ( cut, newCut );

    ValueType newImbalance = GraphUtils<IndexType,ValueType>::computeImbalance( partition, settings.numBlocks );
    EXPECT_EQ( imbalance, newImbalance );

    EXPECT_EQ( comm->sum(newLocalN), N);

    for (IndexType i = 0; i < newLocalN; i++) {
        SCAI_ASSERT_EQ_ERROR( partition.getLocalValues()[i], comm->getRank(), "error for i= " << i <<" and localN= "<< newLocalN );
        //PRINT(comm->getRank() << " : " << i << "- " << partition.getLocalValues()[i] );
    }

    //TODO: move to separate test?
    //benchmarking

    comm->synchronize();

    scai::dmemo::RedistributePlan redistPlan = scai::dmemo::redistributePlanByNewDistribution( distFromPart, inputDist );
    const IndexType sourceSz = redistPlan.getExchangeSourceSize();
    const IndexType targetSz = redistPlan.getExchangeTargetSize();

    IndexType globalSourceSz = comm->sum( sourceSz );
    IndexType globalTargetSz = comm->sum( targetSz );

    PRINT0( "renumbering: " <<renumberPEs  << ", globalSourceSz= " << globalSourceSz << ", globalTargetSz= " << globalTargetSz);
    //PRINT(*comm << " : " << sourceSz << " __ " << targetSz );

    comm->synchronize();

}


TEST_P(auxTest, benchmarkRedistributeFromPartition) {
    std::string fileName = "353off.graph";
    std::string file = graphPath + fileName;

    const IndexType dimensions= 2;
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    const IndexType k = comm->getSize();
    //

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    const IndexType N = graph.getNumRows();
    std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);

    const scai::dmemo::DistributionPtr inputDist  = graph.getRowDistributionPtr();
    const IndexType localN = inputDist->getLocalSize();
    PRINT( comm->getRank() << ": localN= " << localN);

    DenseVector<ValueType> nodeWeights( inputDist, 1 ); //unit weights

    EXPECT_TRUE( coordinates[0].getDistributionPtr()->isEqual( *inputDist ) );

    Settings settings;
    settings.numBlocks = k;
    settings.noRefinement = true;
    settings.dimensions = dimensions;
    settings.verbose = false;
    settings.debugMode = false;
    //settings.initialPartition = InitialPartitioningMethods::SFC;

    const DenseVector<IndexType> initPartition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coordinates, settings);

    scai::dmemo::DistributionPtr intermediateDist = initPartition.getDistributionPtr();

    const bool useRedistributor = GetParam();


    for( bool renumberPEs: vector<bool>({false, true}) ) {

        DenseVector<IndexType> partition = initPartition;

        std::chrono::time_point<std::chrono::steady_clock> start= std::chrono::steady_clock::now();
        //redistribute
        scai::dmemo::DistributionPtr distFromPart = aux<IndexType,ValueType>::redistributeFromPartition(
                    partition,
                    graph,
                    coordinates,
                    nodeWeights,
                    settings,
                    useRedistributor,
                    renumberPEs);

        std::chrono::duration<double> elapTime = std::chrono::steady_clock::now() - start;
        ValueType time = elapTime.count();
        ValueType globTime = comm->max( time );
        PRINT0("max elapsed time with redistributor: " << useRedistributor << " and renumber: " << renumberPEs << " is " << globTime );

        //checks

        const scai::dmemo::DistributionPtr newDist = graph.getRowDistributionPtr();
        EXPECT_TRUE( nodeWeights.getDistribution().isEqual(*newDist) );//, "Distribution mismatch" );
        SCAI_ASSERT_ERROR( coordinates[0].getDistribution().isEqual(*newDist), "Distribution mismatch" );
        SCAI_ASSERT_ERROR( partition.getDistribution().isEqual(*newDist), "Distribution mismatch" );
        SCAI_ASSERT_ERROR( partition.getDistribution().isEqual(*distFromPart), "Distribution mismatch" );

        const IndexType newLocalN = newDist->getLocalSize();
        //PRINT( comm->getRank() <<": new localN= " << newLocalN << " , newLocalN-localN = " << newLocalN-localN );

        //the border nodes, inner nodes, cut and imbalance shoulb be the same
        std::pair<std::vector<IndexType>,std::vector<IndexType>> newborderAndInnerNodes = GraphUtils<IndexType,ValueType>::getNumBorderInnerNodes( graph, partition, settings);

        comm->synchronize();
        //(targetDistribution, sourceDistribution)
        scai::dmemo::RedistributePlan redistPlan = scai::dmemo::redistributePlanByNewDistribution( distFromPart, intermediateDist );
        const IndexType sourceSz = redistPlan.getExchangeSourceSize();
        const IndexType targetSz = redistPlan.getExchangeTargetSize();

        IndexType globalSourceSz = comm->sum( sourceSz );
        IndexType globalTargetSz = comm->sum( targetSz );

        PRINT0( "renumbering: " <<renumberPEs  << ", globalSourceSz= " << globalSourceSz << ", globalTargetSz= " << globalTargetSz);
        //PRINT(*comm << " : " << sourceSz << " -- " << targetSz << " = " << sourceSz-targetSz);

        comm->synchronize();

    }

}

INSTANTIATE_TEST_CASE_P(InstantiationName,
                        auxTest,
                        testing::Values(true, false) );
}
