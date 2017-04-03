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
#include "gtest/gtest.h"

#include <boost/filesystem.hpp>

#include "AuxiliaryFunctions.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class auxTest : public ::testing::Test {

};


TEST_F (auxTest, testMultiLevelStep_dist) {

    const IndexType N = 120;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution( N ));
    const IndexType k = comm->getSize();
    
    scai::lama::CSRSparseMatrix<ValueType> graph;
    
    // create random graph
    scai::common::scoped_array<ValueType> adjArray( new ValueType[ N*N ] );
    
    //initialize matrix with zeros
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            adjArray[i*N+j]=0;
        
    srand(time(NULL));
    IndexType numEdges = int (4*N);
    for(IndexType i=0; i<numEdges; i++){
        // a random position in the matrix
        IndexType x = rand()%N;
        IndexType y = rand()%N;
        adjArray[ x+y*N ]= 1;
        adjArray[ x*N+y ]= 1;
    }
    graph.setRawDenseData( N, N, adjArray.get() );
    EXPECT_TRUE( graph.isConsistent() );
    EXPECT_TRUE( graph.checkSymmetry() );
    ValueType beforel1Norm = graph.l1Norm().Scalar::getValue<ValueType>();
    IndexType beforeNumValues = graph.getNumValues();
    graph.redistribute( distPtr , noDistPtr);
    
    // node weights = 1
    DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);
    //ValueType beforeSumWeigths = uniformWeights.l1Norm().Scalar::getValue<ValueType>();
    IndexType beforeSumWeigths = N;
    //uniformWeights.redistribute( distPtr );
    
    //coordinates at random and redistribute
    std::vector<DenseVector<ValueType>> coords(2);
    for(IndexType i=0; i<2; i++){ 
	coords[i].allocate(N);
	coords[i] = static_cast<ValueType>( 0 );
        // set random coordinates
        for(IndexType j=0; j<N; j++){
            coords[i].setValue(j, rand()%10);
        }
        coords[i].redistribute( distPtr );
    }
    
    struct Settings Settings;
    Settings.numBlocks= k;
    Settings.epsilon = 0.2;
    Settings.multiLevelRounds= 3;
    Settings.coarseningStepsBetweenRefinement = 3;
    Settings.useGeometricTieBreaking = true;
    Settings.dimensions= 2;
    
    DenseVector<IndexType> partition= ParcoRepart<IndexType, ValueType>::hilbertPartition(graph, coords, Settings);
    //partition.redistribute( distPtr );
    
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, partition, uniformWeights, coords, Settings);
    //ITI::aux::multiLevelStep(graph, partition, uniformWeights, coords, Settings);
    
    EXPECT_EQ( graph.l1Norm() , beforel1Norm);
    EXPECT_EQ( graph.getNumValues() , beforeNumValues);
    DenseVector<ValueType> newWeights(N,0);
    assert(uniformWeights.size()==N);
    for(int i=0; i<N; i++){
        newWeights.setValue(i, uniformWeights.getValue(i) );
    }
    //PRINT(newWeights.l1Norm() );
    EXPECT_EQ( newWeights.l1Norm() , beforeSumWeigths );
    
}
//-------------------------------------------------------------------------

TEST_F (auxTest, testInitialPartitions){

    std::string path = "meshes/bubbles/";
    //std::string fileName = "hugetric-00006.graph";
    std::string fileName = "bubbles-00010.graph";
    std::string file = path + fileName;
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
    settings.pixeledDetailLevel = 6;        // for a 16x16 coarsen graph in 2D, 16x16x16 in 3D
    settings.useGeometricTieBreaking =1;
    settings.fileName = fileName;
    settings.useGeometricTieBreaking =1;
    settings.multiLevelRounds = 6;
    settings.minGainForNextRound= 10;
    settings.minBorderNodes = 1;
    
    DenseVector<IndexType> uniformWeights;
    
    ValueType cut;
    ValueType imbalance;
    
    std::string logFile = destPath + "resutls.log";
    std::ofstream logF(logFile);
    logF<< "Results for file " << fileName << std::endl;
    logF<< "node= "<< N << " , edges= "<< edges << " , blocks= "<< k<< std::endl<< std::endl;
    settings.print( logF );
    if( comm->getRank()==0)    settings.print( std::cout );
    logF<< std::endl<< std::endl << "Only initial partition, no MultiLevel or LocalRefinement"<< std::endl << std::endl;
    
    //------------------------------------------- pixeled
    
    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    PRINT0("Get a pixeled partition");
    // get a pixeledPartition
    scai::lama::DenseVector<IndexType> pixeledPartition = ParcoRepart<IndexType, ValueType>::pixelPartition( graph, coordinates, settings);
    
    //aux::print2DGrid( graph, pixeledPartition );
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"pixelPart");
    }
    //cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;
    cut = ParcoRepart<IndexType, ValueType>::computeCut( graph, pixeledPartition);
    imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance( pixeledPartition, k);
    logF<< "-- Initial Pixeled partition " << std::endl;
    logF<< "\tcut: " << cut << " , imbalance= "<< imbalance;
    
    uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, pixeledPartition, uniformWeights, coordinates, settings);
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"finalWithPixel");
    }
    cut = ParcoRepart<IndexType, ValueType>::computeCut( graph, pixeledPartition);
    imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance( pixeledPartition, k);
    logF<< "\tfinal cut= "<< cut  << ", final imbalance= "<< imbalance;
    logF  << std::endl  << std::endl; 
    
    //------------------------------------------- hilbert/sfc
    
    // the partitioning may redistribute the input graph
    graph.redistribute(dist, noDistPointer);
    for(int d=0; d<dimensions; d++){
        coordinates[d].redistribute( dist );
    }    
    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    PRINT0( "Get a hilbert/sfc partition");
    // get a hilbertPartition
    scai::lama::DenseVector<IndexType> hilbertPartition = ParcoRepart<IndexType, ValueType>::hilbertPartition( graph, coordinates, settings);
    
    //aux::print2DGrid( graph, hilbertPartition );
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"hilbertPart");
    }
    cut = ParcoRepart<IndexType, ValueType>::computeCut( graph, hilbertPartition);
    imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance( hilbertPartition, k);
    logF<< "-- Initial Hilbert/sfc partition " << std::endl;
    logF<< "\tcut: " << cut << " , imbalance= "<< imbalance;
    
    uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, hilbertPartition, uniformWeights, coordinates, settings);
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"finalWithHilbert");
    }
    cut = ParcoRepart<IndexType, ValueType>::computeCut( graph, hilbertPartition);
    imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance( hilbertPartition, k);
    logF<< "\tfinal cut= "<< cut << ", final imbalance= "<< imbalance;
    logF  << std::endl  << std::endl; 
    
    //------------------------------------------- spectral
    
    // the partitioning may redistribute the input graph
    graph.redistribute(dist, noDistPointer);
    for(int d=0; d<dimensions; d++){
        coordinates[d].redistribute( dist );
    }
    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    PRINT0("Get a spectral partition");

    // get initial spectral partition
    scai::lama::DenseVector<IndexType> spectralPartition = SpectralPartition<IndexType, ValueType>::getPartition( graph, coordinates, settings);
    
    //aux::print2DGrid( graph, spectralPartition );
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"spectralPart");
    }
    cut = ParcoRepart<IndexType, ValueType>::computeCut( graph, spectralPartition);
    imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance( spectralPartition, k);
    logF<< "-- Initial Spectral partition " << std::endl;
    logF<< "\tcut: " << cut << " , imbalance= "<< imbalance;
    
    uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, spectralPartition, uniformWeights, coordinates, settings);
    
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"finalWithSpectral");
    }
    cut = ParcoRepart<IndexType, ValueType>::computeCut( graph, spectralPartition);
    imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance( spectralPartition, k);
    logF<< "\tfinal cut= "<< cut << ", final imbalance= "<< imbalance;
    logF  << std::endl  << std::endl; 
    
    
    if(comm->getRank()==0){
        std::cout<< "Output files written in " << destPath << std::endl;
    }
    
}

}