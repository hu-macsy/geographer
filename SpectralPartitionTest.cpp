#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>
#include <scai/lama/Vector.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>

#include <scai/solver/GMRES.hpp>
#include <scai/solver/SimpleAMG.hpp>
#include <scai/solver/CG.hpp>
#include <scai/solver/logger/CommonLogger.hpp>
#include <scai/solver/criteria/ResidualThreshold.hpp>
#include <scai/solver/criteria/IterationCount.hpp>

#include <algorithm>
#include <memory>
#include <cstdlib>
#include <numeric>

#include "MeshGenerator.h"
#include "FileIO.h"
#include "gtest/gtest.h"
#include "SpectralPartition.h"
#include "AuxiliaryFunctions.h"


namespace ITI {

class SpectralPartitionTest : public ::testing::Test {
protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";
};

TEST_F(SpectralPartitionTest, testGetLaplacianWithEdgeWeights){
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    IndexType k = comm->getSize();
    //
    IndexType N = 40;
    //CSRSparseMatrix<ValueType> graph(N, N);
    scai::lama::SparseAssemblyStorage<ValueType> graphSt(N, N);
    
    srand(time(NULL));
    
    //TODO: this (rarely) can give disconnected graph
    // random graph with weighted edges
    for(IndexType row=0; row<N; row++){    
        for( IndexType j=0; j<rand()%5+6; j++){
            IndexType col= rand()%N;
            if( col==row ) continue;
            IndexType w = rand()%10+1;
            //PRINT0(row << ", "<< col << ": "<< w);
            graphSt.setValue(row, col, w);
            graphSt.setValue(col, row, w);
        }
        
        // connect this row with the next one so graph is connected
        graphSt.setValue(row, (row+1)%N, rand()%10 +1);
        graphSt.setValue((row+1)%N, row, rand()%10 +1 );
    }
    
    scai::lama::CSRSparseMatrix<ValueType> graph( graphSt);

    scai::dmemo::DistributionPtr blockDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );  
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
    graph.redistribute( blockDistPtr, noDistPtr);
    
    const IndexType localN = blockDistPtr->getLocalSize();
    
    scai::lama::CSRSparseMatrix<ValueType> laplacian = SpectralPartition<IndexType, ValueType>::getLaplacian( graph );
    
    {   // sum of every row must be 0
        scai::hmemo::ReadAccess<ValueType> readLaplVal(laplacian.getLocalStorage().getValues() );
        scai::hmemo::ReadAccess<IndexType> readLaplIA(laplacian.getLocalStorage().getIA() );
        for( IndexType i=0; i<readLaplIA.size()-1; i++){
            IndexType rowSum=0;
            for(IndexType j=readLaplIA[i]; j<readLaplIA[i+1]; j++){
                SCAI_ASSERT(j < readLaplVal.size(), " index " << j << " too big, must be less than " << readLaplVal.size());
                rowSum += readLaplVal[j];
                //PRINT(*comm << ": "<< readLaplVal[j]);
            }
            SCAI_ASSERT_EQ_ERROR(rowSum, 0, "Sum for row " << i <<" is "<< rowSum<< " and not 0");
        }
    }
    
    ValueType fiedlerEigenvalue = -8;
    scai::lama::DenseVector<ValueType> fiedler;
    
    {   // get the getFiedlerVector function
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        fiedler = SpectralPartition<IndexType, ValueType>::getFiedlerVector( graph, fiedlerEigenvalue );
        PRINT0("time to get fiedler vector: " << ( std::chrono::duration<double> (std::chrono::steady_clock::now() -start) ).count() );
        SCAI_ASSERT( fiedlerEigenvalue >0, "fiedler eigenvalue negative: "<< fiedlerEigenvalue);
    }
    
    //TODO: done in a hurry, add prorper tests, assertions
    //prints - assertion
    
    ValueType fiedlerMax = fiedler.max().Scalar::getValue<ValueType>();
    ValueType fiedlerl1Norm = fiedler.l1Norm().Scalar::getValue<ValueType>();
    ValueType fiedlerl2Norm = fiedler.l2Norm().Scalar::getValue<ValueType>();
    ValueType fiedlerMin = fiedler.min().Scalar::getValue<ValueType>();
    
    PRINT0(fiedler.size() << " , max= " << fiedlerMax << " , min= " << fiedlerMin << " , l1Norm= " << fiedlerl1Norm << " , l2Norm= " << fiedlerl2Norm << " , fiedlerEigenvalue= " << fiedlerEigenvalue);
    
    EXPECT_TRUE( graph.getRowDistributionPtr()->isEqual( fiedler.getDistribution() ) );
        
}

TEST_F(SpectralPartitionTest, benchConstructLaplacian) {
	std::string path = "meshes/hugebubbles/";
	std::string fileName = "hugebubbles-00000.graph";
	std::string file = path + fileName;
	const scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );

    scai::lama::CSRSparseMatrix<ValueType> L = SpectralPartition<IndexType, ValueType>::getLaplacian( graph );
}
//------------------------------------------------------------------------------

TEST_F(SpectralPartitionTest, testSpectralPartition){
    //std::string file = "Grid8x8";
    std::string file = graphPath + "rotation-00000.graph";
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
    const IndexType localN = dist->getLocalSize();
    
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed 
        

    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );
    
    //reading coordinates
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    struct Settings Settings;
    Settings.numBlocks= k;
    Settings.epsilon = 0.2;
    
    scai::lama::DenseVector<IndexType> degreeVector = SpectralPartition<IndexType, ValueType>::getDegreeVector( graph);
    SCAI_ASSERT_EQ_ERROR( degreeVector.sum() , 2*edges , "Wrong degree vector sum.");
    
    
    scai::lama::CSRSparseMatrix<ValueType> laplacian = SpectralPartition<IndexType, ValueType>::getLaplacian( graph );
    

    if(laplacian.getNumRows() < 4000 ){
        EXPECT_TRUE( laplacian.checkSymmetry() );
    }
    EXPECT_TRUE( laplacian.isConsistent() );
    
    // check that x = (1, 1, ..., 1 ) is eigenvector with eigenvalue 0
    EXPECT_TRUE( graph.getColDistributionPtr()->isEqual(laplacian.getColDistribution() ) );
    DenseVector<ValueType> x( graph.getColDistributionPtr(), 1 );
    DenseVector<ValueType> y( laplacian * x );
    SCAI_ASSERT_LT_ERROR( y.maxNorm(), Scalar( 1e-8 ), "not a Laplacian matrix" )
    
    ValueType diagonalSum=0;
    for( int r=0; r<laplacian.getNumRows(); r++){
        for( int c=0; c<laplacian.getNumColumns(); c++){
            if( r==c )
                diagonalSum += laplacian.getValue( r, c).Scalar::getValue<ValueType>();
        }
    }
    //PRINT( diagonalSum );
    EXPECT_EQ( diagonalSum , 2*edges);
    
    ValueType sum=0;
    {
        scai::hmemo::ReadAccess<ValueType> readLaplVal(laplacian.getLocalStorage().getValues() );
        for(int i=0; i<laplacian.getLocalStorage().getValues().size(); i++){
            //need read access
            sum += readLaplVal[i];
        }
    }
    // PRINT (sum);
    EXPECT_EQ( sum , 0 );

}
//------------------------------------------------------------------------------

TEST_F(SpectralPartitionTest, testGetPartition){
    //std::string file = "Grid32x32";
    std::string file = graphPath + "trace-00001.graph";
    std::ifstream f(file);
    IndexType dimensions= 2;
    IndexType N, edges;
    f >> N >> edges; 
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    IndexType k = comm->getSize();
    //
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
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
       
    PRINT0("Get a spectral partition");
    // get spectral partition
    settings.pixeledSideLen = 16;    // for a 16x16 coarsen graph
    scai::lama::DenseVector<IndexType> spectralPartition = SpectralPartition<IndexType, ValueType>::getPartition( graph, coordinates, settings);
    
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinates, N, dimensions, "SpectralPartition_+trace01");
    }
    //aux::print2DGrid( graph, spectralPartition );
    
    EXPECT_GE(k-1, spectralPartition.getLocalValues().max() );
    EXPECT_EQ(N, spectralPartition.size());
    EXPECT_EQ(0, spectralPartition.min().getValue<ValueType>());
    EXPECT_EQ(k-1, spectralPartition.max().getValue<ValueType>());
    EXPECT_EQ(graph.getRowDistribution(), spectralPartition.getDistribution());
       
    
}
//------------------------------------------------------------------------------

TEST_F(SpectralPartitionTest, testGetPartitionFromPixeledGraph){
    //std::string file = "Grid32x32";
    std::string file = graphPath + "trace-00008.graph";
    std::ifstream f(file);
    IndexType dimensions= 2;
    IndexType N, edges;
    f >> N >> edges; 
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    IndexType k = comm->getSize();
    //
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
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.pixeledSideLen = 16;    // for a 16x16 coarsen graph

    // get a pixeled-coarsen graph , this is replicated in every PE
    scai::lama::DenseVector<ValueType> pixelWeights;
    scai::lama::CSRSparseMatrix<ValueType> pixelGraph = MultiLevel<IndexType, ValueType>::pixeledCoarsen(graph, coordinates, pixelWeights, settings);
    SCAI_ASSERT( pixelGraph.getRowDistributionPtr()->isReplicated() == 1, "Pixel graph should (?) be replicated.");
        
    int emptyPixels=0;
    for(int i=0; i<pixelWeights.size(); i++){
        if(pixelWeights[i]==0){
            //PRINT0(i);
            ++emptyPixels;
        }
    }
    PRINT0("emptyPixels= " << emptyPixels);
    IndexType numPixels = pixelGraph.getNumRows();
    SCAI_ASSERT( numPixels == pixelGraph.getNumColumns(), "Wrong pixeled graph.");
    SCAI_ASSERT( pixelGraph.isConsistent() == 1 , "Pixeled graph not consistent.");
    if(settings.pixeledSideLen < 32) SCAI_ASSERT( pixelGraph.checkSymmetry() == 1 , "Pixeled graph not symmetric.");
    
    // get the vector using Eigen
    DenseVector<ValueType> eigenVec (numPixels, -1);
    ValueType eigenEigenValue =0;
    
    // get the laplacian of the pixeled graph , since the pixeled graph is replicated so should be the laplacian
    scai::lama::CSRSparseMatrix<ValueType> pixelLaplacian = SpectralPartition<IndexType, ValueType>::getLaplacian( pixelGraph );
    SCAI_ASSERT( pixelLaplacian.isConsistent() == 1 , "Laplacian graph not consistent.");
    SCAI_ASSERT( pixelLaplacian.getNumRows() == numPixels , "Wrong size of the laplacian.");
    ValueType sum=0;
    {// the sum of all elements of the laplacian should be 0
        scai::hmemo::ReadAccess<ValueType> readLaplVal(pixelLaplacian.getLocalStorage().getValues() );
        for(int i=0; i<readLaplVal.size(); i++){
            sum += readLaplVal[i];
        }
    }
    // this does not work with the version of pixel coarsen where we add edge to isolated vertices
    //EXPECT_EQ( sum , 0 );
    
    // the Eigen approach for the pixeled graph
       
    
    DenseVector<ValueType> prod ( pixelGraph*eigenVec);
    
    // get the fiedler vector for the pixeled graph using LAMA
    ValueType eigenvalue;
    scai::lama::DenseVector<ValueType> fiedler = SpectralPartition<IndexType, ValueType>::getFiedlerVector( pixelGraph, eigenvalue );
    SCAI_ASSERT( fiedler.size() == numPixels, "Sizes do not agree.");
    
    
    SCAI_ASSERT_EQ_ERROR( eigenVec.size() , fiedler.size(), "Wrong vector sizes");
    PRINT0("eigenvalue should be similar: Eigen= "<< eigenEigenValue << " , fiedler= "<< eigenvalue);
    
    ValueType eigenl1Norm = eigenVec.l1Norm().Scalar::getValue<ValueType>();
    ValueType fiedlerl1Norm = fiedler.l1Norm().Scalar::getValue<ValueType>();
    PRINT0("l1 norm should be similar: Eigen= "<< eigenl1Norm << " , fiedler= "<< fiedlerl1Norm );
    
    ValueType eigenl2Norm = eigenVec.l2Norm().Scalar::getValue<ValueType>();
    ValueType fiedlerl2Norm = fiedler.l2Norm().Scalar::getValue<ValueType>();
    PRINT0("l2 norm should be similar: Eigen= "<< eigenl2Norm << " , fiedler= "<< fiedlerl2Norm );
    
    ValueType eigenMax = eigenVec.max().Scalar::getValue<ValueType>();
    ValueType fiedlerMax = fiedler.max().Scalar::getValue<ValueType>();
    PRINT0("max should be similar: Eigen= "<< eigenMax  << " , fiedler= "<< fiedlerMax );
    
    ValueType eigenMin = eigenVec.min().Scalar::getValue<ValueType>();
    ValueType fiedlerMin = fiedler.min().Scalar::getValue<ValueType>();
    PRINT0("min should be similar: Eigen= "<< eigenMin << " , fiedler= "<< fiedlerMin );

    EXPECT_TRUE( eigenVec.getDistributionPtr()->isEqual( fiedler.getDistribution() ) );
    EXPECT_TRUE( pixelGraph.getRowDistributionPtr()->isEqual( fiedler.getDistribution() ) );
    
    
    // check 
    DenseVector<ValueType> prodF (pixelGraph*fiedler);
    DenseVector<ValueType> prodF2 ( eigenvalue*fiedler);   
    
    // sort
    scai::lama::DenseVector<IndexType> permutation, permutationF;
    eigenVec.sort(permutation, true);
    fiedler.sort(permutationF, true);
    
    PRINT0("The permutation should be almost the same or the inverse: ");
    for(int i=0; i<fiedler.size(); i++){
        if( i<10 or i>fiedler.size()-10){
            PRINT0(i<<": "<< permutation.getValue(i).Scalar::getValue<ValueType>() << " + " << permutationF.getValue(i).Scalar::getValue<ValueType>() );
        }
    }
    
}

    
}
