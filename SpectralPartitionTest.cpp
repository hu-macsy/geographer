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

#include "../Eigen/Dense"

#include <algorithm>
#include <memory>
#include <cstdlib>
#include <numeric>

#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "gtest/gtest.h"
#include "SpectralPartition.h"
#include "AuxiliaryFunctions.h"


namespace ITI {

class SpectralPartitionTest : public ::testing::Test {

};


TEST_F(SpectralPartitionTest, testSpectralPartition){
    //std::string file = "Grid8x8";
    std::string file = "graphFromQuad3D/graphFromQuad3D_1";
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
    
    /*
    for(int i=0; i<localN; i++){
        PRINT(*comm << ": " << degreeVector.getLocalValues()[i]);
    }
    */
    
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

TEST_F(SpectralPartitionTest, testLamaSolver){
    //std::string file = "Grid16x16";
    std::string file = "meshes/trace/trace-00000.graph";
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
    
    // replicate the laplacian for Eigen
    laplacian.redistribute( noDistPointer, noDistPointer);
    
    
    //
    // From down here everything is local/replicated in every PE
    //
    
    using Eigen::MatrixXd;
    using namespace Eigen;
    
    MatrixXd eigenLapl(N, N);
    for( int r=0; r<laplacian.getNumRows(); r++){
        for( int c=0; c<laplacian.getNumColumns(); c++){
            eigenLapl(c,r) = laplacian.getValue( r, c).Scalar::getValue<ValueType>();
        }
    }
  
    SelfAdjointEigenSolver<MatrixXd> eigensolver( eigenLapl, Eigen::DecompositionOptions::ComputeEigenvectors);
    VectorXd secondEigenVector = eigensolver.eigenvectors().col(1) ;

    DenseVector<ValueType> eigenVec (N, -1);
    for(int i=0; i<secondEigenVector.size(); i++){
        eigenVec.setValue( i, secondEigenVector[i]);
    }
    //redistribute the eigenVec
    eigenVec.redistribute( dist );
    
    
    DenseVector<ValueType> prod (graph*eigenVec);
    DenseVector<ValueType> prod2 ( eigensolver.eigenvalues()[1]*eigenVec);
    
    for(int i=0; i<prod.getLocalValues().size(); i++){
        //PRINT( prod.getLocalValues()[i] << " == "<< prod2.getLocalValues()[i]);
        PRINT0(prod.getLocalValues()[i]/ prod2.getLocalValues()[i]);
    }
    
    //^^^^^ Eigen part
    
    PRINT0("getting the LAMA fiedler vector");    
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    ValueType eigenvalue;
    scai::lama::DenseVector<ValueType> fiedler = SpectralPartition<IndexType, ValueType>::getFiedlerVector( graph, eigenvalue );

    PRINT0("got it! time: " << ( std::chrono::duration<double> (std::chrono::steady_clock::now() -start) ).count() );

    SCAI_ASSERT_EQ_ERROR( eigenVec.size() , fiedler.size(), "Wrong vector sizes");
    PRINT0("eigenvalue should be similar: Eigen= "<< eigensolver.eigenvalues()[1] << " , fiedler= "<< eigenvalue);
    
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
    EXPECT_TRUE( graph.getRowDistributionPtr()->isEqual( fiedler.getDistribution() ) );
    
    // TODO: should it be A*v=l*v? this is not true either for the getFiedlerVector or the vector from Eigen
    
    // check if A*v=l*v
    DenseVector<ValueType> prodF (graph*fiedler);
    DenseVector<ValueType> prodF2 ( eigenvalue*fiedler);
    
    for(int i=0; i<prodF.getLocalValues().size(); i++){
        //PRINT( prod.getLocalValues()[i] << " == "<< prod2.getLocalValues()[i]);
        PRINT0(prodF.getLocalValues()[i]/ prodF2.getLocalValues()[i]);
    }
    
    
    // sort
    //scai::lama::DenseVector<IndexType> permutation;
    //eigenVec.sort(permutation, true);

    scai::lama::DenseVector<IndexType> permutation;
    fiedler.sort(permutation, true);
    
    
    IndexType globalN = N;
    DenseVector<IndexType>  partition;
    
    if (!dist->isReplicated() && comm->getSize() == k) {
        SCAI_REGION( "ParcoRepart.initialPartition.redistribute" )
        
        scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(globalN, comm));
        permutation.redistribute(blockDist);
        scai::hmemo::WriteAccess<IndexType> wPermutation( permutation.getLocalValues() );
        std::sort(wPermutation.get(), wPermutation.get()+wPermutation.size());
        wPermutation.release();
        
        scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, permutation.getLocalValues(), comm));
        
        graph.redistribute(newDistribution, graph.getColDistributionPtr());
        partition = DenseVector<IndexType>(newDistribution, comm->getRank());
    }
    
    ValueType cut = comm->getSize() == 1 ? ParcoRepart<IndexType, ValueType>::computeCut(graph, partition) : comm->sum(ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(graph, false)) / 2;
    
    PRINT0( "cut= " << cut);
    
    aux::print2DGrid( graph, partition );
}
//------------------------------------------------------------------------------

TEST_F(SpectralPartitionTest, testGetPartition){
    //std::string file = "Grid32x32";
    std::string file ="meshes/trace/trace-00001.graph";
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
    settings.pixeledDetailLevel = 4;    //4  for a 16x16 coarsen graph
    scai::lama::DenseVector<IndexType> spectralPartition = SpectralPartition<IndexType, ValueType>::getPartition( graph, coordinates, settings);
    
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, "SpectralPartition_+trace01");
    }
    //aux::print2DGrid( graph, spectralPartition );
    
    
    EXPECT_GE(k-1, spectralPartition.getLocalValues().max() );
    EXPECT_EQ(N, spectralPartition.size());
    EXPECT_EQ(0, spectralPartition.min().getValue<ValueType>());
    EXPECT_EQ(k-1, spectralPartition.max().getValue<ValueType>());
    EXPECT_EQ(graph.getRowDistribution(), spectralPartition.getDistribution());
       
    
}


}