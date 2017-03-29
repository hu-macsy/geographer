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
    // sums the absolute values, make our own sum?
    EXPECT_EQ( sum , 0 );

}
//------------------------------------------------------------------------------

TEST_F(SpectralPartitionTest, testLamaSolver){
    std::string file = "Grid8x8";
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
    
    // replicate the laplacian
    laplacian.redistribute( noDistPointer, noDistPointer);
    
    //
    // From down here everything is local/replicated in every PE
    //
     
    scai::solver::CG cgSolver( "CGTestSolver" );
    scai::lama::NormPtr norm = scai::lama::NormPtr ( new scai::lama::L2Norm ( ) );
    scai::solver::CriterionPtr criterion ( new scai::solver::ResidualThreshold ( norm, 1E-8, scai::solver::ResidualThreshold::Absolute ) );
    cgSolver.setStoppingCriterion ( criterion );
    
    scai::lama::DenseVector<ValueType> solution ( N, 1.0 );
    scai::lama::DenseVector<ValueType> rhs ( N, 0.0 );
    cgSolver.initialize ( laplacian );
    cgSolver.solve ( solution, rhs );
    
    /*
    if( comm->getRank()==0 ){
        for(int i=0; i<solution.size(); i++){
            std::cout<< solution.getLocalValues()[i] << " _ ";
        }
        std::cout << std::endl;
    }
    */
    using Eigen::MatrixXd;
    using namespace Eigen;
    
    MatrixXd eigenLapl(N, N);
    for( int r=0; r<laplacian.getNumRows(); r++){
        for( int c=0; c<laplacian.getNumColumns(); c++){
            eigenLapl(c,r) = laplacian.getValue( r, c).Scalar::getValue<ValueType>();
        }
    }
  
    SelfAdjointEigenSolver<MatrixXd> eigensolver( eigenLapl );
    VectorXd secondEigenVector = eigensolver.eigenvectors().col(2) ;

    DenseVector<ValueType> eigenVec (N, -1);
    for(int i=0; i<secondEigenVector.size(); i++){
        eigenVec.setValue( i, secondEigenVector[i]);
    }
    
    //redistribute the eigenVec
    eigenVec.redistribute( dist );
    
    DenseVector<ValueType> prod (graph*eigenVec);
    DenseVector<ValueType> prod2 ( eigensolver.eigenvalues().col(1)*eigenVec);
    
    //SCAI_ASSERT_EQ_ERROR(prod, prod2, "A*v=lambda*v failed" );
    
    PRINT0("getting the LAMA fiedler vector");    
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    scai::lama::DenseVector<ValueType> fiedler = SpectralPartition<IndexType, ValueType>::getFiedlerVector( graph );

    SCAI_ASSERT_EQ_ERROR( eigenVec.size() , fiedler.size(), "Wrong vector sizes");
    PRINT("l1 norm: Eigen= "<< eigenVec.l1Norm() << " , fiedler= "<< fiedler.l1Norm() );
    PRINT("l2 norm: Eigen= "<< eigenVec.l2Norm() << " , fiedler= "<< fiedler.l2Norm() );
    PRINT("max: Eigen= "<< eigenVec.max() << " , fiedler= "<< fiedler.max() );
    PRINT("min: Eigen= "<< eigenVec.min() << " , fiedler= "<< fiedler.min() );
    
    PRINT0("got it! time: " << ( std::chrono::duration<double> (std::chrono::steady_clock::now() -start) ).count() );

    EXPECT_TRUE( eigenVec.getDistributionPtr()->isEqual( fiedler.getDistribution() ) );
    //EXPECT_TRUE( eigenVec.getDistributionPtr()->isEqual( fiedler.getRowDistribution() ) );
    
    // sort
    scai::lama::DenseVector<IndexType> permutation;
    eigenVec.sort(permutation, true);
    // sort
    scai::lama::DenseVector<IndexType> permutation2;
    fiedler.sort(permutation2, true);
    
    
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
        /*
        if (settings.useGeometricTieBreaking) {
            for (IndexType dim = 0; dim < dimensions; dim++) {
                coordinates[dim].redistribute(newDistribution);
            }
        }
        */
    }
    
    ValueType cut = comm->getSize() == 1 ? ParcoRepart<IndexType, ValueType>::computeCut(graph, partition) : comm->sum(ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(graph, false)) / 2;
    
    PRINT0( "cut= " << cut);
    
    /*
    for(int i=0; i<partition.getLocalValues().size(); i++){
        PRINT(*comm <<", "<< partition.getDistributionPtr()->local2global(i) << ": "<< partition.getLocalValues()[i] );
    }
    */
    aux::print2DGrid( graph, partition );
}
//------------------------------------------------------------------------------

TEST_F(SpectralPartitionTest, testGetPartition){
    std::string file = "Grid32x32";
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
    settings.pixeledDetailLevel = 5;    // for a 16x16 coarsen graph
    scai::lama::DenseVector<IndexType> spectralPartition = SpectralPartition<IndexType, ValueType>::getPartition( graph, coordinates, settings);
    
    if(dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, "SpectralPartition_"+file);
    }
       
    EXPECT_GE(k-1, spectralPartition.getLocalValues().max() );
    EXPECT_EQ(N, spectralPartition.size());
    EXPECT_EQ(0, spectralPartition.min().getValue<ValueType>());
    EXPECT_EQ(k-1, spectralPartition.max().getValue<ValueType>());
    EXPECT_EQ(graph.getRowDistribution(), spectralPartition.getDistribution());
       
    aux::print2DGrid( graph, spectralPartition );
}


}