#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>
#include <scai/lama/Vector.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

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

namespace ITI {

class DISABLED_SpectralPartitionTest : public ::testing::Test {
protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";
};

TEST_F(DISABLED_SpectralPartitionTest, testFiedlerVector) {
    using scai::hmemo::HArray;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    //
    IndexType N = 20;
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));

    //broadcast seed value from root to ensure equal pseudorandom numbers.
    ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
    comm->bcast( seed, 1, 0 );
    srand(seed[0]);

    scai::lama::CSRSparseMatrix<ValueType> graph;// = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDist);

    /**
     * create random graph with weighted edges. We first store the edges in a dense adjacency matrix, then manually convert to CSR.
     * This is necessary since the Lama-supplied conversion to CSR adds zero-valued entries on the diagonal, which confuses constructLaplacian
     */

     {
        std::vector<ValueType> denseAdjacencyMatrix(N*N, 0);

        for (IndexType i = 0; i < N; i++) {
            const IndexType degreeBound = rand()%5+6;
            bool connectedToNextRow = false;

            for( IndexType j=0; j<degreeBound; j++){
                const IndexType col= rand()%N;
                if( col!=i ){
                    const ValueType w = rand()%10+1;
                    denseAdjacencyMatrix[i*N+col] = w;
                    denseAdjacencyMatrix[col*N+i] = w;
                }
            }

            // connect this row with the next one so graph is connected
            const IndexType col = (i+1)%N;
            if (col != i) {
                const ValueType w = rand()%10 +1;
                denseAdjacencyMatrix[i*N+col] = w;
                denseAdjacencyMatrix[col*N+i] = w;
            }

        }

        //convert to CSR
        std::vector<IndexType> newIA(N+1);
        std::vector<IndexType> newJA;
        std::vector<ValueType> newValues;

        for(IndexType i=0; i<N; i++){
            for( IndexType j=0; j<N; j++){
                if( denseAdjacencyMatrix[i*N+j] != 0 ){
                    newJA.push_back(j);
                    newValues.push_back(denseAdjacencyMatrix[i*N+j]);
                }
            }
            newIA[i+1] = newJA.size();
        }

        const IndexType M = newJA.size();
        ASSERT_EQ(M, newValues.size());

        scai::lama::CSRStorage<ValueType> storage(N,N, HArray<IndexType>(N+1,newIA.data()), HArray<IndexType>(M, newJA.data()), HArray<ValueType>(M, newValues.data()));
        graph = scai::lama::CSRSparseMatrix<ValueType>(std::move(storage));
    }
    ValueType fiedlerEigenvalue = -8;
    scai::lama::DenseVector<ValueType> fiedler;

    {   // get the getFiedlerVector function
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        fiedler = SpectralPartition<IndexType, ValueType>::getFiedlerVector( graph, fiedlerEigenvalue );
        PRINT0("time to get fiedler vector: " << ( std::chrono::duration<double> (std::chrono::steady_clock::now() -start) ).count() );
        SCAI_ASSERT( fiedlerEigenvalue >0, "fiedler eigenvalue negative: "<< fiedlerEigenvalue);
    }
    
    //TODO: done in a hurry, add proper tests, assertions
    //prints - assertion

    ValueType fiedlerMax = fiedler.max();
    ValueType fiedlerl1Norm = fiedler.l1Norm();
    ValueType fiedlerl2Norm = fiedler.l2Norm();
    ValueType fiedlerMin = fiedler.min();
    
    PRINT0(fiedler.size() << " , max= " << fiedlerMax << " , min= " << fiedlerMin << " , l1Norm= " << fiedlerl1Norm << " , l2Norm= " << fiedlerl2Norm << " , fiedlerEigenvalue= " << fiedlerEigenvalue);
    
    EXPECT_TRUE( graph.getRowDistributionPtr()->isEqual( fiedler.getDistribution() ) );
        
}

//------------------------------------------------------------------------------

TEST_F(DISABLED_SpectralPartitionTest, DISABLED_testGetPartition){
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
       
    PRINT0("Get a spectral partition");
    // get spectral partition
    settings.pixeledSideLen = 16;    // for a 16x16 coarsen graph
    scai::lama::DenseVector<IndexType> spectralPartition = SpectralPartition<IndexType, ValueType>::getPartition( graph, coordinates, settings);
        
    EXPECT_GE(k-1, scai::utilskernel::HArrayUtils::max(spectralPartition.getLocalValues()) );
    EXPECT_EQ(N, spectralPartition.size());
    EXPECT_EQ(0, spectralPartition.min());
    EXPECT_EQ(k-1, spectralPartition.max());
    EXPECT_EQ(graph.getRowDistribution(), spectralPartition.getDistribution());
}
//------------------------------------------------------------------------------

TEST_F(DISABLED_SpectralPartitionTest, testGetPartitionFromPixeledGraph){
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
    scai::lama::CSRSparseMatrix<ValueType> pixelLaplacian = GraphUtils<IndexType, ValueType>::constructLaplacian( pixelGraph );
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
       
    
    DenseVector<ValueType> prod = scai::lama::eval<DenseVector<ValueType>>( pixelGraph*eigenVec);
    
    // get the fiedler vector for the pixeled graph using LAMA
    ValueType eigenvalue;
    scai::lama::DenseVector<ValueType> fiedler = SpectralPartition<IndexType, ValueType>::getFiedlerVector( pixelGraph, eigenvalue );
    SCAI_ASSERT( fiedler.size() == numPixels, "Sizes do not agree.");
    
    
    SCAI_ASSERT_EQ_ERROR( eigenVec.size() , fiedler.size(), "Wrong vector sizes");
    PRINT0("eigenvalue should be similar: Eigen= "<< eigenEigenValue << " , fiedler= "<< eigenvalue);
    
    ValueType eigenl1Norm = eigenVec.l1Norm();
    ValueType fiedlerl1Norm = fiedler.l1Norm();
    PRINT0("l1 norm should be similar: Eigen= "<< eigenl1Norm << " , fiedler= "<< fiedlerl1Norm );
    
    ValueType eigenl2Norm = eigenVec.l2Norm();
    ValueType fiedlerl2Norm = fiedler.l2Norm();
    PRINT0("l2 norm should be similar: Eigen= "<< eigenl2Norm << " , fiedler= "<< fiedlerl2Norm );
    
    ValueType eigenMax = eigenVec.max();
    ValueType fiedlerMax = fiedler.max();
    PRINT0("max should be similar: Eigen= "<< eigenMax  << " , fiedler= "<< fiedlerMax );
    
    ValueType eigenMin = eigenVec.min();
    ValueType fiedlerMin = fiedler.min();
    PRINT0("min should be similar: Eigen= "<< eigenMin << " , fiedler= "<< fiedlerMin );

    EXPECT_TRUE( eigenVec.getDistributionPtr()->isEqual( fiedler.getDistribution() ) );
    EXPECT_TRUE( pixelGraph.getRowDistributionPtr()->isEqual( fiedler.getDistribution() ) );
    
    
    // check 
    DenseVector<ValueType> prodF = scai::lama::eval<DenseVector<ValueType>>(pixelGraph*fiedler);
    DenseVector<ValueType> prodF2 = scai::lama::eval<DenseVector<ValueType>>( eigenvalue*fiedler);
    
    // sort
    scai::lama::DenseVector<IndexType> permutation, permutationF;
    eigenVec.sort(permutation, true);
    fiedler.sort(permutationF, true);
    
    /*
    PRINT0("The permutation should be almost the same or the inverse: ");
    for(int i=0; i<fiedler.size(); i++){
        if( i<10 or i>fiedler.size()-10){
            PRINT0(i<<": "<< permutation.getValue(i) << " + " << permutationF.getValue(i) );
        }
    }
    */
}

    
}
