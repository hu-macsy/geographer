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
#include "gtest/gtest.h"


using namespace scai;

namespace ITI {

template<typename T>
class MultiLevelTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(MultiLevelTest, testTypes);

//-----------------------------------------------

TYPED_TEST (MultiLevelTest, testCoarseningGrid_2D) {
    using ValueType = TypeParam;

    std::string file = MultiLevelTest<ValueType>::graphPath + "Grid8x8";
    std::ifstream f(file);
    IndexType dimensions= 2, k=8;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed


    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );

    //distribution should be the same
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.nnCoarsening = false;

    //check distributions
    //assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );

    // coarsen the graph
    CSRSparseMatrix<ValueType> coarseGraph;
    DenseVector<IndexType> fineToCoarseMap;
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1.0);
    scai::dmemo::HaloExchangePlan halo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(graph);

    MultiLevel<IndexType, ValueType>::coarsen(graph, uniformWeights, halo, coords, coarseGraph, fineToCoarseMap, settings);

    EXPECT_TRUE(coarseGraph.isConsistent());
    EXPECT_TRUE(coarseGraph.checkSymmetry());
    DenseVector<IndexType> sortedMap(fineToCoarseMap);
    sortedMap.sort(true);
    scai::hmemo::ReadAccess<IndexType> localSortedValues(sortedMap.getLocalValues());
    for (IndexType i = 1; i < localSortedValues.size(); i++) {
        EXPECT_LE(localSortedValues[i-1], localSortedValues[i]);
        EXPECT_TRUE(localSortedValues[i-1] == localSortedValues[i] || localSortedValues[i-1] == localSortedValues[i]-1);
        EXPECT_LE(localSortedValues[i], coarseGraph.getNumRows());
    }
}
//---------------------------------------------------------------------------------------

TYPED_TEST (MultiLevelTest, testGetMatchingGrid_2D) {
    using ValueType = TypeParam;

    //std::string file = graphPath+ "Grid8x8";                         // the easy case
    std::string file = MultiLevelTest<ValueType>::graphPath+ "rotation-00000.graph";     // a harder instance
    std::ifstream f(file);
    IndexType dimensions= 2, k=8;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed


    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );

    //distribution should be the same
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.nnCoarsening = true;

    //check distributions
    //assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1.0);

    std::vector<std::pair<IndexType,IndexType>> matching = MultiLevel<IndexType, ValueType>::maxLocalMatching( graph, uniformWeights, coords, settings.nnCoarsening);
    //assert( matching[0].size() == matching[1].size() );

    // check matching to see if a node appears twice somewhere
    // for an matching as std::vector<std::vector<IndexType>> (2)
    for(int i=0; i<matching.size(); i++) {
        IndexType thisNodeGlob = matching[0].first;
        EXPECT_NE( thisNodeGlob, matching[0].second );
        for(int j=i+1; j<matching.size(); j++) {
            EXPECT_NE( thisNodeGlob, matching[j].first);
            EXPECT_NE( thisNodeGlob, matching[j].second);
        }
    }

    /*
    { // print
        std::cout<<"matched edges for "<< *comm << " (local indices) :" << std::endl;
        for(int i=0; i<matching.size(); i++){
            //std::cout<< i<< ":global  ("<< dist->local2global(matching[0][i])<< ":" << dist->local2global(matching[1][i]) << ") # ";
            std::cout<< i<< ": ("<< matching[i].first << ":" << matching[i].second << ") # ";
        }
    }
    */
}
//------------------------------------------------------------------------------

TYPED_TEST (MultiLevelTest, testComputeGlobalPrefixSum) {
    using ValueType = TypeParam;

    const IndexType globalN = 14764;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    //test for a DenseVector consisting of only 1s
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, globalN) );
    const IndexType localN = dist->getLocalSize();

    DenseVector<IndexType> vector(dist, 1);
    DenseVector<IndexType> prefixSum = MultiLevel<IndexType, ValueType>::computeGlobalPrefixSum(vector);

    ASSERT_EQ(localN, prefixSum.getDistributionPtr()->getLocalSize());
    if (comm->getRank() == 0) {
        ASSERT_EQ(1, prefixSum.getLocalValues()[0]);
    }

    {
        scai::hmemo::ReadAccess<IndexType> rPrefixSum(prefixSum.getLocalValues());
        for (IndexType i = 0; i < localN; i++) {
            EXPECT_EQ(dist->local2Global(i)+1, rPrefixSum[i]);
        }
    }

    //test for a DenseVector consisting of zeros and ones
    DenseVector<IndexType> mixedVector(dist,0);
    {
        scai::hmemo::WriteOnlyAccess<IndexType> wMixed(mixedVector.getLocalValues(), localN);
        for (IndexType i = 0; i < localN; i++) {
            wMixed[i] = i % 2;
        }
    }

    prefixSum = MultiLevel<IndexType, ValueType>::computeGlobalPrefixSum(mixedVector);

    //test for equality with std::partial_sum
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(globalN));
    mixedVector.redistribute(noDistPointer);
    prefixSum.redistribute(noDistPointer);

    scai::hmemo::ReadAccess<IndexType> rMixed(mixedVector.getLocalValues());
    scai::hmemo::ReadAccess<IndexType> rPrefixSum(prefixSum.getLocalValues());
    ASSERT_EQ(globalN, rMixed.size());
    ASSERT_EQ(globalN, rPrefixSum.size());

    std::vector<IndexType> comparison(globalN);
    std::partial_sum(rMixed.get(), rMixed.get()+globalN, comparison.begin());

    for (IndexType i = 0; i < globalN; i++) {
        EXPECT_EQ(comparison[i], rPrefixSum[i]);
    }
    EXPECT_TRUE(std::equal(comparison.begin(), comparison.end(), rPrefixSum.get()));
}
//---------------------------------------------------------------------------------------

TYPED_TEST (MultiLevelTest, testMultiLevelStep_dist) {
    using ValueType = TypeParam;

    //std::string file = graphPath+ "rotation-00000.graph";
    //std::string coordFile = graphPath+ "rotation-00000.graph.xyz";
    std::string file = MultiLevelTest<ValueType>::graphPath+ "trace-00008.graph";
    std::string coordFile = file + ".xyz";
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    const IndexType globalN = graph.getNumRows();

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr distPtr ( graph.getRowDistributionPtr() );
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution( globalN ));

    const IndexType localN = distPtr->getLocalSize();

    struct Settings settings;
    settings.numBlocks= comm->getSize();
    settings.dimensions = 2;

    EXPECT_TRUE( graph.isConsistent() );
    ValueType beforel1Norm = graph.l1Norm();
    IndexType beforeNumValues = graph.getNumValues();

    //broadcast seed value from root to ensure equal pseudorandom numbers.
    ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
    comm->bcast( seed, 1, 0 );
    srand(seed[0]);

    //random partition
    DenseVector<IndexType> partition( distPtr, 0);
    {
        scai::hmemo::WriteAccess<IndexType> wPart(partition.getLocalValues());
        for(IndexType i=0; i<localN; i++) {
            wPart[i] = rand()%settings.numBlocks;
        }
    }

    //changes due to new lama version
    scai::dmemo::RedistributePlan redist = scai::dmemo::redistributePlanByNewOwners(partition.getLocalValues(), partition.getDistributionPtr());
    scai::dmemo::DistributionPtr newDist = redist.getTargetDistributionPtr();

    partition.redistribute( newDist );

    graph.redistribute( newDist, noDistPtr);

    EXPECT_EQ( graph.l1Norm(), beforel1Norm);
    EXPECT_EQ( graph.getNumValues(), beforeNumValues);

    // node weights = 1
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1.0);
    //ValueType beforeSumWeigths = uniformWeights.l1Norm();
    IndexType beforeSumWeights = globalN;

    //coordinates at random and redistribute
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType,ValueType>::readCoords(coordFile, globalN, settings.dimensions);
    for(IndexType i=0; i<settings.dimensions; i++) {
        coords[i].redistribute( newDist );
    }

    settings.epsilon = 0.2;
    settings.multiLevelRounds= 4;
    settings.coarseningStepsBetweenRefinement = 1;
    settings.useGeometricTieBreaking = false;
    settings.dimensions= 2;
    settings.minGainForNextRound = 100;
    settings.minBorderNodes=100;
    settings.nnCoarsening = false;
    Metrics<ValueType> metrics(settings);

    scai::dmemo::HaloExchangePlan halo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(graph);
    typename ITI::CommTree<IndexType,ValueType>::CommTree commTree;
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, partition, uniformWeights, coords, halo, commTree, settings, metrics);

    EXPECT_EQ( graph.l1Norm(), beforel1Norm);
    EXPECT_EQ( graph.getNumValues(), beforeNumValues);
    // l1Norm not supported for IndexType
    EXPECT_EQ( static_cast <DenseVector<ValueType>> (uniformWeights).l1Norm(), beforeSumWeights );

}
//---------------------------------------------------------------------------------------

TYPED_TEST (MultiLevelTest, testPixeledCoarsen_2D) {
    using ValueType = TypeParam;

    for( int dim : { 2, 3 } ) {

        std::string file; // = graphPath + "trace-00008.bgf";
        std::string coordFile; // = graphPath + "trace-00008.graph.xyz";
        IndexType dimensions= dim;
        IndexType k=8;

        if( dim==2 ) {
            file = MultiLevelTest<ValueType>::graphPath + "trace-00008.graph";
        } else if( dim==3 ) {
            file = MultiLevelTest<ValueType>::graphPath + "quadTreeGraph3D_4.graph";
        }
        coordFile = file + ".xyz";

        IndexType N, edges;
        std::ifstream f(file);
        f >> N >> edges;
        f.close();

        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

        // for now local refinement requires k = P
        k = comm->getSize();
        //

        scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
        CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file, comm, ITI::Format::METIS );

        //read the array locally and messed the distribution. Left as a remainder.
        EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
        EXPECT_EQ( edges, (graph.getNumValues())/2 );

        //distribution should be the same
        std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( coordFile, N, dimensions);
        EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
        EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

        struct Settings settings;
        settings.numBlocks= k;
        settings.epsilon = 0.2;


        // coarsen the graph with different coarsening resolution
        for(IndexType i=2; i<7; i++) {
            std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

            settings.pixeledSideLen = std::pow(i,2);
            IndexType sideLen = settings.pixeledSideLen;
            IndexType pixeledGraphSize =  std::pow( sideLen, dimensions);
            IndexType pixeledGraphAdjecencyMatrixSize = pixeledGraphSize*pixeledGraphSize;

            PRINT0("sideLen= "<< settings.pixeledSideLen << " ,pixeledGraphSize= "<< pixeledGraphSize << " , pixeledGraphAdjecencyMatrixSize= " << pixeledGraphAdjecencyMatrixSize );
            if( pixeledGraphSize > N ) {
                if( comm->getRank()==0 ) {
                    std::cout<< " size of pixeledGraph (number of pixels)= "<< pixeledGraphSize << "  > input graph " << N <<". Hmm, not really a coarsening... Breaking..." << std::endl;
                }
                break;
            }

            DenseVector<ValueType> pixelWeights;

            scai::lama::CSRSparseMatrix<ValueType> pixelGraph = MultiLevel<IndexType, ValueType>::pixeledCoarsen(graph, coords, pixelWeights, settings);

            std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - start;
            double maxElapsedTime = comm->max( elapsedSeconds.count() );

            if(comm->getRank()==0 ) {
                std::cout<< "sideLen= "<< sideLen << " , max time: "<< maxElapsedTime << std::endl;
            }

            EXPECT_TRUE(pixelGraph.isConsistent());
            if(pixeledGraphSize < 4000) {
                EXPECT_TRUE(pixelGraph.checkSymmetry());
            }
            SCAI_ASSERT_EQ_ERROR( pixelWeights.sum(), N, "should ne equal");
            EXPECT_LE( pixelGraph.l1Norm(), edges);

            IndexType nnzValues= 2*dimensions*(std::pow(sideLen, dimensions) - std::pow(sideLen, dimensions-1) );

            EXPECT_EQ( nnzValues, pixelGraph.getNumValues() );
            EXPECT_GE( pixelGraph.l1Norm(), 1 );
        }// for i=2:7

    }//for dim
}

//---------------------------------------------------------------------------------------

} // namespace ITI
