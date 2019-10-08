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

#include "GraphUtils.h"
#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "gtest/gtest.h"
//#include "AuxiliaryFunctions.h"
#include "HilbertCurve.h"


using namespace scai;
using scai::lama::DenseVector;

namespace ITI {

template<typename T>
class ParcoRepartTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(ParcoRepartTest, testTypes);

//-----------------------------------------------


TYPED_TEST(ParcoRepartTest, testInitialPartition) {
    using ValueType = TypeParam;

    std::string fileName = "bigtrace-00000.graph";
    std::string file = ParcoRepartTest<ValueType>::graphPath + fileName;
    std::ifstream f(file);
    IndexType dimensions= 2;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    IndexType k = comm->getSize();
    //
    PRINT0("nodes= "<< N << " and k= "<< k );

    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    graph.redistribute(dist, noDistPointer);

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));

    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ(edges, (graph.getNumValues())/2 );

    struct Settings settings;
    settings.numBlocks= k;
    settings.dimensions = dimensions;
    settings.epsilon = 0.2;
    settings.pixeledSideLen = 16;
    settings.useGeometricTieBreaking = 1;
    settings.dimensions = dimensions;
    settings.initialPartition = ITI::Tool::geoSFC;
    settings.noRefinement = true;

    //get sfc partition
    std::vector<scai::lama::DenseVector<ValueType>> uniformWeights(1, DenseVector<ValueType>(graph.getRowDistributionPtr(), 1));
    DenseVector<IndexType> hilbertInitialPartition = ParcoRepart<IndexType, ValueType>::partitionGraph( coords, uniformWeights, settings );
    //ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coords, N, dimensions, "hilbertPartition");

    EXPECT_GE(k-1, scai::utilskernel::HArrayUtils::max(hilbertInitialPartition.getLocalValues()) );
    EXPECT_EQ(N, hilbertInitialPartition.size());
    EXPECT_EQ(0, hilbertInitialPartition.min());
    EXPECT_EQ(k-1, hilbertInitialPartition.max());

    // after the first partitioning coordinates are redistributed
    // redistribution needed because sort works only for block distribution
    coords[0].redistribute(dist);
    coords[1].redistribute(dist);
    graph.redistribute(dist, noDistPointer);

    for( int i=3; i<6; i++) {
        settings.pixeledSideLen = std::pow(i,2);
        DenseVector<IndexType> pixelInitialPartition = ParcoRepart<IndexType, ValueType>::pixelPartition(coords, settings);

        EXPECT_GE(k-1, scai::utilskernel::HArrayUtils::max(pixelInitialPartition.getLocalValues()) );
        EXPECT_EQ(N, pixelInitialPartition.size());
        EXPECT_EQ(0, pixelInitialPartition.min());
        EXPECT_EQ(k-1, pixelInitialPartition.max());
    }
}
//---------------------------------------------------------------------------------------

TYPED_TEST(ParcoRepartTest, testMetisWrapper) {
    using ValueType = TypeParam;

    std::string fileName = "bigtrace-00000.graph";
    //std::string fileName = "Grid16x16";
    std::string file = ParcoRepartTest<ValueType>::graphPath + fileName;
    std::ifstream f(file);
    IndexType dimensions= 2;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );

    // for now local refinement requires k = P
    const IndexType k = comm->getSize();
    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();
    const IndexType localN = blockDist->getLocalSize();

    //
    PRINT0("nodes= "<< N << " and k= "<< k );

    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(1, scai::lama::DenseVector<ValueType>(blockDist, 1));

    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*blockDist));
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*graph.getRowDistributionPtr() ));
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ(edges, (graph.getNumValues())/2 );


    IndexType vtxDist[numPEs+1];
    for(int i=0; i<numPEs+1; i++) {
        vtxDist[i] = 0;
    }
    vtxDist[thisPE+1] = localN;

//PRINT( *comm << ": " << vtxDist[thisPE+1] << " === " << blockDist->getLocalSize());

    comm->sumImpl( vtxDist, vtxDist, numPEs+1, scai::common::TypeTraits<IndexType>::stype);

    // get the prefix sum
    for(int i=1; i<numPEs+1; i++) {
        vtxDist[i] = vtxDist[i-1]+vtxDist[i];
    }

    //
    // setting xadj=ia and adjncy=ja values, these are the local values of every processor
    //

    const scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();

    scai::hmemo::ReadAccess<IndexType> ia( localMatrix.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localMatrix.getJA() );
    IndexType iaSize= ia.size();

    IndexType xadj[ iaSize ];
    IndexType adjncy[ja.size()];

    for(int i=0; i<iaSize ; i++) {
        xadj[i]= ia[i];
        SCAI_ASSERT( xadj[i] >=0, "negative value for i= "<< i << " , val= "<< xadj[i]);
    }

    for(int i=0; i<ja.size(); i++) {
        adjncy[i]= ja[i];
        SCAI_ASSERT( adjncy[i] >=0, "negative value for i= "<< i << " , val= "<< adjncy[i]);
        SCAI_ASSERT( adjncy[i] <N, "too large value for i= "<< i << " , val= "<< adjncy[i]);
    }
    ia.release();
    ja.release();

    //
    // convert the coordinates
    //

    ValueType* xyzLocal = new ValueType[localN*dimensions];

    for(int d=0; d<dimensions; d++) {
        scai::hmemo::ReadAccess<ValueType> localCoords( coords[d].getLocalValues() );
        for( int i=0; i<localN; i++) {
            xyzLocal[dimensions*i+d] = ValueType( localCoords[i] );
        }
    }

    //
    // convert the node weights
    //

    // this required from parmetis but why int node weights and not double?

    IndexType* vwgt = new IndexType[localN];
    {
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights[0].getLocalValues() );
        SCAI_ASSERT_EQ_ERROR( localN, localWeights.size(), "Local weights size mismatch. Are node weights distributed correctly?");
        for(unsigned int i=0; i<localN; i++) {
            vwgt[i] = IndexType (localWeights[i]);
        }
    }

    Settings settings;
    settings.numBlocks = k;
    settings.noRefinement=  true;  //does not work when we do local refinement
    settings.minSamplingNodes = -1; // used as flag in order to use all local nodes to minimize random behavior

    Metrics<ValueType> metrics1(settings);
    Metrics<ValueType> metrics2(settings);

    //scai::lama::DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType,ValueType>::partitionGraph( vtxDist, xadj, adjncy, localMatrix.getJA().size(), vwgt, dimensions, xyzLocal, settings, metrics1 );
    std::vector<IndexType> localPartition = ITI::ParcoRepart<IndexType,ValueType>::partitionGraph( vtxDist, xadj, adjncy, localMatrix.getJA().size(), vwgt, xyzLocal, settings, metrics1 );

    scai::lama::DenseVector<IndexType> partition( graph.getRowDistributionPtr(), scai::hmemo::HArray<IndexType>(  localPartition.size(), localPartition.data()) );

    metrics1.getAllMetrics(graph, partition, nodeWeights, settings);

    scai::lama::DenseVector<IndexType> partition2 = ITI::ParcoRepart<IndexType,ValueType>::partitionGraph( graph, coords, nodeWeights, settings, metrics2);
    partition2.redistribute( graph.getRowDistributionPtr() );

    metrics2.getAllMetrics(graph, partition2, nodeWeights, settings);

    if( comm->getRank()==0) {
        std::cout<< "Metrics for first partition:"<< std::endl;
        metrics1.print( std::cout );
        std::cout<< std::endl <<"Metrics for second partition:"<< std::endl;
        metrics2.print( std::cout );
    }

    EXPECT_NEAR( metrics1.MM["finalImbalance"], metrics2.MM["finalImbalance"], 0.0001 );
    EXPECT_EQ( metrics1.MM["finalCut"], metrics2.MM["finalCut"] );
    EXPECT_EQ( metrics1.MM["maxCommVolume"], metrics2.MM["maxCommVolume"] );
    EXPECT_EQ( metrics1.MM["totalCommVolume"], metrics2.MM["totalCommVolume"] );

}

//---------------------------------------------------------------------------------------

TYPED_TEST(ParcoRepartTest, testPartitionBalanceDistributed) {
    using ValueType = TypeParam;

    IndexType nroot = 11;
    IndexType n = nroot * nroot * nroot;
    IndexType dimensions = 3;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    IndexType k = comm->getSize();

    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

    scai::lama::CSRSparseMatrix<ValueType> a = scai::lama::zero<CSRSparseMatrix<ValueType>>(dist, noDistPointer);
    std::vector<ValueType> maxCoord(dimensions, nroot);
    std::vector<IndexType> numPoints(dimensions, nroot);

    std::vector<DenseVector<ValueType>> coordinates(dimensions);
    for(IndexType i=0; i<dimensions; i++) {
        coordinates[i] = DenseVector<ValueType>(dist, 0);
    }

    MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist(a, coordinates, maxCoord, numPoints, dimensions);

    const ValueType epsilon = 0.05;

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = epsilon;
    settings.dimensions = dimensions;
    settings.minGainForNextRound = 10;
    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, settings, metrics);

    EXPECT_GE(k-1, scai::utilskernel::HArrayUtils::max(partition.getLocalValues()) );
    EXPECT_EQ(n, partition.size());
    EXPECT_EQ(0, partition.min());
    EXPECT_EQ(k-1, partition.max());
    EXPECT_EQ(a.getRowDistribution(), partition.getDistribution());

    const scai::lama::DenseVector<ValueType> nodeWeights;
    const ValueType imbalance = GraphUtils<IndexType, ValueType>::computeImbalance(partition, k, nodeWeights);
    EXPECT_LE(imbalance, epsilon);

    const ValueType cut = GraphUtils<IndexType, ValueType>::computeCut(a, partition, true);

    if (comm->getRank() == 0) {
        std::cout << "Commit " << version << ": Partitioned graph with " << n << " nodes into " << k << " blocks with a total cut of " << cut << std::endl;
    }
}
//---------------------------------------------------------------------------------------

TYPED_TEST(ParcoRepartTest, testImbalance) {
    using ValueType = TypeParam;

    const IndexType n = 10000;
    const IndexType k = 20;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

    //generate random partition
    scai::lama::DenseVector<IndexType> part(dist, 0);
    for (IndexType i = 0; i < n; i++) {
        IndexType blockId = rand() % k;
        part.setValue(i, blockId);
    }

    //sanity check for partition generation
    ASSERT_GE(part.min(), 0);
    ASSERT_LE(part.max(), k-1);

    const scai::lama::DenseVector<ValueType> nodeWeights;
    ValueType imbalance = GraphUtils<IndexType, ValueType>::computeImbalance(part, k, nodeWeights);
    EXPECT_GE(imbalance, 0);

    // test perfectly balanced partition
    for (IndexType i = 0; i < n; i++) {
        IndexType blockId = i % k;
        part.setValue(i, blockId);
    }
    imbalance = GraphUtils<IndexType, ValueType>::computeImbalance(part, k, nodeWeights);
    EXPECT_EQ(0, imbalance);

    //test maximally imbalanced partition
    for (IndexType i = 0; i < n; i++) {
        IndexType blockId = 0;
        part.setValue(i, blockId);
    }

    imbalance = GraphUtils<IndexType, ValueType>::computeImbalance(part, k, nodeWeights);
    EXPECT_EQ((n/std::ceil(n/k))-1, imbalance);
}
//---------------------------------------------------------------------------------------

TYPED_TEST(ParcoRepartTest, testCut) {
    using ValueType = TypeParam;

    const IndexType n = 1000;
    const IndexType k = 10;

    //define distributions
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

    //generate random complete matrix
    auto a = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDistPointer);
    scai::lama::MatrixCreator::fillRandom(a, 1);

    //generate balanced distributed partition
    scai::lama::DenseVector<IndexType> part(dist, 0);
    for (IndexType i = 0; i < n; i++) {
        IndexType blockId = i % k;
        part.setValue(i, blockId);
    }

    //cut should be 10*900 / 2
    const IndexType blockSize = n / k;
    const ValueType cut = GraphUtils<IndexType, ValueType>::computeCut(a, part, false);
    EXPECT_EQ(k*blockSize*(n-blockSize) / 2, cut);

    //now convert distributed into replicated partition vector and compare again
    part.redistribute(noDistPointer);
    a.redistribute(noDistPointer, noDistPointer);
    const ValueType replicatedCut = GraphUtils<IndexType, ValueType>::computeCut(a, part, false);
    EXPECT_EQ(k*blockSize*(n-blockSize) / 2, replicatedCut);
}
//---------------------------------------------------------------------------------------

TYPED_TEST(ParcoRepartTest, testTwoWayCut) {
    using ValueType = TypeParam;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    std::string file = ParcoRepartTest<ValueType>::graphPath + "Grid32x32";
    const IndexType k = comm->getSize();
    //const ValueType epsilon = 0.05;
    //const IndexType iterations = 1;

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );

    scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const IndexType n = inputDist->getGlobalSize();
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

    const IndexType localN = inputDist->getLocalSize();

    //generate random partition
    scai::lama::DenseVector<IndexType> part(inputDist, 0);
    for (IndexType i = 0; i < localN; i++) {
        IndexType blockId = rand() % k;
        IndexType globalID = inputDist->local2Global(i);
        part.setValue(globalID, blockId);
    }

    //redistribute according to partition
    //scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(*inputDist, part.getLocalValues()));
    scai::dmemo::DistributionPtr newDistribution = scai::dmemo::generalDistributionByNewOwners( *inputDist, part.getLocalValues() );

    graph.redistribute(newDistribution, graph.getColDistributionPtr());
    part.redistribute(newDistribution);

    //get communication scheme
    scai::lama::DenseVector<IndexType> mapping(k, 0);
    for (IndexType i = 0; i < k; i++) {
        mapping.setValue(i, i);
    }

    struct Settings settings;
    settings.numBlocks= k;

    //std::vector<DenseVector<IndexType>> scheme = ParcoRepart<IndexType, ValueType>::computeCommunicationPairings(graph, part, mapping);
    scai::lama::CSRSparseMatrix<ValueType> blockGraph =  GraphUtils<IndexType, ValueType>::getBlockGraph( graph, part, k);
    EXPECT_TRUE( blockGraph.isConsistent() );
    EXPECT_TRUE( blockGraph.checkSymmetry() );
    std::vector<DenseVector<IndexType>> scheme = ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local(blockGraph, settings);

    const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    const scai::hmemo::HArray<IndexType>& localData = part.getLocalValues();
    scai::dmemo::HaloExchangePlan partHalo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(graph);
    scai::hmemo::HArray<IndexType> haloData;
    //comm->updateHalo( haloData, localData, partHalo );
    partHalo.updateHalo( haloData, localData, *comm);

    ValueType localCutSum = 0;
    for (IndexType round = 0; round < scheme.size(); round++) {
        scai::hmemo::ReadAccess<IndexType> commAccess(scheme[round].getLocalValues());
        ASSERT_EQ(k, commAccess.size());
        IndexType partner = commAccess[scheme[round].getDistributionPtr()->global2Local(comm->getRank())];

        if (partner != comm->getRank()) {
            for (IndexType j = 0; j < ja.size(); j++) {
                IndexType haloIndex = partHalo.global2Halo(ja[j]);
                if (haloIndex != scai::invalidIndex && haloData[haloIndex] == partner) {
                    localCutSum++;
                }
            }
        }
    }
    const ValueType globalCut = GraphUtils<IndexType, ValueType>::computeCut(graph, part, false);

    EXPECT_EQ(globalCut, comm->sum(localCutSum) / 2);
}
//---------------------------------------------------------------------------------------
//TODO: this test should be merged with some test in GraphUtils?
TYPED_TEST(ParcoRepartTest, testCommunicationScheme_local) {
    using ValueType = TypeParam;

    /**
     * Check for:
     * 1. Basic Sanity: All ids are valid
     * 2. Completeness: All PEs with a common edge communicate
     * 3. Symmetry: In each round, partner[partner[i]] == i holds
     * 4. Efficiency: Pairs don't communicate more than once
     */

    const IndexType n = 300;
    const IndexType p = 30;//purposefully not a power of two, to check what happens
    const IndexType k = p;

    //fill random matrix

    srand(time(NULL));

    auto a = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(n,n);
    scai::lama::MatrixCreator::fillRandom(a, 0.01);  // not symmetric
    CSRSparseMatrix<ValueType> aT = scai::lama::eval<scai::lama::CSRSparseMatrix<ValueType>>(transpose(a));

    a = scai::lama::eval<scai::lama::CSRSparseMatrix<ValueType>>(a+aT);

    //go over elements and turn them to integers.
    //this way a.checkSymmetry does not fail due to floating point errors
    for(int i=0; i<n; i++ ) {
        for(int j=0; j<n;  j++ ) {
            ValueType val = a.getLocalStorage().getValue(i,j);
            if(val!=0)//if [i,j] is a non-zero value
                a.setValue( i, j, int (100*val) );
        }
    }

    //PRINT("num of edges= " << a.getNumValues()/2 );
    EXPECT_TRUE( a.isConsistent() );
    EXPECT_TRUE( a.checkSymmetry() );

    //generate random partition
    scai::lama::DenseVector<IndexType> part(n, 0);
    for (IndexType i = 0; i < n; i++) {
        IndexType blockId = rand() % k;
        part.setValue(i, blockId);
    }

    struct Settings settings;
    settings.numBlocks= k;

    scai::lama::CSRSparseMatrix<ValueType> blockGraph =  GraphUtils<IndexType, ValueType>::getBlockGraph_dist( a, part, k);
    EXPECT_TRUE( blockGraph.isConsistent() );
    EXPECT_TRUE( blockGraph.checkSymmetry() );

    std::vector<DenseVector<IndexType>> scheme = ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local(blockGraph, settings);

    IndexType rounds = scheme.size();
    //PRINT("num edges of the blockGraph= "<< blockGraph.getNumValues()/2 );
    //PRINT("#rounds= "<< rounds );

    std::vector<std::vector<bool> > communicated(p);
    for (IndexType i = 0; i < p; i++) {
        communicated[i].resize(p, false);
    }

    for (IndexType round = 0; round < rounds; round++) {
        EXPECT_EQ(scheme[round].size(), p);
        for (IndexType i = 0; i < p; i++) {
            const IndexType partner = scheme[round].getValue(i);

            //sanity
            EXPECT_GE(partner, 0);
            EXPECT_LT(partner, p);

            if (partner != i) {
                //symmetry
                ValueType partnerOfPartner = scheme[round].getValue(partner);
                EXPECT_EQ(i, partnerOfPartner);

                //efficiency
                EXPECT_FALSE(communicated[i][partner]) << i << " and " << partner << " already communicated.";

                communicated[i][partner] = true;
            }
        }
    }

    //completeness.
    {
        const CSRStorage<ValueType>& localStorage = blockGraph.getLocalStorage();
        scai::hmemo::ReadAccess<IndexType> ia( localStorage.getIA() );
        scai::hmemo::ReadAccess<IndexType> ja( localStorage.getJA() );
        EXPECT_EQ(ia.size(), k+1);

        for(IndexType i=0; i<k; i++) {
            const IndexType endCols = ia[i+1];
            for (IndexType j = ia[i]; j < endCols; j++) {
                EXPECT_TRUE(communicated[i][ja[j]]) << i << " and " << ja[j] << " did not communicate";
            }
        }
    }
}
//---------------------------------------------------------------------------------------

TYPED_TEST (ParcoRepartTest, testBorders_Distributed) {
    using ValueType = TypeParam;

    std::string file = ParcoRepartTest<ValueType>::graphPath + "Grid32x32";
    IndexType dimensions= 2;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    IndexType k = comm->getSize();
    //

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    IndexType globalN = graph.getNumRows();

    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, globalN) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(globalN));

    graph.redistribute(dist, noDistPointer);

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), globalN, dimensions);

    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.multiLevelRounds = 3;
    //settings.initialPartition = InitialPartitioningMethods::Multisection;
    settings.initialPartition = Tool::geoKmeans;
    Metrics<ValueType> metrics(settings);

    // get partition
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);
    ASSERT_EQ(globalN, partition.size());

    scai::dmemo::DistributionPtr newDist = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr partDist = partition.getDistributionPtr();
    //IndexType newLocalN = newDist->getLocalSize();
    ASSERT_TRUE( newDist->isEqual( *partDist ) );

    //get the border nodes
    scai::lama::DenseVector<IndexType> border = GraphUtils<IndexType, ValueType>::getBorderNodes( graph, partition);

    const scai::hmemo::ReadAccess<IndexType> localBorder(border.getLocalValues());
    for(IndexType i=0; i<newDist->getLocalSize(); i++) {
        EXPECT_GE(localBorder[i], 0);
        EXPECT_LE(localBorder[i], 1);
    }

    //partition.redistribute(dist); //not needed now

    // print
    int numX= 32, numY= 32;         // 2D grid dimensions
    ASSERT_EQ(globalN, numX*numY);
    IndexType partViz[numX][numY];
    IndexType bordViz[numX][numY];
    for(int i=0; i<numX; i++)
        for(int j=0; j<numY; j++) {
            partViz[i][j]=partition.getValue(i*numX+j);
            bordViz[i][j]=border.getValue(i*numX+j);
        }

    //test getBlockGraph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType, ValueType>::getBlockGraph( graph, partition, k);
    EXPECT_TRUE(blockGraph.checkSymmetry() );

    comm->synchronize();

    if(comm->getRank()==0 ) {
        std::cout<<"----------------------------"<< " Partition  "<< *comm << std::endl;
        for(int i=0; i<numX; i++) {
            for(int j=0; j<numY; j++) {
                if(bordViz[i][j]==1)
                    std::cout<< "\033[1;31m"<< partViz[i][j] << "\033[0m" <<"-";
                else
                    std::cout<< partViz[i][j]<<"-";
            }
            std::cout<< std::endl;
        }

        // print
        //scai::hmemo::ReadAccess<IndexType> blockGraphRead( blockGraph );
        std::cout<< *comm <<" , Block Graph"<< std::endl;
        for(IndexType row=0; row<k; row++) {
            std::cout<< row << "|\t";
            for(IndexType col=0; col<k; col++) {
                std::cout << col<< ": " << blockGraph( row,col) <<" - ";
            }
            std::cout<< std::endl;
        }
    }
    comm->synchronize();
}

//------------------------------------------------------------------------------

TYPED_TEST (ParcoRepartTest, testPEGraph_Distributed) {
    using ValueType = TypeParam;

    std::string file = ParcoRepartTest<ValueType>::graphPath + "Grid16x16";
    IndexType dimensions= 2, k=8;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    IndexType N = dist->getGlobalSize();
    IndexType edges = graph.getNumValues()/2;
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));

    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );

    //distribution should be the same
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));

    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition(dist, -1);
    partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    //get the PE graph
    scai::lama::CSRSparseMatrix<ValueType> PEgraph =  GraphUtils<IndexType, ValueType>::getPEGraph( graph);
    EXPECT_EQ( PEgraph.getNumColumns(), comm->getSize() );
    EXPECT_EQ( PEgraph.getNumRows(), comm->getSize() );

    // in the distributed version each PE has only one row, its own
    // the getPEGraph uses a BLOCK distribution
    scai::dmemo::DistributionPtr distPEs ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, comm->getSize() ) );
    EXPECT_TRUE( PEgraph.getRowDistribution().isEqual( *distPEs )  );
    EXPECT_EQ( PEgraph.getLocalNumRows(), 1);
    EXPECT_EQ( PEgraph.getLocalNumColumns(), comm->getSize());
    //print
    /*
    std::cout<<"----------------------------"<< " PE graph  "<< *comm << std::endl;
    for(IndexType i=0; i<PEgraph.getNumRows(); i++){
        std::cout<< *comm<< ":";
        for(IndexType j=0; j<PEgraph.getNumColumns(); j++){
            std::cout<< PEgraph(i,j) << "-";
        }
        std::cout<< std::endl;
    }
    */
}

//------------------------------------------------------------------------------

TYPED_TEST (ParcoRepartTest, testGetBlockGraph_2D) {
    using ValueType = TypeParam;

    std::string file = ParcoRepartTest<ValueType>::graphPath+ "Grid16x16";
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
    //graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed

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
    settings.dimensions = dimensions;
    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    //check distributions
    assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );

    //test getBlockGraph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType, ValueType>::getBlockGraph( graph, partition, k);
    EXPECT_TRUE( blockGraph.isConsistent() );
    EXPECT_TRUE( blockGraph.checkSymmetry() );
    /*
    { // print
    //scai::hmemo::ReadAccess<IndexType> blockGraphRead( blockGraph );
    std::cout<< *comm <<" , Block Graph"<< std::endl;
    for(IndexType row=0; row<k; row++){
        for(IndexType col=0; col<k; col++){
            std::cout<< comm->getRank()<< ":("<< row<< ","<< col<< "):" << blockGraph( row,col) <<" - ";
        }
        std::cout<< std::endl;
    }
    }
    */
}
//------------------------------------------------------------------------------

TYPED_TEST (ParcoRepartTest, testGetBlockGraph_3D) {
    using ValueType = TypeParam;

    std::vector<IndexType> numPoints= { 4, 4, 4};
    std::vector<ValueType> maxCoord= { 42, 11, 160};
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];
    std::cout<<"Building mesh of size "<< numPoints[0]<< "x"<< numPoints[1]<< "x"<< numPoints[2] << " , N=" << N <<std::endl;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));
    //
    IndexType k = comm->getSize();
    //
    std::vector<DenseVector<ValueType>> coords(3);
    for(IndexType i=0; i<3; i++) {
        coords[i].allocate(dist);
        coords[i] = static_cast<ValueType>( 0.0 );
    }

    auto adjM = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>( dist, noDistPointer);

    // create the adjacency matrix and the coordinates
    MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist(adjM, coords, maxCoord, numPoints, 3);

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = 3;
    //settings.debugMode = true;
    //settings.verbose = true;
    Metrics<ValueType> metrics(settings);

    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(adjM, coords, settings, metrics);

    //check distributions
    assert( partition.getDistribution().isEqual( adjM.getRowDistribution()) );
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );

    //test getBlockGraph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType, ValueType>::getBlockGraph( adjM, partition, k);
    EXPECT_TRUE( blockGraph.isConsistent() );
    EXPECT_TRUE( blockGraph.checkSymmetry() );

    //The code below is not doing anything.
    //Also, it does not compile with the new lama version (01/19) and I am not sure how to adapt it.
    //So I am commenting it out.
    //TODO: fix or remove
    /*
    //get halo (buildPartHalo) and check if block graphs is correct
    scai::dmemo::HaloExchangePlan partHalo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(adjM);
    scai::hmemo::HArray<IndexType> reqIndices = partHalo.getRequiredIndexes();
    scai::hmemo::HArray<IndexType> provIndices = partHalo.getProvidesIndexes();

    const scai::hmemo::ReadAccess<IndexType> reqIndicesRead( reqIndices);
    const scai::hmemo::ReadAccess<IndexType> provIndicesRead( provIndices);
    */

    /*
    for(IndexType i=0; i< reqIndicesRead.size(); i++){
        PRINT(i <<": " << *comm <<" , req= "<<  reqIndicesRead[i] );
    }
    for(IndexType i=0; i< provIndicesRead.size(); i++){
        PRINT(i <<": " << *comm <<" , prov= "<<  provIndicesRead[i] );
    }
    */
}
//------------------------------------------------------------------------------
/* with the 8x8  grid and k=16 the block graph is a 4x4 grid. With the hilbert curve it looks like this:
 *
 *  5 - 6 - 9 - 10
 *  |   |   |   |
 *  4 - 7 - 8 - 11
 *  |   |   |   |
 *  3 - 2 - 13- 12
 *  |   |   |   |
 *  0 - 1 - 14- 15
*/
TYPED_TEST (ParcoRepartTest, testGetLocalWeightedGraphColoring_2D) {
    using ValueType = TypeParam;

    std::string file = ParcoRepartTest<ValueType>::graphPath+ "Grid8x8";
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
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed


    //read the array locally and messed the distribution. Left as a reminder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );

    //reading coordinates
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    Metrics<ValueType> metrics(settings);

    //get the partition
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, settings, metrics);

    //check distributions
    assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );

    //get getBlockGraph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType, ValueType>::getBlockGraph( graph, partition, k);

    IndexType colors;
    std::vector< std::vector<IndexType>>  coloring = GraphUtils<IndexType, ValueType>::mecGraphColoring( blockGraph, colors); // our implementation

    std::vector<DenseVector<IndexType>> communication = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph, settings);

    // as many rounds as colors
    EXPECT_EQ(colors, communication.size());
    for(IndexType i=0; i<communication.size(); i++) {
        // every round k entries
        EXPECT_EQ( k, communication[i].size());
        for(IndexType j=0; j<k; j++) {
            EXPECT_LE(communication[i](j), k);
            EXPECT_GE(communication[i](j), 0);
        }
    }

}
//-------------------------------------------------------------------------------

//WARNING: disabled test because the setRawData() seems to not work properly

TYPED_TEST (ParcoRepartTest, 
    DISABLED_testGetLocalCommunicationWithColoring_2D) {
    using ValueType = TypeParam;

    std::string file = ParcoRepartTest<ValueType>::graphPath + "Grid16x16";
    std::ifstream f(file);
    IndexType dimensions= 2;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    //IndexType k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed


    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 );


    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size(), coords[1].getLocalValues().size() );

    struct Settings settings;

    //scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, k, 0.2);

    //check distributions
    //assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );

    //test getBlockGraph
    //scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getBlockGraph( graph, partition, k);

    // build block array by hand

    // two cases


    {   // case 1                     0  1  2  3  4  5
        ValueType adjArray[36] = {  0, 1, 0, 1, 0, 1, //0
                                    1, 0, 1, 0, 1, 0, //1
                                    0, 1, 0, 1, 1, 0, //2
                                    1, 0, 1, 0, 0, 1, //3
                                    0, 1, 1, 0, 0, 1, //4
                                    1, 0, 0, 1, 1, 0  //5
                                 };

        scai::lama::CSRSparseMatrix<ValueType> blockGraph;
        blockGraph.setRawDenseData( 6, 6, adjArray);
        SCAI_ASSERT( blockGraph.checkSymmetry(), true );
        PRINT0(">> "<< blockGraph.getLocalStorage().getValues().size());
        PRINT0(">> "<< blockGraph.getLocalStorage().getIA().size());
        PRINT0(">> "<< blockGraph.getLocalStorage().getJA().size());
        for(int i=0; i<6; i++) {
            for(int j=0; j<6; j++)
                PRINT0(i<<", " << j << " = " << blockGraph.getValue(i,j));
        }
        // get the communication pairs
        std::vector<DenseVector<IndexType>> commScheme = ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( blockGraph,settings );

        // print the pairs

        for(IndexType i=0; i<commScheme.size(); i++) {
            for(IndexType j=0; j<commScheme[i].size(); j++) {
                PRINT( "round :"<< i<< " , PEs talking: "<< j << " with "<< commScheme[i].getValue(j));
            }
            std::cout << std::endl;
        }

    }


    {   // case 2
        ValueType adjArray4[16] = { 0, 1, 0, 1,
                                    1, 0, 1, 0,
                                    0, 1, 0, 1,
                                    1, 0, 1, 0
                                  };
        scai::lama::CSRSparseMatrix<ValueType> blockGraph;
        blockGraph.setRawDenseData( 4, 4, adjArray4);

        SCAI_ASSERT( blockGraph.checkSymmetry(), true );

        // get the communication pairs
        std::vector<DenseVector<IndexType>> commScheme = ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( blockGraph, settings);

        // print the pairs
        /*
        for(IndexType i=0; i<commScheme.size(); i++){
            for(IndexType j=0; j<commScheme[i].size(); j++){
                PRINT( "round :"<< i<< " , PEs talking: "<< j << " with "<< commScheme[i].getValue(j));
            }
            std::cout << std::endl;
        }
        */
    }

    {   // case 3
        ValueType adjArray2[4] = {  0, 1,
                                    1, 0
                                 };
        scai::lama::CSRSparseMatrix<ValueType> blockGraph;
        //TODO: aparently CSRSparseMatrix.getNumValues() counts also 0 when setting via a setRawDenseData despite
        // the documentation claiming otherwise. use l1Norm for unweigthed graphs
        blockGraph.setRawDenseData( 2, 2, adjArray2);

        // get the communication pairs
        std::vector<DenseVector<IndexType>> commScheme = ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( blockGraph, settings);
    }
}
//------------------------------------------------------------------------------

TYPED_TEST(ParcoRepartTest, testPixelNeighbours) {
    using ValueType = TypeParam;

    srand(time(NULL));

    for(IndexType dimension: {
                2,3
            }) {
        IndexType numEdges = 0;
        IndexType sideLen = rand()%30 +10;
        IndexType totalSize = std::pow(sideLen,dimension);
        //std::cout<< "dim= "<< dimension << " and sideLen= "<< sideLen << std::endl;

        for(IndexType thisPixel=0; thisPixel<totalSize; thisPixel++) {
            std::vector<IndexType> pixelNgbrs = ParcoRepart<IndexType, ValueType>::neighbourPixels( thisPixel, sideLen, dimension);
            numEdges += pixelNgbrs.size();
            SCAI_ASSERT(pixelNgbrs.size() <= 2*dimension, "Wrong number of neighbours");
            SCAI_ASSERT(pixelNgbrs.size() >= dimension, "Wrong number of neighbours");
            /*
            std::cout<<" neighbours of pixel " << thisPixel  <<std::endl;
            for(int i=0; i<pixelNgbrs.size(); i++){
                std::cout<< pixelNgbrs[i] << " ,";
            }
            std::cout<< std::endl;
            */
        }
        SCAI_ASSERT_EQUAL_ERROR(numEdges/2,  dimension*( std::pow(sideLen,dimension)- std::pow(sideLen, dimension-1)) );
    }
}
//------------------------------------------------------------------------------

TYPED_TEST(ParcoRepartTest, testRedistributeFromPartition) {
    using ValueType = TypeParam;

    //std::string file = ParcoRepartTest<ValueType>::graphPath + "Grid32x32";
    std::string fileName = "bigtrace-00000.graph";
    std::string file = ParcoRepartTest<ValueType>::graphPath + fileName;
    IndexType dimensions= 2;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    IndexType k = comm->getSize();
    //

    //read graph and coordinates
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );
    IndexType globalN = graph.getNumRows();

    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, globalN) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(globalN));

    graph.redistribute(dist, noDistPointer);

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), globalN, dimensions);

    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());

    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    settings.multiLevelRounds = 3;
    //settings.initialPartition = InitialPartitioningMethods::Multisection;
    settings.initialPartition = Tool::geoKmeans;
    settings.noRefinement = true;
    Metrics<ValueType> metrics(settings);

    // set uniform weights
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(1, DenseVector<ValueType>(graph.getRowDistributionPtr(), 1));
    ASSERT_EQ(1, nodeWeights.size());

    // get partition
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, nodeWeights, settings, metrics);
    ASSERT_EQ(globalN, partition.size());

    bool useRedistributor = false;

    std::chrono::time_point<std::chrono::system_clock> beforeRedistribution =  std::chrono::system_clock::now();
    scai::dmemo::DistributionPtr retDist = aux<IndexType, ValueType>::redistributeFromPartition( partition, graph, coords, nodeWeights, settings, useRedistributor);
    std::chrono::duration<double> redistTime = std::chrono::system_clock::now() - beforeRedistribution;
    ValueType maxTime = comm->max( redistTime.count() );
    if( comm->getRank()==0)
        std::cout << "Time to redistribute (useRedistributor= " << useRedistributor <<"): " << maxTime << std::endl;

    EXPECT_TRUE( coords[0].getDistribution().isEqual(nodeWeights[0].getDistribution()) );
    EXPECT_TRUE( coords[0].getDistribution().isEqual(graph.getRowDistribution()) );
    EXPECT_TRUE( coords[0].getDistribution().isEqual(partition.getDistribution()) );


    useRedistributor = !useRedistributor;
    beforeRedistribution =  std::chrono::system_clock::now();
    scai::dmemo::DistributionPtr retDist2 = aux<IndexType, ValueType>::redistributeFromPartition( partition, graph, coords, nodeWeights, settings, useRedistributor);
    redistTime = std::chrono::system_clock::now() - beforeRedistribution;
    maxTime = comm->max( redistTime.count() );
    if( comm->getRank()==0)
        std::cout << "Time to redistribute (useRedistributor= " << useRedistributor <<"): " << maxTime << std::endl;

    EXPECT_TRUE( retDist->isEqual(*retDist2) );

}
//------------------------------------------------------------------------------


/**
* TODO: test for correct error handling in case of inconsistent distributions
*/

} //namespace
