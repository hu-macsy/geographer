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

#include "ParcoRepart.h"
#include "MeshGenerator.h"
#include "FileIO.h"
#include "LocalRefinement.h"
#include "GraphUtils.h"
#include "gtest/gtest.h"


using namespace scai;

namespace ITI {

template<typename T>
class LocalRefinementTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(LocalRefinementTest, testTypes);

//---------------------------------------------------------------------------------------

TYPED_TEST(LocalRefinementTest, testFiducciaMattheysesDistributed) {
    using ValueType = TypeParam;

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType k = comm->getSize();
    ValueType epsilon = 0.1;

    //srand(2); //WARNING/TODO 04/03: hangs for p=4
    //srand(3); //WARNING/TODO 04/03: hangs for p=6
    //srand(4); //WARNING/TODO 04/03: hangs for p=6
    //srand(9); //WARNING/TODO 04/03: hangs for p=4
    srand(11); //WARNING/TODO 04/03: hangs for p=5


    IndexType dimensions = 3;

    IndexType nroot = 16;
    IndexType n = nroot * nroot * nroot;

    scai::dmemo::DistributionPtr inputDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

    auto graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(inputDist, noDistPointer);
    std::vector<ValueType> maxCoord(dimensions, nroot);
    std::vector<IndexType> numPoints(dimensions, nroot);

    std::vector<DenseVector<ValueType>> coordinates(dimensions);
    for(IndexType i=0; i<dimensions; i++) {
        coordinates[i].allocate(inputDist);
        coordinates[i] = static_cast<ValueType>( 0 );
    }

    //MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist(graph, coordinates, maxCoord, numPoints, dimensions);
    MeshGenerator<IndexType, ValueType>::createRandomStructured3DMesh_dist(graph, coordinates, maxCoord, numPoints);

    /*
    	//try reading from file instead of generating the mesh
    	std::string file = graphPath + "Grid8x8";
    	CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    	const IndexType n = graph.getNumRows();
    	std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), n, dimensions);
    	scai::dmemo::DistributionPtr inputDist  = graph.getRowDistributionPtr();
    */

    ASSERT_EQ(n, inputDist->getGlobalSize());

    const IndexType localN = inputDist->getLocalSize();

    //generate random partition
    scai::lama::DenseVector<IndexType> part(inputDist, 0);
    for (IndexType i = 0; i < localN; i++) {
        IndexType blockId = rand() % k;
        //IndexType blockId = comm->getRank(); //simpler, balanced partition
        IndexType globalID = inputDist->local2Global(i);
        part.setValue(globalID, blockId);
    }
    //test initial partion for imbalance
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1.0);
    ValueType initialImbalance = GraphUtils<IndexType,ValueType>::computeImbalance(part, k, uniformWeights);

    // If initial partition is highly imbalanced local refinement cannot fix it.
    // TODO: should the final partion be balanced no matter how imbalanced is the initial one???
    // set as epsilon the initial imbalance

    if(initialImbalance > epsilon) {
        PRINT0("Warning, initial random partition too imbalanced: "<< initialImbalance);
        PRINT0("Setting as epsilon the initial imbalance " << initialImbalance);
        epsilon = initialImbalance;
    }

    //redistribute according to partition
    scai::dmemo::DistributionPtr newDistribution = scai::dmemo::generalDistributionByNewOwners(part.getDistribution(), part.getLocalValues() );

    graph.redistribute(newDistribution, graph.getColDistributionPtr());
    part.redistribute(newDistribution);

    for (IndexType dim = 0; dim < dimensions; dim++) {
        coordinates[dim].redistribute(newDistribution);
    }

    Settings settings;
    settings.numBlocks= k;
    settings.epsilon = epsilon;

    //get block graph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType,ValueType>::getBlockGraph( graph, part, settings.numBlocks);

    //color block graph and get a communication schedule
    std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph, settings);

    //get random node weights
    DenseVector<ValueType> weights(graph.getRowDistributionPtr(), 1);
    //WARNING: fillRange does not exist anymore, changed to 1
    //weights.fillRandom(1,1);
    // setRandom creates too big numbers and weights.sum() < 0 because (probably) sum does not fit in int?
    //weights.setRandom(graph.getRowDistributionPtr(), 1);
    //ValueType totalWeight = n*(n+1)/2;
    ValueType totalWeight = n;	//TODO: assuming unit weights
    ValueType minNodeWeight = weights.min();
    ValueType maxNodeWeight = weights.max();

    EXPECT_EQ(weights.sum(), totalWeight );
    if (comm->getRank() == 0) {
        std::cout << "Max node weight: " << maxNodeWeight << std::endl;
        std::cout << "Min node weight: " << minNodeWeight << std::endl;
    }
    //DenseVector<IndexType> nonWeights = DenseVector<IndexType>(0, 1);

    std::vector<IndexType> localBorder = GraphUtils<IndexType,ValueType>::getNodesWithNonLocalNeighbors(graph);

    //get distances
    std::vector<ValueType> distances = LocalRefinement<IndexType,ValueType>::distancesFromBlockCenter(coordinates);

    ValueType cut = GraphUtils<IndexType,ValueType>::computeCut(graph, part, true);
    DenseVector<IndexType> origin(graph.getRowDistributionPtr(), comm->getRank());
    ASSERT_GE(cut, 0);
    const IndexType iterations = 10;


    for (IndexType i = 0; i < iterations; i++) {

        typename ITI::CommTree<IndexType,ValueType>::CommTree commTree;
        std::vector<ValueType> gainPerRound = LocalRefinement<IndexType, ValueType>::distributedFMStep(graph, part, localBorder, weights, coordinates, distances, origin, commTree, communicationScheme, settings);
        IndexType gain = 0;
        for (IndexType roundGain : gainPerRound) gain += roundGain;

        //check correct gain calculation
        const ValueType newCut = GraphUtils<IndexType,ValueType>::computeCut(graph, part, true);
        EXPECT_EQ(cut - gain, newCut) << "Old cut " << cut << ", gain " << gain << " newCut " << newCut;

        EXPECT_LE(newCut, cut);
        cut = newCut;
    }

    //check for balance
    ValueType imbalance = GraphUtils<IndexType,ValueType>::computeImbalance(part, k, weights);
    PRINT0("final imbalance: " << imbalance);
    // TODO: I do not know, both assertion fail from time to time...
    // at least return a solution less imbalanced than the initial one
    EXPECT_LE(imbalance, initialImbalance);
    EXPECT_LE( imbalance, settings.epsilon);
}

//---------------------------------------------------------------------------------------
TYPED_TEST(LocalRefinementTest, testOriginArray) {
    using ValueType = TypeParam;

    std::string fileName = "bubbles-00010.graph";
    std::string file = LocalRefinementTest<ValueType>::graphPath + fileName;

    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file);
    std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords(std::string(file + ".xyz"), graph.getNumRows(), 2);

    //prepare ancillary data structures
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    DenseVector<IndexType> part(dist, comm->getRank());
    std::vector<IndexType> localBorder = GraphUtils<IndexType,ValueType>::getNodesWithNonLocalNeighbors(graph);
    DenseVector<ValueType> weights(dist, 1);
    std::vector<ValueType> distances = LocalRefinement<IndexType,ValueType>::distancesFromBlockCenter(coordinates);
    DenseVector<IndexType> origin(dist, comm->getRank());
    Settings settings;
    settings.numBlocks= comm->getSize();
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType,ValueType>::getBlockGraph( graph, part, settings.numBlocks);
    std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph, settings);

    ValueType gain = 0;
    IndexType iter = 0;
    do {
        typename ITI::CommTree<IndexType,ValueType>::CommTree commTree;
        std::vector<ValueType> gainPerRound = LocalRefinement<IndexType, ValueType>::distributedFMStep(graph, part, localBorder, weights, coordinates, distances, origin, commTree, communicationScheme, settings);
        gain = std::accumulate(gainPerRound.begin(), gainPerRound.end(), 0);
        if (comm->getRank() == 0) std::cout << "Found gain " << gain << " with " << gainPerRound.size() << " colors." << std::endl;
        iter++;
    } while(gain > 100);

    //check for equality of redistributed values and origin
    scai::dmemo::DistributionPtr newDist = graph.getRowDistributionPtr();

    ASSERT_TRUE(graph.getRowDistribution().isEqual(origin.getDistribution()));
    scai::hmemo::ReadAccess<IndexType> rOrigin(origin.getLocalValues());

    for (IndexType i = 0; i < newDist->getLocalSize(); i++) {
        IndexType origOwner = dist->getAnyOwner(newDist->local2Global(i));
        EXPECT_EQ(rOrigin[i], origOwner);
    }
}

//---------------------------------------------------------------------------------------

TYPED_TEST(LocalRefinementTest, testGetInterfaceNodesDistributed) {
    using ValueType = TypeParam;

    const IndexType dimX = 10;
    const IndexType dimY = 10;
    const IndexType dimZ = 10;
    const IndexType n = dimX*dimY*dimZ;

    //define distributions
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    const IndexType k = comm->getSize();

    auto a = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(n,n);
    // WARNING: an error in the next line when run with p=7
    scai::lama::MatrixCreator::buildPoisson(a, 3, 19, dimX,dimY,dimZ);

    scai::dmemo::DistributionPtr dist = a.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

    //we want replicated columns
    a.redistribute(dist, noDistPointer);

    //generate balanced distributed partition
    scai::lama::DenseVector<IndexType> part(dist, 0);
    for (IndexType i = 0; i < n; i++) {
        IndexType blockId = i % k;
        part.setValue(i, blockId);
    }

    //redistribute according to partition
    //01/19: changes because of new lama version
    //scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(*dist, part.getLocalValues()));
    scai::dmemo::DistributionPtr newDist = scai::dmemo::generalDistributionByNewOwners( *dist, part.getLocalValues() );

    a.redistribute(newDist, a.getColDistributionPtr());
    part.redistribute(newDist);

    //get communication scheme
    scai::lama::DenseVector<IndexType> mapping(k, 0);
    for (IndexType i = 0; i < k; i++) {
        mapping.setValue(i, i);
    }

    Settings settings;
    settings.numBlocks= comm->getSize();
    //std::vector<DenseVector<IndexType>> scheme = ParcoRepart<IndexType, ValueType>::computeCommunicationPairings(a, part, mapping);
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = GraphUtils<IndexType,ValueType>::getBlockGraph( a, part, k);
    std::vector<DenseVector<IndexType>> scheme = ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( blockGraph, settings );
    std::vector<IndexType> localBorder = GraphUtils<IndexType,ValueType>::getNodesWithNonLocalNeighbors(a);

    IndexType thisBlock = comm->getRank();

    for (IndexType round = 0; round < scheme.size(); round++) {
        scai::hmemo::ReadAccess<IndexType> commAccess(scheme[round].getLocalValues());
        IndexType partner = commAccess[scheme[round].getDistributionPtr()->global2Local(comm->getRank())];

        if (partner == thisBlock) {
            scai::dmemo::HaloExchangePlan partHalo = GraphUtils<IndexType,ValueType>::buildNeighborHalo(a);
            scai::hmemo::HArray<IndexType> haloData;
            //01/19: changes because of new lama version
            //comm->updateHalo( haloData, part.getLocalValues(), partHalo );
            partHalo.updateHalo( haloData, part.getLocalValues(), *comm );

        } else {
            IndexType otherBlock = partner;

            std::vector<IndexType> interfaceNodes;
            std::vector<IndexType> roundMarkers;
            std::tie(interfaceNodes, roundMarkers) = LocalRefinement<IndexType, ValueType>::getInterfaceNodes(a, part, localBorder, otherBlock, 10);
            IndexType lastRoundMarker = roundMarkers[roundMarkers.size()-1];

            //last round marker can only be zero if set is empty
            EXPECT_LE(lastRoundMarker, interfaceNodes.size());
            if (interfaceNodes.size() > 0) {
                EXPECT_GT(lastRoundMarker, 0);
            }

            //check for uniqueness
            std::vector<IndexType> sortedCopy(interfaceNodes);
            std::sort(sortedCopy.begin(), sortedCopy.end());
            auto it = std::unique(sortedCopy.begin(), sortedCopy.end());
            EXPECT_EQ(sortedCopy.end(), it);

            scai::hmemo::HArray<IndexType> localData = part.getLocalValues();
            scai::hmemo::ReadAccess<IndexType> partAccess(localData);

            //test whether all returned nodes are of the specified block
            for (IndexType node : interfaceNodes) {
                ASSERT_TRUE(newDist->isLocal(node));
                EXPECT_EQ(thisBlock, partAccess[newDist->global2Local(node)]);
            }

            //test whether rounds are consistent: first nodes should have neighbors of otherBlock, later nodes not
            //test whether last round marker is set correctly: nodes before last round marker should have neighbors in set, nodes afterwards need not
            //TODO: extend test case to check for other round markers
            const CSRStorage<ValueType>& localStorage = a.getLocalStorage();
            const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
            const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

            scai::dmemo::HaloExchangePlan partHalo = GraphUtils<IndexType,ValueType>::buildNeighborHalo(a);
            scai::hmemo::HArray<IndexType> haloData;
            //comm->updateHalo( haloData, localData, partHalo ); //01/19: changes because of new lama version
            partHalo.updateHalo( haloData, localData, *comm);

            bool inFirstRound = true;
            for (IndexType i = 0; i < interfaceNodes.size(); i++) {
                ASSERT_TRUE(newDist->isLocal(interfaceNodes[i]));
                IndexType localID = newDist->global2Local(interfaceNodes[i]);
                bool directNeighbor = false;
                for (IndexType j = ia[localID]; j < ia[localID+1]; j++) {
                    IndexType neighbor = ja[j];
                    if (newDist->isLocal(neighbor)) {
                        if (partAccess[newDist->global2Local(neighbor)] == thisBlock && i < lastRoundMarker) {
                            EXPECT_EQ(1, std::count(interfaceNodes.begin(), interfaceNodes.end(), neighbor));
                        } else if (partAccess[newDist->global2Local(neighbor)] == otherBlock) {
                            directNeighbor = true;
                        }
                    } else {
                        IndexType haloIndex = partHalo.global2Halo(neighbor);
                        if (haloIndex != scai::invalidIndex && haloData[haloIndex] == otherBlock) {
                            directNeighbor = true;
                        }
                    }
                }

                if (directNeighbor) {
                    EXPECT_TRUE(inFirstRound);
                    EXPECT_LT(i, lastRoundMarker);
                } else {
                    inFirstRound = false;
                }

                if (i == 0) {
                    EXPECT_TRUE(directNeighbor);
                }
            }
        }
    }
}
//----------------------------------------------------------

TYPED_TEST(LocalRefinementTest, testDistancesFromBlockCenter) {
    using ValueType = TypeParam;

    const IndexType nroot = 16;
    const IndexType n = nroot * nroot * nroot;
    const IndexType dimensions = 3;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

    auto a = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDistPointer);
    std::vector<ValueType> maxCoord(dimensions, nroot);
    std::vector<IndexType> numPoints(dimensions, nroot);

    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

    std::vector<DenseVector<ValueType>> coordinates(dimensions);
    for(IndexType i=0; i<dimensions; i++) {
        coordinates[i].allocate(coordDist);
        coordinates[i] = static_cast<ValueType>( 0 );
    }

    MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist(a, coordinates, maxCoord, numPoints, dimensions);

    const IndexType localN = dist->getLocalSize();

    std::vector<ValueType> distances = LocalRefinement<IndexType, ValueType>::distancesFromBlockCenter(coordinates);
    EXPECT_EQ(localN, distances.size());
    const ValueType maxPossibleDistance = pow(dimensions*(nroot*nroot),0.5);

    for (IndexType i = 0; i < distances.size(); i++) {
        EXPECT_LE(distances[i], maxPossibleDistance);
    }
}
//---------------------------------------------------------------------------------------



}// namespace ITI

