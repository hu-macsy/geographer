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
#include "gtest/gtest.h"

#include "AuxiliaryFunctions.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class MultiLevelTest : public ::testing::Test {

};


TEST_F (MultiLevelTest, testCoarseningGrid_2D) {
     std::string file = "Grid8x8";
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
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    struct Settings Settings;
    Settings.numBlocks= k;
    Settings.epsilon = 0.2;

    //check distributions
    //assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );

    // coarsen the graph
    CSRSparseMatrix<ValueType> coarseGraph;
    DenseVector<IndexType> fineToCoarseMap;
    DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);

    MultiLevel<IndexType, ValueType>::coarsen(graph, uniformWeights, coarseGraph, fineToCoarseMap);
    
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

TEST_F (MultiLevelTest, testGetMatchingGrid_2D) {
    //std::string file = "Grid8x8";                         // the easy case
    std::string file = "./meshes/rotation/rotation-00000.graph";     // a harder instance
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
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    struct Settings Settings;
    Settings.numBlocks= k;
    Settings.epsilon = 0.2;

    //check distributions
    //assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);

    std::vector<std::pair<IndexType,IndexType>> matching = MultiLevel<IndexType, ValueType>::maxLocalMatching( graph, uniformWeights );
    //assert( matching[0].size() == matching[1].size() );
    
    // check matching to see if a node appears twice somewhere
    // for an matching as std::vector<std::vector<IndexType>> (2)
    for(int i=0; i<matching.size(); i++){
        IndexType thisNodeGlob = matching[0].first;
        assert( thisNodeGlob!= matching[0].second );
            for(int j=i+1; j<matching.size(); j++){
                assert( thisNodeGlob != matching[j].first);
                assert( thisNodeGlob != matching[j].second);
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

TEST_F (MultiLevelTest, testComputeGlobalPrefixSum) {
	const IndexType globalN = 14764;
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	//test for a DenseVector consisting of only 1s
	scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, globalN) );
	const IndexType localN = dist->getLocalSize();

	DenseVector<IndexType> vector(dist, 1);
	DenseVector<IndexType> prefixSum = MultiLevel<IndexType, ValueType>::computeGlobalPrefixSum<IndexType>(vector);

	ASSERT_EQ(localN, prefixSum.getDistributionPtr()->getLocalSize());
	if (comm->getRank() == 0) {
		ASSERT_EQ(1, prefixSum.getLocalValues()[0]);
	}

	{
		scai::hmemo::ReadAccess<IndexType> rPrefixSum(prefixSum.getLocalValues());
		for (IndexType i = 0; i < localN; i++) {
			EXPECT_EQ(dist->local2global(i)+1, rPrefixSum[i]);
		}
	}

	//test for a DenseVector consisting of zeros and ones
	DenseVector<IndexType> mixedVector(dist);
	{
		scai::hmemo::WriteOnlyAccess<IndexType> wMixed(mixedVector.getLocalValues(), localN);
		for (IndexType i = 0; i < localN; i++) {
			wMixed[i] = i % 2;
		}
	}

	prefixSum = MultiLevel<IndexType, ValueType>::computeGlobalPrefixSum<IndexType>(mixedVector);

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

TEST_F (MultiLevelTest, testMultiLevelStep_dist) {

    const IndexType N = 500;
    
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
    IndexType numEdges = int (1.12*N);
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
    
    //random partition
    DenseVector<IndexType> partition( N , 0);
    for(IndexType i=0; i<N; i++){
        partition.setValue(i, rand()%k );
    }
    partition.redistribute( distPtr );
    
    struct Settings Settings;
    Settings.numBlocks= k;
    Settings.epsilon = 0.2;
    Settings.multiLevelRounds= 10;
    Settings.coarseningStepsBetweenRefinement = 3;
    Settings.useGeometricTieBreaking = true;
    Settings.dimensions= 2;
    
    //ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, partition, uniformWeights, coords, Settings);
    ITI::aux::multiLevelStep(graph, partition, uniformWeights, coords, Settings);
    
    EXPECT_EQ( graph.l1Norm() , beforel1Norm);
    EXPECT_EQ( graph.getNumValues() , beforeNumValues);
    EXPECT_EQ( uniformWeights.l1Norm() , beforeSumWeigths );
    
}
//---------------------------------------------------------------------------------------
} // namespace ITI