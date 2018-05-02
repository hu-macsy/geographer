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

class MultiLevelTest : public ::testing::Test {
protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";

};


TEST_F (MultiLevelTest, testCoarseningGrid_2D) {
     std::string file = graphPath + "Grid8x8";
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
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1.0);
    scai::dmemo::Halo halo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(graph);

    MultiLevel<IndexType, ValueType>::coarsen(graph, uniformWeights, halo, coarseGraph, fineToCoarseMap);
    
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
    std::string file = graphPath+ "rotation-00000.graph";     // a harder instance
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
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1.0);

    std::vector<std::pair<IndexType,IndexType>> matching = MultiLevel<IndexType, ValueType>::maxLocalMatching( graph, uniformWeights );
    //assert( matching[0].size() == matching[1].size() );
    
    // check matching to see if a node appears twice somewhere
    // for an matching as std::vector<std::vector<IndexType>> (2)
    for(int i=0; i<matching.size(); i++){
        IndexType thisNodeGlob = matching[0].first;
        EXPECT_NE( thisNodeGlob, matching[0].second );
            for(int j=i+1; j<matching.size(); j++){
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
	DenseVector<IndexType> mixedVector(dist,0);
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

    

    std::string file = graphPath+ "rotation-00000.graph";
    std::string coordFile = graphPath+ "rotation-00000.graph.xyz";

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file );
    const IndexType N = graph.getNumRows();

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr distPtr ( graph.getRowDistributionPtr() );
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution( N ));
    
    struct Settings settings;
    settings.numBlocks= comm->getSize();
    settings.dimensions = 2;

    EXPECT_TRUE( graph.isConsistent() );
    //EXPECT_TRUE( graph.checkSymmetry() );
    ValueType beforel1Norm = graph.l1Norm();
    IndexType beforeNumValues = graph.getNumValues();
    
    //broadcast seed value from root to ensure equal pseudorandom numbers.
    ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
    comm->bcast( seed, 1, 0 );
    srand(seed[0]);

    //random partition
	DenseVector<IndexType> partition( N , 0);
	{
	    scai::hmemo::WriteAccess<IndexType> wPart(partition.getLocalValues());
	    for(IndexType i=0; i<N; i++){
	        wPart[i] = rand()%settings.numBlocks;
        }
	    const IndexType localSum = std::accumulate(wPart.get(), wPart.get() + N, 0);
	    const IndexType globalSum = comm->sum(localSum);
	    ASSERT_EQ(globalSum, settings.numBlocks*localSum);
	}

	scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(partition.getLocalValues(), comm));
	partition.redistribute( newDist );

    graph.redistribute( newDist , noDistPtr);

    EXPECT_EQ( graph.l1Norm() , beforel1Norm);
    EXPECT_EQ( graph.getNumValues() , beforeNumValues);

    // node weights = 1
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(graph.getRowDistributionPtr(), 1.0);
    //ValueType beforeSumWeigths = uniformWeights.l1Norm();
    IndexType beforeSumWeights = N;
    
    //coordinates at random and redistribute
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType,ValueType>::readCoords(coordFile, N, settings.dimensions);
    for(IndexType i=0; i<settings.dimensions; i++){
        coords[i].redistribute( newDist );
    }

    settings.epsilon = 0.2;
    settings.multiLevelRounds= 6;
    settings.coarseningStepsBetweenRefinement = 3;
    settings.useGeometricTieBreaking = true;
    settings.dimensions= 2;
    settings.minGainForNextRound = 10;
    
    scai::dmemo::Halo halo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(graph);
    ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(graph, partition, uniformWeights, coords, halo, settings);
    
    EXPECT_EQ( graph.l1Norm() , beforel1Norm);
    EXPECT_EQ( graph.getNumValues() , beforeNumValues);
    // l1Norm not supported for IndexType
    EXPECT_EQ( static_cast <DenseVector<ValueType>> (uniformWeights).l1Norm() , beforeSumWeights );
    
}
//--------------------------------------------------------------------------------------- 

TEST_F (MultiLevelTest, testPixeledCoarsen_2D) {
    std::string file = graphPath + "trace-00008.bgf";
    std::string coordFile = graphPath + "trace-00008.graph.xyz";
    
    // kept for further checking
    std::string txtFile = graphPath+ "trace-00008.graph";
    std::ifstream f(txtFile);
    IndexType dimensions= 2, k=8;
    IndexType N, edges;
    f >> N >> edges; 
    f.close();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );  
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( file, ITI::Format::BINARY );
    //distrubute graph
    //graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed 
        

    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 ); 
    
    //distribution should be the same
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( coordFile, N, dimensions);
    EXPECT_TRUE(coords[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = 0.2;

    //check distributions
    //assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );

    // coarsen the graph
    for(IndexType i=2; i<7; i++){
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        
        settings.pixeledSideLen = std::pow(i,2);
        IndexType sideLen = settings.pixeledSideLen;
        IndexType pixeledGraphSize =  std::pow( sideLen, dimensions);
        IndexType pixeledGraphAdjecencyMatrixSize = pixeledGraphSize*pixeledGraphSize;
        
                
        PRINT0("sideLen= "<< settings.pixeledSideLen << " ,pixeledGraphSize= "<< pixeledGraphSize << " , pixeledGraphAdjecencyMatrixSize= " << pixeledGraphAdjecencyMatrixSize );
        if( pixeledGraphSize > N ){
            std::cout<< " size of pixeledGraph (number of pixels)= "<< pixeledGraphSize << "  > input grap " << N <<" .Hmm, not really a coarsening... Breaking..." << std::endl;
            break;
        }
        
        DenseVector<ValueType> pixelWeights;
        
        scai::lama::CSRSparseMatrix<ValueType> pixelGraph = MultiLevel<IndexType, ValueType>::pixeledCoarsen(graph, coords, pixelWeights, settings);
        
        std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - start;
        double maxElapsedTime = comm->max( elapsedSeconds.count() );
        
        if(comm->getRank()==0 ){
            std::cout<< "sideLen= "<< sideLen << " , max time: "<< maxElapsedTime << std::endl;
        }
        
        EXPECT_TRUE(pixelGraph.isConsistent());
        if(pixeledGraphSize < 4000){
            EXPECT_TRUE(pixelGraph.checkSymmetry());
        }
        SCAI_ASSERT_EQ_ERROR( pixelWeights.sum(), N , "should ne equal");
        EXPECT_LE( pixelGraph.l1Norm(), edges);
        
        IndexType nnzValues= 2*dimensions*(std::pow(sideLen, dimensions) - std::pow(sideLen, dimensions-1) );
        
        EXPECT_EQ( nnzValues , pixelGraph.getNumValues() );
        EXPECT_GE( pixelGraph.l1Norm(), 1 );
    }
}

//---------------------------------------------------------------------------------------

} // namespace ITI
