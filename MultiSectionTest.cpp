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

#include "gtest/gtest.h"

#include "MeshGenerator.h"
#include "FileIO.h"
#include "AuxiliaryFunctions.h"
#include "MultiSection.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class MultiSectionTest : public ::testing::Test {

};


TEST_F(MultiSectionTest, testGetPartitionNonUniformFromFile){
    
    const IndexType dimensions = 2;
    const IndexType k = std::pow( 4, dimensions);

    std::string path = "./";
    std::string fileName = "Grid64x64";
    //std::string fileName = "trace-00003.graph";
    std::string file = path + fileName;
    std::ifstream f(file);
    IndexType N, edges;
    f >> N >> edges; 
    
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr("BLOCK", comm, N) );
    const scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));
    const IndexType localN = dist->getLocalSize();
    PRINT0("Number of vertices= " << N << " and k= "<< k );
    
    //
    // get the adjacency matrix and the coordinates
    //
    scai::lama::CSRSparseMatrix<ValueType> adjM = FileIO<IndexType, ValueType>::readGraph(file );
    adjM.redistribute(dist, noDistPointer);
    
    std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    
    EXPECT_TRUE(coordinates[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ( adjM.getNumColumns(), adjM.getNumRows());
    EXPECT_EQ( edges, (adjM.getNumValues())/2 );   
    
    //
    //create weights locally
    //
    scai::lama::DenseVector<ValueType> nodeWeights( dist );
    IndexType actualTotalWeight = 0;
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1;
            //localPart[i] = rand()%9+rand()%7;
            actualTotalWeight += localPart[i];         
        }
    }
    actualTotalWeight =  comm->sum(actualTotalWeight);
    
    Settings settings;
    settings.dimensions = dimensions;
    settings.numBlocks = k;
    
    //
    // get the partition with multisection and one with bisection
    //
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    scai::lama::DenseVector<IndexType> partitionMS =  MultiSection<IndexType, ValueType>::getPartitionNonUniform( adjM, coordinates, nodeWeights, settings);

    std::chrono::duration<double> partitionMSTime = std::chrono::system_clock::now() - startTime;

    startTime = std::chrono::system_clock::now();
    
    settings.bisect = 1;
    scai::lama::DenseVector<IndexType> partitionBS =  MultiSection<IndexType, ValueType>::getPartitionNonUniform( adjM, coordinates, nodeWeights, settings);
    
    std::chrono::duration<double> partitionBSTime = std::chrono::system_clock::now() - startTime;
    
    if (comm->getRank() == 0) {
        std::cout<< "Time to partition with multisection: "<< partitionMSTime.count() << std::endl;
        std::cout<< "Time to partition with bisection: "<< partitionBSTime.count() << std::endl;
    }
    
    //
    // assertions - prints
    //  

    for(IndexType i=0; i<localN; i++){
        SCAI_ASSERT( partitionMS.getLocalValues()[i]!=-1 , "In PE " << *comm << " local point " << i << " has no partition." );
        SCAI_ASSERT( partitionBS.getLocalValues()[i]!=-1 , "In PE " << *comm << " local point " << i << " has no partition." );
    }
    
    const ValueType cutMS = GraphUtils::computeCut(adjM, partitionMS, true);
    const ValueType cutBS = GraphUtils::computeCut(adjM, partitionBS, true);
    
    const ValueType imbalanceMS = GraphUtils::computeImbalance<IndexType, ValueType>(partitionMS, k);
    const ValueType imbalanceBS = GraphUtils::computeImbalance<IndexType, ValueType>(partitionBS, k);
    
    PRINT0( "Multisection:  cut= " << cutMS << " , imbalance= "<< imbalanceMS);
    PRINT0( "Bisection: cut= " << cutBS << " , imbalance= " << imbalanceBS );
    
}
//---------------------------------------------------------------------------------------


TEST_F(MultiSectionTest, testGetRectangles){
 
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen= 30;
    IndexType dim = 3;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();  
    
    //create weights locally
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1;
            //localPart[i] = rand()%7*comm->getRank()+2;
        }
    }

    Settings settings;
    settings.dimensions = dim;
    settings.numBlocks = 125;
    
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    std::shared_ptr<rectCell<IndexType,ValueType>> root= MultiSection<IndexType, ValueType>::getRectangles( nodeWeights, sideLen, settings);
    
    std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> rectangles = root->getAllLeaves();
    
    std::chrono::duration<double> partitionTime = std::chrono::system_clock::now() - startTime;
    
    if (comm->getRank() == 0) {
        std::cout<< "Time to partition: "<< partitionTime.count() << std::endl;
    }
    
    // assertions - prints
    
    SCAI_ASSERT( rectangles.size()==settings.numBlocks , "Returned number of rectangles is wrong. Should be "<< settings.numBlocks<< " but it is "<< rectangles.size() );
    
    ValueType totalWeight = 0;
    IndexType totalVolume = 0;
    ValueType minWeight = LONG_MAX, maxWeight = 0;
    
    for(int r=0; r<settings.numBlocks; r++){
        struct rectangle thisRectangle = rectangles[r]->getRect();
        
        if( comm->getRank()==0 and settings.numBlocks<20){
            thisRectangle.print();
        }
        
        // the sum of all the volumes must be equal the volume of the grid: sideLen^dim
        ValueType thisVolume = 1;
        for(int d=0; d<dim; d++){
            thisVolume = thisVolume * (thisRectangle.top[d]-thisRectangle.bottom[d]+1);
        }
        totalVolume += thisVolume;
        
        ValueType thisWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, thisRectangle, sideLen, settings);
        SCAI_ASSERT( thisWeight==thisRectangle.weight, "wrong weight calculation");
        SCAI_ASSERT( thisVolume==thisWeight , "This rectangle's area= "<< thisVolume << " and should be equal to its weight that is= "<< thisWeight);
        
        if( thisWeight<minWeight ){
            minWeight = thisWeight;
        }
        if( thisWeight>maxWeight ){
            maxWeight = thisWeight;
        }
        
        totalWeight += thisWeight;
        
        // check if this rectangle overlaps with some other rectangle
        for(int r2=0; r2<settings.numBlocks; r2++){
            if( r2==r) continue; // do not self-check
            
            struct rectangle otherRectangle = rectangles[r2]->getRect();
            bool overlap = true;
            for(int d=0; d<dim; d++){
                if( thisRectangle.bottom[d]>=otherRectangle.top[d] or thisRectangle.top[d]<=otherRectangle.bottom[d] ){
                    // if they do not overlap in dimension then they cannot overlap 
                    overlap = false;
                    break;
                }
            }
            
            if( comm->getRank()==0 and overlap ){
                //PRINT0("Found overlapping rectangles:");
                thisRectangle.print();
                otherRectangle.print();
                throw std::runtime_error("The rectangles above overlap.");
            }
        }
    }
    PRINT0( "averageWeight= "<< N/settings.numBlocks );
    PRINT0( "minWeight= "<< minWeight << " , maxWeight= "<< maxWeight );
    
    //all points are covered by a rectangle
    ValueType sumWeight = nodeWeights.sum().Scalar::getValue<ValueType>();
    SCAI_ASSERT( totalWeight==sumWeight , "sum of all rectangles weight= "<< totalWeight << " and should be equal the sum of weights= "<< sumWeight);
    // this works even when weights are not 1
    SCAI_ASSERT( totalVolume==N , "total volume= "<< totalVolume << " and should be equal the number of points= "<< N);
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, test1DPartitionGreedy){
 
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen= 20;
    IndexType dim = 2;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    ValueType origTotalWeight = 0;
    
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            //localPart[i] = 1;
            localPart[i] = rand()%4*comm->getRank()+2;
            origTotalWeight += localPart[i];
        }
    }
    
    origTotalWeight = comm->sum(origTotalWeight);
    
    IndexType k= 5;
    
    Settings settings;
    settings.dimensions = dim;
    
    rectangle bBox;
    bBox.bottom = {0,0};
    bBox.top = { (ValueType) sideLen, (ValueType) sideLen};
    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );

    // the 1D partition for all dimensions
    for( int dimensionToPartition=0; dimensionToPartition<dim; dimensionToPartition++){
        std::vector<IndexType> dim2proj = {dimensionToPartition};
        
        // get the projection in one dimension
        std::vector<std::vector<ValueType>> projection = MultiSection<IndexType, ValueType>::projection( nodeWeights, root, dim2proj, sideLen, settings);

        for( int proj=0; proj<projection.size(); proj++){
            std::vector<ValueType> weightPerPart;
            std::vector<IndexType> part1D;
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DGreedy( projection[proj],  k, settings);
            
            //assertions - checks - prints
            
            SCAI_ASSERT( !std::is_sorted(part1D.end(), part1D.begin()) , "part1D is not sorted" );
            for(int i=0; i<part1D.size(); i++){
                SCAI_ASSERT( part1D[i]>=0, "Wrong partition index " << part1D[i] << " in position "<< i );
                SCAI_ASSERT( part1D[i]<sideLen, "Wrong partition index " << part1D[i] << " in position "<< i );
                if(i+1<part1D.size()){
                    SCAI_ASSERT( part1D[i]!=part1D[i+1], "part1D[i]== part1D[i+1]== " << part1D[i] << ". Maybe this should not happen");
                }
            }
            
            // vectors are of expected size
            SCAI_ASSERT( part1D.size()==k, "part1D.size()= "<< part1D.size() << " and is should be = " << k );
            SCAI_ASSERT( weightPerPart.size()==k, "weightPerPart.size()= "<< weightPerPart.size() << " and is should be = " << k );
            
            // calculate min and max weights
            ValueType minWeight=LONG_MAX, maxWeight=0;
            for(int i=0; i<weightPerPart.size(); i++){
                if( weightPerPart[i]<minWeight ){
                    minWeight = weightPerPart[i];
                }
                if( weightPerPart[i]>maxWeight ){
                    maxWeight = weightPerPart[i];
                }
            }
            
            ValueType maxOverMin = maxWeight/minWeight;
            PRINT0("max weight = "<< maxWeight << " , min weight = " << minWeight << " , max/min= " << maxOverMin);
            
            ValueType totalWeight = std::accumulate(weightPerPart.begin(), weightPerPart.end(), 0.0);
            ValueType averageWeight = totalWeight/k;
        
            SCAI_ASSERT( totalWeight==origTotalWeight, "totalWeight= "<< totalWeight << " should be= "<< origTotalWeight );
            
            PRINT0("max weight / average =" << maxWeight/averageWeight<< " , min / average =" << minWeight/averageWeight);
            PRINT0("Average weight= "<< averageWeight);
            
            // print weigths for each part
            for(int i=0; i<weightPerPart.size(); i++){
                PRINT0("part "<< i <<" weight: "<< weightPerPart[i]);
            }    
            /*
            if(comm->getRank() ==0){
                for(int i=0; i<part1D.size(); i++){
                    std::cout<< *comm <<": "<< part1D[i] << std::endl;
                }
            }
            */
        }
        // TODO: add more tests
    }
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, test1DPartitionOptimal){
 
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    const IndexType N= 500;
    const ValueType w= 50;
    const IndexType k= 5;
    const IndexType optimalWeight = w* N/k;  // make sure N/k is int
    
    std::vector<ValueType> nodeWeights( N, w);
    
    ValueType origTotalWeight = N*w;
    
    std::vector<ValueType> weightPerPart;
    std::vector<IndexType> part1D;
    
    Settings settings;
    
    //get optimal 1D partitioning
    std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DOptimal( nodeWeights, k, settings);
    
    //assertions - prints
/*    
for(int i=0; i<part1D.size(); i++){
    PRINT0(i<< ": " << part1D[i] << " ++ " << weightPerPart[i] );
}
  */  
    SCAI_ASSERT( part1D.size()==weightPerPart.size() , "Wrong size of returned vectors: part1D.size()= " << part1D.size() << " and weightPerPart.size()= "<< weightPerPart.size());
    
    //PRINT("0: from [0 to" << part1D[0] <<") with weight " <<  weightPerPart[0] );
    for(int i=0; i<part1D.size()-1; i++){
        PRINT0( i << ": from ["<< part1D[i] << " to " << part1D[i+1] -1<<"] with weight " << weightPerPart[i]);
    }
    PRINT(k-1 << ": from ["<< part1D.back() << " to " << N-1 << "] with weight " << weightPerPart.back() );
    
    ValueType totalWeight = std::accumulate(weightPerPart.begin(), weightPerPart.end(), 0.0);
    ValueType averageWeight = totalWeight/k;
        
    SCAI_ASSERT( totalWeight==origTotalWeight, "totalWeight= "<< totalWeight << " should be= "<< origTotalWeight );
    
    SCAI_ASSERT( *std::max_element(weightPerPart.begin(), weightPerPart.end())==optimalWeight, "Partition not optimal." )
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, test1DPartitionOptimalRandomWeights){
 
    const IndexType N= 87;
    const IndexType k= 13;
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    //
    //create random local weights
    //
    
    std::random_device rnd_dvc;
    std::mt19937 mersenne_engine(rnd_dvc());
    std::uniform_real_distribution<ValueType> dist(7.0, 13.0);
    auto gen = std::bind(dist, mersenne_engine);
    
    std::vector<ValueType> nodeWeights(N);
    std::generate( begin(nodeWeights), end(nodeWeights), gen);

    //
//    std::vector<ValueType> nodeWeights= {10.2484 , 10.9272 , 10.5296 , 10.5115 , 10.039 , 10.1079 , 10.0137 , 10.4012 , 10.7798 , 10.8752 , 10.0367 , 10.5632 , 10.904 , 10.4084 , 10.7319 , 10.9817 , 10.3689 , 10.4523 , 10.8339};
    //
/*    
for(int i=0; i<nodeWeights.size(); i++){
        std::cout << nodeWeights[i] << " , ";
}
std::cout << std::endl;
*/
    
    const ValueType origTotalWeight = std::accumulate( nodeWeights.begin(), nodeWeights.end(), 0.0);
    const ValueType optimalWeight = origTotalWeight/k;
    //PRINT0(origTotalWeight << " @@ " << optimalWeight);
    
    std::vector<ValueType> weightPerPart, weightPerPartGreedy, weightPerPartMine;
    std::vector<IndexType> part1D, part1DGreedy, part1DMine;
    
    Settings settings;
    
    //get greedy 1D partitioning    
    std::tie( part1DGreedy, weightPerPartGreedy) = MultiSection<IndexType, ValueType>::partition1DGreedy( nodeWeights, k, settings);

    //get optimal 1D partitioning
    std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DOptimal( nodeWeights, k, settings);
    
    //
    //assertions - prints
    // 
    ValueType maxWeightOpt = *std::max_element(weightPerPart.begin(), weightPerPart.end());
    ValueType maxWeightGrd = *std::max_element(weightPerPartGreedy.begin(), weightPerPartGreedy.end());
    
    SCAI_ASSERT( maxWeightOpt - maxWeightGrd < 0.00001, "Greedy solution better than \"optimal\"... :( ,  maxWeightOpt= " << maxWeightOpt << " , maxWeightGrd= " << maxWeightGrd );
    SCAI_ASSERT_EQ_ERROR( weightPerPartGreedy.size(), k , "Return vector has wrong size");
    SCAI_ASSERT_EQ_ERROR( part1DGreedy.size(), k , "Return vector has wrong size");
       
    SCAI_ASSERT_EQ_ERROR( weightPerPart.size(), k , "Return vector has wrong size");
    SCAI_ASSERT_EQ_ERROR( part1D.size(), k , "Return vector has wrong size");
    
    const ValueType totalWeight = std::accumulate(weightPerPart.begin(), weightPerPart.end(), 0.0);        
    SCAI_ASSERT( totalWeight==origTotalWeight, "totalWeight= "<< totalWeight << " should be= "<< origTotalWeight );
    const ValueType totalWeightGreedy = std::accumulate(weightPerPartGreedy.begin(), weightPerPartGreedy.end(), 0.0);        
    SCAI_ASSERT( totalWeightGreedy-origTotalWeight<0.000001, "totalWeight= "<< totalWeightGreedy << " should be= "<< origTotalWeight );
    
    SCAI_ASSERT( part1D.size()==weightPerPart.size() , "Wrong size of returned vectors: part1D.size()= " << part1D.size() << " and weightPerPart.size()= "<< weightPerPart.size());
    
    //PRINT("0: from [0 to" << part1D[0] <<") with weight " <<  weightPerPart[0] );
    for(int i=0; i<part1D.size()-1; i++){
        PRINT0( i << " OPT: ["<< part1D[i] << " to " << part1D[i+1] -1<<"] with weight " << weightPerPart[i]);
        //PRINT0( i << " GRD: ["<< part1DGreedy[i] << " to " << part1DGreedy[i+1] -1<<"] with weight " << weightPerPartGreedy[i]);
    }
    PRINT(k-1 << " OPT: ["<< part1D.back() << " to " << N-1 << "] with weight " << weightPerPart.back() );
    PRINT(k-1 << " GRD: ["<< part1DGreedy.back() << " to " << N-1 << "] with weight " << weightPerPartGreedy.back() );
    
    PRINT("max weight, OPT: "<<maxWeightOpt << " , GRD: " << maxWeightGrd );
  
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testProbeFunction ){
    
    const IndexType N = 88;
    const IndexType k = 17;
    const IndexType w = 2;
    std::vector<ValueType> weights( N, w);
    
    //create the prefix sum array
    //
    std::vector<ValueType> prefixSum( N+1 , 0);
    prefixSum[0] = 0;
    
    for(IndexType i=1; i<N+1; i++ ){
        prefixSum[i] = prefixSum[i-1] + weights[i-1]; 
    }
    
    IndexType realBottleneck = N*w/k +1;

    for( int target=10; target<300; target+=5){
        bool existsPart = MultiSection<IndexType, ValueType>::probe( prefixSum, k, target);
        
        if(target<realBottleneck){
            SCAI_ASSERT( !existsPart , "There should not exist a partition with bottleneck " << target);
        }
        if(target>=realBottleneck){
            SCAI_ASSERT( existsPart , "There should exist a partition with bottleneck " << target);
        }
    }
    
    // create random weights
    std::random_device rnd_dvc;
    std::mt19937 mersenne_engine(rnd_dvc());
    std::uniform_real_distribution<ValueType> dist(1, 16);
    auto gen = std::bind(dist, mersenne_engine);
    std::generate( begin(weights), end(weights), gen);
    
    for(IndexType i=1; i<N; i++ ){
        prefixSum[i] = prefixSum[i-1] + weights[i]; 
    }
    
    //find the bottlneck, all larger values must haave a partitioning
    for( int target=40; target<prefixSum.back()/2; target+=10){
        int bottleneck = prefixSum.back();
        bool existsPart = MultiSection<IndexType, ValueType>::probe( prefixSum, k, target);
        
        if( target>bottleneck){
            SCAI_ASSERT( existsPart , "There should exist a partition with bottleneck " << target );
        }
        if( existsPart ){
            //PRINT("Found partition with bottleneck " << target );
            bottleneck = target;
        }
    }
    
    ////
    std::vector<ValueType> nodeWeights2= {10.2484 , 10.9272 , 10.5296 , 10.5115 , 10.039 , 10.1079 , 10.0137 , 10.4012 , 10.7798 , 10.8752 , 10.0367 , 10.5632 , 10.904 , 10.4084 , 10.7319 , 10.9817 , 10.3689 , 10.4523 , 10.8339};
    std::vector<ValueType> prefixSum2( nodeWeights2.size()+1, 0);
    prefixSum2[0] = 0;
    
    for(IndexType i=1; i<=nodeWeights2.size(); i++ ){
        prefixSum2[i] = prefixSum2[i-1] + nodeWeights2[i-1]; 
    }
    
    ValueType target2 = 30.0;
    bool existsPart = false;
    std::vector<IndexType> splitters;
    while( !existsPart ){
        std::tie(existsPart,splitters) = MultiSection<IndexType, ValueType>::probeAndGetSplitters( prefixSum2, 6, target2);
        //PRINT(" -- " << target2);
        target2 += 1.0;
    }
    //PRINT(" ++ " << target2-1);
    //aux::printVector(splitters);
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testRectTree){
    
    rectangle initialRect;
    initialRect.bottom = { 0, 0};
    initialRect.top = { 100, 100};
    
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(initialRect) );

    rectangle r1;
    r1.bottom = { 0,0};
    r1.top = { 50, 50};
    root->insert( r1 );
    
    rectangle r2;
    r2.bottom = { 70, 90};
    r2.top = {100, 100};
    root->insert( r2 );

    rectangle r3;
    r3.bottom = {10,20};
    r3.top = {20, 40};
    root->insert( r3 );
    
    rectangle r4;
    r4.bottom = {10, 70};
    r4.top = {40, 80};
    root->insert( r4 );
    
    rectangle r5;
    r5.bottom = {12, 20};
    r5.top = {18, 40};
    root->insert( r5 );
    
    rectangle r6;
    r6.bottom = {30, 10};
    r6.top = {40, 35};
    root->insert( r6 );
    
    /*              root
     *             / |  \
     *           r1  r2  r4
     *          / \
     *         r3  r6
     *          |  
     *         r5   
     */
    
    //checks
    SCAI_ASSERT( root->getSubtreeSize()==7, "Wrong subtree size.");
    SCAI_ASSERT( root->getNumLeaves()==4, "Wrong number of leaves.");
    SCAI_ASSERT( root->getNumLeaves()==root->indexLeaves(0), "Wrong leaf indexing");
    
    std::vector<std::vector<ValueType>> points = {  
                                                    {33.0, 31},     // in r6
                                                    {13.4, 28.5},   // in r5
                                                    {75.0, 91.0},   // in r2
                                                    {12.0, 71.5},   // in r4                                                    
                                                    {15.0, 30},     // in r5
                                                    {33.1, 33.1}    // in r6
                                                };
                                                    
    std::shared_ptr<rectCell<IndexType,ValueType>> retRect; 
    
    for(int p=0; p<points.size(); p++){
        try{
            retRect = root->getContainingLeaf( points[p] );
        }catch( const std::logic_error& e ){
            PRINT( e.what() );
            std::cout<< "point ("<< points[p][0] << ", " << points[p][1] << ") is not in any rectangle" << std::endl;
            std::terminate();
        }

        SCAI_ASSERT( retRect->getRect().bottom[0]<=points[p][0] and retRect->getRect().top[0]>=points[p][0] , "Wrong rectangle");
        SCAI_ASSERT( retRect->getRect().bottom[1]<=points[p][1] and retRect->getRect().top[1]>=points[p][1] , "Wrong rectangle");
        
    }

    std::vector<ValueType> point(2); 
    
    //random points
    srand(time(NULL));
    for(int r=0; r<100; r++){
        point[0] = rand()%101;
        point[1] = rand()%101;
        
        try{
            retRect = root->getContainingLeaf( point );
        }catch( const std::logic_error& e ){
            //PRINT("Null pointer for point ("<<point[0] << ", "<< point[1]<< ")" );
            continue;
        }
        SCAI_ASSERT( retRect->getRect().bottom[0]<=point[0] and retRect->getRect().top[0]>=point[0] , "Wrong rectangle");
        SCAI_ASSERT( retRect->getRect().bottom[1]<=point[1] and retRect->getRect().top[1]>=point[1] , "Wrong rectangle");
    }

    //all the points
    int numOfPointsIn_r2 =0;
    int numOfPointsIn_r4 =0;
    int numOfPointsIn_r6 =0;
    int numOfPointsNotInAnyRect = 0;
    for( int x=0; x<=100; x++){
        for(int y=0; y<=100; y++){
            point[0]=x;
            point[1]=y;
            
            try{
                retRect = root->getContainingLeaf( point );
            }catch( const std::logic_error& e ){
                //PRINT("Null pointer for point ("<<point[0] << ", "<< point[1]<< ")" );
                numOfPointsNotInAnyRect++;
                continue;
            }
            
            if(retRect->getRect()==r2){
                numOfPointsIn_r2++;
            }
            if(retRect->getRect()==r4){
                numOfPointsIn_r4++;
            }
            if(retRect->getRect()==r6){
                numOfPointsIn_r6++;
            }
        }
    }
    
    int r2Vol, r4Vol, r6Vol;
    r2Vol = (r2.top[0]-r2.bottom[0]+1)*(r2.top[1]-r2.bottom[1]+1);
    r4Vol = (r4.top[0]-r4.bottom[0]+1)*(r4.top[1]-r4.bottom[1]+1);
    r6Vol = (r6.top[0]-r6.bottom[0]+1)*(r6.top[1]-r6.bottom[1]+1);
    SCAI_ASSERT( numOfPointsIn_r2==r2Vol, "Wrong number of points= " << numOfPointsIn_r2 <<" in rectangle r2= " << r2Vol);
    SCAI_ASSERT( numOfPointsIn_r4==r4Vol, "Wrong number of points in rectangle r4.");
    SCAI_ASSERT( numOfPointsIn_r6==r6Vol, "Wrong number of points in rectangle r6.");
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testGetRectangleWeightNonUniform){
    
    const IndexType N = 16;
    const IndexType dimensions = 2;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N+1) );
    
    scai::lama::DenseVector<ValueType> nodeWeights(N+1);
    std::vector<scai::lama::DenseVector<ValueType>> coordinates(dimensions);
    
    // put weights only in the line x=y
    int w=4;
    for(int d=0; d<dimensions; d++){
        coordinates[d].allocate( N+1 );
        for(int i=0; i<=N; i++){
            nodeWeights[i] = w;
            coordinates[d].setValue(i,i);   
        }
        coordinates[d].redistribute( blockDist );
    }

    nodeWeights.redistribute( blockDist );
    
    Settings settings;
    
    rectangle bBox; 
    
    std::vector<ValueType> maxCoord = { N, N };
    
    // all the points
    bBox.bottom = { 0, 0};
    bBox.top = { N, N };
    ValueType bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, bBox, maxCoord, settings);
    SCAI_ASSERT( bBoxWeight==17*w, "Weight of the bounding box not correct, should be " << 16*w << " but it is "<< bBoxWeight);
    
    bBox.bottom = { 0, 0};
    bBox.top = { N/2, N/2 };
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, bBox, maxCoord, settings);
    SCAI_ASSERT( bBoxWeight==9*w, "Weight of the bounding box not correct, should be " << 9*w  << " but it is "<< bBoxWeight);
    
    bBox.bottom = { N/2, N/2 };
    bBox.top = { N, N };
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, bBox, maxCoord, settings);
    SCAI_ASSERT( bBoxWeight==9*w, "Weight of the bounding box not correct, should be " << 9*w << " but it is "<< bBoxWeight);
    
    bBox.bottom = { 3, 6 };
    bBox.top = { 7, 11 };
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, bBox, maxCoord, settings);
    SCAI_ASSERT( bBoxWeight==2*w, "Weight of the bounding box not correct, should be " <<  2*w << " but it is "<< bBoxWeight);
    
    bBox.bottom = { 3, 1 };
    bBox.top = { 7, 11 };
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, bBox, maxCoord, settings);
    SCAI_ASSERT( bBoxWeight==5*w, "Weight of the bounding box not correct, should be "<< 4*w  << " but it is "<< bBoxWeight);
    
    bBox.bottom = { 2, 6 };
    bBox.top = { 5, 11 };
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, bBox, maxCoord, settings);
    SCAI_ASSERT( bBoxWeight==0, "Weight of the bounding box not correct, should be 0 but it is "<< bBoxWeight);
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testGetRectangleWeight){

    IndexType N= 16;
    IndexType sideLen = 4;
    ValueType nodeW[N] = {  1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 0, 1, 5,
                            3, 4, 5, 6
    };
    scai::lama::DenseVector<ValueType> nodeWeights(N, nodeW);
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    
    nodeWeights.redistribute( blockDist );
    
    Settings settings;
    
    rectangle bBox;        // the sub-rectangles of the initial grid
    
    bBox.bottom = {0, 0};       //  1, 2, 3
    bBox.top = {2, 2};          //  5, 6, 7
                                //  9, 0, 1
    
    ValueType bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, bBox, sideLen, settings);
    SCAI_ASSERT( bBoxWeight==34, "Weight of the bounding box not correct, should be 14 but it is "<< bBoxWeight);
    
    bBox.bottom = {2, 0};       //  9, 0, 1, 5      rows 2 and 3
    bBox.top = {3, 3};          //  3, 4, 5, 6  
    
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, bBox, sideLen, settings);
    SCAI_ASSERT( bBoxWeight==33, "Weight of the bounding box not correct, should be 33 but it is "<< bBoxWeight);
    
                                //  3, 4
    bBox.bottom = {0, 2};       //  7, 8            column 3
    bBox.top = {3, 3};          //  1, 5
                                //  5, 6
    
    bBoxWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( nodeWeights, bBox, sideLen, settings);
    SCAI_ASSERT( bBoxWeight==39, "Weight of the bounding box not correct, should be 16 but it is "<< bBoxWeight);
}
//---------------------------------------------------------------------------------------    
    
TEST_F(MultiSectionTest, test1DProjection){
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen = 5;
    IndexType dim = 3;
    IndexType N= std::pow( sideLen+1, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    //create local weights
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1;
        }
    }
    
    // test projection in all dimensions
    
    for(int d=0; d<dim; d++){
        rectangle bBox;
        // for all dimensions i: bottom[i]<top[i] 
        // WARNING: this test case throw Point out of bounds exception
        bBox.bottom = {0, 0, 0};
        bBox.top =    {5, 5, 5};
        
        // create the root of the tree that contains the whole grid
        std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
        
        ValueType bBoxVolume = 1;
        for(int d=0; d<dim; d++){
            bBoxVolume = bBoxVolume * (bBox.top[d]-bBox.bottom[d]+1);
        }
        
        Settings settings;
        settings.dimensions = dim;
        std::vector<IndexType> dim2proj = {d};
        
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projection( nodeWeights, root, dim2proj, sideLen+1, settings);
        std::vector<ValueType> projection = projections[0];
        
        //assertions
        const IndexType projLength = bBox.top[dim2proj[0]]-bBox.bottom[dim2proj[0]]+1;
        SCAI_ASSERT( projLength==projection.size(), "Length of projection is not correct");
        
        ValueType projSum = std::accumulate( projection.begin(), projection.end(), 0.0);
        SCAI_ASSERT( bBoxVolume==projSum, "Volume of bounding box= "<< bBoxVolume<< " and should be equal to the sum of the projection which is "<< projSum );
        
        ValueType bBoxSlice = bBoxVolume/(bBox.top[dim2proj[0]]-bBox.bottom[dim2proj[0]]+1);
        for(int i=0; i<projection.size(); i++){
            //this only works when dim=2 and nodeWeights=1
            SCAI_ASSERT( projection[i]==bBoxSlice , "projection["<< i<<"]= "<< projection[i] << " should be equal to "<< bBoxSlice );
        }
    }
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testGetRectanglesNonUniform){
 
    const IndexType dimensions = 3;
    const std::vector<ValueType> minCoords= {0, 0, 0};    
    const std::vector<ValueType> maxCoords= {20, 20, 20};
    const IndexType N = (maxCoords[0]+1)*(maxCoords[1]+1)*(maxCoords[2]+1); 
    const IndexType k = std::pow( 4, dimensions);
    
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr("BLOCK", comm, N) );
    const scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));
    const IndexType localN = dist->getLocalSize();
    
    // in this version the adjacency matrix is not used in the getRectanglesNonUniform
    scai::lama::CSRSparseMatrix<ValueType> adjM(dist, noDistPointer);
    
    //
    //create weights locally
    //
    scai::lama::DenseVector<ValueType> nodeWeights( dist );
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1;
            //localPart[i] = rand()%7*comm->getRank()+2;
        }
    }
    
    //
    // create integer coordinates in the grid every two points (not all)
    //
    std::vector<DenseVector<IndexType>> coordinates(dimensions);
    
    for(IndexType i=0; i<dimensions; i++){
        coordinates[i].allocate(N);
        coordinates[i] = static_cast<IndexType>( 0 );
    }
    
    int p=0;
    for(int x=0; x<=maxCoords[0]; x=x+1){
        for(int y=0; y<=maxCoords[1]; y=y+1){
            for(int z=0; z<=maxCoords[2]; z=z+1){
                coordinates[0][p] = x;
                coordinates[1][p] = y;
                coordinates[2][p] = z;
                ++p;
            }
        }
    }
    SCAI_ASSERT( p==N, "Wrong number of coordinates created.");
    
    coordinates[0].redistribute( dist );
    coordinates[1].redistribute( dist );
    coordinates[2].redistribute( dist );
    
    // copy local coordinates to a vector<vector<IndexType>>
    std::vector<std::vector<IndexType>> coords( localN, std::vector<IndexType>( dimensions, 0) );
    {
        for (IndexType d = 0; d < dimensions; d++) {
            const scai::utilskernel::LArray<IndexType>& localPartOfCoords = coordinates[d].getLocalValues();
            for (IndexType i = 0; i < localN; i++) {
                coords[i][d] = IndexType (localPartOfCoords[i]);
            }
        }
    }
    
    Settings settings;
    settings.dimensions = dimensions;
    settings.numBlocks = k;
    
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    std::shared_ptr<rectCell<IndexType,ValueType>> root = MultiSection<IndexType, ValueType>::getRectanglesNonUniform( adjM, coords, nodeWeights, minCoords, maxCoords, settings);

    std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> rectangles = root->getAllLeaves();
    
    std::chrono::duration<double> partitionTime = std::chrono::system_clock::now() - startTime;
    
    if (comm->getRank() == 0) {
        std::cout<< "Time to partition: "<< partitionTime.count() << std::endl;
    }
    
    //
    // assertions - prints
    //
    SCAI_ASSERT( rectangles.size()==settings.numBlocks , "Returned number of rectangles is wrong. Should be "<< settings.numBlocks<< " but it is "<< rectangles.size() );
    
    ValueType totalWeight = 0;
    IndexType totalVolume = 0;
    ValueType minWeight = LONG_MAX, maxWeight = 0;
    
    for(int r=0; r<settings.numBlocks; r++){
        struct rectangle thisRectangle = rectangles[r]->getRect();
        
        if( comm->getRank()==0 and settings.numBlocks<20){
            thisRectangle.print();
        }
        
        // the sum of all the volumes must be equal the volume of the grid
        ValueType thisVolume = 1;
        for(int d=0; d<dimensions; d++){
            thisVolume = thisVolume * (thisRectangle.top[d]-thisRectangle.bottom[d]+1);
            SCAI_ASSERT( thisRectangle.top[d]<=maxCoords[d] , "Rectangle's top coordinate is out of bounds: thisRectangle.top[d]= " << thisRectangle.top[d] << " ,  maxCoords[d]= " << maxCoords[d] );
        }
        totalVolume += thisVolume;

        ValueType thisWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, thisRectangle, maxCoords, settings);
        SCAI_ASSERT( thisWeight==thisRectangle.weight, "wrong weight calculation: thisWeight= " << thisWeight << " , thisRectangle.weight= " << thisRectangle.weight);
        SCAI_ASSERT( thisVolume==thisWeight , "This rectangle's area= "<< thisVolume << " and should be equal to its weight that is= "<< thisWeight);   // that is true only when all weights are 1
        
        if( thisWeight<minWeight ){
            minWeight = thisWeight;
        }
        if( thisWeight>maxWeight ){
            maxWeight = thisWeight;
        }
        
        totalWeight += thisWeight;
        
        // check if this rectangle overlaps with some other rectangle
        for(int r2=0; r2<settings.numBlocks; r2++){
            if( r2==r) continue; // do not self-check
            
            struct rectangle otherRectangle = rectangles[r2]->getRect();
            bool overlap = true;
            for(int d=0; d<dimensions; d++){
                if( thisRectangle.bottom[d]>=otherRectangle.top[d] or thisRectangle.top[d]<=otherRectangle.bottom[d] ){
                    // if they do not overlap in dimension then they cannot overlap 
                    overlap = false;
                    break;
                }
            }
            
            if( comm->getRank()==0 and overlap ){
                //PRINT0("Found overlapping rectangles:");
                thisRectangle.print();
                otherRectangle.print();
                throw std::runtime_error("The rectangles above overlap.");
            }
        }
    }
    PRINT0( "averageWeight= "<< N/settings.numBlocks );
    PRINT0( "minWeight= "<< minWeight << " , maxWeight= "<< maxWeight );
    
    //all points are covered by a rectangle
    ValueType sumWeight = nodeWeights.sum().Scalar::getValue<ValueType>();
    SCAI_ASSERT( totalWeight==sumWeight , "sum of all rectangles weight= "<< totalWeight << " and should be equal the sum of weights= "<< sumWeight);
    // this works even when weights are not 1
    SCAI_ASSERT( totalVolume==N , "total volume= "<< totalVolume << " and should be equal the number of points= "<< N);    
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testGetRectanglesNonUniformFile){
    
    const IndexType dimensions = 2;
    const IndexType k = std::pow( 4, dimensions);

    std::string path = "meshes/trace/";
    std::string fileName = "trace-00010.graph";
    std::string file = path + fileName;
    std::ifstream f(file);
    IndexType N, edges;
    f >> N >> edges; 
    
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr("BLOCK", comm, N) );
    const scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));
    const IndexType localN = dist->getLocalSize();
    PRINT0("Number of vertices= " << N);
        
    //
    // get the adjacency matrix and the coordinates
    //
    scai::lama::CSRSparseMatrix<ValueType> adjM = FileIO<IndexType, ValueType>::readGraph(file );
    adjM.redistribute(dist, noDistPointer);
    
    std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);
    
    EXPECT_TRUE(coordinates[0].getDistributionPtr()->isEqual(*dist));
    EXPECT_EQ( adjM.getNumColumns(), adjM.getNumRows());
    EXPECT_EQ( edges, (adjM.getNumValues())/2 );   
    
    //
    //create weights locally
    //
    scai::lama::DenseVector<ValueType> nodeWeights( dist );
    IndexType actualTotalWeight = 0;
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            //localPart[i] = 1;
            localPart[i] = rand()%7*comm->getRank()+2;
            actualTotalWeight += localPart[i];         
        }
    }
    actualTotalWeight =  comm->sum(actualTotalWeight);

    //
    // find min and max and scale local coordinates to a IndexType DenseVector
    //
    std::vector<ValueType> minCoords( dimensions, std::numeric_limits<ValueType>::max() );
    std::vector<ValueType> maxCoords( dimensions, std::numeric_limits<ValueType>::lowest() );
    std::vector<ValueType> scaledMin( dimensions, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> scaledMax( dimensions, std::numeric_limits<ValueType>::lowest());
    
    std::vector<std::vector<IndexType>> scaledCoords( localN , std::vector<IndexType>(dimensions,0) );
    // scale= N^(1/d): this way the scaled max is N^(1/d) and this is also the maximum size of the projection arrays
    IndexType scale = std::pow( N, 1.0/dimensions);
    
    {   
        // gel local min/max
        for (IndexType d = 0; d < dimensions; d++) {
            //scaledCoords[d].allocate( dist );
            //scaledCoords[d] = static_cast<ValueType>( 0 );
            const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[d].getLocalValues();
            
            for (IndexType i = 0; i < localN; i++) {
                ValueType coord = localPartOfCoords[i];
                if (coord < minCoords[d]) minCoords[d] = ValueType (coord);
                if (coord > maxCoords[d]) maxCoords[d] = ValueType (coord);
                //wCoordsIndex[i] = coord;
            }
        }
        // get global min/max
        for (IndexType d = 0; d < dimensions; d++) {
            minCoords[d] = comm->min(minCoords[d]);
            maxCoords[d] = comm->max(maxCoords[d])+1;
            scaledMin[d] = 0;
            scaledMax[d] = scale;
        }
        
        for (IndexType d = 0; d < dimensions; d++) {
            //get local parts of coordinates
            const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[d].getLocalValues();
            
            for (IndexType i = 0; i < localN; i++) {
                ValueType normalizedCoord = (localPartOfCoords[i] - minCoords[d])/(maxCoords[d]-minCoords[d]);
                IndexType scaledCoord =  normalizedCoord * scale; 
                scaledCoords[i][d] = scaledCoord;
                
                if (scaledCoord < scaledMin[d]) scaledMin[d] = scaledCoord;
                if (scaledCoord > scaledMax[d]) scaledMax[d] = scaledCoord;
            }
            scaledMin[d] = comm->min( scaledMin[d] );
            scaledMax[d] = comm->max( scaledMax[d] );
        }
    }
    
    for (IndexType d=0; d<dimensions; d++) {
        SCAI_ASSERT( scaledMax[d]<= scale, "Scaled maximum value "<< scaledMax[d] << " is too large. should be less than " << scale );
        if( scaledMin[d]!=0 ){
            //TODO: it works even if scaledMin is not 0 but the projection arrays will start from 0 and the first 
            //      elements will just always be 0.
            PRINT(":");
            throw std::logic_error("Minimum scaled value should be 0 but it is " + std::to_string(scaledMin[d]) );
        }
    }
    SCAI_ASSERT( localN==scaledCoords.size(), "Wrong size of scaled coordinates vector: localN= "<< localN << " and scaledCoords.size()= " << scaledCoords.size() );
    
    //
    // get tree of rectangles
    //
    Settings settings;
    settings.dimensions = dimensions;
    settings.numBlocks = k;
    
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    std::shared_ptr<rectCell<IndexType,ValueType>> root = MultiSection<IndexType, ValueType>::getRectanglesNonUniform( adjM, scaledCoords, nodeWeights, scaledMin, scaledMax, settings);
    
    std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> rectangles = root->getAllLeaves();

    std::chrono::duration<double> partitionTime = std::chrono::system_clock::now() - startTime;
    
    if (comm->getRank() == 0) {
        std::cout<< "Time to partition: "<< partitionTime.count() << std::endl;
    }
    
    //
    // assertions - prints
    //  

    IndexType totalWeightRect = 0;
    ValueType minWeight = LONG_MAX, maxWeight = 0;
    
    for(int r=0; r<settings.numBlocks; r++){
        struct rectangle thisRectangle = rectangles[r]->getRect();
        
        ValueType thisWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( scaledCoords, nodeWeights, thisRectangle, scaledMax, settings);
        /*
        //re-scale rectangle
        for (IndexType d=0; d<dimensions; d++) {
            thisRectangle.bottom[d] = (thisRectangle.bottom[d]/scale)*(maxCoords[d]-minCoords[d]) + minCoords[d];
            thisRectangle.top[d] = (thisRectangle.top[d]/scale)*(maxCoords[d]-minCoords[d]) + minCoords[d];
        }
        ValueType thisWeight = MultiSection<IndexType, ValueType>::getRectangleWeight( coordinates, nodeWeights, thisRectangle, maxCoords, settings);
        */
        SCAI_ASSERT( thisWeight==thisRectangle.weight, "wrong weight calculation: thisWeight= " << thisWeight << " , thisRectangle.weight= " << thisRectangle.weight);
        
        if( thisWeight<minWeight ){
            minWeight = thisWeight;
        }
        if( thisWeight>maxWeight ){
            maxWeight = thisWeight;
        }
        
        totalWeightRect += thisRectangle.weight;
        
        // check if this rectangle overlaps with some other rectangle
        for(int r2=0; r2<settings.numBlocks; r2++){
            if( r2==r) continue; // do not self-check
            
            struct rectangle otherRectangle = rectangles[r2]->getRect();
            bool overlap = true;
            for(int d=0; d<dimensions; d++){
                if( thisRectangle.bottom[d]>=otherRectangle.top[d] or thisRectangle.top[d]<=otherRectangle.bottom[d] ){
                    // if they do not overlap in dimension then they cannot overlap 
                    overlap = false;
                    break;
                }
            }
            
            if( comm->getRank()==0 and overlap ){
                //PRINT0("Found overlapping rectangles:");
                thisRectangle.print();
                otherRectangle.print();
                throw std::runtime_error("The rectangles above overlap.");
            }
        }
    }
    PRINT0( "averageWeight= "<< actualTotalWeight/k );
    PRINT0( "minWeight= "<< minWeight << " , maxWeight= "<< maxWeight );
    
    SCAI_ASSERT( totalWeightRect==actualTotalWeight , "Wrong total weight:  totalWeightRect= "<< totalWeightRect << " and actual weight is " << actualTotalWeight );
    
}
//---------------------------------------------------------------------------------------


/* example for this test case (for all maxCoord but only 2 dimensions):
 * a grid separated in 3 rectangles with coresponding leaf ids
 *  -----------------
 *  |   1   |       |
 *  | bBox3 |       |
 *  --------|       |
 *  |       |   2   |
 *  |   0   | bBox1 |
 *  | bB0x2 |       |
 *  |       |       |
 *  -----------------
 * 
 * */
TEST_F(MultiSectionTest, test1DProjectionNonUniform_2D){
    
    const IndexType dimensions = 2;
    const std::vector<ValueType> maxCoord= {14, 20};
    const IndexType N = (maxCoord[0]+1)*(maxCoord[1]+1);
    
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr("BLOCK", comm, N) );
    const scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));
    const IndexType localN = dist->getLocalSize();
    
    scai::lama::CSRSparseMatrix<ValueType> adjM(dist, noDistPointer);
  
    std::vector<DenseVector<IndexType>> coordinates(dimensions);
    
    // create all integer cordinates in the grid
    for(IndexType i=0; i<dimensions; i++){
        coordinates[i].allocate(N);
        coordinates[i] = static_cast<ValueType>( 0 );
    }
    int p=0;
    for(int i=0; i<=maxCoord[0]; i++){
        for(int j=0; j<=maxCoord[1]; j++){  
            coordinates[0][p] = i;
            coordinates[1][p] = j;
            ++p;
        }
    }
    SCAI_ASSERT( p==N, "Wrong number of points.");
    
    coordinates[0].redistribute( dist );
    coordinates[1].redistribute( dist );
    
    //set local weights + convert local part of coordinates to a vector<vector<IndexType>>
    scai::lama::DenseVector<ValueType> nodeWeights( dist );
    std::vector<std::vector<IndexType>> localPoints( localN, std::vector<IndexType>( dimensions, 0) );
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1; //rand()%10;
        }
        
        for(int d=0; d<dimensions; d++){
            scai::hmemo::ReadAccess<IndexType> localCoordR( coordinates[d].getLocalValues() );
            for(int i=0; i<localN; i++){
                localPoints[i][d] = localCoordR[i];
            }
        }
        
    }
    SCAI_ASSERT( coordinates[0].getLocalValues().size()==nodeWeights.getLocalValues().size(), "Wrong local sizes for coordinates and weights vectors");
    
    struct Settings settings;
    //settings.numBlocks= k;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    
    rectangle bBox;         //not a leaf => no projection calculated
    // for all dimensions i: bottom[i]<top[i] 
    bBox.bottom = {0, 0};
    bBox.top = maxCoord;
    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
    
    rectangle bBox0;        //not a leaf => no projection calculated
    bBox0.bottom = {0, 0};
    bBox0.top = { std::floor(maxCoord[0]/2-1), maxCoord[1] };
    ValueType bBox0Weight = (bBox0.top[0]-bBox0.bottom[0]+1)*(bBox0.top[1]-bBox0.bottom[1]+1);
    bBox0.weight = bBox0Weight;
    root->insert(bBox0);
    //if(comm->getRank()==0) bBox0.print();    
    
    rectangle bBox1;        // leafID=2 => projection[2]
    bBox1.bottom = { std::floor(maxCoord[0]/2), 0 };
    bBox1.top = maxCoord;
    ValueType bBox1Weight = (bBox1.top[0]-bBox1.bottom[0]+1)*(bBox1.top[1]-bBox1.bottom[1]+1);
    bBox1.weight = bBox1Weight;
    root->insert(bBox1);
    //if(comm->getRank()==0) bBox1.print();    
    
    rectangle bBox2;        // leafID=0 => projection[0]
    bBox2.bottom = {0, 0};
    bBox2.top = { std::floor(maxCoord[0]/2-1), std::floor(maxCoord[1]*0.75 -1) };
    ValueType bBox2Weight = (bBox2.top[0]-bBox2.bottom[0]+1)*(bBox2.top[1]-bBox2.bottom[1]+1);
    bBox2.weight = bBox2Weight;
    root->insert( bBox2 );
    //if(comm->getRank()==0)  bBox2.print();        
    
    rectangle bBox3;        // leafID=1 => projection[1]
    bBox3.bottom = {0, std::floor(maxCoord[1]*0.75) };
    bBox3.top = { std::floor(maxCoord[0]/2 -1) , maxCoord[1] };
    ValueType bBox3Weight = (bBox3.top[0]-bBox3.bottom[0]+1)*(bBox3.top[1]-bBox3.bottom[1]+1);
    bBox.weight = bBox3Weight;
    root->insert( bBox3 );
    //if(comm->getRank()==0)  bBox3.print();         
    
    SCAI_ASSERT( root->getNumLeaves()==3 , "Tree must have 3 leaves but has "<< root->getNumLeaves() );
    
    // bBox0 not leaf, bBox1=projections[2], bBox2=projections[0] , bBox3=projections[1]

    ValueType totalGridWeight = N;
    SCAI_ASSERT( totalGridWeight==nodeWeights.sum() , "Wrong sum of node weights: "<< nodeWeights.sum() );
    
    for(int d=0; d<dimensions; d++){
        //dim2proj.size() = number of leaves/rectangles
        std::vector<IndexType> dim2proj = {d, d, d};
        SCAI_ASSERT( dim2proj.size()==root->getNumLeaves(), "Wrong size of vecctor dim2proj or number of leaves");
        
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projectionNonUniform( localPoints , nodeWeights, root, dim2proj, settings);
        
        ValueType projectionsTotalWeight =0 ;
        
        for( int i=0; i<projections.size(); i++){
                projectionsTotalWeight += std::accumulate( projections[i].begin(), projections[i].end(), 0.0 );
        }
        SCAI_ASSERT( projectionsTotalWeight==totalGridWeight , "Wrong sum of projections weights: projectionsTotalWeight= " << projectionsTotalWeight << " , totalGridWeight= " << totalGridWeight);
    
        SCAI_ASSERT( projections[0].size()==(bBox2.top[d]-bBox2.bottom[d]+1), "Wrong size for projection 0");
        SCAI_ASSERT( projections[1].size()==(bBox3.top[d]-bBox3.bottom[d]+1), "Wrong size for projection 1");
        SCAI_ASSERT( projections[2].size()==(bBox1.top[d]-bBox1.bottom[d]+1), "Wrong size for projection 2");
        
        SCAI_ASSERT( projections.size()==3, "projections size must be 2 but it is "<< projections.size() );
        ValueType proj0Weight = std::accumulate( projections[0].begin(), projections[0].end(), 0.0 );
        ValueType proj1Weight = std::accumulate( projections[1].begin(), projections[1].end(), 0.0 );
        ValueType proj2Weight = std::accumulate( projections[2].begin(), projections[2].end(), 0.0 );

        SCAI_ASSERT( totalGridWeight == proj1Weight+proj0Weight+proj2Weight , "Total weight is "<< totalGridWeight << " but the sum of the two projections is "<<  proj1Weight+proj0Weight+proj2Weight );
        SCAI_ASSERT( proj0Weight==bBox2Weight, "Weight of first rectangle is "<< bBox2Weight << " but the weight of the projection is "<< proj0Weight);
        SCAI_ASSERT( proj1Weight==bBox3Weight, "Weight of second rectangle is "<< bBox3Weight << " but the weight of the projection is "<< proj1Weight);
        SCAI_ASSERT( proj2Weight==bBox1Weight, "Weight of third rectangle is "<< bBox1Weight << " but the weight of the projection is "<< proj2Weight);
    } 
    
    //add extra rectangle to check block graph
    // graph should be:    0 - 2    =>  0 1 1 1
    //                     | \ |        1 0 1 1
    //                     3 - 1        1 1 0 0
    //                                  1 1 0 0
    
    rectangle bBox4;
    bBox4.bottom =  { std::floor(maxCoord[0]/2), 0 };
    bBox4.top = { std::floor(maxCoord[0]/3 -1) , maxCoord[1] };
    ValueType bBox4Weight = (bBox4.top[0]-bBox4.bottom[0]+1)*(bBox4.top[1]-bBox4.bottom[1]+1);
    bBox.weight = bBox4Weight;
    root->insert( bBox4 );
    //if(comm->getRank()==0)  bBox4.print();     

    scai::lama::CSRSparseMatrix<ValueType> blockGraph = MultiSection<IndexType, ValueType>::getBlockGraphFromTree_local(root);

    IndexType numRows = blockGraph.getNumRows() , numCols = blockGraph.getNumColumns();
    PRINT0( numRows<< " x "<< numCols );
    SCAI_ASSERT_EQ_ERROR( numRows, numCols, "Number of rows and colums must be equal");        EXPECT_TRUE( blockGraph.isConsistent() );
    EXPECT_TRUE( blockGraph.checkSymmetry() );

    if( comm->getRank()==0 ){
        for(int i=0; i<numRows; i++){
            for(int j=0; j<numCols; j++){
                std::cout<< blockGraph.getValue(i,j).Scalar::getValue<IndexType>() << " ";
            }
            std::cout << std::endl;
        }
    }
}
//---------------------------------------------------------------------------------------
/* same example as in the 2D case
 */
TEST_F(MultiSectionTest, test1DProjectionNonUniform_3D){
    
    const IndexType dimensions = 3;
    const std::vector<ValueType> maxCoord= {10, 15, 20};
    const IndexType N = (maxCoord[0]+1)*(maxCoord[1]+1)*(maxCoord[2]+1);
    
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr("BLOCK", comm, N) );
    const scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));
    const IndexType localN = dist->getLocalSize();
    
    scai::lama::CSRSparseMatrix<ValueType> adjM(dist, noDistPointer);
  
    std::vector<DenseVector<ValueType>> coordinates(dimensions);
    
    // create all integer cordinates in the grid
    for(IndexType i=0; i<dimensions; i++){
        coordinates[i].allocate(N);
        coordinates[i] = static_cast<ValueType>( 0 );
    }
    int p=0;
    for(int x=0; x<=maxCoord[0]; x++){
        for(int y=0; y<=maxCoord[1]; y++){
            for(int z=0; z<=maxCoord[2]; z++){
                coordinates[0][p] = x;
                coordinates[1][p] = y;
                coordinates[2][p] = z;
                ++p;
            }
        }
    }
    SCAI_ASSERT( p==N, "Wrong number of coordinates created");
    
    coordinates[0].redistribute( dist );
    coordinates[1].redistribute( dist );
    coordinates[2].redistribute( dist );
    
    //copy local coordinates to a vector<vector<>> structure
    std::vector<std::vector<IndexType>> coordsIndex( localN, std::vector<IndexType>(dimensions,0) );
    {
        for (IndexType d = 0; d < dimensions; d++) {
            const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[d].getLocalValues();
            
            for (IndexType i = 0; i < localN; i++) {
                coordsIndex[i][d] = localPartOfCoords[i];
            }
        }
    }

    //set local weights
    scai::lama::DenseVector<ValueType> nodeWeights( dist );
    {
        scai::hmemo::WriteAccess<ValueType> localPart(nodeWeights.getLocalValues());
        srand(time(NULL));
        for(int i=0; i<localN; i++){
            localPart[i] = 1; //rand()%10;
        }
    }
    SCAI_ASSERT( coordinates[0].getLocalValues().size()==nodeWeights.getLocalValues().size(), "Wrong local sizes for coordinates and weights vectors");
    
    struct Settings settings;
    settings.epsilon = 0.2;
    settings.dimensions = dimensions;
    
    rectangle bBox;         //not a leaf => no projection calculated
    // for all dimensions i: bottom[i]<top[i] 
    bBox.bottom = {0, 0, 0};
    bBox.top = maxCoord;
    
    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
    
    rectangle bBox0;        //not a leaf => no projection calculated
    bBox0.bottom = {0, 0, 0};
    bBox0.top = { std::floor(maxCoord[0]/2-1), maxCoord[1], maxCoord[2] };
    bBox0.weight = (bBox0.top[0]-bBox0.bottom[0])*(bBox0.top[1]-bBox0.bottom[1])*(bBox0.top[2]-bBox0.bottom[2]);
    root->insert(bBox0);
    
    rectangle bBox1;        // leafID=2 => projection[2]
    bBox1.bottom = { std::floor(maxCoord[0]/2), 0 , 0};
    bBox1.top = maxCoord;
    ValueType bBox1Weight = (bBox1.top[0]-bBox1.bottom[0]+1)*(bBox1.top[1]-bBox1.bottom[1]+1)*(bBox1.top[2]-bBox1.bottom[2]+1);
    bBox1.weight = bBox1Weight;
    root->insert(bBox1);
    
    rectangle bBox2;        // leafID=0 => projection[0]
    bBox2.bottom = {0, 0, 0};
    bBox2.top = { std::floor(maxCoord[0]/2-1), std::floor(maxCoord[1]*0.75-1), maxCoord[2] };
    ValueType bBox2Weight = (bBox2.top[0]-bBox2.bottom[0]+1)*(bBox2.top[1]-bBox2.bottom[1]+1)*(bBox2.top[2]-bBox2.bottom[2]+1);
    bBox2.weight = bBox2Weight;
    root->insert( bBox2 );

    rectangle bBox3;        // leafID=1 => projection[1]
    bBox3.bottom = {0, std::floor(maxCoord[1]*0.75), 0};
    bBox3.top = { std::floor(maxCoord[0]/2-1) , maxCoord[1], maxCoord[2] };
    ValueType bBox3Weight = (bBox3.top[0]-bBox3.bottom[0]+1)*(bBox3.top[1]-bBox3.bottom[1]+1)*(bBox3.top[2]-bBox3.bottom[2]+1);
    bBox3.weight = bBox3Weight;
    root->insert( bBox3 );
    
    SCAI_ASSERT( root->getNumLeaves()==3 , "Tree must have 3 leaves but has "<< root->getNumLeaves() );

    // bBox0 not leaf, bBox1=projections[2], bBox2=projections[0] , bBox3=projections[1]

    ValueType totalGridWeight = N;
    SCAI_ASSERT( totalGridWeight==nodeWeights.sum() , "Wrong sum of node weights: "<< nodeWeights.sum().Scalar::getValue<IndexType>() );
    
    for(int d=0; d<dimensions; d++){
        //dim2proj.size() = number of leaves/rectangles
        std::vector<IndexType> dim2proj = {d, d, d};
        SCAI_ASSERT( dim2proj.size()==root->getNumLeaves(), "Wrong size of vecctor dim2proj or number of leaves");
        
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projectionNonUniform( coordsIndex, nodeWeights, root, dim2proj, settings);
    
        SCAI_ASSERT( projections[0].size()==(bBox2.top[d]-bBox2.bottom[d]+1), "Wrong size for projection 0");
        SCAI_ASSERT( projections[1].size()==(bBox3.top[d]-bBox3.bottom[d]+1), "Wrong size for projection 1");
        SCAI_ASSERT( projections[2].size()==(bBox1.top[d]-bBox1.bottom[d]+1), "Wrong size for projection 2");
        
        SCAI_ASSERT( projections.size()==3, "projections size must be 3 but it is "<< projections.size() );
        
        ValueType proj0Weight = std::accumulate( projections[0].begin(), projections[0].end(), 0.0 );
        ValueType proj1Weight = std::accumulate( projections[1].begin(), projections[1].end(), 0.0 );
        ValueType proj2Weight = std::accumulate( projections[2].begin(), projections[2].end(), 0.0 );

        SCAI_ASSERT( totalGridWeight == proj1Weight+proj0Weight+proj2Weight , "Total weight is "<< totalGridWeight << " but the sum of the two projections is "<<  proj1Weight+proj0Weight+proj2Weight );
        SCAI_ASSERT( proj0Weight==bBox2Weight, "Weight of first rectangle is "<< bBox2Weight << " but the weight of the projection is "<< proj0Weight);
        SCAI_ASSERT( proj1Weight==bBox3Weight, "Weight of second rectangle is "<< bBox3Weight << " but the weight of the projection is "<< proj1Weight);
        SCAI_ASSERT( proj2Weight==bBox1Weight, "Weight of third rectangle is "<< bBox1Weight << " but the weight of the projection is "<< proj2Weight);
    }    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testInbBox){
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType sideLen =10;
    IndexType dim = 3;
    IndexType N= std::pow( sideLen, dim );   // for a N^dim grid
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::lama::DenseVector<ValueType> nodeWeights( blockDist );
    IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();
    
    // for all dimensions i: first[i]<second[i] 
    rectangle bBox;
    bBox.bottom = {2,3,1};
    bBox.top = {6,8,6};
    
    //bBox area = number of points in the box
    IndexType bBoxArea = 1;
    for(int i=0; i<dim; i++){
        bBoxArea *= (bBox.top[i]-bBox.bottom[i]+1);
    }
    
    IndexType numPoints=0;
    for(int i=0; i<N; i++){
        std::vector<IndexType> coords = MultiSection<IndexType,ValueType>::indexToCoords<IndexType>(i, sideLen, dim);
        if( MultiSection<IndexType, ValueType>::inBBox(coords, bBox) == true){
            ++numPoints;
        }
    }
    PRINT0("bBox area= "<< bBoxArea << " , num of points in bBox= "<< numPoints);
    SCAI_ASSERT( numPoints==bBoxArea, "numPoints= " << numPoints << " should equal to bBoxArea= "<< bBoxArea);
    
    
}
//---------------------------------------------------------------------------------------

TEST_F(MultiSectionTest, testIndexTo){
    
    IndexType sideLen=4;
    IndexType gridSize;
    IndexType dim;
    
    // 2D case
    dim = 2;
    gridSize= sideLen*sideLen;
    for(int i=0; i<gridSize; i++){
        std::vector<IndexType> p = MultiSection<IndexType, ValueType>::indexToCoords<IndexType>( i , sideLen, dim );
        //PRINT(i<< ": " << p[0] << " , "<< p[1]);
        EXPECT_LE( p[0], sideLen);
        EXPECT_LE( p[1], sideLen);
        EXPECT_GE( p[0], 0);
        EXPECT_GE( p[1], 0);
    }
    
    // 3D case
    dim = 3;
    gridSize= sideLen*sideLen*sideLen;
    for(int i=0; i<gridSize; i++){
        std::vector<IndexType> t = MultiSection<IndexType, ValueType>::indexToCoords<IndexType>( i , sideLen, dim );
        //PRINT(i<< ": " << t[0] << " , "<< t[1] << " , "<< t[2] );
        EXPECT_LE( t[0], sideLen);
        EXPECT_LE( t[1], sideLen);
        EXPECT_LE( t[2], sideLen);
        EXPECT_GE( t[0], 0);
        EXPECT_GE( t[1], 0);
        EXPECT_GE( t[2], 0);
    }
}

}
