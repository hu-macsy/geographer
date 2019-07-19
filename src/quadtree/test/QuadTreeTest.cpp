/*
 * QuadTreeTest.cpp
 *
 *  Created on: 28.05.2014
 *      Author: Moritz v. Looz (moritz.looz-corswarem@kit.edu)
 */

#include <stack>
#include <cmath>
#include <algorithm>
#include <random>

#include <scai/lama/matrix/all.hpp>

#include "QuadTreeTest.h"
#include "../../ParcoRepart.h"
#include "../../FileIO.h"
#include "../../GraphUtils.h"
#include "../../Settings.h"
#include "../../Metrics.h"

#include "../QuadTreeCartesianEuclid.h"
#include "../QuadTreePolarEuclid.h"
#include "../KDTreeEuclidean.h"

#include <boost/filesystem.hpp>

namespace ITI {

TEST_F(QuadTreeTest, testGetGraphFromForestRandom_2D) {

    // every forest[i] is a pointer to the root of a tree
    std::vector<std::shared_ptr<const SpatialCell>> forest;

    IndexType n= 20;
    //vector<Point<ValueType> > positions(n);
    //vector<index> content(n);

    Point<ValueType> min(0.0, 0.0);
    Point<ValueType> max(1.0, 1.0);
    index capacity = 1;

    QuadTreeCartesianEuclid quad(min, max, true, capacity);
    QuadTreeCartesianEuclid quad2(min, max, true, capacity);
    index i=0;
    srand(time(NULL));

    for (i = 0; i < n; i++) {
        Point<ValueType> pos = Point<ValueType>({ValueType(rand()) / RAND_MAX, ValueType(rand()) / RAND_MAX});
        Point<ValueType> pos2 = Point<ValueType>({ValueType(rand()) / RAND_MAX, ValueType(rand()) / RAND_MAX});
        quad.addContent(i, pos);
        quad2.addContent(i, pos2);
    }

    IndexType globIndexing = quad2.indexSubtree(quad.indexSubtree(0));

    forest.push_back(quad.getRoot());
    forest.push_back(quad2.getRoot());

    IndexType numTrees = forest.size();


    // ^^ forest created ^^


    // graphNgbrsPtrs[i]= a set with pointers to the neighbours of -i- in the CSR matrix/graph
    std::vector< std::set<std::shared_ptr<const SpatialCell>>> graphNgbrsPtrs( globIndexing );
    //WARNING: this kind of edges must be symmetric
    graphNgbrsPtrs[forest[0]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[1]) );
    graphNgbrsPtrs[forest[1]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[0]) );

    PRINT("num trees= " << numTrees << ", globIndex= " << globIndexing);
    int dimension = 2;
    std::vector<std::vector<ValueType>> coords( dimension );
    scai::lama::CSRSparseMatrix<ValueType> graph= SpatialTree::getGraphFromForest<IndexType, ValueType>( graphNgbrsPtrs,  forest, coords);

    // checkSymmetry is really expensive for big graphs, used only for small instances
    graph.checkSymmetry();
    graph.isConsistent();
}
//-------------------------------------------------------------------------------------------------


TEST_F(QuadTreeTest, testGetGraphFromForestByHand_2D) {

    // every forest[i] is a pointer to the root of a tree
    std::vector<std::shared_ptr<const SpatialCell>> forest;

    IndexType n= 2;
    vector<Point<ValueType> > positions(n);
    vector<index> content(n);

    Point<ValueType> min(0.0, 0.0);
    Point<ValueType> max(1.0, 1.0);
    index capacity = 1;
    index i=0;

    QuadTreeCartesianEuclid quad0(min, max, true, capacity);
    quad0.addContent( i++, Point<ValueType>({0.4, 0.3}) );
    quad0.addContent( i++, Point<ValueType>({0.4, 0.8}) );

    QuadTreeCartesianEuclid quad1( Point<ValueType>({1.0, 0.0}), Point<ValueType>({2.0, 1.0}), true, capacity);
    quad1.addContent(i++, Point<ValueType>({1.3, 0.2}));
    quad1.addContent(i++, Point<ValueType>({1.3, 0.8}));

    QuadTreeCartesianEuclid quad2( Point<ValueType>({0.0, 1.0}), Point<ValueType>({1.0, 2.0}), true, capacity);
    quad2.addContent( i++, Point<ValueType>({0.6, 1.1}) );
    quad2.addContent( i++, Point<ValueType>({0.6, 1.8}) );

    QuadTreeCartesianEuclid quad3( Point<ValueType>({1.0, 1.0}), Point<ValueType>({2.0, 2.0}), true, capacity);
    quad3.addContent( i++, Point<ValueType>({1.3, 1.2}) );
    quad3.addContent( i++, Point<ValueType>({1.3, 1.8}) );

    IndexType globIndexing=0;
    globIndexing = quad0.indexSubtree(globIndexing);
    globIndexing = quad1.indexSubtree(globIndexing);
    globIndexing = quad2.indexSubtree(globIndexing);
    globIndexing = quad3.indexSubtree(globIndexing);

    forest.push_back(quad0.getRoot());
    forest.push_back(quad1.getRoot());
    forest.push_back(quad2.getRoot());
    forest.push_back(quad3.getRoot());

    IndexType numTrees = forest.size();
    for(i=0; i<numTrees; i++) {
        PRINT(i << ",forest root id= "<< forest[i]->getID());
    }

    // ^^ forest created ^^


    // graphNgbrsPtrs[i]= a set with pointers to the neighbours of -i- in the CSR matrix/graph
    // graphNgbrsPtrs.size() == size of the forest , all nodes on every tree
    std::vector< std::set<std::shared_ptr<const SpatialCell>>> graphNgbrsPtrs( globIndexing );
    //WARNING: this kind of edges must be symmetric

    // quad0 connects with quad1 and 2
    graphNgbrsPtrs[forest[0]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[1]) );
    graphNgbrsPtrs[forest[1]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[0]) );
    graphNgbrsPtrs[forest[0]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[2]) );
    graphNgbrsPtrs[forest[2]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[0]) );

    // quad1 connects with 0 and 3
    graphNgbrsPtrs[forest[1]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[3]) );
    graphNgbrsPtrs[forest[3]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[1]) );

    graphNgbrsPtrs[forest[2]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[3]) );
    graphNgbrsPtrs[forest[3]->getID()].insert( std::shared_ptr<const SpatialCell> (forest[2]) );

    int dimension = 2;
    std::vector<std::vector<ValueType>> coords( dimension );
    PRINT("num trees= " << numTrees << ", globInde= " << globIndexing);
    scai::lama::CSRSparseMatrix<ValueType> graph= SpatialTree::getGraphFromForest<IndexType, ValueType>( graphNgbrsPtrs,  forest, coords);

    EXPECT_EQ(coords.size(), dimension);
    EXPECT_EQ(coords[0].size(), graph.getNumRows());

    // check coords are the same
    for(int d=0; d<coords.size(); d++) {
        for(int i=0; i<coords[d].size(); i++) {
            std::cout << coords[d][i] << ", ";
        }
        std::cout<< std::endl;
    }

    // checkSymmetry is really expensive for big graphs, used only for small instances
    graph.checkSymmetry();
    graph.isConsistent();

    //print graph
    /*
    for(int i=0; i<graph.getNumRows(); i++){
        std::cout << i <<": ";
        for(int j=0; j<graph.getNumColumns(); j++){
            std::cout<< j << ":"<< graph(i,j).Scalar::getValue<ValueType>() << " , ";
        }
        std::cout<< std::endl;
    }
    */
    PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() );
}


TEST_F(QuadTreeTest, testGetGraphMatrixFromTree_3D) {
    count n = 3500;

    vector<Point<ValueType> > positions(n);
    vector<index> content(n);

    Point<ValueType> min(0.0, 0.0, 0.0);
    Point<ValueType> max(1.0, 1.0, 1.0);
    index capacity = 1;

    QuadTreeCartesianEuclid quad(min, max, true, capacity);
    index i=0;
    srand(time(NULL));

    for (i = 0; i < n; i++) {
        Point<ValueType> pos = Point<ValueType>({ValueType(rand()) / RAND_MAX, ValueType(rand()) / RAND_MAX, ValueType(rand()) / RAND_MAX});
        positions[i] = pos;
        content[i] = i;
        quad.addContent(i, pos);
    }

    //PRINT("Num of leaves= N = "<< quad.countLeaves() );
    index N= quad.countLeaves();

    // index the tree
    index treeSize = quad.indexSubtree(0);

    // A set for every node in the tree, graphNgbrsCells[i] contains shared_ptrs to every neighbour
    // of -i- in the output graph, not the quad tree.
    std::vector< std::set<std::shared_ptr<const SpatialCell>>> graphNgbrsCells( treeSize );
    int dimension = 3;
    std::vector<std::vector<ValueType>> coords( dimension );

    scai::lama::CSRSparseMatrix<ValueType> graph= quad.getTreeAsGraph<IndexType, ValueType>( graphNgbrsCells, coords );

    // checkSymmetry is really expensive for big graphs, used only for small instances
    //graph.checkSymmetry();
    graph.isConsistent();

    //EXPECT_EQ( graph.getNumRows(), graph.getNumColumns() );
    ASSERT_EQ( graph.getNumRows(), N);
    ASSERT_EQ( graph.getNumColumns(), N);

    const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    // 50 is too large upper bound (is it?). Should be around 24 for 3D and 8 (or 10) for 2D
    //TODO: maybe 30 is not so large... find another way to do it or skip it entirely
    IndexType upBound= 50;
    std::vector<IndexType> degreeCount( upBound*2, 0 );

    for(IndexType i=0; i<N; i++) {
        IndexType nodeDegree = ia[i+1] -ia[i];
        if( nodeDegree > upBound) {
            //throw std::warning( "Node with large degree, degree= "+  std::to_string(nodeDegree) + " > current upper bound= " + std::to_string(upBound) );
            // throw as a warning for now
            PRINT("WARNING: degree too high= "<< nodeDegree);
        }
        ++degreeCount[nodeDegree];
    }

    IndexType numEdges = 0;
    IndexType maxDegree = 0;
    std::cout<< "\t Num of nodes"<< std::endl;
    for(int i=0; i<degreeCount.size(); i++) {
        if(  degreeCount[i] !=0 ) {
            //PRINT("degree " << i << ":   "<< degreeCount[i]);
            numEdges += i*degreeCount[i];
            maxDegree = i;
        }
    }
    EXPECT_EQ(numEdges, graph.getNumValues() );

    ValueType averageDegree = ValueType( numEdges)/N;

    PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() << ", average degree= "<< averageDegree << ", max degree= "<< maxDegree);

}
//-------------------------------------------------------------------------------------------------

TEST_F(QuadTreeTest, testGetGraphMatrixFromTree_Distributed_3D) {

    count n = 500;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    vector<Point<ValueType> > positions(n);
    vector<index> content(n);

    Point<ValueType> min(0.0, 0.0, 0.0);
    Point<ValueType> max(1.0, 1.0, 1.0);
    index capacity = 1;

    QuadTreeCartesianEuclid quad(min, max, true, capacity);
    index i=0;

    //broadcast seed value from root to ensure equal pseudorandom numbers.
    ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
    comm->bcast( seed, 1, 0 );
    srand(seed[0]);

    for (i = 0; i < n; i++) {
        Point<ValueType> pos = Point<ValueType>({ValueType(rand()) / RAND_MAX, ValueType(rand()) / RAND_MAX, ValueType(rand()) / RAND_MAX});
        positions[i] = pos;
        content[i] = i;
        quad.addContent(i, pos);
    }

    PRINT("Num of leaves = N = "<< quad.countLeaves() );
    index N = quad.countLeaves();
    // index the tree
    index treeSize = quad.indexSubtree(0);
    ASSERT_GT(treeSize, 0);

    // A set for every node in the tree, graphNgbrsCells[i] contains shared_ptrs to every neighbour
    // of -i- in the output graph, not the quad tree.
    std::vector< std::set<std::shared_ptr<const SpatialCell>>> graphNgbrsCells( treeSize );
    int dimension = 3;
    std::vector<std::vector<ValueType>> coords( dimension );

    scai::lama::CSRSparseMatrix<ValueType> graph= quad.getTreeAsGraph<IndexType,ValueType>(graphNgbrsCells, coords);
    /*
    //print graph
    for(int i=0; i<graph.getNumRows(); i++){
        std::cout << i <<": \t";
        for(int j=0; j<graph.getNumColumns(); j++){
            std::cout<< j << ":"<< graph.getValue(i,j).Scalar::getValue<ValueType>() << " , ";
        }
        std::cout<< std::endl;
    }
    */

    // checkSymmetry is really expensive for big graphs, use only for small instances
    graph.checkSymmetry();
    graph.isConsistent();

    ASSERT_EQ(coords[0].size(), N);

    ASSERT_EQ( graph.getNumRows(), N);
    ASSERT_EQ( graph.getNumColumns(), N);
    {
        const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
        const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
        const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

        // 20 is too large upper bound. Should be around 24 for 3D and 8 (or 10) for 2D
        //TODO: maybe 30 is not so large... find another way to do it or skip it entirely
        IndexType upBound= 50;
        std::vector<IndexType> degreeCount( upBound, 0 );
        for(IndexType i=0; i<N; i++) {
            IndexType nodeDegree = ia[i+1] -ia[i];
            if( nodeDegree > upBound) {
                throw std::logic_error( "Node with large degree, degree= "+  std::to_string(nodeDegree) + " > current upper bound= " + std::to_string(upBound) );
            }
            ++degreeCount[nodeDegree];
        }

        IndexType numEdges = 0;
        //IndexType maxDegree = 0; //not used
        //std::cout<< "\t Num of nodes"<< std::endl;
        for(int i=0; i<degreeCount.size(); i++) {
            if(  degreeCount[i] !=0 ) {
                //std::cout << "degree " << i << ":   "<< degreeCount[i]<< std::endl;
                numEdges += i*degreeCount[i];
                //maxDegree = i;
            }
        }
        EXPECT_EQ(numEdges, graph.getNumValues() );

        //ValueType averageDegree = ValueType( numEdges)/N;
        //PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() << ", average degree= "<< averageDegree << ", max degree= "<< maxDegree);
    }

    // communicate/distribute graph

    IndexType k = comm->getSize();

    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));

    graph.redistribute( dist, noDistPointer);

    //TODO: change coords data type to vector<ValueType> ? or the way we copy to a DenseVector
    std::vector<DenseVector<ValueType>> coordsDV(dimension);

    for(int d=0; d<dimension; d++) {
        coordsDV[d].allocate(coords[d].size() );
        for(IndexType j=0; j<coords[d].size(); j++) {
            coordsDV[d].setValue(j, coords[d][j]);
        }
        coordsDV[d].redistribute(dist);
    }

    EXPECT_EQ(coordsDV[0].getLocalValues().size(), graph.getLocalNumRows() );

    const ValueType epsilon = 0.05;
    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = epsilon;
    settings.dimensions = dimension;
    settings.minGainForNextRound = 5;
    settings.storeInfo = false;

    struct Metrics metrics(settings);

    EXPECT_EQ( coords[0].size(), N);
    EXPECT_EQ( graph.getNumRows(), N);
    EXPECT_EQ( graph.getNumColumns(), N);

    scai::lama::DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coordsDV, settings, metrics);

    const ValueType imbalance = GraphUtils<IndexType, ValueType>::computeImbalance(partition, k);
    EXPECT_LE(imbalance, epsilon);

    const ValueType cut = GraphUtils<IndexType, ValueType>::computeCut(graph, partition, true);

    if (comm->getRank() == 0) {
        std::cout << "Commit " << version << ": Partitioned graph with " << N << " nodes into " << k << " blocks with a total cut of " << cut << std::endl;
    }

}


TEST_F(QuadTreeTest, testGetGraphMatrixFromTree_2D) {

    index n=8;
    vector<Point<ValueType> > positions(n);
    vector<index> content(n);

    Point<ValueType> min(0.0, 0.0);
    Point<ValueType> max(2.0, 2.0);
    index capacity = 1;

    // the quadtree
    QuadTreeCartesianEuclid quad(min, max, true, capacity);

    index i=0;
    // 2D points
    quad.addContent(i++, Point<ValueType>({0.2, 0.2}) );
    quad.addContent(i++, Point<ValueType>({0.8, 0.7}) );
    quad.addContent(i++, Point<ValueType>({1.4, 0.7}) );
    quad.addContent(i++, Point<ValueType>({1.8, 0.3}) );

    quad.addContent(i++, Point<ValueType>({0.2, 0.8}) );
    quad.addContent(i++, Point<ValueType>({0.2, 0.6}) );

    quad.addContent(i++, Point<ValueType>({0.7, 1.1}) );
    quad.addContent(i++, Point<ValueType>({0.2, 1.6}) );

    PRINT("Num of leaves= N = "<< quad.countLeaves() );
    index N= quad.countLeaves();
    // index the tree
    index treeSize = quad.indexSubtree(0);

    // A set for every node in the tree, graphNgbrsCells[i] contains shared_ptrs to every neighbour
    // of -i- in the output graph, not the quad tree.
    std::vector< std::set<std::shared_ptr<const SpatialCell>>> graphNgbrsCells( treeSize );
    int dimension = 2;
    std::vector<std::vector<ValueType>> coords( dimension );

    scai::lama::CSRSparseMatrix<ValueType> graph= quad.getTreeAsGraph<IndexType, ValueType>(graphNgbrsCells, coords);

    /*
    //print graph
    for(int i=0; i<graph.getNumRows(); i++){
        std::cout << i <<": \t";
        for(int j=0; j<graph.getNumColumns(); j++){
            std::cout<< j << ":"<< graph.getValue(i,j).Scalar::getValue<ValueType>() << " , ";
        }
        std::cout<< std::endl;
    }
    */

    // checkSymmetry is really expensive for big graphs, used only for small instances
    graph.checkSymmetry();
    graph.isConsistent();

    ASSERT_EQ( graph.getNumRows(), N);
    ASSERT_EQ( graph.getNumColumns(), N);

    const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    // 20 is too large upper bound. Should be around 24 for 3D and 8 (or 10) for 2D
    //TODO: maybe 20 is not so large... find another way to do it or skip it entirely
    IndexType upBound= 20;
    std::vector<IndexType> degreeCount( upBound, 0 );
    for(IndexType i=0; i<N; i++) {
        IndexType nodeDegree = ia[i+1] -ia[i];
        if( nodeDegree > upBound) {
            throw std::logic_error( "Node with large degree, degree= "+  std::to_string(nodeDegree) + " > current upper bound= " + std::to_string(upBound) );
        }
        /*
        PRINT("node "<< i << " has edges with nodes (from "<< ia[i]<< " to "<< ia[i+1]<<"): ");
        for(int ii=ia[i]; ii<ia[i+1]; ii++){
            std::cout<< ja[ii] << ", ";
        }
        std::cout<< std::endl;
        */
        ++degreeCount[nodeDegree];
    }

    IndexType numEdges = 0;
    IndexType maxDegree = 0;
    std::cout<< "\t Num of nodes"<< std::endl;
    for(int i=0; i<degreeCount.size(); i++) {
        if(  degreeCount[i] !=0 ) {
            std::cout << "degree " << i << ":   "<< degreeCount[i]<< std::endl;
            numEdges += i*degreeCount[i];
            maxDegree = i;
        }
    }
    EXPECT_EQ(numEdges, graph.getNumValues() );

    ValueType averageDegree = ValueType( numEdges)/N;

    PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() << ", average degree= "<< averageDegree << ", max degree= "<< maxDegree);

}



TEST_F(QuadTreeTest, testGetGraphMatrixFromTree_Distributed_2D) {

    count n = 100;

    vector<Point<ValueType> > positions(n);
    vector<index> content(n);

    Point<ValueType> min(0.0, 0.0);
    Point<ValueType> max(1000.0, 1000.0);
    index capacity = 1;

    QuadTreeCartesianEuclid quad(min, max, true, capacity);
    index i=0;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    //broadcast seed value from root to ensure equal pseudorandom numbers.
    ValueType seed[1] = {static_cast<ValueType>(time(NULL))};
    comm->bcast( seed, 1, 0 );
    srand(seed[0]);

    for (i = 0; i < n; i++) {
        Point<ValueType> pos = Point<ValueType>({ max[0]*(ValueType(rand()) / RAND_MAX), max[1]*(ValueType(rand()) / RAND_MAX) });
        positions[i] = pos;
        content[i] = i;
        quad.addContent(i, pos);
    }

    // 2D points
    quad.addContent(i++, Point<ValueType>({818, 170 }) );
    quad.addContent(i++, Point<ValueType>({985, 476 }) );
    quad.addContent(i++, Point<ValueType>({128, 174 }) );
    quad.addContent(i++, Point<ValueType>({771, 11 }) );
    quad.addContent(i++, Point<ValueType>({614, 458 }) );
    quad.addContent(i++, Point<ValueType>({10, 91 }) );
    quad.addContent(i++, Point<ValueType>({740, 930 }) );
    quad.addContent(i++, Point<ValueType>({749, 945 }) );
    quad.addContent(i++, Point<ValueType>({249, 945 }) );
    quad.addContent(i++, Point<ValueType>({430, 845 }) );
    quad.addContent(i++, Point<ValueType>({430, 825 }) );

    PRINT("Num of leaves= N = "<< quad.countLeaves() );
    index N= quad.countLeaves();
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));
    // index the tree
    index treeSize = quad.indexSubtree(0);

    // A set for every node in the tree, graphNgbrsCells[i] contains shared_ptrs to every neighbour
    // of -i- in the output graph, not the quad tree.
    std::vector< std::set<std::shared_ptr<const SpatialCell>>> graphNgbrsCells( treeSize );
    int dimension = 2;
    std::vector<std::vector<ValueType>> coords( dimension );

    scai::lama::CSRSparseMatrix<ValueType> graph= quad.getTreeAsGraph<IndexType, ValueType>(graphNgbrsCells, coords);

    // checkSymmetry is really expensive for big graphs, use only for small instances
    if(N<3000) {
        graph.checkSymmetry();
    }
    graph.isConsistent();

    ASSERT_EQ(coords[0].size(), N);

    ASSERT_EQ( graph.getNumRows(), N);
    ASSERT_EQ( graph.getNumColumns(), N);
    {
        const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
        const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
        const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

        // 20 is too large upper bound. Should be around 24 for 3D and 8 (or 10) for 2D
        //TODO: maybe 30 is not so large... find another way to do it or skip it entirely
        IndexType upBound= 20;
        std::vector<IndexType> degreeCount( upBound, 0 );
        for(IndexType i=0; i<N; i++) {
            IndexType nodeDegree = ia[i+1] -ia[i];
            if( nodeDegree > upBound) {
                throw std::logic_error( "Node with large degree, degree= "+  std::to_string(nodeDegree) + " > current upper bound= " + std::to_string(upBound) );
            }
            ++degreeCount[nodeDegree];
        }

        IndexType numEdges = 0;
        //IndexType maxDegree = 0; //not used
        //std::cout<< "\t Num of nodes"<< std::endl;
        for(int i=0; i<degreeCount.size(); i++) {
            if(  degreeCount[i] !=0 ) {
                //std::cout << "degree " << i << ":   "<< degreeCount[i]<< std::endl;
                numEdges += i*degreeCount[i];
                //maxDegree = i;
            }
        }
        EXPECT_EQ(numEdges, graph.getNumValues() );

        //ValueType averageDegree = ValueType( numEdges)/N;
        //PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() << ", average degree= "<< averageDegree << ", max degree= "<< maxDegree);
    }

    // communicate/distribute graph

    IndexType k = comm->getSize();

    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));

    graph.redistribute( dist, noDistPointer);

    //TODO: change coords data type to vector<ValueType> ? or the way we copy to a DenseVector
    std::vector<DenseVector<ValueType>> coordsDV(dimension);

    for(int d=0; d<dimension; d++) {
        coordsDV[d].allocate(coords[d].size() );
        for(IndexType j=0; j<coords[d].size(); j++) {
            coordsDV[d].setValue(j, coords[d][j]);
        }
        coordsDV[d].redistribute(dist);
    }

    EXPECT_EQ(coordsDV[0].getLocalValues().size(), graph.getLocalNumRows() );

    // write coords in files for visualization purposes
    std::string destPath = "./partResults/fromQuadTree/blocks_"+std::to_string(k)+"/";
    boost::filesystem::create_directories( destPath );

    const ValueType epsilon = 0.05;
    struct Settings settings;
    settings.numBlocks= k;
    settings.epsilon = epsilon;
    settings.dimensions = dimension;
    settings.useGeometricTieBreaking = 1;
	settings.initialPartition = ITI::Tool::geoSFC;
	settings.noRefinement = true;
	
	struct Metrics metrics(settings);
	
	std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(1, DenseVector<ValueType>(graph.getRowDistributionPtr(), 1));

    ValueType cut, maxCut= N;
    ValueType imbalance;

    IndexType np = 3;
    scai::dmemo::DistributionPtr bestDist = dist;
    scai::lama::DenseVector<IndexType> sfcPartition;

    for(int detail= 0; detail<np; detail++) {
        settings.pixeledSideLen= std::pow( 2, detail + np );
        sfcPartition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( coordsDV, nodeWeights, settings, metrics );
        scai::dmemo::DistributionPtr newDist = sfcPartition.getDistributionPtr();
        sfcPartition.redistribute(newDist);
        graph.redistribute(newDist, noDist);
        cut = GraphUtils<IndexType, ValueType>::computeCut(graph, sfcPartition, true);
        if (cut<maxCut) {
            maxCut = cut;
            bestDist = sfcPartition.getDistributionPtr();
        }
    }

    graph.redistribute(bestDist, noDist);

    for(int d=0; d<dimension; d++) {
        coordsDV[d].redistribute(bestDist);
    }

    if(dimension==2) {
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordsDV, dimension, destPath+"pixel");
    }

    //redistribute
    graph.redistribute( dist, noDistPointer);
    for(int d=0; d<dimension; d++) {
        coordsDV[d].redistribute(dist);
    }

    scai::lama::DenseVector<IndexType> hilbertPartition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( coordsDV, nodeWeights, settings, metrics );
    scai::dmemo::DistributionPtr newDist = hilbertPartition.getDistributionPtr();
    graph.redistribute(newDist, noDist);
    hilbertPartition.redistribute(newDist);

    ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordsDV, dimension, destPath+"hilbert");

    cut = ITI::GraphUtils<IndexType, ValueType>::computeCut(graph, hilbertPartition, true);
    imbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance(hilbertPartition, k);

    if( imbalance>epsilon ) {
        PRINT0("WARNING, imbalance: "<< imbalance <<" more than epislon: "<< epsilon);
    }
    if (comm->getRank() == 0) {
        std::cout << "Commit " << version << ": Partitioned graph with " << N << " nodes into " << k << " blocks with a total cut of " << cut << std::endl;
    }

}


//

TEST_F(QuadTreeTest, DISABLED_testCartesianEuclidQuery) {
    count n = 10000;

    assert(n > 0);

    std::vector<Point<ValueType> > positions(n);
    std::vector<index> content(n);

    QuadTreeCartesianEuclid quad({0,0}, {1,1}, true);
    for (index i = 0; i < n; i++) {
        Point<ValueType> pos = Point<ValueType>({ValueType(rand()) / RAND_MAX, ValueType(rand()) / RAND_MAX});
        positions[i] = pos;
        content[i] = i;
        quad.addContent(i, pos);
    }



    EXPECT_EQ(n, quad.size());
    quad.recount();
    EXPECT_EQ(n, quad.size());

    quad.trim();

    for (index i = 0; i < 200; i++) {
        index query = (ValueType(rand()) / RAND_MAX)*(n);
        ValueType acc = ValueType(rand()) / RAND_MAX ;
        auto edgeProb = [acc](ValueType distance) -> ValueType {return acc;};
        std::vector<index> near;
        quad.getElementsProbabilistically(positions[query], edgeProb, near);
        EXPECT_NEAR(near.size(), acc*n, std::max(ValueType(acc*n*0.5), ValueType(10.0)));
    }

    for (index i = 0; i < 200; i++) {
        index query = (ValueType(rand()) / RAND_MAX)*(n);
        ValueType threshold = ValueType(rand()) / RAND_MAX;
        auto edgeProb = [threshold](ValueType distance) -> ValueType {return distance <= threshold ? 1 : 0;};
        std::vector<index> near;
        quad.getElementsProbabilistically(positions[query], edgeProb, near);
        std::vector<index> circleDenizens;
        quad.getElementsInEuclideanCircle(positions[query], threshold, circleDenizens);
        EXPECT_EQ(near.size(), circleDenizens.size());
    }

    //TODO: some test about appropriate subtrees and leaves

    auto edgeProb = [](ValueType distance) -> ValueType {return 1;};
    std::vector<index> near;
    quad.getElementsProbabilistically(positions[0], edgeProb, near);
    EXPECT_EQ(n, near.size());

    auto edgeProb2 = [](ValueType distance) -> ValueType {return 0;};
    near.clear();
    quad.getElementsProbabilistically(positions[0], edgeProb2, near);
    EXPECT_EQ(0, near.size());
}




TEST_F(QuadTreeTest, DISABLED_testPolarEuclidQuery) {
    /**
     * setup of data structures and constants
     */
    ValueType maxR = 2;
    count n = 10000;
    std::vector<ValueType> angles(n);
    std::vector<ValueType> radii(n);
    std::vector<index> content(n);

    ValueType minPhi = 0;
    ValueType maxPhi = 2*M_PI;
    ValueType minR = 0;

    /**
     * get random number generators
     */

    std::uniform_real_distribution<ValueType> phidist{minPhi, maxPhi};
    std::uniform_real_distribution<ValueType> rdist{minR, maxR};

    /**
     * fill vectors
     */
    for (index i = 0; i < n; i++) {
        angles[i] = (ValueType(rand()) / RAND_MAX)*(2*M_PI);
        radii[i] = (ValueType(rand()) / RAND_MAX)*(maxR-minR)+minR;
        content[i] = i;
    }

    const bool splitTheoretical = true;
    QuadTreePolarEuclid tree(angles, radii, content, splitTheoretical);
    EXPECT_EQ(n, tree.size());

    tree.trim();

    for (index i = 0; i < 200; i++) {
        index query = (ValueType(rand()) / RAND_MAX)*(n);
        ValueType acc = ValueType(rand()) / RAND_MAX ;
        auto edgeProb = [acc](ValueType distance) -> ValueType {return acc;};
        std::vector<index> near;
        tree.getElementsProbabilistically({angles[query], radii[query]}, edgeProb, near);
        EXPECT_NEAR(near.size(), acc*n, std::max(ValueType(acc*n*0.5), ValueType(10.0)));
    }

    //TODO: some test about appropriate subtrees and leaves

    auto edgeProb = [](ValueType distance) -> ValueType {return 1;};
    std::vector<index> near;
    tree.getElementsProbabilistically({angles[0], radii[0]}, edgeProb, near);
    EXPECT_EQ(n, near.size());

    auto edgeProb2 = [](ValueType distance) -> ValueType {return 0;};
    near.clear();
    tree.getElementsProbabilistically({angles[0], radii[0]}, edgeProb2, near);
    EXPECT_EQ(0, near.size());
}

TEST_F(QuadTreeTest, testQuadTreePolarEuclidInsertion) {
    /**
     * setup of data structures and constants
     */
    ValueType maxR = 2;
    count n = 1000;
    std::vector<ValueType> angles(n);
    std::vector<ValueType> radii(n);
    std::vector<index> content(n);

    ValueType minPhi = 0;
    ValueType maxPhi = 2*M_PI;
    ValueType minR = 0;

    /**
     * get random number generators
     */

    std::uniform_real_distribution<ValueType> phidist{minPhi, maxPhi};
    std::uniform_real_distribution<ValueType> rdist{minR, maxR};

    /**
     * fill vectors
     */
    for (index i = 0; i < n; i++) {
        angles[i] = (ValueType(rand()) / RAND_MAX)*(2*M_PI);
        radii[i] = (ValueType(rand()) / RAND_MAX)*(maxR-minR)+minR;
        content[i] = i;
    }

    QuadTreePolarEuclid tree(angles, radii, content);
    EXPECT_EQ(n, tree.size());

    /**
     * elements are returned
     */
    std::vector<index> returned = tree.getElements();
    EXPECT_EQ(n, returned.size());
    sort(returned.begin(), returned.end());
    for (index i = 0; i < returned.size(); i++) {
        EXPECT_EQ(i, returned[i]);
    }
}

TEST_F(QuadTreeTest, testQuadNodePolarEuclidDistanceBounds) {
    Point<ValueType> query = {3.81656, 1.18321};
    Point<ValueType> lowerLeft = {1.5708, 0};
    Point<ValueType> upperRight = {2.35619, 0.706942};
    Point<ValueType> interior = {2.35602,0.129449};
    Point<ValueType> projected = {2.35619,0.129449};

    QuadNodePolarEuclid testNode(lowerLeft, upperRight);
    ASSERT_TRUE(testNode.responsible(interior));
    EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(query[0], query[1], interior[0], interior[1]));

    EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(1.5708, 0, 3.81656, 1.18321));
    EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(2.35619, 0.706942, 3.81656, 1.18321));
    EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(1.5708, 0.706942, 3.81656, 1.18321));
    EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(1.5708, 0.706942, 3.81656, 1.18321));
}


TEST_F(QuadTreeTest, testQuadNodeCartesianDistances) {
    Point<ValueType> lower({0.24997519780061023, 0.7499644402803205});
    Point<ValueType> upper({0.49995039560122045, 0.99995258704042733});

    ASSERT_LE(lower[0], upper[0]);
    ASSERT_LE(lower[1], upper[1]);
    ASSERT_EQ(2, lower.getDimensions());
    ASSERT_EQ(2, upper.getDimensions());

    Point<ValueType> query({0.81847946542324035, 0.91885035291473593});

    QuadNodeCartesianEuclid node(lower, upper, 1000);
    //count steps = 100;
    Point<ValueType> posAtMin = lower;
    ValueType minDistance = posAtMin.distance(query);

    Point<ValueType> p(0.49969783875749996, 0.87199796797360407);

    EXPECT_TRUE(node.responsible(p));
    ValueType distanceQueryToCell = node.distances(query).first;
    ValueType distanceQueryToPoint = query.distance(p);

    EXPECT_LE(distanceQueryToCell, distanceQueryToPoint);
    EXPECT_LE(distanceQueryToCell, minDistance);
}

TEST_F(QuadTreeTest, DISABLED_benchCartesianQuadProbabilisticQueryUniform) {
    const index maxDim = 10;
    const count n = 50000;
    std::vector<Point<ValueType> > points;
    auto edgeProb = [n](ValueType distance) -> ValueType {return std::min<ValueType>(1, (1/(distance*n)));};

    for (index dim = 1; dim < maxDim; dim++) {
        std::vector<ValueType> minCoords(dim, 0);
        std::vector<ValueType> maxCoords(dim, 1);

        std::vector<Point<ValueType> > coordVector;
        QuadTreeCartesianEuclid quad(minCoords, maxCoords);
        for (index i = 0; i < n; i++) {
            std::vector<ValueType> coords(dim);
            for (index j = 0; j < dim; j++) {
                coords[j] = ValueType(rand()) / RAND_MAX;
            }
            quad.addContent(i, coords);
            coordVector.push_back(coords);
        }

        count numResults = 0;

        for (index i = 0; i < n; i++) {
            std::vector<index> result;
            quad.getElementsProbabilistically(coordVector[i], edgeProb, result);
            numResults += result.size();
        }
    }
}

TEST_F(QuadTreeTest, DISABLED_benchCartesianKDProbabilisticQueryUniform) {
    const index maxDim = 10;
    const count n = 50000;
    std::vector<Point<ValueType> > points;
    auto edgeProb = [n](ValueType distance) -> ValueType {return std::min<ValueType>(1, (1/(distance*n)));};

    for (index dim = 1; dim < maxDim; dim++) {
        std::vector<ValueType> minCoords(dim, 0);
        std::vector<ValueType> maxCoords(dim, 1);

        std::vector<Point<ValueType> > coordVector;
        KDTreeEuclidean<true> tree(minCoords, maxCoords);
        for (index i = 0; i < n; i++) {
            std::vector<ValueType> coords(dim);
            for (index j = 0; j < dim; j++) {
                coords[j] = ValueType(rand()) / RAND_MAX;
            }
            tree.addContent(i, coords);
            coordVector.push_back(coords);
        }

        count numResults = 0;

        for (index i = 0; i < n; i++) {
            std::vector<index> result;
            tree.getElementsProbabilistically(coordVector[i], edgeProb, result);
            numResults += result.size();
        }
    }
}

TEST_F(QuadTreeTest, DISABLED_benchPolarQuadProbabilisticQueryUniform) {
    const count n = 50000;
    std::vector<Point<ValueType> > points;
    auto edgeProb = [n](ValueType distance) -> ValueType {return std::min<ValueType>(1, (1/(distance*n)));};

    std::vector<ValueType> minCoords(2, 0);
    std::vector<ValueType> maxCoords(2);
    maxCoords[0] = 2*M_PI;
    maxCoords[1] = 1;

    std::vector<Point<ValueType> > coordVector;
    QuadTreePolarEuclid quad(minCoords, maxCoords);
    for (index i = 0; i < n; i++) {
        Point<ValueType> coords = {(ValueType(rand()) / RAND_MAX)*2*M_PI, ValueType(rand()) / RAND_MAX};
        quad.addContent(i, coords);
        coordVector.push_back(coords);
    }

    count numResults = 0;

    for (index i = 0; i < n; i++) {
        std::vector<index> result;
        quad.getElementsProbabilistically(coordVector[i], edgeProb, result);
        numResults += result.size();
    }
}

TEST_F(QuadTreeTest, DISABLED_benchPolarKDProbabilisticQueryUniform) {
    const count n = 50000;
    std::vector<Point<ValueType> > points;
    auto edgeProb = [n](ValueType distance) -> ValueType {return std::min<ValueType>(1, (1/(distance*n)));};

    std::vector<ValueType> minCoords({0,0});
    std::vector<ValueType> maxCoords({2*M_PI, 1});

    std::vector<Point<ValueType> > coordVector;
    KDTreeEuclidean<false> tree(minCoords, maxCoords);
    for (index i = 0; i < n; i++) {
        std::vector<ValueType> coords = {(ValueType(rand()) / RAND_MAX)*2*M_PI, ValueType(rand()) / RAND_MAX};
        tree.addContent(i, coords);
        coordVector.push_back(coords);
    }

    count numResults = 0;

    for (index i = 0; i < n; i++) {
        std::vector<index> result;
        tree.getElementsProbabilistically(coordVector[i], edgeProb, result);
        numResults += result.size();
    }
}



} /* namespace ITI */
