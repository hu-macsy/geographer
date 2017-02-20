/*
 * QuadTreeTest.cpp
 *
 *  Created on: 28.05.2014
 *      Author: Moritz v. Looz (moritz.looz-corswarem@kit.edu)
 */

#include <stack>
#include <cmath>
#include <algorithm>

#include <scai/lama/matrix/all.hpp>

#include "QuadTreeTest.h"
//#include "../../MeshIO.h"
#include "../../ParcoRepart.h"

#include "../QuadTreeCartesianEuclid.h"
#include "../QuadTreePolarEuclid.h"
#include "../KDTreeEuclidean.h"

namespace ITI {

typedef double ValueType;
typedef int IndexType;
  

TEST_F(QuadTreeTest, testGetGraphFromForest){
    
    // every forest[i] is a pointer to the root of a tree
    std::vector<std::shared_ptr<SpatialCell>> forest;
    
    IndexType n= 10;
    vector<Point<double> > positions(n);
    vector<index> content(n);

    Point<double> min(0.0, 0.0);
    Point<double> max(1.0, 1.0);
    index capacity = 1;
    
    QuadTreeCartesianEuclid quad(min, max, true, capacity);
    index i=0;
    srand(time(NULL));
    
    for (i = 0; i < n; i++) {
        Point<double> pos = Point<double>({double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX});
        positions[i] = pos;
        content[i] = i;
        quad.addContent(i, pos);
    }
	
    forest.push_back(quad.getRoot());
    // make more trees and pass them to the forest
    // CARE though, indexing should be one for all trees, maybe a forestIndex() routine or just a for 
    IndexType globIndexing=0;
    for(IndexType i=0; i<forest.size(); i++){
        // no tested, should index all trees properly
        globIndexing = forest[i]->indexSubtree( globIndexing );
    }
    
    
    IndexType forestSize = forest.size();
    
    // graphNgbrsPtrs[i]= a set with pointers to the neighbours of -i- in the CSR matrix/graph
    std::vector< std::set<std::shared_ptr<SpatialCell>>> graphNgbrsPtrs(forestSize);
    
    scai::lama::CSRSparseMatrix<ValueType> graph= SpatialTree::getGraphFromForest<IndexType, ValueType>( graphNgbrsPtrs, forest);
}



TEST_F(QuadTreeTest, testGetGraphMatrixFromTree_3D) {
	count n = 2500;

	vector<Point<double> > positions(n);
	vector<index> content(n);

        Point<double> min(0.0, 0.0, 0.0);
        Point<double> max(1.0, 1.0, 1.0);
        index capacity = 1;
        
	QuadTreeCartesianEuclid quad(min, max, true, capacity);
        index i=0;
        srand(time(NULL));
    
        for (i = 0; i < n; i++) {
		Point<double> pos = Point<double>({double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX});
		positions[i] = pos;
		content[i] = i;
		quad.addContent(i, pos);
	}
	
	PRINT("Num of leaves= N = "<< quad.countLeaves() );
	index N= quad.countLeaves();
        
        // index the tree
        index treeSize = quad.indexSubtree(0);
        
        // A set for every node in the tree, graphNgbrsCells[i] contains shared_ptrs to every neighbour
        // of -i- in the output graph, not the quad tree.
        std::vector< std::set<std::shared_ptr<SpatialCell>>> graphNgbrsCells( treeSize );
        
	scai::lama::CSRSparseMatrix<double> graph= quad.getTreeAsGraph<int, double>( graphNgbrsCells );

        // checkSymmetry is really expensive for big graphs, used only for small instances
	// graph.checkSymmetry();
	graph.isConsistent();
        //EXPECT_EQ( graph.getNumRows(), graph.getNumColumns() );
	EXPECT_EQ( graph.getNumRows(), N);
	EXPECT_EQ( graph.getNumColumns(), N);

	const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
        
        // 20 is too large upper bound. Should be around 24 for 3D and 8 (or 10) for 2D
        //TODO: maybe 30 is not so large... find another way to do it or skip it entirely
        IndexType upBound= 50;
        std::vector<IndexType> degreeCount( upBound, 0 );
        for(IndexType i=0; i<N; i++){
            IndexType nodeDegree = ia[i+1] -ia[i];
            if( nodeDegree > upBound){
                throw std::logic_error( "Node with large degree, degree= "+  std::to_string(nodeDegree) + " > current upper bound= " + std::to_string(upBound) );
            }
            ++degreeCount[nodeDegree];
        }
        
        IndexType numEdges = 0;
        IndexType maxDegree = 0;
        std::cout<< "\t Num of nodes"<< std::endl;
        for(int i=0; i<degreeCount.size(); i++){
            if(  degreeCount[i] !=0 ){
                std::cout << "degree " << i << ":   "<< degreeCount[i]<< std::endl;
                numEdges += i*degreeCount[i];
                maxDegree = i;
            }
        }
        EXPECT_EQ(numEdges, graph.getNumValues() );
        
        ValueType averageDegree = ValueType( numEdges)/N;
        
        PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() << ", average degree= "<< averageDegree << ", max degree= "<< maxDegree);  
        
}


TEST_F(QuadTreeTest, testGetGraphMatrixFromTree_Distributed_3D) {

        count n = 2500;

	vector<Point<double> > positions(n);
	vector<index> content(n);

        Point<double> min(0.0, 0.0, 0.0);
        Point<double> max(1.0, 1.0, 1.0);
        index capacity = 1;
        
	QuadTreeCartesianEuclid quad(min, max, true, capacity);
        index i=0;
        srand(time(NULL));
    
        for (i = 0; i < n; i++) {
		Point<double> pos = Point<double>({double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX});
		positions[i] = pos;
		content[i] = i;
		quad.addContent(i, pos);
	}
	
	PRINT("Num of leaves= N = "<< quad.countLeaves() );
	index N= quad.countLeaves();
        // index the tree
        index treeSize = quad.indexSubtree(0);
        
        // A set for every node in the tree, graphNgbrsCells[i] contains shared_ptrs to every neighbour
        // of -i- in the output graph, not the quad tree.
        std::vector< std::set<std::shared_ptr<SpatialCell>>> graphNgbrsCells( treeSize );
        
	scai::lama::CSRSparseMatrix<double> graph= quad.getTreeAsGraph<int, double>(graphNgbrsCells);
        
        // checkSymmetry is really expensive for big graphs, used only for small instances
	// graph.checkSymmetry();
	graph.isConsistent();

	EXPECT_EQ( graph.getNumRows(), N);
	EXPECT_EQ( graph.getNumColumns(), N);

	const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
        
        // 20 is too large upper bound. Should be around 24 for 3D and 8 (or 10) for 2D
        //TODO: maybe 30 is not so large... find another way to do it or skip it entirely
        IndexType upBound= 50;
        std::vector<IndexType> degreeCount( upBound, 0 );
        for(IndexType i=0; i<N; i++){
            IndexType nodeDegree = ia[i+1] -ia[i];
            if( nodeDegree > upBound){
                throw std::logic_error( "Node with large degree, degree= "+  std::to_string(nodeDegree) + " > current upper bound= " + std::to_string(upBound) );
            }
            ++degreeCount[nodeDegree];
        }
        
        IndexType numEdges = 0;
        IndexType maxDegree = 0;
        std::cout<< "\t Num of nodes"<< std::endl;
        for(int i=0; i<degreeCount.size(); i++){
            if(  degreeCount[i] !=0 ){
                std::cout << "degree " << i << ":   "<< degreeCount[i]<< std::endl;
                numEdges += i*degreeCount[i];
                maxDegree = i;
            }
        }
        EXPECT_EQ(numEdges, graph.getNumValues() );
        
        ValueType averageDegree = ValueType( numEdges)/N;
        
        PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() << ", average degree= "<< averageDegree << ", max degree= "<< maxDegree);  
    
        // communicate distribute graph
    
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

        IndexType k = comm->getSize();

        scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));
        
        graph.redistribute( dist, noDistPointer);
        
        /*
        //TODO: coordinates vector is missing
        scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coordinates, Settings);
        
        ParcoRepart<IndexType, ValueType> repart;
        EXPECT_LE(repart.computeImbalance(partition, k), epsilon);

        const ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(a, partition, true);

        if (comm->getRank() == 0) {
            std::cout << "Commit " << version << ": Partitioned graph with " << n << " nodes into " << k << " blocks with a total cut of " << cut << std::endl;
        }
        */
}


TEST_F(QuadTreeTest, testGetGraphMatrixFromTree_2D) {
    
    index n=8;
    vector<Point<double> > positions(n);
    vector<index> content(n);
    
    Point<double> min(0.0, 0.0);
    Point<double> max(2.0, 2.0);
    index capacity = 1;
        
    // the quadtree 
    QuadTreeCartesianEuclid quad(min, max, true, capacity);
    
    index i=0;
    // 2D points
    quad.addContent(i++, Point<double>({0.2, 0.2}) );
    quad.addContent(i++, Point<double>({0.8, 0.7}) );
    quad.addContent(i++, Point<double>({1.4, 0.7}) );
    quad.addContent(i++, Point<double>({1.8, 0.3}) );
    
    quad.addContent(i++, Point<double>({0.2, 0.8}) );
    quad.addContent(i++, Point<double>({0.2, 0.6}) );
    
    quad.addContent(i++, Point<double>({0.7, 1.1}) );
    quad.addContent(i++, Point<double>({0.2, 1.6}) );
    
    PRINT("Num of leaves= N = "<< quad.countLeaves() );
    index N= quad.countLeaves();
    // index the tree
    index treeSize = quad.indexSubtree(0);
        
    // A set for every node in the tree, graphNgbrsCells[i] contains shared_ptrs to every neighbour
    // of -i- in the output graph, not the quad tree.
    std::vector< std::set<std::shared_ptr<SpatialCell>>> graphNgbrsCells( treeSize );
        
    scai::lama::CSRSparseMatrix<double> graph= quad.getTreeAsGraph<int, double>(graphNgbrsCells);
    
    // checkSymmetry is really expensive for big graphs, used only for small instances
    graph.checkSymmetry();
    graph.isConsistent();
    
    EXPECT_EQ( graph.getNumRows(), N);
    EXPECT_EQ( graph.getNumColumns(), N);
    
    const scai::lama::CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    
    // 20 is too large upper bound. Should be around 24 for 3D and 8 (or 10) for 2D
    //TODO: maybe 20 is not so large... find another way to do it or skip it entirely
    IndexType upBound= 20;
    std::vector<IndexType> degreeCount( upBound, 0 );
    for(IndexType i=0; i<N; i++){
        IndexType nodeDegree = ia[i+1] -ia[i];
        if( nodeDegree > upBound){
            throw std::logic_error( "Node with large degree, degree= "+  std::to_string(nodeDegree) + " > current upper bound= " + std::to_string(upBound) );
        }
        ++degreeCount[nodeDegree];
    }
    
    IndexType numEdges = 0;
    IndexType maxDegree = 0;
        std::cout<< "\t Num of nodes"<< std::endl;    
    for(int i=0; i<degreeCount.size(); i++){
        if(  degreeCount[i] !=0 ){
            std::cout << "degree " << i << ":   "<< degreeCount[i]<< std::endl;
            numEdges += i*degreeCount[i];
            maxDegree = i;
        }
    }
    EXPECT_EQ(numEdges, graph.getNumValues() );
    
    ValueType averageDegree = ValueType( numEdges)/N;
        
    PRINT("num edges= "<< graph.getNumValues() << " , num nodes= " << graph.getNumRows() << ", average degree= "<< averageDegree << ", max degree= "<< maxDegree);   
        
	
    
}
    
TEST_F(QuadTreeTest, testCartesianEuclidQuery) {
	count n = 10000;

	assert(n > 0);

	std::vector<Point<double> > positions(n);
	std::vector<index> content(n);

	QuadTreeCartesianEuclid quad({0,0}, {1,1}, true);
	for (index i = 0; i < n; i++) {
		Point<double> pos = Point<double>({double(rand()) / RAND_MAX, double(rand()) / RAND_MAX});
		positions[i] = pos;
		content[i] = i;
		quad.addContent(i, pos);
	}



	EXPECT_EQ(n, quad.size());
	quad.recount();
	EXPECT_EQ(n, quad.size());

	quad.trim();

	for (index i = 0; i < 200; i++) {
		index query = (double(rand()) / RAND_MAX)*(n);
		double acc = double(rand()) / RAND_MAX ;
		auto edgeProb = [acc](double distance) -> double {return acc;};
		std::vector<index> near;
		quad.getElementsProbabilistically(positions[query], edgeProb, near);
		EXPECT_NEAR(near.size(), acc*n, std::max(acc*n*0.25, 10.0));
	}

	for (index i = 0; i < 200; i++) {
		index query = (double(rand()) / RAND_MAX)*(n);
		double threshold = double(rand()) / RAND_MAX;
		auto edgeProb = [threshold](double distance) -> double {return distance <= threshold ? 1 : 0;};
		std::vector<index> near;
		quad.getElementsProbabilistically(positions[query], edgeProb, near);
		std::vector<index> circleDenizens;
		quad.getElementsInEuclideanCircle(positions[query], threshold, circleDenizens);
		EXPECT_EQ(near.size(), circleDenizens.size());
	}

	//TODO: some test about appropriate subtrees and leaves

	auto edgeProb = [](double distance) -> double {return 1;};
	std::vector<index> near;
	quad.getElementsProbabilistically(positions[0], edgeProb, near);
	EXPECT_EQ(n, near.size());

	auto edgeProb2 = [](double distance) -> double {return 0;};
	near.clear();
	quad.getElementsProbabilistically(positions[0], edgeProb2, near);
	EXPECT_EQ(0, near.size());
}


//

TEST_F(QuadTreeTest, testPolarEuclidQuery) {
	/**
	 * setup of data structures and constants
	 */
	double maxR = 2;
	count n = 10000;
	std::vector<double> angles(n);
	std::vector<double> radii(n);
	std::vector<index> content(n);

	double minPhi = 0;
	double maxPhi = 2*M_PI;
	double minR = 0;

	/**
	 * get random number generators
	 */

	std::uniform_real_distribution<double> phidist{minPhi, maxPhi};
	std::uniform_real_distribution<double> rdist{minR, maxR};

	/**
	 * fill vectors
	 */
	for (index i = 0; i < n; i++) {
		angles[i] = (double(rand()) / RAND_MAX)*(2*M_PI);
		radii[i] = (double(rand()) / RAND_MAX)*(maxR-minR)+minR;
		content[i] = i;
	}

	const bool splitTheoretical = true;
	QuadTreePolarEuclid tree(angles, radii, content, splitTheoretical);
	EXPECT_EQ(n, tree.size());

	tree.trim();

	for (index i = 0; i < 200; i++) {
		index query = (double(rand()) / RAND_MAX)*(n);
		double acc = double(rand()) / RAND_MAX ;
		auto edgeProb = [acc](double distance) -> double {return acc;};
		std::vector<index> near;
		tree.getElementsProbabilistically({angles[query], radii[query]}, edgeProb, near);
		EXPECT_NEAR(near.size(), acc*n, std::max(acc*n*0.25, 10.0));
	}

	//TODO: some test about appropriate subtrees and leaves

	auto edgeProb = [](double distance) -> double {return 1;};
	std::vector<index> near;
	tree.getElementsProbabilistically({angles[0], radii[0]}, edgeProb, near);
	EXPECT_EQ(n, near.size());

	auto edgeProb2 = [](double distance) -> double {return 0;};
	near.clear();
	tree.getElementsProbabilistically({angles[0], radii[0]}, edgeProb2, near);
	EXPECT_EQ(0, near.size());
}

TEST_F(QuadTreeTest, testQuadTreePolarEuclidInsertion) {
	/**
	 * setup of data structures and constants
	 */
	double maxR = 2;
	count n = 1000;
	std::vector<double> angles(n);
	std::vector<double> radii(n);
	std::vector<index> content(n);

	double minPhi = 0;
	double maxPhi = 2*M_PI;
	double minR = 0;

	/**
	 * get random number generators
	 */

	std::uniform_real_distribution<double> phidist{minPhi, maxPhi};
	std::uniform_real_distribution<double> rdist{minR, maxR};

	/**
	 * fill vectors
	 */
	for (index i = 0; i < n; i++) {
		angles[i] = (double(rand()) / RAND_MAX)*(2*M_PI);
		radii[i] = (double(rand()) / RAND_MAX)*(maxR-minR)+minR;
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
	Point<double> query = {3.81656, 1.18321};
	Point<double> lowerLeft = {1.5708, 0};
	Point<double> upperRight = {2.35619, 0.706942};
	Point<double> interior = {2.35602,0.129449};
	Point<double> projected = {2.35619,0.129449};

	QuadNodePolarEuclid testNode(lowerLeft, upperRight);
	ASSERT_TRUE(testNode.responsible(interior));
	EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(query[0], query[1], interior[0], interior[1]));

	EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(1.5708, 0, 3.81656, 1.18321));
	EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(2.35619, 0.706942, 3.81656, 1.18321));
	EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(1.5708, 0.706942, 3.81656, 1.18321));
	EXPECT_LE(testNode.distances(query).first, testNode.euclidDistancePolar(1.5708, 0.706942, 3.81656, 1.18321));
}


TEST_F(QuadTreeTest, testQuadNodeCartesianDistances) {
	Point<double> lower({0.24997519780061023, 0.7499644402803205});
	Point<double> upper({0.49995039560122045, 0.99995258704042733});

	ASSERT_LE(lower[0], upper[0]);
	ASSERT_LE(lower[1], upper[1]);
	ASSERT_EQ(2, lower.getDimensions());
	ASSERT_EQ(2, upper.getDimensions());

	Point<double> query({0.81847946542324035, 0.91885035291473593});

	QuadNodeCartesianEuclid node(lower, upper, 1000);
	//count steps = 100;
	Point<double> posAtMin = lower;
	double minDistance = posAtMin.distance(query);

	Point<double> p(0.49969783875749996, 0.87199796797360407);

	EXPECT_TRUE(node.responsible(p));
	double distanceQueryToCell = node.distances(query).first;
	double distanceQueryToPoint = query.distance(p);

	EXPECT_LE(distanceQueryToCell, distanceQueryToPoint);
	EXPECT_LE(distanceQueryToCell, minDistance);
}

TEST_F(QuadTreeTest, DISABLED_benchCartesianQuadProbabilisticQueryUniform) {
	const index maxDim = 10;
	const count n = 50000;
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	for (index dim = 1; dim < maxDim; dim++) {
		std::vector<double> minCoords(dim, 0);
		std::vector<double> maxCoords(dim, 1);

		std::vector<Point<double> > coordVector;
		QuadTreeCartesianEuclid quad(minCoords, maxCoords);
		for (index i = 0; i < n; i++) {
			std::vector<double> coords(dim);
			for (index j = 0; j < dim; j++) {
				coords[j] = double(rand()) / RAND_MAX;
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
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	for (index dim = 1; dim < maxDim; dim++) {
		std::vector<double> minCoords(dim, 0);
		std::vector<double> maxCoords(dim, 1);

		std::vector<Point<double> > coordVector;
		KDTreeEuclidean<true> tree(minCoords, maxCoords);
		for (index i = 0; i < n; i++) {
			std::vector<double> coords(dim);
			for (index j = 0; j < dim; j++) {
				coords[j] = double(rand()) / RAND_MAX;
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
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	std::vector<double> minCoords(2, 0);
	std::vector<double> maxCoords(2);
	maxCoords[0] = 2*M_PI;
	maxCoords[1] = 1;

	std::vector<Point<double> > coordVector;
	QuadTreePolarEuclid quad(minCoords, maxCoords);
	for (index i = 0; i < n; i++) {
		Point<double> coords = {(double(rand()) / RAND_MAX)*2*M_PI, double(rand()) / RAND_MAX};
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
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	std::vector<double> minCoords({0,0});
	std::vector<double> maxCoords({2*M_PI, 1});

	std::vector<Point<double> > coordVector;
	KDTreeEuclidean<false> tree(minCoords, maxCoords);
	for (index i = 0; i < n; i++) {
		std::vector<double> coords = {(double(rand()) / RAND_MAX)*2*M_PI, double(rand()) / RAND_MAX};
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
