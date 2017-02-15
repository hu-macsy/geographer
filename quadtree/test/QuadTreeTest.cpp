/*
 * QuadTreeTest.cpp
 *
 *  Created on: 28.05.2014
 *      Author: Moritz v. Looz (moritz.looz-corswarem@kit.edu)
 */

#include <stack>
#include <cmath>
#include <algorithm>

#include "QuadTreeTest.h"

#include "../QuadTreeCartesianEuclid.h"
#include "../QuadTreePolarEuclid.h"
#include "../KDTreeEuclidean.h"

namespace ITI {

TEST_F(QuadTreeTest, testCartesianEuclidQuery) {
	count n = 10000;

	assert(n > 0);

	vector<Point<double> > positions(n);
	vector<index> content(n);

	QuadTreeCartesianEuclid<index> quad({0,0}, {1,1}, true);
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
		vector<index> near;
		quad.getElementsProbabilistically(positions[query], edgeProb, near);
		EXPECT_NEAR(near.size(), acc*n, std::max(acc*n*0.25, 10.0));
	}

	for (index i = 0; i < 200; i++) {
		index query = (double(rand()) / RAND_MAX)*(n);
		double threshold = double(rand()) / RAND_MAX;
		auto edgeProb = [threshold](double distance) -> double {return distance <= threshold ? 1 : 0;};
		vector<index> near;
		quad.getElementsProbabilistically(positions[query], edgeProb, near);
		vector<index> circleDenizens;
		quad.getElementsInEuclideanCircle(positions[query], threshold, circleDenizens);
		EXPECT_EQ(near.size(), circleDenizens.size());
	}

	//TODO: some test about appropriate subtrees and leaves

	auto edgeProb = [](double distance) -> double {return 1;};
	vector<index> near;
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
	vector<double> angles(n);
	vector<double> radii(n);
	vector<index> content(n);

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
	QuadTreePolarEuclid<index> tree(angles, radii, content, splitTheoretical);
	EXPECT_EQ(n, tree.size());

	tree.trim();

	for (index i = 0; i < 200; i++) {
		index query = (double(rand()) / RAND_MAX)*(n);
		double acc = double(rand()) / RAND_MAX ;
		auto edgeProb = [acc](double distance) -> double {return acc;};
		vector<index> near;
		tree.getElementsProbabilistically({angles[query], radii[query]}, edgeProb, near);
		EXPECT_NEAR(near.size(), acc*n, std::max(acc*n*0.25, 10.0));
	}

	//TODO: some test about appropriate subtrees and leaves

	auto edgeProb = [](double distance) -> double {return 1;};
	vector<index> near;
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
	vector<double> angles(n);
	vector<double> radii(n);
	vector<index> content(n);

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

	QuadTreePolarEuclid<index> tree(angles, radii, content);
	EXPECT_EQ(n, tree.size());

	/**
	 * elements are returned
	 */
	vector<index> returned = tree.getElements();
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

	QuadNodePolarEuclid<index> testNode(lowerLeft, upperRight);
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

	QuadNodeCartesianEuclid<index> node(lower, upper, 1000);
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

TEST_F(QuadTreeTest, benchCartesianQuadProbabilisticQueryUniform) {
	const index maxDim = 10;
	const count n = 50000;
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	for (index dim = 1; dim < maxDim; dim++) {
		std::vector<double> minCoords(dim, 0);
		std::vector<double> maxCoords(dim, 1);

		std::vector<Point<double> > coordVector;
		QuadTreeCartesianEuclid<index> quad(minCoords, maxCoords);
		for (index i = 0; i < n; i++) {
			vector<double> coords(dim);
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

TEST_F(QuadTreeTest, benchCartesianKDProbabilisticQueryUniform) {
	const index maxDim = 10;
	const count n = 50000;
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	for (index dim = 1; dim < maxDim; dim++) {
		std::vector<double> minCoords(dim, 0);
		std::vector<double> maxCoords(dim, 1);

		std::vector<Point<double> > coordVector;
		KDTreeEuclidean<index, true> tree(minCoords, maxCoords);
		for (index i = 0; i < n; i++) {
			vector<double> coords(dim);
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

TEST_F(QuadTreeTest, benchPolarQuadProbabilisticQueryUniform) {
	const count n = 50000;
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	std::vector<double> minCoords(2, 0);
	std::vector<double> maxCoords(2);
	maxCoords[0] = 2*M_PI;
	maxCoords[1] = 1;

	std::vector<Point<double> > coordVector;
	QuadTreePolarEuclid<index> quad(minCoords, maxCoords);
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

TEST_F(QuadTreeTest, benchPolarKDProbabilisticQueryUniform) {
	const count n = 50000;
	std::vector<Point<double> > points;
	auto edgeProb = [n](double distance) -> double {return std::min<double>(1, (1/(distance*n)));};

	std::vector<double> minCoords({0,0});
	std::vector<double> maxCoords({2*M_PI, 1});

	std::vector<Point<double> > coordVector;
	KDTreeEuclidean<index,false> tree(minCoords, maxCoords);
	for (index i = 0; i < n; i++) {
		vector<double> coords = {(double(rand()) / RAND_MAX)*2*M_PI, double(rand()) / RAND_MAX};
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
