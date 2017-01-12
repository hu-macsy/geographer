#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>

#include "MeshIO.h"
#include "ParcoRepart.h"
#include "gtest/gtest.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class ParcoRepartTest : public ::testing::Test {

};


TEST_F(ParcoRepartTest, testMinimumNeighborDistanceDistributed) {
  IndexType nroot = 7;
  IndexType n = nroot * nroot * nroot;
  IndexType dimensions = 3;
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

  scai::lama::CSRSparseMatrix<ValueType>a(dist, noDistPointer);
  std::vector<ValueType> maxCoord(dimensions, nroot);
  std::vector<IndexType> numPoints(dimensions, nroot);

  scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  scai::hmemo::ContextPtr contexPtr = scai::hmemo::Context::getHostPtr();
  
  std::vector<DenseVector<ValueType>> coordinates(dimensions);
  for(IndexType i=0; i<dimensions; i++){ 
	  coordinates[i].allocate(coordDist);
	  coordinates[i] = static_cast<ValueType>( 0 );
  }
  
  MeshIO<IndexType, ValueType>::createStructured3DMesh(a, coordinates, maxCoord, numPoints);
  a.redistribute(dist, noDistPointer);

//  for (IndexType i = 0; i < nroot; i++) {
//    for (IndexType j = 0; j < nroot; j++) {
//      //this is slightly wasteful, since it also iterates over indices of other processors
//      // no need to check if local, since setValue skips non-local coordinates. index must always use the global value
//        coordinates[0].setValue(i*nroot + j, i);
//        coordinates[1].setValue(i*nroot + j, j);
//    }
//  }
  
  const ValueType minDistance = ParcoRepart<IndexType, ValueType>::getMinimumNeighbourDistance(a, coordinates, dimensions);
  EXPECT_LE(minDistance, nroot*1.5);
  EXPECT_GE(minDistance, 1);
}


TEST_F(ParcoRepartTest, testPartitionBalanceLocal) {
  IndexType nroot = 8;
  IndexType n = nroot * nroot * nroot;
  IndexType k = 8;
  IndexType dimensions = 3;
  const ValueType epsilon = 0.05;

  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));
  
  scai::lama::CSRSparseMatrix<ValueType>a(noDistPointer, noDistPointer);
  std::vector<ValueType> maxCoord(dimensions, nroot);
  std::vector<IndexType> numPoints(dimensions, nroot);

  std::vector<DenseVector<ValueType>> coordinates(dimensions);
  for(IndexType i=0; i<dimensions; i++){
	  coordinates[i].allocate(noDistPointer);
	  coordinates[i] = static_cast<ValueType>( 0 );
  }
  
  MeshIO<IndexType, ValueType>::createStructured3DMesh(a, coordinates, maxCoord, numPoints);
  
  scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, k, epsilon);
  
  EXPECT_EQ(n, partition.size());
  EXPECT_EQ(0, partition.min().getValue<IndexType>());
  EXPECT_EQ(k-1, partition.max().getValue<IndexType>());
  EXPECT_TRUE(partition.getDistribution().isReplicated());//for now
  
  ParcoRepart<IndexType, ValueType> repart;
  EXPECT_LE(repart.computeImbalance(partition, k), epsilon);
}

TEST_F(ParcoRepartTest, testPartitionBalanceDistributed) {
  IndexType nroot = 8;
  IndexType n = nroot * nroot * nroot;
  IndexType dimensions = 3;
  
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

  IndexType k = comm->getSize();

  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));
  
  scai::lama::CSRSparseMatrix<ValueType>a(dist, noDistPointer);
  std::vector<ValueType> maxCoord(dimensions, nroot);
  std::vector<IndexType> numPoints(dimensions, nroot);

  scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

  std::vector<DenseVector<ValueType>> coordinates(dimensions);
  for(IndexType i=0; i<dimensions; i++){ 
	  coordinates[i].allocate(coordDist);
	  coordinates[i] = static_cast<ValueType>( 0 );
  }
  
  MeshIO<IndexType, ValueType>::createStructured3DMesh(a, coordinates, maxCoord, numPoints);
  a.redistribute(dist, noDistPointer);

  const ValueType epsilon = 0.05;

  scai::lama::DenseVector<IndexType> partition(n, k+1);
  partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, k, epsilon);
  
  EXPECT_GE(k-1, partition.getLocalValues().max() );
  EXPECT_EQ(n, partition.size());
  EXPECT_EQ(0, partition.min().getValue<ValueType>());
  EXPECT_EQ(k-1, partition.max().getValue<ValueType>());
  EXPECT_EQ(a.getRowDistribution(), partition.getDistribution());

  ParcoRepart<IndexType, ValueType> repart;
  EXPECT_LE(repart.computeImbalance(partition, k), epsilon);
}

TEST_F(ParcoRepartTest, testImbalance) {
  const IndexType n = 10000;
  const IndexType k = 10;

  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );

  //generate random partition
  scai::lama::DenseVector<IndexType> part(dist);
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = rand() % k;
    part.setValue(i, blockId);
  }

  //sanity check for partition generation
  ASSERT_GE(part.min().getValue<ValueType>(), 0);
  ASSERT_LE(part.max().getValue<ValueType>(), k-1);

  ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_GE(imbalance, 0);

  // test perfectly balanced partition
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = i % k;
    part.setValue(i, blockId);
  }
  imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_EQ(0, imbalance);

  //test maximally imbalanced partition
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = 0;
    part.setValue(i, blockId);
  }
  imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_EQ((n/std::ceil(n/k))-1, imbalance);
}

TEST_F(ParcoRepartTest, testCut) {
  const IndexType n = 1000;
  const IndexType k = 10;

  //define distributions
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

  //generate random complete matrix
  scai::lama::CSRSparseMatrix<ValueType>a(dist, noDistPointer);
  scai::lama::MatrixCreator::fillRandom(a, 1);

  //generate balanced distributed partition
  scai::lama::DenseVector<IndexType> part(dist);
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = i % k;
    part.setValue(i, blockId);
  }

  //cut should be 10*900 / 2
  const IndexType blockSize = n / k;
  const ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  EXPECT_EQ(k*blockSize*(n-blockSize) / 2, cut);

  //now convert distributed into replicated partition vector and compare again
  part.redistribute(noDistPointer);
  a.redistribute(noDistPointer, noDistPointer);
  const ValueType replicatedCut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  EXPECT_EQ(k*blockSize*(n-blockSize) / 2, replicatedCut);
}


TEST_F(ParcoRepartTest, testFiducciaMattheysesLocal) {
  const IndexType n = 1000;
  const IndexType k = 10;
  const ValueType epsilon = 0.05;
  const IndexType iterations = 1;

  //generate random matrix
  scai::lama::CSRSparseMatrix<ValueType>a(n,n);
  scai::lama::MatrixCreator::buildPoisson(a, 3, 19, 10,10,10);

  //we want a replicated matrix
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));
  a.redistribute(noDistPointer, noDistPointer);

  //generate random partition
  scai::lama::DenseVector<IndexType> part(n, 0);
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = rand() % k;
    part.setValue(i, blockId);
  }

  ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  for (IndexType i = 0; i < iterations; i++) {
    ValueType gain = ParcoRepart<IndexType, ValueType>::replicatedMultiWayFM(a, part, k, epsilon);

    //check correct gain calculation
    const ValueType newCut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
    EXPECT_EQ(cut - gain, newCut) << "Old cut " << cut << ", gain " << gain << " newCut " << newCut;
    EXPECT_LE(newCut, cut);
    cut = newCut;
  }
  
  //generate balanced partition
  for (IndexType i = 0; i < n; i++) {
    IndexType blockId = i % k;
    part.setValue(i, blockId);
  }

  //check correct cut with balanced partition
  cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  ValueType gain = ParcoRepart<IndexType, ValueType>::replicatedMultiWayFM(a, part, k, epsilon);
  const ValueType newCut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
  EXPECT_EQ(cut - gain, newCut);
  EXPECT_LE(newCut, cut);

  //check for balance
  ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
  EXPECT_LE(imbalance, epsilon);
}

TEST_F(ParcoRepartTest, testFiducciaMattheysesDistributed) {
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	const IndexType dimX = 10;
	const IndexType dimY = 10;
	const IndexType dimZ = 10;
	const IndexType n = dimX*dimY*dimZ;
	const IndexType k = comm->getSize() > 1 ? comm->getSize() : 10;
	const ValueType epsilon = 0.05;
	const IndexType iterations = 1;

	//generate random matrix
	scai::lama::CSRSparseMatrix<ValueType>a(n,n);
	scai::lama::MatrixCreator::buildPoisson(a, 3, 19, dimX,dimY,dimZ);

	scai::dmemo::DistributionPtr inputDist = a.getRowDistributionPtr();
	scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

	a.redistribute(inputDist, noDistPointer);//need replicated columns for FM;

	ASSERT_EQ(n, inputDist->getGlobalSize());

	//generate balanced partition
	scai::lama::DenseVector<IndexType> part(inputDist);

	for (IndexType i = 0; i < n; i++) {
		IndexType blockId = i % k;
		part.setValue(i, blockId);
	}

	ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
	for (IndexType i = 0; i < iterations; i++) {
		ValueType gain = ParcoRepart<IndexType, ValueType>::distributedFMStep(a, part, k, epsilon);

		//check correct gain calculation
		const ValueType newCut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
		EXPECT_EQ(cut - gain, newCut) << "Old cut " << cut << ", gain " << gain << " newCut " << newCut;
		EXPECT_LE(newCut, cut);
		cut = newCut;
	}

	//generate balanced partition
	for (IndexType i = 0; i < n; i++) {
		IndexType blockId = i % k;
		part.setValue(i, blockId);
	}

	//check correct cut with balanced partition
	cut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
	ValueType gain = ParcoRepart<IndexType, ValueType>::distributedFMStep(a, part, k, epsilon);
	const ValueType newCut = ParcoRepart<IndexType, ValueType>::computeCut(a, part, true);
	EXPECT_EQ(cut - gain, newCut);
	EXPECT_LE(newCut, cut);

	//check for balance
	ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(part, k);
	EXPECT_LE(imbalance, epsilon);
}

TEST_F(ParcoRepartTest, testCommunicationScheme) {
	/**
	 * Check for:
	 * 1. Basic Sanity: All ids are valid
	 * 2. Completeness: All PEs with a common edge communicate
	 * 3. Symmetry: In each round, partner[partner[i]] == i holds
	 * 4. Efficiency: Pairs don't communicate more than once
	 */

	const IndexType n = 1000;
	const IndexType p = 129;//purposefully not a power of two, to check what happens
	const IndexType k = p;

	//fill random matrix
	scai::lama::CSRSparseMatrix<ValueType>a(n,n);
	scai::lama::MatrixCreator::fillRandom(a, 0.0001);

	//generate random partition
	scai::lama::DenseVector<IndexType> part(n, 0);
	for (IndexType i = 0; i < n; i++) {
		IndexType blockId = rand() % k;
		part.setValue(i, blockId);
	}

	//create trivial mapping
	scai::lama::DenseVector<IndexType> mapping(k, 0);
	for (IndexType i = 0; i < k; i++) {
		mapping.setValue(i, i);
	}

	std::vector<DenseVector<IndexType>> scheme = ParcoRepart<IndexType, ValueType>::computeCommunicationPairings(a, part, mapping);

	//EXPECT_LE(scheme.size(), p);

	std::vector<std::vector<bool> > communicated(p);
	for (IndexType i = 0; i < p; i++) {
		communicated[i].resize(p, false);
	}

	for (IndexType round = 0; round < scheme.size(); round++) {
		EXPECT_EQ(scheme[round].size(), p);
		for (IndexType i = 0; i < p; i++) {
			//Scalar partner = scheme[round].getValue(i);
			const IndexType partner = scheme[round].getValue(i).getValue<IndexType>();

			//sanity
			EXPECT_GE(partner, 0);
			EXPECT_LT(partner, p);

			if (partner != i) {
				//symmetry
				Scalar partnerOfPartner = scheme[round].getValue(partner);
				EXPECT_EQ(i, partnerOfPartner.getValue<IndexType>());

				//efficiency
				EXPECT_FALSE(communicated[i][partner]) << i << " and " << partner << " already communicated.";

				communicated[i][partner] = true;
			}
		}
	}

	//completeness. For now checking all pairs. TODO: update to only check edges
	for (IndexType i = 0; i < p; i++) {
		for (IndexType j = 0; j < i; j++) {
			EXPECT_TRUE(communicated[i][j]) << i << " and " << j << " did not communicate";
		}
	}
}

TEST_F(ParcoRepartTest, testGetInterfaceNodesLocal) {
	/**
	 * test first with complete matrix. Unsuitable, I know, but we don't have a mesh generator yet.
	 * TODO: We now have a mesh generator. Update with it.
	 */

	  const IndexType n = 1000;
	  const IndexType k = 10;

	  //define distributions
	  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
	  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

	  //generate random complete matrix
	  scai::lama::CSRSparseMatrix<ValueType>a(dist, noDistPointer);
	  scai::lama::MatrixCreator::fillRandom(a, 1);

	  //generate balanced distributed partition
	  scai::lama::DenseVector<IndexType> part(dist);
	  for (IndexType i = 0; i < n; i++) {
	    IndexType blockId = i % k;
	    part.setValue(i, blockId);
	  }

	  IndexType thisBlock = 0;
	  IndexType otherBlock = 1;

	  std::vector<IndexType> interfaceNodes;
	  IndexType lastRoundMarker;
	  std::tie(interfaceNodes, lastRoundMarker) = ParcoRepart<IndexType, ValueType>::getInterfaceNodes(a, part, thisBlock, otherBlock, 1);

	  scai::hmemo::ReadAccess<IndexType> partAccess(part.getLocalValues());

	  for (IndexType node : interfaceNodes) {
		  ASSERT_TRUE(dist->isLocal(node));
		  EXPECT_EQ(thisBlock, partAccess[dist->global2local(node)]);
	  }
}

/**
* TODO: test for correct error handling in case of inconsistent distributions
*/

} //namespace
