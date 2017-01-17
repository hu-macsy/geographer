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

  scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, k, epsilon);

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
	const IndexType k = comm->getSize();
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

	const IndexType localN = inputDist->getLocalSize();

	IndexType blockId = comm->getRank();

	for (IndexType i = 0; i < localN; i++) {
		IndexType globalID = inputDist->local2global(i);
		assert(part.getDistributionPtr()->isLocal(globalID));
		part.setValue(globalID, blockId);
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
//----------------------------------------------------------

TEST_F (ParcoRepartTest, testBorders_Distributed) {
    std::string file = "Grid16x16";
    std::ifstream f(file);
    IndexType dimensions= 2, k=4;
    IndexType N, edges;
    f >> N >> edges; 
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );  
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    CSRSparseMatrix<ValueType> graph(dist, noDistPointer);
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, file );
    graph.redistribute(dist, noDistPointer);
    
    std::vector<DenseVector<ValueType>> coords(2);
    coords[0].allocate(N);
    coords[1].allocate(N);
    coords[0]= static_cast<ValueType>( 0 );
    coords[1]= static_cast<ValueType>( 0 );
    coords[0].redistribute(dist);
    coords[1].redistribute(dist);
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D( std::string(file + ".xyz"), coords, N);
    
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ(edges, (graph.getNumValues())/2 );   
    
    // get partition
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, k, 0.2);
  
    //get the border nodes
    scai::lama::DenseVector<IndexType> border(dist, 0);
    border = ParcoRepart<IndexType,ValueType>::getBorderNodes( graph , partition);
    
    for(IndexType i=0; i<dist->getLocalSize(); i++){
        EXPECT_GE(border.getLocalValues()[i] , 0);
        EXPECT_LE(border.getLocalValues()[i] , 1);
    }
    
    //partition.redistribute(dist); //not needed now
    
    // print
    int numX= 16, numY= 16;         // 2D grid dimensions
    IndexType partViz[numX][numY];   
    IndexType bordViz[numX][numY]; 
    for(int i=0; i<numX; i++)
        for(int j=0; j<numY; j++){
            partViz[i][j]=partition.getValue(i*numX+j).getValue<IndexType>();
            bordViz[i][j]=border.getValue(i*numX+j).getValue<IndexType>();
        }
            
    std::cout<<"----------------------------"<< " Partition  "<< *comm << std::endl;    
    for(int i=0; i<numX; i++){
        for(int j=0; j<numY; j++){
            if(bordViz[i][j]==1) 
                std::cout<< "\033[1;31m"<< partViz[i][j] << "\033[0m" <<"-";
            else
                std::cout<< partViz[i][j]<<"-";
        }
        std::cout<< std::endl;
    }
    
}

//------------------------------------------------------------------------------


TEST_F (ParcoRepartTest, testPEGraph_Distributed) {
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
    CSRSparseMatrix<ValueType> graph( N , N);           
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed 
    
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 ); 
    std::vector<DenseVector<ValueType>> coords(2);
    coords[0].allocate(N);
    coords[1].allocate(N);
    coords[0]= static_cast<ValueType>( 0 );
    coords[1]= static_cast<ValueType>( 0 );
    
    //fromFile2Coords_2D is not distributed, must redistribute
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D( std::string(file + ".xyz"), coords, N);
    coords[0].redistribute(dist);
    coords[1].redistribute(dist);
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    scai::lama::DenseVector<IndexType> partition(dist, -1);
    partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, k, 0.2);

    //get the PE graph
    scai::lama::CSRSparseMatrix<ValueType> PEgraph =  ParcoRepart<IndexType, ValueType>::getPEGraph( graph); 
    EXPECT_EQ( PEgraph.getNumColumns(), comm->getSize() );
    EXPECT_EQ( PEgraph.getNumRows(), comm->getSize() );
    
    
    //print
    std::cout<<"----------------------------"<< " PE graph  "<< *comm << std::endl;    
    for(IndexType i=0; i<PEgraph.getNumRows(); i++){
        std::cout<< *comm<< ":";
        for(IndexType j=0; j<PEgraph.getNumColumns(); j++){
            std::cout<< PEgraph(i,j).Scalar::getValue<ValueType>() << "-";
        }
        std::cout<< std::endl;
    }
    
}

//------------------------------------------------------------------------------

TEST_F (ParcoRepartTest, testGetLocalBlockGraphEdges_2D) {
    std::string file = "Grid16x16";
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
    CSRSparseMatrix<ValueType> graph( N , N);           
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed 
        

    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 ); 
    std::vector<DenseVector<ValueType>> coords(2);
    coords[0].allocate(N);
    coords[1].allocate(N);
    coords[0]= static_cast<ValueType>( 0 );
    coords[1]= static_cast<ValueType>( 0 );
    
    //fromFile2Coords_2D is not distributed, must redistribute
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D( std::string(file + ".xyz"), coords, N);
    coords[0].redistribute(dist);
    coords[1].redistribute(dist);
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    // get partition
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, k, 0.2);
    
    //check distributions
    assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );    
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition 
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );
    
    // test getLocalBlockGraphEdges
    IndexType max = partition.max().Scalar::getValue<IndexType>();
    std::vector<std::vector<IndexType> > edgesBlock =  ParcoRepart<IndexType, ValueType>::getLocalBlockGraphEdges( graph, partition);

    for(IndexType i=0; i<edgesBlock[0].size(); i++){
        std::cout<<  __FILE__<< " ,"<<__LINE__ <<" , "<< i <<":  _ PE number: "<< comm->getRank() << " , edge ("<< edgesBlock[0][i]<< ", " << edgesBlock[1][i] << ")" << std::endl;
        EXPECT_LE( edgesBlock[0][i] , max);
        EXPECT_LE( edgesBlock[1][i] , max);
        EXPECT_GE( edgesBlock[0][i] , 0);
        EXPECT_GE( edgesBlock[1][i] , 0);
    }

}

//------------------------------------------------------------------------------

TEST_F (ParcoRepartTest, testGetLocalBlockGraphEdges_3D) {
    IndexType dimensions= 3, k=8;
    std::vector<IndexType> numPoints= {4, 4, 4};
    std::vector<ValueType> maxCoord= {4,4,4};
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    // for now local refinement requires k = P
    k = comm->getSize();
    //
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );  
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    
    CSRSparseMatrix<ValueType> graph( N , N); 
    std::vector<DenseVector<ValueType>> coords(3, DenseVector<ValueType>(N, 0));
    
    MeshIO<IndexType, ValueType>::createStructured3DMesh(graph, coords, maxCoord, numPoints);
    graph.redistribute(dist, noDistPointer); // needed because createStructured3DMesh is not distributed 
    coords[0].redistribute(dist);
    coords[1].redistribute(dist);
    coords[2].redistribute(dist);
    
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, k, 0.2);
    
    //check distributions
    assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition 
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );
    
    // test getLocalBlockGraphEdges
    IndexType max = partition.max().Scalar::getValue<IndexType>();
    std::vector<std::vector<IndexType> > edgesBlock =  ParcoRepart<IndexType, ValueType>::getLocalBlockGraphEdges( graph, partition);

    for(IndexType i=0; i<edgesBlock[0].size(); i++){
        std::cout<<  __FILE__<< " ,"<<__LINE__ <<" , "<< i <<":  __"<< *comm<< " , >> edge ("<< edgesBlock[0][i]<< ", " << edgesBlock[1][i] << ")" << std::endl;
        EXPECT_LE( edgesBlock[0][i] , max);
        EXPECT_LE( edgesBlock[1][i] , max);
        EXPECT_GE( edgesBlock[0][i] , 0);
        EXPECT_GE( edgesBlock[1][i] , 0);
    }
}


//------------------------------------------------------------------------------

TEST_F (ParcoRepartTest, testGetBlockGraph_2D) {
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
    CSRSparseMatrix<ValueType> graph( N , N);           
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed 
        

    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 ); 
    std::vector<DenseVector<ValueType>> coords(2);
    coords[0].allocate(N);
    coords[1].allocate(N);
    coords[0]= static_cast<ValueType>( 0 );
    coords[1]= static_cast<ValueType>( 0 );
    
    //fromFile2Coords_2D is not distributed, must redistribute
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D( std::string(file + ".xyz"), coords, N);
    coords[0].redistribute(dist);
    coords[1].redistribute(dist);
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, k, 0.2);
    
    //check distributions
    assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition 
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );
    
    //test getBlockGraph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getBlockGraph( graph, partition, k);
    
    { // print
    //scai::hmemo::ReadAccess<IndexType> blockGraphRead( blockGraph );
    std::cout<< *comm <<" , Block Graph"<< std::endl;
    for(IndexType row=0; row<k; row++){
        for(IndexType col=0; col<k; col++){
            std::cout<< comm->getRank()<< ":("<< row<< ","<< col<< "):" << blockGraph( row,col).Scalar::getValue<ValueType>() <<" - ";
        }
        std::cout<< std::endl;
    }
    }
}

//------------------------------------------------------------------------------
/* with the 8x8 grid and k=16 the block graph is a 4x4 grid. With the hilbert curve it looks like this:
 * 
 *  5 - 6 - 9 - 10
 *  |   |   |   |
 *  4 - 7 - 8 - 11
 *  |   |   |   |
 *  3 - 2 - 13- 12
 *  |   |   |   |
 *  0 - 1 - 14- 15
*/
TEST_F (ParcoRepartTest, testGetLocalGraphColoring_2D) {
     std::string file = "Grid8x8";
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
    CSRSparseMatrix<ValueType> graph( N , N);           
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, file );
    //distrubute graph
    graph.redistribute(dist, noDistPointer); // needed because readFromFile2AdjMatrix is not distributed 
        

    //read the array locally and messed the distribution. Left as a remainder.
    EXPECT_EQ( graph.getNumColumns(), graph.getNumRows());
    EXPECT_EQ( edges, (graph.getNumValues())/2 ); 
    std::vector<DenseVector<ValueType>> coords(2);
    coords[0].allocate(N);
    coords[1].allocate(N);
    coords[0]= static_cast<ValueType>( 0 );
    coords[1]= static_cast<ValueType>( 0 );
    
    //fromFile2Coords_2D is not distributed, must redistribute
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D( std::string(file + ".xyz"), coords, N);
    coords[0].redistribute(dist);
    coords[1].redistribute(dist);
    EXPECT_EQ(coords[0].getLocalValues().size() , coords[1].getLocalValues().size() );
    
    //get the partition
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords, k, 0.2);
    
    //check distributions
    assert( partition.getDistribution().isEqual( graph.getRowDistribution()) );
    // the next assertion fails in "this version" (commit a2fc03ab73f3af420123c491fbf9afb84be4a0c4) because partition 
    // redistributes the graph nodes so every block is in one PE (k=P) but does NOT redistributes the coordinates.
    //assert( partition.getDistribution().isEqual( coords[0].getDistribution()) );
    
    //get getBlockGraph
    scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getBlockGraph( graph, partition, k);
    
    // test graph coloring
    // get a CSRSparseMatrix from the HArray
    scai::lama::SparseAssemblyStorage<ValueType> myStorage( k, k );
    {
    //scai::hmemo::ReadAccess<IndexType> blockGraphRead( blockGraph );
    for(IndexType row=0; row<k; row++){
        for(IndexType col=row; col<k; col++){
            //myStorage.setValue(row, col, blockGraphRead[ row*k +col]); //Matrix[i,j] = HArray[i*k +j]
            myStorage.setValue(row, col, static_cast<ValueType>( blockGraph( row,col).Scalar::getValue<IndexType>()) ); 
        }
    }
    }
    CSRSparseMatrix<ValueType> blockGraphMatrix( myStorage );
    
    scai::lama::CSRSparseMatrix<ValueType>  dummy = ParcoRepart<IndexType, ValueType>::getGraphEdgeColoring_local(blockGraphMatrix);
}

/**
* TODO: test for correct error handling in case of inconsistent distributions
*/

} //namespace
