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
#include <chrono>

#include "ParcoRepart.h"
#include "HilbertCurve.h"
#include "MeshIO.h"
#include "gtest/gtest.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class ParcoRepartTest : public ::testing::Test {

};

/* A test originated to test the DenseVector::gather and DenseVector::scatter functions.
 * */

TEST_F(ParcoRepartTest, testGatherScatter_Distributed_2D){
    IndexType nroot = 8;
    IndexType n = nroot * nroot;
    IndexType k = 4;
    IndexType dimensions = 2;
    IndexType recursionDepth = 8;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));
    scai::lama::CSRSparseMatrix<ValueType> a(dist, noDistPointer);
    scai::lama::MatrixCreator::fillRandom(a, 0.2);//TODO: make this a proper heterogenuous mesh  
    
    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
    std::vector<DenseVector<ValueType>> coordinates(dimensions);

    for(IndexType i=0; i<dimensions; i++){ 
        coordinates[i].allocate(coordDist);
        coordinates[i] = static_cast<ValueType>( 0 );
    }

    IndexType index;
    for (IndexType i = 0; i < nroot; i++) {
        for (IndexType j = 0; j < nroot; j++) {
        //this is slightly wasteful, since it also iterates over indices of other processors
        index = i*nroot+j;
        coordinates[0].setValue( index, i);
        coordinates[1].setValue( index, j);
        }
    }

    const IndexType localN = dist->getLocalSize();
    const IndexType globalN = dist->getGlobalSize();
    std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    
    for (IndexType i = 0; i < globalN; i++) {
	for (IndexType dim = 0; dim < dimensions; dim++) {
            ValueType coord = coordinates[dim].getValue(i).Scalar::getValue<ValueType>();
            if (coord < minCoords[dim]) minCoords[dim] = coord;
            if (coord > maxCoords[dim]) maxCoords[dim] = coord;
	}
    }
	
    scai::lama::DenseVector<ValueType> hilbertIndices(dist);
    for (IndexType i = 0; i < localN; i++) {
	IndexType globalIndex = dist->local2global(i);
	ValueType globalHilbertIndex = HilbertCurve<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, globalIndex, recursionDepth, minCoords, maxCoords);
        //std::cout<< __FILE__<< ",  "<< __LINE__<< " , hilbertIndex="<< globalHilbertIndex <<std::endl;
	hilbertIndices.setValue(globalIndex, globalHilbertIndex);              
    }
    
    /*
    for(IndexType i=0; i<n ; i++){
        ValueType x = coordinates[0].getValue(i).Scalar::getValue<ValueType>();
        ValueType y = coordinates[1].getValue(i).Scalar::getValue<ValueType>();
        ValueType hilbI = hilbertIndices.getValue(i).Scalar::getValue<ValueType>();
        if(dist->isLocal(i))
            std::cout<<" __"<<*comm << ": <"<< x<< ", "<< y<< ">  , hilbI="<< hilbI<<std::endl;
    }
    */

    scai::lama::DenseVector<IndexType> permutation(dist);
    for(IndexType i=0; i<n; i++)
        permutation.setValue(i ,i);
    
    /*
    for(IndexType i=0; i<n; i++)
        //if( dist->isLocal( i ) )
            std::cout<< __FILE__<< ",  "<< __LINE__<<" __"<< *comm<< " , permutation["<< i<<"]= "<< permutation.getValue(i).Scalar::getValue<ValueType>()<< std::endl;
    */
    
        
    hilbertIndices.sort(permutation, true);
    
    /*
    for(IndexType i=0; i<localN; i++)
        std::cout<< __FILE__<< ",  "<< __LINE__<<" __"<< *comm<< " , permutation["<< i<<"]= "<< permutation.getLocalValues()[i] << std::endl;
    */    

    /*
    std::cout<< __FILE__<< ",  "<< __LINE__<< "  After sorting, NO gather"<< std::endl;

    for(IndexType i=0; i<n ; i++){
        ValueType x = coordinates[0].getValue(i).Scalar::getValue<ValueType>();
        ValueType y = coordinates[1].getValue(i).Scalar::getValue<ValueType>();
        ValueType hilbI = hilbertIndices.getValue(i).Scalar::getValue<ValueType>();
        if(dist->isLocal(i))
            std::cout<<" __"<<*comm << ": <"<< x<< ", "<< y<< ">  , hilbI="<< hilbI<<std::endl;
    }
    
    
    
    std::cout<< __FILE__<< ",  "<< __LINE__<< "  After sorting, WITH gather"<< std::endl;

    //permutation.gather( permutation, permutation, scai::utilskernel::binary::COPY);
    
    for(IndexType i=0; i<localN; i++)
        std::cout<< __FILE__<< ",  "<< __LINE__<<" __"<< *comm<< " , permutation["<< i<<"]= "<< permutation.getLocalValues()[i] << std::endl;
    */
    
    
    //
    scai::lama::DenseVector<IndexType> result(dist);
    //
    
    std::vector<DenseVector<ValueType>> coords_gathered( 2);
    for(IndexType i=0; i<dimensions; i++){ 
        coords_gathered[i].allocate(coordDist);
        coords_gathered[i] = static_cast<ValueType>( 0 );
    }
    
    
    coords_gathered[0].gather( coordinates[0], permutation, scai::utilskernel::binary::COPY);
    coords_gathered[1].gather( coordinates[1], permutation, scai::utilskernel::binary::COPY);
    
    
    for(IndexType i=0; i<localN ; i++){
        ValueType x = coords_gathered[0].getLocalValues()[i];
        ValueType y = coords_gathered[1].getLocalValues()[i];
        ValueType hilbI = hilbertIndices.getLocalValues()[i];
        
        std::cout<< i<<"- global="<< dist->local2global(i) << " __"<<*comm << ": <"<< x<< ", "<< y<< ">  , hilbI="<< hilbI<< " , and perm="<< permutation.getLocalValues()[i]<< std::endl;
        
        /* // wrong calculaation for part. we need inverse[perm[i]]. We can get it with:
         *  DenseVector<IndexType> tmpPerm = permutation;
         *  tmpPerm.sort( permPerm, true);
         *  part = int( permPerm.getLocalValues()[i] *k/n);
         * 
         * IndexType part = int( permutation.getLocalValues()[i] *k / n );
         */
        /*
        result.getLocalValues()[i] = part;
        
        std::cout<< i<<"- global="<< dist->local2global(i) << " __"<<*comm << ": <"<< x<< ", "<< y<< ">  , hilbI="<< hilbI<< " , and perm="<< permutation.getLocalValues()[i]<< " , result=" << part<< std::endl;
        */
        
    }
    
}


/*Create a structured 2D grid and partitions it in k parts. 
 * Also print/visualize in C-style way.
 */

TEST_F(ParcoRepartTest, testPartitionBalanceDistributed) {
  IndexType numX = 32, numY = 32;
  IndexType n = numX*numY;
  IndexType k = 8;
  IndexType dimensions = 2;
  
  scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
  scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));
  scai::lama::CSRSparseMatrix<ValueType> a(dist, noDistPointer);
  scai::lama::MatrixCreator::fillRandom(a, 0.2);//TODO: make this a proper heterogenuous mesh  
  
  scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
  std::vector<DenseVector<ValueType>> coordinates(dimensions);

  for(IndexType i=0; i<dimensions; i++){ 
      coordinates[i].allocate(coordDist);
      coordinates[i] = static_cast<ValueType>( 0 );
  }

  //create the grid
  IndexType i=0;
  ValueType indexX, indexY;
  ValueType DX= 1, DY=1;
  for(indexX= 0; indexX<numX; indexX++) {
    for(indexY= 0; indexY<numY; indexY++) {
	coordinates[0].setValue(i, indexX);
	coordinates[1].setValue(i, indexY);
        ++i;
    }
 }
  
  ValueType epsilon = 0.05;

  scai::lama::DenseVector<IndexType> partition(dist);
  partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, dimensions,  k, epsilon);
  IndexType localN = dist->getLocalSize();
  
  std::cout<< __FILE__<< " ,"<<__LINE__<<" == "<< *dist<< "==  __"<< *comm<< "local_max="<< partition.getLocalValues().max() <<std::endl ;
  
  EXPECT_GE(k-1, partition.getLocalValues().max() );
  EXPECT_EQ(n, partition.size());
  EXPECT_EQ(0, partition.min().getValue<ValueType>());
  EXPECT_EQ(k-1, partition.max().getValue<ValueType>());
  EXPECT_EQ(a.getRowDistribution(), partition.getDistribution());

  std::vector<IndexType> subsetSizes(k, 0);
  scai::utilskernel::LArray<ValueType> localPartition = partition.getLocalValues();
  
  
  for (IndexType i = 0; i < localPartition.size(); i++) {
    ValueType partID = localPartition[i];
    EXPECT_LE(partID, k-1); 
    EXPECT_GE(partID, 0);
    subsetSizes[partID] += 1;
  }
  IndexType optSize = std::ceil(n / k);

//if we don't have the full partition locally, 
  if (!partition.getDistribution().isReplicated()) {
    //sum block sizes over all processes
    for (IndexType partID = 0; partID < k; partID++) {
      subsetSizes[partID] = comm->sum(subsetSizes[partID]);
    }
  }
  
  EXPECT_LE(*std::max_element(subsetSizes.begin(), subsetSizes.end()), (1+epsilon)*optSize);
  
  /*
   * for replicating the local values check /home/harry/scai_lama/scai/dmemo/test/DistributionTest.cpp line 310.
   */

  
  
// print
    IndexType partViz2[numX][numY];   
    for(int i=0; i<numX; i++)
        for(int j=0; j<numY; j++)
            partViz2[i][j]=partition.getValue(i*numX+j).getValue<IndexType>();
            //partViz2[i][j]=partition.getLocalValues()[ dist->global2local(i*nroot+j) ];
            
    std::cout<<"----------------------------"<< " Partition  "<< *comm << std::endl;    
    for(int i=0; i<numX; i++){
        for(int j=0; j<numY; j++)
        std::cout<< partViz2[i][j]<<"-";
        std::cout<< std::endl;
      }

    for(int i=0; i<k; i++)
        std::cout<< "part "<< i<< " has #elements="<< subsetSizes[i]<< std::endl;
      


}

//----------------------------------------------------------------------------------------

TEST_F(ParcoRepartTest, testPartitionBalanceStructured_Distributed_3D) {
    IndexType k = 8;
    IndexType dimensions = 3;
  
    std::vector<IndexType> numPoints= {15, 13, 22};
    std::vector<ValueType> maxCoord= {100,180,130};
    IndexType numberOfPoints= numPoints[0]*numPoints[1]*numPoints[2];
    std::vector<DenseVector<ValueType>> coords(3, DenseVector<ValueType>(numberOfPoints, 0));
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, numberOfPoints) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(numberOfPoints));
    scai::lama::CSRSparseMatrix<ValueType> adjM(dist, noDistPointer);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    MeshIO<IndexType, ValueType>::createStructured3DMesh(adjM, coords, maxCoord, numPoints);
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout<<__FILE__<< "  "<< __LINE__<< " , time for creating structured3DMesh: "<< duration <<std::endl;

    ValueType epsilon = 0.05;

    scai::lama::DenseVector<IndexType> partition(dist);
    partition = ParcoRepart<IndexType, ValueType>::partitionGraph(adjM, coords, dimensions,  k, epsilon);
    IndexType localN = dist->getLocalSize();  

    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - t2 ).count();
    std::cout<<__FILE__<< "  "<< __LINE__<< " , time for partitioning: "<< duration <<std::endl;

    std::cout<< __FILE__<< " ,"<<__LINE__<<" == "<< *dist<< "==  __"<< *comm<< "local_max="<< partition.getLocalValues().max() <<std::endl ;
  
    EXPECT_GE(k-1, partition.getLocalValues().max() );
    EXPECT_EQ(numberOfPoints, partition.size());
    EXPECT_EQ(0, partition.min().getValue<ValueType>());
    EXPECT_EQ(k-1, partition.max().getValue<ValueType>());
    EXPECT_EQ(adjM.getRowDistribution(), partition.getDistribution());

    std::vector<IndexType> subsetSizes(k, 0);
    scai::utilskernel::LArray<ValueType> localPartition = partition.getLocalValues();
  
  
    for (IndexType i = 0; i < localPartition.size(); i++) {
        ValueType partID = localPartition[i];
        EXPECT_LE(partID, k-1); 
        EXPECT_GE(partID, 0);
        subsetSizes[partID] += 1;
    }
    IndexType optSize = std::ceil(numberOfPoints / k);

    //if we don't have the full partition locally, 
    if (!partition.getDistribution().isReplicated()) {
        //sum block sizes over all processes
        for (IndexType partID = 0; partID < k; partID++) {
            subsetSizes[partID] = comm->sum(subsetSizes[partID]);
        }
    }
  
    EXPECT_LE(*std::max_element(subsetSizes.begin(), subsetSizes.end()), (1+epsilon)*optSize);

    for(int i=0; i<k; i++)
        std::cout<< "part "<< i<< " has #elements="<< subsetSizes[i]<< std::endl;
      
  /*
   * for replicating the local values check /home/harry/scai_lama/scai/dmemo/test/DistributionTest.cpp line 310.
   */
    const ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(adjM, partition, true);
    std::cout<< "cut "<< " is ="<< cut<< std::endl;
}
//----------------------------------------------------------------------------------------

TEST_F(ParcoRepartTest, testPartitionBalanceStructured_Local_3D) {
    IndexType k = 8;
    IndexType dimensions = 3;
  
    std::vector<IndexType> numPoints= {15, 13, 22};
    std::vector<ValueType> maxCoord= {100,180,130};
    IndexType numberOfPoints= numPoints[0]*numPoints[1]*numPoints[2];
    std::vector<DenseVector<ValueType>> coords(3, DenseVector<ValueType>(numberOfPoints, 0));
    
    //scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    //scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, numberOfPoints) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(numberOfPoints));
    scai::lama::CSRSparseMatrix<ValueType> adjM(noDistPointer, noDistPointer);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    MeshIO<IndexType, ValueType>::createStructured3DMesh(adjM, coords, maxCoord, numPoints);
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout<<__FILE__<< "  "<< __LINE__<< " , time for creating structured3DMesh: "<< duration <<std::endl;

    ValueType epsilon = 0.05;

    scai::lama::DenseVector<IndexType> partition(numberOfPoints, -1);
    partition = ParcoRepart<IndexType, ValueType>::partitionGraph(adjM, coords, dimensions,  k, epsilon);
    //IndexType localN = dist->getLocalSize();  

    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - t2 ).count();
    std::cout<<__FILE__<< "  "<< __LINE__<< " , time for partitioning: "<< duration <<std::endl;

    std::cout<< __FILE__<< " ,"<<__LINE__<< "==  __"<< "local_max="<< partition.getLocalValues().max() <<std::endl ;
  
    EXPECT_GE(k-1, partition.getLocalValues().max() );
    EXPECT_EQ(numberOfPoints, partition.size());
    EXPECT_EQ(0, partition.min().getValue<ValueType>());
    EXPECT_EQ(k-1, partition.max().getValue<ValueType>());
    EXPECT_EQ(adjM.getRowDistribution(), partition.getDistribution());

    std::vector<IndexType> subsetSizes(k, 0);
    scai::utilskernel::LArray<ValueType> localPartition = partition.getLocalValues();
  
  
    for (IndexType i = 0; i < localPartition.size(); i++) {
        ValueType partID = localPartition[i];
        EXPECT_LE(partID, k-1); 
        EXPECT_GE(partID, 0);
        subsetSizes[partID] += 1;
    }
    IndexType optSize = std::ceil(numberOfPoints / k);

    /*
    //if we don't have the full partition locally, 
    if (!partition.getDistribution().isReplicated()) {
        //sum block sizes over all processes
        for (IndexType partID = 0; partID < k; partID++) {
            subsetSizes[partID] = comm->sum(subsetSizes[partID]);
        }
    }
    */
    EXPECT_LE(*std::max_element(subsetSizes.begin(), subsetSizes.end()), (1+epsilon)*optSize);

    for(int i=0; i<k; i++)
        std::cout<< "part "<< i<< " has #elements="<< subsetSizes[i]<< std::endl;
      
  /*
   * for replicating the local values check /home/harry/scai_lama/scai/dmemo/test/DistributionTest.cpp line 310.
   */
    const ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut(adjM, partition, true);
    std::cout<< "cut "<< " is ="<< cut<< std::endl;
}


/**
* TODO: test for correct error handling in case of inconsistent distributions
*/

} //namespace

