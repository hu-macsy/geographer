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

#include "ParcoRepart.h"
#include "gtest/gtest.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class ParcoRepartTest : public ::testing::Test {

};



TEST_F(ParcoRepartTest, testPartitionBalanceDistributed) {
  IndexType nroot = 8;
  IndexType n = nroot * nroot;
  IndexType k = 4;
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

IndexType index;
  for (IndexType i = 0; i < nroot; i++) {
    for (IndexType j = 0; j < nroot; j++) {
      //this is slightly wasteful, since it also iterates over indices of other processors
      index = i*nroot+j;
      coordinates[0].setValue( index, i);
      coordinates[1].setValue( index, j);
    }
  }
  
  ValueType epsilon = 0.05;

  scai::lama::DenseVector<IndexType> partition(n, 121);
  partition = ParcoRepart<IndexType, ValueType>::partitionGraph(a, coordinates, dimensions,  k, epsilon);

  IndexType localN = dist->getLocalSize();
  
  std::cout<< __FILE__<< " ,"<<__LINE__<<" __"<< *comm<<": local="<< localN<<" , global="<< dist->getGlobalSize()<<std::endl ;
  
  for(IndexType i=0; i<n ; i++){
    IndexType index = dist->global2local(i);
    IndexType part = partition(i).Scalar::getValue<IndexType>();
    if(dist->isLocal(index))
        std::cout<<" __"<<*comm << ": <"<< i<<", "<< index<< "> , part="<< part <<std::endl;
  }
  
  std::cout<< __FILE__<< " ,"<<__LINE__<<" __"<< *comm<<": local_max="<< partition.getLocalValues().max() <<std::endl ;
  
  EXPECT_GE(k-1, partition.getLocalValues().max() );
  EXPECT_EQ(n, partition.size());
  EXPECT_EQ(0, partition.min().getValue<ValueType>());
  EXPECT_EQ(k-1, partition.max().getValue<ValueType>());
  EXPECT_EQ(a.getRowDistribution(), partition.getDistribution());

  std::vector<IndexType> subsetSizes(k, 0);
  scai::utilskernel::LArray<ValueType> localPartition = partition.getLocalValues();
  
  std::cout<< __FILE__<< " ,"<<__LINE__<< std::endl;
  
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
  
  std::cout<< __FILE__<< " ,"<<__LINE__<< std::endl;
  
  EXPECT_LE(*std::max_element(subsetSizes.begin(), subsetSizes.end()), (1+epsilon)*optSize);
  
  /*
   * for replicating the local values check /home/harry/scai_lama/scai/dmemo/test/DistributionTest.cpp line 310.
   */

  
  
  
    IndexType partViz2[nroot][nroot];   
    for(int i=0; i<nroot; i++)
        for(int j=0; j<nroot; j++)
            partViz2[i][j]=partition.getValue(i*nroot+j).getValue<IndexType>();
    std::cout<<"----------------------------"<< " Partition  "<< *comm << std::endl;    
    for(int i=0; i<nroot; i++){
        for(int j=0; j<nroot; j++)
            std::cout<< partViz2[i][j]<<"-";
        std::cout<< std::endl;
    }

    for(int i=0; i<k; i++){
      std::cout<< "part "<< i<< " has #elements="<< subsetSizes[i]<< std::endl;
    }
  
  
  
  
  

}


/**
* TODO: test for correct error handling in case of inconsistent distributions
*/

} //namespace

