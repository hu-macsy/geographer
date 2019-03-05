
#include "AuxiliaryFunctions.h"


namespace ITI {


template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr aux<IndexType,ValueType>::redistributeFromPartition( 
                DenseVector<IndexType>& partition,
                CSRSparseMatrix<ValueType>& graph,
                std::vector<DenseVector<ValueType>>& coordinates,
                DenseVector<ValueType>& nodeWeights,
                Settings settings, 
                bool useRedistributor,
                bool renumberPEs ){

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();
    const IndexType globalN = coordinates[0].getDistributionPtr()->getGlobalSize();
    const IndexType localN = partition.getDistributionPtr()->getLocalSize();
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

    SCAI_ASSERT_EQ_ERROR( graph.getNumRows(), globalN, "Mismatch in graph and coordinates size" );
    SCAI_ASSERT_EQ_ERROR( nodeWeights.getDistributionPtr()->getGlobalSize(), globalN , "Mismatch in nodeWeights vector" );
    SCAI_ASSERT_EQ_ERROR( partition.size(), globalN, "Mismatch in partition size");
    SCAI_ASSERT_EQ_ERROR( partition.min(), 0, "Minimum entry in partition should be 0" );
    SCAI_ASSERT_EQ_ERROR( partition.max(), numPEs-1, "Maximum entry in partition must be equal the number of processors.")

    //possible optimization: go over your local partition, calculate size of each local block and claim the PE rank of the majority block

    if( renumberPEs ){
		scai::hmemo::ReadAccess<IndexType> rPart( partition.getLocalValues() );
		//std::map<IndexType,IndexType> blockSizes;
		//scai::lama::SparseVector<IndexType> blockSizes( numPEs, 0 );
		std::vector<IndexType> blockSizes( numPEs, 0 );
		for (IndexType i = 0; i < localN; i++) {
			blockSizes[ rPart[i] ] += (IndexType) 1;
		}
for(int ii=0; ii<numPEs; ii++ ){
	PRINT( thisPE << ": for block " << ii << " size= " << blockSizes[ii] );
}
		//sort block IDs based on their local size
		std::vector<IndexType> indices( numPEs );
  		std::iota( indices.begin(), indices.end(), 0);
  		std::sort( indices.begin(), indices.end(), [&](IndexType i1, IndexType i2) {return blockSizes[i1] > blockSizes[i2];});
  		std::sort( blockSizes.begin(), blockSizes.end(), std::greater<IndexType>() ); //sort also block sizes

  		SCAI_ASSERT_EQ_ERROR( std::accumulate(indices.begin(), indices.end(), 0), numPEs*(numPEs-1)/2, "wrong indices vector" );//not needed?
	  		
  		//for profiling/debugging, count the non-zero values, i.e., number of blocks owned by this PE
  		const IndexType numBlocksOwned = std::count_if( blockSizes.begin(), blockSizes.end(), [&](IndexType bSize){ return bSize>0;} );
  		PRINT(*comm<<": owns "<< numBlocksOwned << " blocks");

  		//the claimed PE id and size for the claimed block  		
  		std::vector<IndexType> allClaimedIDs( numPEs, 0 );
  		std::vector<IndexType> allClaimedSizes( numPEs, 0 );

  		//the claimed PE id and size for the claimed block  		
  		std::vector<IndexType> gatheredClaimedIDs( numPEs, 0 );
  		std::vector<IndexType> gatheredClaimedSizes( numPEs, 0 );

  		//TODO: check bitset
  		std::vector<bool> mappedPEs ( numPEs, false); //none is mapped
  		std::vector<bool> availableIDs( numPEs, true); //all PE IDs are available to claim

  		IndexType preference = 0; //the index of the block I want to claim, start with the first, heaviest block

  		while( std::accumulate(mappedPEs.begin(), mappedPEs.end(), 0)!=numPEs ){ //all 1's

	  		//TODO: send 2-3 claimed indices at once so you have less global sum rounds
	  		
	  		//need to set to 0 because of the global sum 
	  		allClaimedIDs.assign( numPEs, 0);
	  		allClaimedSizes.assign( numPEs, 0);

  			//set the ID you claim and its weight

	  		//if I am already mapped send previous choice
	  		if( mappedPEs[thisPE] ){
	  			//set again otherwise the global sum will destroy the old values
	  			allClaimedIDs[thisPE] = indices[preference]; //the block id that I claim
	  			allClaimedSizes[thisPE] = blockSizes[preference];
	  		}else{
	  			//no point of asking an ID of a block I own nothing
	  			while( preference<numBlocksOwned ){
	  				//if the prefered ID is available
	  				if( availableIDs[ indices[preference] ] ){
	  					allClaimedIDs[thisPE] = indices[preference]; //the block id that I claim
	  					allClaimedSizes[thisPE] = blockSizes[preference]; //and its weight
	  					break;
					}else{
						preference++;
					}
	  			}

	  			//that means that all the block IDs that I want are already taken. pick one ID at random
	  			if( preference==numBlocksOwned){
	  				PRINT( thisPE <<": all preferences are taken, pick ID at random" );
	  				srand( thisPE );
	  				IndexType ind = rand()%numPEs;
	  				while( not availableIDs[ind] ){
	  					ind = (ind+1)%numPEs;
	  				}
	  				allClaimedIDs[thisPE] = ind;
	  				allClaimedSizes[thisPE] = thisPE;	  				
	  			}
	  		}

	  		SCAI_ASSERT_LE_ERROR( preference, numBlocksOwned, "index out of bounds" );

			PRINT( thisPE <<": claims ID " << allClaimedIDs[thisPE] << " with size " << allClaimedSizes[thisPE] );

for( IndexType i=0; i<numPEs;i++ )
	PRINT0( i << ": is PE i mapped: " << mappedPEs[i] << " -- is ID i available: " << availableIDs[i] );
	  		//global sum to gather all claimed IDs for all PEs
	  		comm->sumImpl( allClaimedIDs.data(), allClaimedIDs.data(), numPEs, scai::common::TypeTraits<IndexType>::stype );
	  		comm->sumImpl( allClaimedSizes.data(), allClaimedSizes.data(), numPEs, scai::common::TypeTraits<IndexType>::stype );
	  		//comm->sumImpl( mappedPEs.data(), mappedPEs.data(), numPEs, scai::common::TypeTraits<bool>::stype );

//for( IndexType i=0; i<numPEs;i++ )
//	PRINT0( i << ": " << allClaimedIDs[i] );

	  		//go over the gathered claims and resolve conflicts

	  		//TODO, try: lexicographic sorting, first based in claimedID and then by size
	  		std::vector<IndexType> indicesAfterSum( numPEs );
	  		std::iota( indicesAfterSum.begin(), indicesAfterSum.end(), 0); 
	  		std::sort( indicesAfterSum.begin(), indicesAfterSum.end(), 
	  			[&](IndexType i1, IndexType i2) {
	  				if( allClaimedIDs[i1]==allClaimedIDs[i2] ){
	  					return allClaimedSizes[i1]<allClaimedSizes[i2];
	  				}
	  				return allClaimedIDs[i1]<allClaimedIDs[i2];
	  			});
	  		SCAI_ASSERT_EQ_ERROR( std::accumulate(indicesAfterSum.begin(), indicesAfterSum.end(), 0), numPEs*(numPEs-1)/2, "wrong indices vector" );

	  		for( IndexType i=0; i<numPEs-1; i++ ){
	  			IndexType index = indicesAfterSum[i]; //go over according to sorting
	  			IndexType nextIndex = indicesAfterSum[i+1];
	  			//since we sort them, only adjacent claimed IDs can have a conflict
				PRINT0( index << ": claims ID " << allClaimedIDs[index] << " with size " << allClaimedSizes[index] );	  			
	  			if( allClaimedIDs[index]==allClaimedIDs[nextIndex] ){
	  				// coflict: the first one will stay since it has larger size
	  				SCAI_ASSERT_LE_ERROR( allClaimedSizes[index], allClaimedSizes[nextIndex], "for index " << index << "; sorting gonne wrong?");
				//TODO: will this work if there are more than 2 consecutive PEs that claim the same ID?
	  			}else{
PRINT0(">>>>> mapped " << index << " to " << allClaimedIDs[index] );
	  				mappedPEs[index]= true; //this PE got mapped for this round
	  				availableIDs[ allClaimedIDs[index]  ] = false; 
	  			}
	  		}//for i<numPEs-1

	  		//check last PE that is not included in the loop
	  		{
	  	//this way, the last index always takes what it asked for
		  		const IndexType lastIndex = indicesAfterSum.back();
		  		//const IndexType secondlastIndex = indicesAfterSum[numPEs-2];
		  		PRINT0( lastIndex << ": claims ID " << allClaimedIDs[lastIndex] << " with size " << allClaimedSizes[lastIndex] );
		  		//if( allClaimedIDs[lastIndex]==allClaimedIDs[secondlastIndex] ){
		  			// coflict: the first one will stay since it has larger size
		  		//	SCAI_ASSERT_LE_ERROR( allClaimedSizes[secondlastIndex], allClaimedSizes[lastIndex], "for index " << lastIndex << "; sorting gonne wrong?");
		  		//}
		  		mappedPEs[lastIndex] = true; //this PE got mapped for this round
		  		availableIDs[ allClaimedIDs[lastIndex] ] = false;  //ID becomes unavailable
	  		}

	  		PRINT0("there are " << std::accumulate(mappedPEs.begin(), mappedPEs.end(), 0) << " mapped PEs");

	  		//if this PE did not manage to get mapped, change your prefered ID
	  		//if( not mappedPEs[thisPE] ){
	  		//	preference++;
	  		//}

	  	}//while //TODO/WARNING:not sure at all that this will always finish
	
		IndexType newRank = allClaimedIDs[thisPE];
		scai::dmemo::CommunicatorPtr newComm = comm->split(0, newRank);
		//comm->setSizeAndRank( numPEs, );
		std::cout << *comm << ": new rank claimed= " << newRank << ", new communicator = " << *newComm << std::endl;

	}


    scai::dmemo::DistributionPtr distFromPartition;

    if( useRedistributor ){
        scai::dmemo::RedistributePlan resultRedist = scai::dmemo::redistributePlanByNewOwners(partition.getLocalValues(), partition.getDistributionPtr());
        //auto resultRedist = scai::dmemo::redistributePlanByNewOwners(partition.getLocalValues(), partition.getDistributionPtr());

        partition = DenseVector<IndexType>(resultRedist.getTargetDistributionPtr(), comm->getRank());
        scai::dmemo::RedistributePlan redistributor = scai::dmemo::redistributePlanByNewDistribution(resultRedist.getTargetDistributionPtr(), graph.getRowDistributionPtr());
        
        for (IndexType d=0; d<settings.dimensions; d++) {
            coordinates[d].redistribute(redistributor);
        }
        nodeWeights.redistribute(redistributor);    
        graph.redistribute( redistributor, noDist );

        distFromPartition = resultRedist.getTargetDistributionPtr();
    }else{
        // create new distribution from partition
        distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues());

        partition.redistribute( distFromPartition );
        graph.redistribute( distFromPartition, noDist );
        nodeWeights.redistribute( distFromPartition );

        // redistribute coordinates
        for (IndexType d = 0; d < settings.dimensions; d++) {
            //assert( coordinates[dim].size() == globalN);
            coordinates[d].redistribute( distFromPartition );
        }
    }

    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    SCAI_ASSERT_ERROR( nodeWeights.getDistribution().isEqual(*inputDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( coordinates[0].getDistribution().isEqual(*inputDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( partition.getDistribution().isEqual(*inputDist), "Distribution mismatch" );

    return distFromPartition;
}                




}//namespace ITI 