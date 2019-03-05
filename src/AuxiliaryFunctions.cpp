
#include "AuxiliaryFunctions.h"

//#include <scai/hmemo/WriteAccess.hpp>

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

	//----------------------------------------------------------------
	// renumber blocks according to which block is the majority in every PE
	// in order to reduce redistribution costs
	//

    if( renumberPEs ){
		scai::hmemo::ReadAccess<IndexType> rPart( partition.getLocalValues() );
		//std::map<IndexType,IndexType> blockSizes;
		//scai::lama::SparseVector<IndexType> blockSizes( numPEs, 0 );
		std::vector<IndexType> blockSizes( numPEs, 0 );
		for (IndexType i = 0; i < localN; i++) {
			blockSizes[ rPart[i] ] += (IndexType) 1;
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
		std::vector<IndexType> finalMapping( numPEs, 0 );

  		//TODO: check bitset
  		std::vector<bool> mappedPEs ( numPEs, false); //none is mapped
  		std::vector<bool> availableIDs( numPEs, true); //all PE IDs are available to claim

  		IndexType preference = 0; //the index of the block I want to claim, start with the first, heaviest block

  		while( std::accumulate(mappedPEs.begin(), mappedPEs.end(), 0)!=numPEs ){ //all 1's

	  		//TODO: send 2-3 claimed indices at once so you have less global sum rounds

	  		//need to set to 0 because of the global sum 
	  		allClaimedIDs.assign( numPEs, 0);
	  		allClaimedSizes.assign( numPEs, 0);

			//TODO: find a better way to solve this problem
  			//set the ID you claim and its weight

	  		//if I am already mapped send previous choice
	  		if( mappedPEs[thisPE] ){
	  			//set again otherwise the global sum will destroy the old values
	  			allClaimedIDs[thisPE] = finalMapping[thisPE]; //the block id that I claim
	  			allClaimedSizes[thisPE] = thisPE;  //it does not matter
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
	  			//TODO: not so great solution, can pick non local blocks
	  			//that means that all the block IDs that I want are already taken. pick one ID at random
	  			if( preference==numBlocksOwned){
	  				srand( thisPE*thisPE ); //TODO: do something better here
	  				IndexType ind = rand()%numPEs;
	  				while( not availableIDs[ind] ){
	  					ind = (ind+1)%numPEs;
	  				}
	  				allClaimedIDs[thisPE] = ind;
	  				allClaimedSizes[thisPE] = thisPE;	  				
	  				PRINT( thisPE <<": all preferences are taken, pick ID " << ind << " at random" );
	  			}
	  		}
	  		SCAI_ASSERT_LE_ERROR( preference, numBlocksOwned, "index out of bounds" );

			//PRINT( thisPE <<": claims ID " << allClaimedIDs[thisPE] << " with size " << allClaimedSizes[thisPE] );

	  		//global sum to gather all claimed IDs for all PEs
	  		comm->sumImpl( allClaimedIDs.data(), allClaimedIDs.data(), numPEs, scai::common::TypeTraits<IndexType>::stype );
	  		comm->sumImpl( allClaimedSizes.data(), allClaimedSizes.data(), numPEs, scai::common::TypeTraits<IndexType>::stype );

	  		//go over the gathered claims and resolve conflicts

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
	  			if( allClaimedIDs[index]==allClaimedIDs[nextIndex] ){
	  				// coflict: the last one will be mapped since it has larger size
	  				SCAI_ASSERT_LE_ERROR( allClaimedSizes[index], allClaimedSizes[nextIndex], "for index " << index << "; sorting gonne wrong?");
	  			}else{
	  				finalMapping[index] = allClaimedIDs[index];
	  				mappedPEs[index]= true; //this PE got mapped for this round
	  				availableIDs[ allClaimedIDs[index]  ] = false; 
	  			}
	  		}//for i<numPEs-1

	  		{
	  			//this way, the last index always takes what it asked for
		  		const IndexType lastIndex = indicesAfterSum.back();
		  		//PRINT0( lastIndex << ": claims ID " << allClaimedIDs[lastIndex] << " with size " << allClaimedSizes[lastIndex] );
		  		finalMapping[lastIndex] = allClaimedIDs[lastIndex];
		  		mappedPEs[lastIndex] = true; //this PE got mapped for this round
		  		availableIDs[ allClaimedIDs[lastIndex] ] = false;  //ID becomes unavailable
	  		}

	  		PRINT0("there are " << std::accumulate(mappedPEs.begin(), mappedPEs.end(), 0) << " mapped PEs");

	  	}//while //TODO/WARNING:not sure at all that this will always finish

	  	//here. allClaimedIDs should have the final claimed ID for every PE
	  	SCAI_ASSERT_EQ_ERROR( std::accumulate(finalMapping.begin(), finalMapping.end(), 0), numPEs*(numPEs-1)/2, "wrong indices vector" );

	  	//instead of renumber the PEs, renumber the blocks
		std::vector<IndexType> blockRenumbering( numPEs ); //this should be k, but the whole functions works only when k=p

	  	//reverse the renumbering from PEs to blocks: if PE 3 claimed ID 5, then renumber block 5 to 3
	  	for( IndexType i=0; i<numPEs; i++){
	  		blockRenumbering[ finalMapping[i] ] = i;
			//PRINT0("block " << finalMapping[i] << " is mapped to " << i );
	  	}

	  	//go over local partition and renumber
		scai::hmemo::WriteAccess<IndexType> partAccess( partition.getLocalValues() );

	  	for( IndexType i=0; i<localN; i++){
	  		partAccess[i] = blockRenumbering[ partAccess[i] ];
	  	}

	}// if( renumberPEs )


	//----------------------------------------------------------------
	// create the new distribution and redistribute data
	//

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