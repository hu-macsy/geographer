#include "AuxiliaryFunctions.h"
#include "scai/partitioning/Partitioning.hpp"
#include <numeric>


namespace ITI {

template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr aux<IndexType,ValueType>::redistributeFromPartition(
    DenseVector<IndexType>& partition,
    CSRSparseMatrix<ValueType>& graph,
    std::vector<DenseVector<ValueType>>& coordinates,
    std::vector<DenseVector<ValueType>>& nodeWeights,
    Settings settings,
    bool useRedistributor,
    bool renumberPEs ) {

    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();
    const IndexType globalN = coordinates[0].getDistributionPtr()->getGlobalSize();
    const IndexType localN = partition.getDistributionPtr()->getLocalSize();

    SCAI_ASSERT_EQ_ERROR( partition.getDistribution(), graph.getRowDistribution(), "Distributions do not agree");
    SCAI_ASSERT_EQ_ERROR( graph.getNumRows(), globalN, "Mismatch in graph and coordinates size" );
    SCAI_ASSERT_EQ_ERROR( nodeWeights[0].getDistributionPtr()->getGlobalSize(), globalN, "Mismatch in nodeWeights vector" );
    SCAI_ASSERT_EQ_ERROR( partition.size(), globalN, "Mismatch in partition size");
    SCAI_ASSERT_EQ_ERROR( partition.min(), 0, "Minimum entry in partition should be 0" );
    SCAI_ASSERT_EQ_ERROR( partition.max(), numPEs-1, "Maximum entry in partition must be equal the number of processors.")

    //----------------------------------------------------------------
    // renumber blocks according to which block is the majority in every PE
    // in order to reduce redistribution costs
    //

    if( renumberPEs and partition.max()==numPEs-1) {
        scai::hmemo::ReadAccess<IndexType> rPart( partition.getLocalValues() );
        std::vector<IndexType> blockSizes( numPEs, 0 );
        for (IndexType i = 0; i < localN; i++) {
            blockSizes[ rPart[i] ] += (IndexType) 1;
        }

        //sort block IDs based on their local size
        std::vector<IndexType> indices( numPEs );
        std::iota( indices.begin(), indices.end(), 0);
        std::sort( indices.begin(), indices.end(), [&](IndexType i1, IndexType i2) {
            return blockSizes[i1] > blockSizes[i2];
        });
        std::sort( blockSizes.begin(), blockSizes.end(), std::greater<IndexType>() ); //sort also block sizes

        SCAI_ASSERT_EQ_ERROR( std::accumulate(indices.begin(), indices.end(), 0), numPEs*(numPEs-1)/2, "wrong indices vector" );//not needed?

        //for profiling/debugging, count the non-zero values, i.e., number of blocks owned by this PE
        const IndexType numBlocksOwned = std::count_if( blockSizes.begin(), blockSizes.end(), [&](IndexType bSize) {
            return bSize>0;
        } );
        if( settings.debugMode ) {
            PRINT(*comm<<": owns "<< numBlocksOwned << " blocks");
        }

        //the claimed PE id and size for the claimed block
        std::vector<IndexType> allClaimedIDs( numPEs, 0 );
        std::vector<IndexType> allClaimedSizes( numPEs, 0 );
        std::vector<IndexType> finalMapping( numPEs, 0 );

        //TODO: check bitset
        std::vector<bool> mappedPEs ( numPEs, false); //none is mapped
        std::vector<bool> availableIDs( numPEs, true); //all PE IDs are available to claim

        IndexType preference = 0; //the index of the block I want to claim, start with the first, heaviest block

        while( std::accumulate(mappedPEs.begin(), mappedPEs.end(), 0)!=numPEs ) { //all 1's

            //TODO: send 2-3 claimed indices at once so you have less global sum rounds

            //need to set to 0 because of the global sum
            allClaimedIDs.assign( numPEs, 0);
            allClaimedSizes.assign( numPEs, 0);

            //TODO: find a better way to solve this problem
            //set the ID you claim and its weight

            //if I am already mapped send previous choice
            if( mappedPEs[thisPE] ) {
                //set again otherwise the global sum will destroy the old values
                allClaimedIDs[thisPE] = finalMapping[thisPE]; //the block id that I claim
                allClaimedSizes[thisPE] = thisPE;  //it does not matter
            } else {
                //no point of asking an ID of a block I own nothing
                while( preference<numBlocksOwned ) {
                    //if the prefered ID is available
                    if( availableIDs[ indices[preference] ] ) {
                        allClaimedIDs[thisPE] = indices[preference]; //the block id that I claim
                        allClaimedSizes[thisPE] = blockSizes[preference]; //and its weight
                        break;
                    } else {
                        preference++;
                    }
                }
                //TODO: not so great solution, can pick non local blocks
                //that means that all the block IDs that I want are already taken. pick one ID at random
                if( preference==numBlocksOwned) {
                    srand( thisPE*thisPE ); //TODO: do something better here
                    IndexType ind = rand()%numPEs;
                    while( not availableIDs[ind] ) {
                        ind = (ind+1)%numPEs;
                    }
                    allClaimedIDs[thisPE] = ind;
                    allClaimedSizes[thisPE] = thisPE;
                    PRINT( thisPE <<": all preferences are taken, pick ID " << ind << " at random" );
                }
            }
            SCAI_ASSERT_LE_ERROR( preference, numBlocksOwned, "index out of bounds" );

            if( settings.debugMode ) {
                PRINT( thisPE <<": claims ID " << allClaimedIDs[thisPE] << " with size " << allClaimedSizes[thisPE] );
            }

            //global sum to gather all claimed IDs for all PEs
            comm->sumImpl( allClaimedIDs.data(), allClaimedIDs.data(), numPEs, scai::common::TypeTraits<IndexType>::stype );
            comm->sumImpl( allClaimedSizes.data(), allClaimedSizes.data(), numPEs, scai::common::TypeTraits<IndexType>::stype );

            //go over the gathered claims and resolve conflicts

            std::vector<IndexType> indicesAfterSum( numPEs );
            std::iota( indicesAfterSum.begin(), indicesAfterSum.end(), 0);
            std::sort( indicesAfterSum.begin(), indicesAfterSum.end(),
            [&](IndexType i1, IndexType i2) {
                if( allClaimedIDs[i1]==allClaimedIDs[i2] ) {
                    return allClaimedSizes[i1]<allClaimedSizes[i2];
                }
                return allClaimedIDs[i1]<allClaimedIDs[i2];
            });
            SCAI_ASSERT_EQ_ERROR( std::accumulate(indicesAfterSum.begin(), indicesAfterSum.end(), 0), numPEs*(numPEs-1)/2, "wrong indices vector" );

            for( IndexType i=0; i<numPEs-1; i++ ) {
                IndexType index = indicesAfterSum[i]; //go over according to sorting
                IndexType nextIndex = indicesAfterSum[i+1];
                //since we sort them, only adjacent claimed IDs can have a conflict
                if( allClaimedIDs[index]==allClaimedIDs[nextIndex] ) {
                    // coflict: the last one will be mapped since it has larger size
                    SCAI_ASSERT_LE_ERROR( allClaimedSizes[index], allClaimedSizes[nextIndex], "for index " << index << "; sorting gonne wrong?");
                } else {
                    finalMapping[index] = allClaimedIDs[index];
                    mappedPEs[index]= true; //this PE got mapped for this round
                    availableIDs[ allClaimedIDs[index]  ] = false;
                    //PRINT0( index << ": claims ID " << allClaimedIDs[index] << " with size " << allClaimedSizes[index] );
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

            //PRINT0("there are " << std::accumulate(mappedPEs.begin(), mappedPEs.end(), 0) << " mapped PEs");

        }//while //TODO/WARNING:not sure at all that this will always finish

        //here. allClaimedIDs should have the final claimed ID for every PE
        SCAI_ASSERT_EQ_ERROR( std::accumulate(finalMapping.begin(), finalMapping.end(), 0), numPEs*(numPEs-1)/2, "wrong indices vector" );

        //instead of renumber the PEs, renumber the blocks
        std::vector<IndexType> blockRenumbering( numPEs ); //this should be k, but the whole functions works only when k=p

        //if every PE got the same ID as the one already has
        bool nothingChanged = true;

        //reverse the renumbering from PEs to blocks: if PE 3 claimed ID 5, then renumber block 5 to 3
        for( IndexType i=0; i<numPEs; i++) {
            blockRenumbering[ finalMapping[i] ] = i;
            if( settings.debugMode ) {
                MSG0("block " << finalMapping[i] << " is mapped to " << i );
            }
            if( finalMapping[i]!=i )
                nothingChanged = false;
        }

        //go over local partition and renumber if some IDs changed
        if( not nothingChanged ) {
            scai::hmemo::WriteAccess<IndexType> partAccess( partition.getLocalValues() );
            for( IndexType i=0; i<localN; i++) {
                partAccess[i] = blockRenumbering[ partAccess[i] ];
            }
        }

    }// if( renumberPEs )

    if( settings.debugMode ) {
        IndexType numSamePart = 0;
        scai::hmemo::ReadAccess<IndexType> partAccess( partition.getLocalValues() );
        for( IndexType i=0; i<localN; i++) {
            if( partAccess[i]==thisPE ) {
                numSamePart++;
            }
        }
        ValueType percentSame = ((ValueType)numSamePart)/localN;
        PRINT( comm->getRank() <<": renumber= "<< renumberPEs << ", percent of points with part=rank: "<< percentSame <<", numPoints= "<< numSamePart );
        IndexType totalNotMovedPoints = comm->sum( numSamePart );
        PRINT0(">>> the total number of non-migrating points is " << totalNotMovedPoints );
    }

    //----------------------------------------------------------------
    // create the new distribution and redistribute data
    //

    scai::dmemo::DistributionPtr distFromPartition;

    if( useRedistributor ) {
        scai::dmemo::RedistributePlan resultRedist = scai::dmemo::redistributePlanByNewOwners(partition.getLocalValues(), partition.getDistributionPtr());
        distFromPartition = resultRedist.getTargetDistributionPtr();
        SCAI_ASSERT_EQ_ERROR( partition.getDistribution(), graph.getRowDistribution(), "Distributions do not agree");
        
        scai::dmemo::RedistributePlan redistributor = scai::dmemo::redistributePlanByNewDistribution( distFromPartition, graph.getRowDistributionPtr());
    
        redistributeInput( redistributor, partition, graph, coordinates, nodeWeights);      
    } else {
        // create new distribution from partition
        distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues());

        redistributeInput( distFromPartition, partition, graph, coordinates, nodeWeights);       
    }

    const scai::dmemo::DistributionPtr rowDist = graph.getRowDistributionPtr();
    SCAI_ASSERT_ERROR( nodeWeights[0].getDistribution().isEqual(*rowDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( coordinates[0].getDistribution().isEqual(*rowDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( partition.getDistribution().isEqual(*rowDist), "Distribution mismatch" );

    return distFromPartition;
}//redistributeFromPartition
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void aux<IndexType,ValueType>::redistributeInput(
    const scai::dmemo::DistributionPtr targetDistribution,
    DenseVector<IndexType>& partition,
    CSRSparseMatrix<ValueType>& graph,
    std::vector<DenseVector<ValueType>>& coordinates,
    std::vector<DenseVector<ValueType>>& nodeWeights){

    for (IndexType d=0; d<coordinates.size(); d++) {
        coordinates[d].redistribute( targetDistribution );
    }

    for(int w=0; w<nodeWeights.size(); w++){
        nodeWeights[w].redistribute( targetDistribution );
    }

    //column are not distributed
    const IndexType globalN = coordinates[0].getDistributionPtr()->getGlobalSize();
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

    graph.redistribute( targetDistribution, noDist );
    partition.redistribute( targetDistribution );
} 

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void  aux<IndexType,ValueType>::redistributeInput(
    const scai::dmemo::RedistributePlan redistributor,
    DenseVector<IndexType>& partition,
    CSRSparseMatrix<ValueType>& graph,
    std::vector<DenseVector<ValueType>>& coordinates,
    std::vector<DenseVector<ValueType>>& nodeWeights){

    for (IndexType d=0; d<coordinates.size(); d++) {
        coordinates[d].redistribute(redistributor);
    }
    for(int w=0; w<nodeWeights.size(); w++){
        nodeWeights[w].redistribute(redistributor);
    }

    //column are not distributed
    const IndexType globalN = coordinates[0].getDistributionPtr()->getGlobalSize();
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

    graph.redistribute( redistributor, noDist );
    partition.redistribute(redistributor);
    //30/09/19: below is older code that seems wrong
    //partition = DenseVector<IndexType>( distFromPartition, comm->getRank());
} 

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
IndexType aux<IndexType, ValueType>::toMetisInterface(
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const struct Settings &settings,
    std::vector<IndexType>& vtxDist, 
    std::vector<IndexType>& xadj,
    std::vector<IndexType>& adjncy,
    std::vector<ValueType>& vwgt,
    std::vector<double>& tpwgts,
    IndexType &wgtFlag,
    IndexType &numWeights,
    std::vector<double>& ubvec,
    std::vector<double>& xyzLocal,
    std::vector<IndexType>& options){
    SCAI_REGION( "aux.toMetisInterface");

    const scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const IndexType N = graph.getNumRows();
    const IndexType localN= dist->getLocalSize();    
    const IndexType size = comm->getSize();
    
    if( not checkConsistency( graph, coords, nodeWeights, settings)){
        PRINT0("Input not consistent.\nAborting...");
        return -1;
    }

    SCAI_ASSERT_ERROR( dist->isBlockDistributed(comm), "to convert to metis interface the input must have a block or generalBlock distribution");
        
    // get local range of indices

    IndexType lb2=N+1, ub2=-1;
    {
        scai::hmemo::HArray<IndexType> myGlobalIndexes;
        dist->getOwnedIndexes( myGlobalIndexes );
        scai::hmemo::ReadAccess<IndexType> rIndices( myGlobalIndexes );
        SCAI_ASSERT_EQ_ERROR( localN, myGlobalIndexes.size(), "Local size mismatch" );

        for( int i=0; i<localN; i++) {
            if( rIndices[i]<lb2 ) lb2=rIndices[i];
            if( rIndices[i]>ub2 ) ub2=rIndices[i];
        }
        ++ub2;  // we need max+1
    }
    //PRINT(comm->getRank() << ": "<< lb2 << " - "<< ub2);    

    scai::hmemo::HArray<IndexType> sendVtx(size+1, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvVtx(size+1);

    //TODO: use a sumArray instead of shiftArray
    for(IndexType round=0; round<comm->getSize(); round++) {
        SCAI_REGION("ParcoRepart.getBlockGraph.shiftArray");
        {   // write your part
            scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendVtx );
            sendPartWrite[0]=0;
            sendPartWrite[comm->getRank()+1]=ub2;
        }
        comm->shiftArray(recvVtx, sendVtx, 1);
        sendVtx.swap(recvVtx);
    }

    scai::hmemo::ReadAccess<IndexType> recvPartRead( recvVtx );

    // vtxDist is an array of size numPEs and is replicated in every processor
    //IndexType* tmpVtxDist = new IndexType[ size+1 ];
    vtxDist.resize( size+1 );
    vtxDist[0]= 0;

    for(int i=0; i<recvPartRead.size()-1; i++) {
        vtxDist[i+1]= recvPartRead[i+1];
    }

    //for(IndexType i=0; i<recvPartRead.size(); i++){
    //  PRINT(*comm<< " , " << i <<": " << vtxDist[i]);
    //}
    
    recvPartRead.release();

    //
    // set the input parameters for parmetis
    //

    // ndims: the number of dimensions
    IndexType ndims = settings.dimensions;

    // setting xadj=ia and adjncy=ja values, these are the local values of every processor
    const scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();

    scai::hmemo::ReadAccess<IndexType> ia( localMatrix.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localMatrix.getJA() );
    IndexType iaSize= ia.size();

    xadj.resize( iaSize ); 
    adjncy.resize( ja.size() );

    for(int i=0; i<iaSize ; i++) {
        xadj[i]= ia[i];
        SCAI_ASSERT( xadj[i] >=0, "negative value for i= "<< i << " , val= "<< xadj[i]);
    }

    for(int i=0; i<ja.size(); i++) {
        adjncy[i]= ja[i];
        SCAI_ASSERT( adjncy[i] >=0, "negative value for i= "<< i << " , val= "<< adjncy[i]);
        SCAI_ASSERT( adjncy[i] <N, "too large value for i= "<< i << " , val= "<< adjncy[i]);
    }
    ia.release();
    ja.release();

    // wgtflag is for the weight and can take 4 values. Here, 0 is for no weights.
    wgtFlag= 0;

    // the numbers of weights each vertex has.
    numWeights = 1;

    // if node weights are given

    if( nodeWeights[0].getLocalValues().size()!=0  ) {
        numWeights = nodeWeights.size();
        vwgt.resize( localN*numWeights );

        for( unsigned int w=0; w<numWeights; w++ ) {
            scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights[w].getLocalValues() );
            SCAI_ASSERT_EQ_ERROR( localN, localWeights.size(), "Local weights size mismatch. Are node weights distributed correctly?");

            //all weights for each vertex are stored contiguously
            for(unsigned int i=0; i<localN; i++) {
                int index = i*numWeights + w;
                vwgt[index] = localWeights[i];
            }
        }
        wgtFlag = 2;    //weights only in vertices
    }

    // ubvec: array of size ncon to specify imbalance for every vertex weight.
    // 1 is perfect balance and nparts perfect imbalance. Here 1 for now
    ubvec.resize( numWeights );
    for(unsigned int i=0; i<numWeights; i++) {
        ubvec[i] = ValueType(settings.epsilon + 1); //same balance for all constraints
    }

    // nparts: the number of parts to partition (=k)
    IndexType nparts= settings.numBlocks;

    // tpwgts: array of size ncons*nparts, that is used to specify the fraction of
    // vertex weight that should be distributed to each sub-domain for each balance
    // constraint. Here we want equal sizes, so every value is 1/nparts.
    
    tpwgts.resize( nparts*numWeights );

    ValueType total = 0.0;
    for(int i=0; i<nparts*numWeights ; i++) {
        tpwgts[i] = ValueType(1.0)/nparts;
        total += tpwgts[i];
    }

    SCAI_ASSERT_LT_ERROR( std::abs(total-numWeights), 1e-6, "Wrong tpwgts assignment");


    // the xyz array for coordinates of size dim*localN contains the local coords
    xyzLocal.resize( ndims*localN );

    std::vector<scai::hmemo::HArray<ValueType>> localPartOfCoords( ndims );
    for(int d=0; d<ndims; d++) {
        localPartOfCoords[d] = coords[d].getLocalValues();
    }
    for(unsigned int i=0; i<localN; i++) {
        SCAI_ASSERT_LE_ERROR( ndims*(i+1), ndims*localN, "Too large index, localN= " << localN );
        for(int d=0; d<ndims; d++) {
            xyzLocal[ndims*i+d] = ValueType(localPartOfCoords[d][i]);
        }
    }

    // options: array of integers for passing arguments.
    // Here, options[0]=0 for the default values.
    options.resize(1, 0);

    return localN;

}//toMetisInterface
//---------------------------------------------------------------------------------------

//overloaded version with commTree
template<typename IndexType, typename ValueType>
IndexType aux<IndexType, ValueType>::toMetisInterface(
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const ITI::CommTree<IndexType,ValueType> &commTree,
    const struct Settings &settings,
    std::vector<IndexType>& vtxDist, 
    std::vector<IndexType>& xadj,
    std::vector<IndexType>& adjncy,
    std::vector<ValueType>& vwgt,
    std::vector<double>& tpwgts,
    IndexType &wgtFlag,
    IndexType &numWeights,
    std::vector<double>& ubvec,
    std::vector<double>& xyzLocal,
    std::vector<IndexType>& options){
    SCAI_REGION( "aux.toMetisInterface");

    //create as without the commTree
    const IndexType localN = aux<IndexType,ValueType>::toMetisInterface(
        graph, coords, nodeWeights, settings, vtxDist, xadj, adjncy,
        vwgt, tpwgts, wgtFlag, numWeights, ubvec, xyzLocal, options );

    //overwrite only affected data

    std::vector<std::vector<ValueType>> blockSizes = commTree.getBalanceVectors();
    SCAI_ASSERT_EQ_ERROR( blockSizes.size(), numWeights, "Wrong number of weights");
    SCAI_ASSERT_EQ_ERROR( blockSizes[0].size(), settings.numBlocks, "Wrong size of weights" );

    // tpwgts: array of size numWeights*nparts, that is used to specify the fraction of
    // vertex weight that should be distributed to each sub-domain for each balance
    // constraint. Here, perhaps we do NOT want equal sizes

    //the total weight of all blocks for each weight
    std::vector<ValueType> blockWeightsSum(numWeights);
    for( int w=0; w<numWeights; w++ ){
        blockWeightsSum[w] = std::accumulate( blockSizes[w].begin(), blockSizes[w].end(), 0.0 );
    }

    const IndexType nparts= settings.numBlocks; //metis naming
    assert( tpwgts.size()== nparts*numWeights );

    ValueType total = 0.0;
    for( int w=0; w<numWeights; w++ ){
        for(int k=0; k<nparts; k++){
            int index = k*numWeights+w;
            tpwgts[index] = blockSizes[w][k]/blockWeightsSum[w];
            total += tpwgts[index];
        }
    }

    SCAI_ASSERT_LT_ERROR( std::abs(total-numWeights), 1e-6, "Wrong tpwgts assignment");

    return localN;
}//toMetisInterface
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void ITI::aux<IndexType, ValueType>::checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input) {
    SCAI_REGION( "aux.checkLocalDegreeSymmetry" )

    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const IndexType localN = inputDist->getLocalSize();

    const CSRStorage<ValueType>& storage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
    const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());

    std::vector<IndexType> inDegree(localN, 0);
    std::vector<IndexType> outDegree(localN, 0);
    for (IndexType i = 0; i < localN; i++) {
        IndexType globalI = inputDist->local2Global(i);
        const IndexType beginCols = localIa[i];
        const IndexType endCols = localIa[i+1];

        for (IndexType j = beginCols; j < endCols; j++) {
            IndexType globalNeighbor = localJa[j];

            if (globalNeighbor != globalI && inputDist->isLocal(globalNeighbor)) {
                IndexType localNeighbor = inputDist->global2Local(globalNeighbor);
                outDegree[i]++;
                inDegree[localNeighbor]++;
            }
        }
    }

    for (IndexType i = 0; i < localN; i++) {
        if (inDegree[i] != outDegree[i]) {
            //now check in detail:
            IndexType globalI = inputDist->local2Global(i);
            for (IndexType j = localIa[i]; j < localIa[i+1]; j++) {
                IndexType globalNeighbor = localJa[j];
                if (inputDist->isLocal(globalNeighbor)) {
                    IndexType localNeighbor = inputDist->global2Local(globalNeighbor);
                    bool foundBackEdge = false;
                    for (IndexType y = localIa[localNeighbor]; y < localIa[localNeighbor+1]; y++) {
                        if (localJa[y] == globalI) {
                            foundBackEdge = true;
                        }
                    }
                    if (!foundBackEdge) {
                        throw std::runtime_error("Local node " + std::to_string(globalI) + " has edge to local node " + std::to_string(globalNeighbor)
                                                 + " but no back edge found.");
                    }
                }
            }
        }
    }
}//checkLocalDegreeSymmetry
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
bool ITI::aux<IndexType, ValueType>::checkConsistency(
	const CSRSparseMatrix<ValueType> &input,
    const std::vector<DenseVector<ValueType>> &coordinates,
    const std::vector<DenseVector<ValueType>> &nodeWeights,
    const Settings settings){
	
	SCAI_REGION( "aux.checkConsistency" )

    const IndexType k = settings.numBlocks;
    const ValueType epsilon = settings.epsilon;
    const IndexType dimensions = coordinates.size();
	const IndexType n = input.getNumRows();

    /*
    * check input arguments for sanity
    */

    for( int d=0; d<dimensions; d++) {
        if (n != coordinates[d].size()) {
            throw std::runtime_error("Matrix has " + std::to_string(n) + " rows, but " + std::to_string(coordinates[0].size())
                                     + " coordinates are given.");
        }
    }

    if (n != input.getNumColumns()) {
        throw std::runtime_error("Matrix must be quadratic.");
    }

    if (!input.isConsistent()) {
        throw std::runtime_error("Input matrix inconsistent");
    }

    if (k > n) {
        throw std::runtime_error("Creating " + std::to_string(k) + " blocks from " + std::to_string(n) + " elements is impossible.");
    }

    if (epsilon < 0) {
        throw std::runtime_error("Epsilon " + std::to_string(epsilon) + " is invalid.");
    }

    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();

    if (!coordDist->isEqual( *inputDist) ) {
        throw std::runtime_error( "Coordinate and graph distributions should be equal.");
    }

    if (settings.repartition && nodeWeights.size() > 1) {
        throw std::logic_error("Repartitioning not implemented for multiple node weights.");
    }

    if (settings.initialPartition == ITI::Tool::geoMS && nodeWeights.size() > 1) {
        throw std::logic_error("MultiSection not implemented for multiple weights.");
    }

    for (IndexType i = 0; i < nodeWeights.size(); i++) {
        assert(nodeWeights[i].getDistribution().isEqual(*inputDist));
    }
    
    {
        SCAI_REGION("aux.checkConsistency.synchronize")
        comm->synchronize();
    }

	return true;
}//checkConsistency
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
bool aux<IndexType, ValueType>::alignDistributions(
        scai::lama::CSRSparseMatrix<ValueType> &graph,
        std::vector<scai::lama::DenseVector<ValueType>> &coords,
        std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        scai::lama::DenseVector<IndexType>& partition,
        const Settings settings){
    //-----------------------------------------
    //
    // if distribution do not agree, redistribute

    const scai::dmemo::DistributionPtr graphDist = graph.getRowDistributionPtr();
    scai::dmemo::CommunicatorPtr comm = graphDist->getCommunicatorPtr();

    bool willRedistribute = false;

    if( not coords[0].getDistributionPtr()->isEqual(*graphDist) ){
        PRINT0("Coordinate and graph distribution do not agree; will redistribute input");
        willRedistribute = true;
    }
    if( not nodeWeights[0].getDistributionPtr()->isEqual(*graphDist) ){
        PRINT0("nodeWeights and graph distribution do not agree; will redistribute input");
        willRedistribute = true;        
    }
    if( not partition.getDistributionPtr()->isEqual(*graphDist) ){
        PRINT0("nodeWeights and graph distribution do not agree; will redistribute input");
        willRedistribute = true;        
    }
    if( not graphDist->isBlockDistributed(comm) ){
        PRINT0("Input does not have a suitable distribution; will redistribute");
        willRedistribute = true;            
    }

    if( willRedistribute ){

        const scai::dmemo::DistributionPtr newGenBlockDist = GraphUtils<IndexType, ValueType>::genBlockRedist(graph);
        
        aux<IndexType,ValueType>::redistributeInput( newGenBlockDist, partition, graph, coords, nodeWeights);
    }
    return willRedistribute;
}//alignDistributions
//------------------------------------------------------------------------

//Force that vertices have equal, unit weights by accepting the input size as a number, not the node weights.
//In this case, the input load is equal the size of the graph.

template <typename IndexType, typename ValueType>
std::vector<ValueType> aux<IndexType, ValueType>::blockSizesForMemory( 
    const std::vector<std::vector<ValueType>> &inBlockSizes,
    const IndexType inputSize,
    const IndexType maxMemoryCapacity ) {

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType myRank = comm->getRank();

    // if( commTree.isProportional[1] and maxMemoryCapacity==0){
    //     throw std::logic_error( "Memory per PE is given as a percentage but the maximum memory of the system is not given");
    // }

    //first weight is computational speed and its sum is the total computational speed of the system
    //the total memory capacity is needed if memory is given as a percentage
    SCAI_ASSERT_EQ_ERROR( inBlockSizes.size(), 2, 
        "We need 2 weights per PE for computational load and memory. Was a description of the system provided? use option --blockSizesFile or --topologyFile" );

    const IndexType numBlocks = inBlockSizes[0].size();
    std::vector<IndexType> blockIndices( numBlocks );
    std::iota( blockIndices.begin(), blockIndices.end(), 0 );

    auto cpuOverMem = [&](IndexType i, IndexType j)->bool{
        const ValueType iCpuOverMem = inBlockSizes[0][i]/inBlockSizes[1][i];
        const ValueType jCpuOverMem = inBlockSizes[0][j]/inBlockSizes[1][j];;
        return iCpuOverMem>jCpuOverMem;
    };

    std::sort( blockIndices.begin(), blockIndices.end(), cpuOverMem );

    const ValueType totalCompSpeed = std::accumulate( inBlockSizes[0].begin(), inBlockSizes[0].end(), 0.0 );
    const ValueType totalMemCapacity = std::accumulate( inBlockSizes[1].begin(), inBlockSizes[1].end(), 0.0 );
    if(myRank==0){
        std::cout <<"total computational speed of the system is "<< totalCompSpeed << " and memory " << totalMemCapacity << std::endl;
    }
    
    ValueType assignedLoad = 0.0; //for debugging
    ValueType speedLeft = totalCompSpeed;
    ValueType loadLeft = inputSize;
    ValueType prevAssignedWeight = 0;

    std::vector<ValueType> retBlockSizes(numBlocks);
    bool filledFastPEs = false; //the fast PEs are filled first

    for( auto i : blockIndices){
        const std::vector<ValueType> &thisBlock = { inBlockSizes[0][i], inBlockSizes[1][i]};

        const ValueType thisCompSpeedPerCent = thisBlock[0]/speedLeft;
        //we add the percentage this block need to take from the excess load so far.
        const ValueType optWeight = thisCompSpeedPerCent*loadLeft;

        //memory constraint can be given as a percentage or an absolute number
        ValueType memCapacity=0.0;
        if( maxMemoryCapacity>0 ) {
            memCapacity = thisBlock[1]/totalMemCapacity*maxMemoryCapacity;
        }else{
            memCapacity = thisBlock[1];
        }

        ValueType excessLoad = 0;
        //if the opt weight fits
        if( optWeight<memCapacity ){
            retBlockSizes[i] = optWeight; //give the optimum
            if(not filledFastPEs){
                //print that only once
                MSG0("in " << __FUNCTION__ << ", from block " << i << " and on, blocks get their optWeight= " << optWeight );
            }
            filledFastPEs = true;
        }else{
            assert(not filledFastPEs);
            retBlockSizes[i] = memCapacity; //fill it
            excessLoad = optWeight-memCapacity;
            MSG0("in " << __FUNCTION__ << ", block " << i << ", optWeight= " << optWeight << ", memCapacity= " << memCapacity << 
                ", assigned= " << retBlockSizes[i] << ", excess load is " << excessLoad );
        }
        speedLeft -= thisBlock[0];
        loadLeft -= retBlockSizes[i];
        assignedLoad += retBlockSizes[i];

        if( std::abs(retBlockSizes[i]-prevAssignedWeight) > 0.01 ){
            MSG0("new assigned weight= " << retBlockSizes[i] << " for block " << i );
            prevAssignedWeight = retBlockSizes[i];
        }

    }//for blockIndices

    //all input is assigned; take care of floating errors
    SCAI_ASSERT_LT_ERROR( std::abs(assignedLoad-inputSize), 5, "Not all load was distributed, input does not fit to system");

    return retBlockSizes;
}//blockSizesForMemory
//------------------------------------------------------------------------

template class aux<IndexType, double>;
//template class aux<long long unsigned int, double>; //required for the parhip wrapper
template class aux<IndexType, float>;

}//namespace ITI
