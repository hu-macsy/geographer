#include <unordered_set>
#include <numeric>

#include "LocalRefinement.h"
#include "GraphUtils.h"
#include "HaloPlanFns.h"
#include "PrioQueue.h"

#include <scai/utilskernel/TransferUtils.hpp>
#include <scai/dmemo/mpi/MPIException.hpp>

using scai::hmemo::HArray;

namespace ITI {

template<typename IndexType, typename ValueType>
std::vector<ValueType> ITI::LocalRefinement<IndexType, ValueType>::distributedFMStep(
    CSRSparseMatrix<ValueType>& input,
    DenseVector<IndexType>& part,
    std::vector<IndexType>& nodesWithNonLocalNeighbors,
    DenseVector<ValueType> &nodeWeights,
    std::vector<DenseVector<ValueType>> &coordinates,
    std::vector<ValueType> &distances,
    DenseVector<IndexType> &origin,
    const std::vector<DenseVector<IndexType>>& communicationScheme,
    Settings settings) {

    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

    SCAI_REGION( "LocalRefinement.distributedFMStep" )
    const IndexType globalN = input.getRowDistributionPtr()->getGlobalSize();
    scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();

    if (part.getDistributionPtr()->getLocalSize() != input.getRowDistributionPtr()->getLocalSize()) {
        throw std::runtime_error("Distributions of input matrix and partitions must be equal, for now.");
    }

    if (!input.getColDistributionPtr()->isReplicated()) {
        throw std::runtime_error("Column distribution needs to be replicated.");
    }

    if (settings.useGeometricTieBreaking) {
        for (IndexType dim = 0; dim < coordinates.size(); dim++) {
            if (coordinates[dim].getDistributionPtr()->getLocalSize() != input.getRowDistributionPtr()->getLocalSize()) {
                throw std::runtime_error("Coordinate distribution must be equal to matrix distribution");
            }
        }
        assert(distances.size() == input.getRowDistributionPtr()->getLocalSize());
    }

    if (settings.epsilon < 0) {
        throw std::runtime_error("Epsilon must be >= 0, not " + std::to_string(settings.epsilon));
    }

    if (settings.numBlocks != comm->getSize()) {
        throw std::runtime_error("Called with " + std::to_string(comm->getSize()) + " processors, but " + std::to_string(settings.numBlocks) + " blocks.");
    }

    //TODO: opt size
    //const IndexType optSize_old = ceil(double(globalN) / settings.numBlocks);
    const IndexType optSize =  nodeWeights.sum() / settings.numBlocks;
    const IndexType maxAllowableBlockSize = optSize*(1+settings.epsilon);

    //for now, we are assuming equal numbers of blocks and processes
    const IndexType localBlockID = comm->getRank();

    const bool nodesWeighted = nodeWeights.getDistributionPtr()->getGlobalSize() > 0;
    if (nodesWeighted && nodeWeights.getDistributionPtr()->getLocalSize() != input.getRowDistributionPtr()->getLocalSize()) {
        throw std::runtime_error("Node weights have " + std::to_string(nodeWeights.getDistributionPtr()->getLocalSize()) + " local values, should be "
                                 + std::to_string(input.getRowDistributionPtr()->getLocalSize()));
    }

    ValueType gainSum = 0;
    std::vector<ValueType> gainPerRound(communicationScheme.size(), 0);

    //copy into usable data structure with iterators
    std::vector<IndexType> myGlobalIndices(input.getRowDistributionPtr()->getLocalSize());
    {
        const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
        scai::hmemo::HArray<IndexType> ownIndices;
        inputDist->getOwnedIndexes(ownIndices);
        scai::hmemo::ReadAccess<IndexType> rIndices(ownIndices);
        for (IndexType j = 0; j < myGlobalIndices.size(); j++) {
            myGlobalIndices[j] = rIndices[j];
        }
    }

    std::chrono::duration<double> beforeLoop = std::chrono::steady_clock::now() - startTime;
    if(settings.verbose or settings.debugMode) {
        ValueType t1 = comm->max(beforeLoop.count());
        PRINT0("time elapsed before main loop: " << t1 );
        PRINT0("number of rounds/loops: " << communicationScheme.size() );
    }

    //main loop, one iteration for each color of the graph coloring
    for (IndexType color = 0; color < communicationScheme.size(); color++) {
        SCAI_REGION( "LocalRefinement.distributedFMStep.loop" )
        std::chrono::time_point<std::chrono::steady_clock> startColor =  std::chrono::steady_clock::now();

        const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
        const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
        const scai::dmemo::DistributionPtr commDist = communicationScheme[color].getDistributionPtr();

        const IndexType localN = inputDist->getLocalSize();

        if (!communicationScheme[color].getDistributionPtr()->isLocal(comm->getRank())) {
            throw std::runtime_error("Scheme value for " + std::to_string(comm->getRank()) + " must be local.");
        }

        scai::hmemo::ReadAccess<IndexType> commAccess(communicationScheme[color].getLocalValues());
        IndexType partner = commAccess[commDist->global2Local(comm->getRank())];

        //check symmetry of communication scheme
        assert(partner < comm->getSize());
        if (commDist->isLocal(partner)) {
            IndexType partnerOfPartner = commAccess[commDist->global2Local(partner)];
            if (partnerOfPartner != comm->getRank()) {
                throw std::runtime_error("Process " + std::to_string(comm->getRank()) + ": Partner " + std::to_string(partner) + " has partner "
                                         + std::to_string(partnerOfPartner) + ".");
            }
            if(settings.debugMode){
                std::cout<< "Comm round "<< color <<": PE " << comm->getRank() << " is paired with " << partner << std::endl;
            }
        }

        /*
         * check for validity of partition
         */
        {
            SCAI_REGION( "LocalRefinement.distributedFMStep.loop.checkPartition" )
            scai::hmemo::ReadAccess<IndexType> partAccess(part.getLocalValues());
            for (IndexType j = 0; j < localN; j++) {
                if (partAccess[j] != localBlockID) {
                    throw std::runtime_error("Block ID "+std::to_string(partAccess[j])+" found on process "+std::to_string(localBlockID)+".");
                }
            }
            for ([[maybe_unused]] IndexType node : nodesWithNonLocalNeighbors) {
                assert(inputDist->isLocal(node));
            }
        }

        ValueType gainThisRound = 0;

        scai::dmemo::HaloExchangePlan graphHalo;
        CSRStorage<ValueType> haloMatrix;
        HArray<ValueType> nodeWeightHaloData;

        if (partner != comm->getRank()) {
            //processor is active this round

            /*
             * get indices of border nodes with breadth-first search
             */
            std::vector<IndexType> interfaceNodes;
            std::vector<IndexType> roundMarkers;
            std::tie(interfaceNodes, roundMarkers)= getInterfaceNodes(input, part, nodesWithNonLocalNeighbors, partner, settings.minBorderNodes);

            const IndexType lastRoundMarker = roundMarkers[roundMarkers.size()-1];
            const IndexType secondRoundMarker = roundMarkers[1];

            /*
             * now swap indices of nodes in border region with partner processor.
             * For this, first find out the length of the swap array.
             */

            SCAI_REGION_START( "LocalRefinement.distributedFMStep.loop.prepareSets" )
            //swap size of border region and total block size.
            IndexType blockSize = part.getDistributionPtr()->getLocalSize();

            const ValueType blockWeightSum = scai::utilskernel::HArrayUtils::sum(nodeWeights.getLocalValues());

            ValueType swapField[5];
            swapField[0] = interfaceNodes.size();
            swapField[1] = secondRoundMarker;
            swapField[2] = lastRoundMarker;
            swapField[3] = blockSize;
            swapField[4] = blockWeightSum;
            comm->swap(swapField, 5, partner);
            //want to isolate raw array accesses as much as possible, define named variables and only use these from now
            const IndexType otherSize = swapField[0];
            const IndexType otherSecondRoundMarker = swapField[1];
            const IndexType otherLastRoundMarker = swapField[2];
            //const IndexType otherBlockSize = swapField[3];
//WARNING/TODO: this assumes that node weights (and thus block weights) are integers
            const IndexType otherBlockWeightSum = swapField[4];

            if (interfaceNodes.size() == 0) {
                if (otherSize != 0) {
                    std::cout<<"this PE: " << comm->getRank() << ", partner: " << partner << std::endl;
                    throw std::runtime_error("Partner PE has a border region, but this PE doesn't. Looks like the block indices were allocated inconsistently.");
                } else {
                    /*
                     * These processes don't share a border and thus have no communication to do with each other. How did they end up in a communication scheme?
                     * We could skip the loop entirely. However, the remaining instructions are very fast anyway.
                     */
                }
            }

            //the two border regions might have different sizes. Swapping array is sized for the maximum of the two.
            const IndexType swapLength = std::max(otherSize, IndexType (interfaceNodes.size()));
            IndexType swapNodes[swapLength];
            for (IndexType i = 0; i < swapLength; i++) {
                if (i < interfaceNodes.size()) {
                    swapNodes[i] = interfaceNodes[i];
                } else {
                    swapNodes[i] = -1;
                }
            }

            //now swap border region
            comm->swap(swapNodes, swapLength, partner);

            //read interface nodes of partner process from swapped array.
            std::vector<IndexType> requiredHaloIndices(otherSize);
            std::copy(swapNodes, swapNodes+otherSize, requiredHaloIndices.begin());

            //if we need more halo indices than there are non-local indices at all, something went wrong.
            assert(requiredHaloIndices.size() <= globalN - inputDist->getLocalSize());

            //swap distances used for tie breaking
            ValueType distanceSwap[swapLength];
            if (settings.useGeometricTieBreaking) {
                for (IndexType i = 0; i < interfaceNodes.size(); i++) {
                    distanceSwap[i] = distances[inputDist->global2Local(interfaceNodes[i])];
                }
                comm->swap(distanceSwap, swapLength, partner);
            }

            /*
             * Build Halo to cover border region of other PE.
             * This uses a special halo builder method that doesn't require communication, since the required and provided indices are already known.
             */
            [[maybe_unused]] const IndexType numValues = input.getLocalStorage().getValues().size();
            {
                scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
                scai::hmemo::HArrayRef<IndexType> arrProvidedIndexes( interfaceNodes );
                graphHalo = buildWithPartner( *inputDist, arrRequiredIndexes, arrProvidedIndexes, partner );
            }

            //all required halo indices are in the halo
            for ([[maybe_unused]] IndexType node : requiredHaloIndices) {
                assert(graphHalo.global2Halo(node) != scai::invalidIndex);
            }

            /*
             * Exchange Halo. This only requires communication with the partner process.
             */
            haloMatrix.exchangeHalo( graphHalo, input.getLocalStorage(), *comm );

            //local part should stay unchanged, check edge number as proxy for that
            assert(input.getLocalStorage().getValues().size() == numValues);

            //Here we only exchange one BFS-Round less than gathered, to make sure that all neighbors of the considered edges are still in the halo.
            std::vector<IndexType> borderRegionIDs(interfaceNodes.begin(), interfaceNodes.begin()+lastRoundMarker);
            std::vector<bool> assignedToSecondBlock(lastRoundMarker, 0);//nodes from own border region are assigned to first block
            std::copy(requiredHaloIndices.begin(), requiredHaloIndices.begin()+otherLastRoundMarker, std::back_inserter(borderRegionIDs));
            assignedToSecondBlock.resize(borderRegionIDs.size(), 1);//nodes from other border region are assigned to second block
            assert(borderRegionIDs.size() == lastRoundMarker + otherLastRoundMarker);

            const IndexType borderRegionSize = borderRegionIDs.size();

            /*
             * If nodes are weighted, exchange Halo for node weights
             */
            std::vector<ValueType> borderNodeWeights = {};
            if (nodesWeighted) {
                const HArray<ValueType>& localWeights = nodeWeights.getLocalValues();
                assert(localWeights.size() == localN);
                graphHalo.updateHalo( nodeWeightHaloData, localWeights, *comm );
                borderNodeWeights.resize(borderRegionSize,-1);
                for (IndexType i = 0; i < borderRegionSize; i++) {
                    const IndexType globalI = borderRegionIDs[i];
                    const IndexType localI = inputDist->global2Local(globalI);
                    if (localI != scai::invalidIndex) {
                        borderNodeWeights[i] = localWeights[localI];
                    } else {
                        const IndexType localI = graphHalo.global2Halo(globalI);
                        assert(localI != scai::invalidIndex);
                        borderNodeWeights[i] = nodeWeightHaloData[localI];
                    }
                    assert(borderNodeWeights[i] >= 0);
                }
            }

            //origin data, for redistribution in uncoarsening step
            scai::hmemo::HArray<IndexType> originData;
            graphHalo.updateHalo(originData, origin.getLocalValues(), *comm);

//WARNING/TODO: this assumes that node weights (and thus block weights) are integers?
            //block sizes and capacities
            std::pair<IndexType, IndexType> blockSizes = {blockWeightSum, otherBlockWeightSum};
            std::pair<IndexType, IndexType> maxBlockSizes = {maxAllowableBlockSize, maxAllowableBlockSize};

            //second round markers
            std::pair<IndexType, IndexType> secondRoundMarkers = {secondRoundMarker, otherSecondRoundMarker};

            //tie breaking keys
            std::vector<ValueType> tieBreakingKeys(borderRegionSize, 0);

            if (settings.useGeometricTieBreaking) {
                for (IndexType i = 0; i < lastRoundMarker; i++) {
                    tieBreakingKeys[i] = -distances[inputDist->global2Local(interfaceNodes[i])];
                }
                for (IndexType i = lastRoundMarker; i < borderRegionSize; i++) {
                    tieBreakingKeys[i] = -distanceSwap[i-lastRoundMarker];
                }
            }

            if (settings.useDiffusionTieBreaking) {
                std::vector<ValueType> load = twoWayLocalDiffusion(input, haloMatrix, graphHalo, borderRegionIDs, secondRoundMarkers, assignedToSecondBlock, settings);
                for (IndexType i = 0; i < borderRegionSize; i++) {
                    tieBreakingKeys[i] = std::abs(load[i]);
                }
            }

            SCAI_REGION_END( "LocalRefinement.distributedFMStep.loop.prepareSets" )

            /*
             * execute FM locally
             */
            /*
            borderNodesOwners: the owner PE for every node that participates in the LR. this is actually part[i]
            commTree: the physical network
            inside, precalculate the pairwise distances for every owner. We expect that the number
            of PEs involved is low so it makes sense to precompute the distances.
            Maybe distances can be computed here and given as an input
            */
            ValueType gain = twoWayLocalFM(input, haloMatrix, graphHalo, borderRegionIDs, borderNodeWeights, assignedToSecondBlock, maxBlockSizes, blockSizes, tieBreakingKeys, settings);

            {
                SCAI_REGION( "LocalRefinement.distributedFMStep.loop.swapFMResults" )
                /*
                 * Communicate achieved gain.
                 * Since only two values are swapped, the tracing results measure the latency and synchronization overhead,
                 * the difference in running times between the two local FM implementations.
                 */
                swapField[0] = gain;
                swapField[1] = otherBlockWeightSum;
                comm->swap(swapField, 2, partner);
            }
            const ValueType otherGain = swapField[0];
            const ValueType otherSecondBlockWeightSum = swapField[1];

            if (otherSecondBlockWeightSum > maxBlockSizes.first) {
                //If a block is too large after the refinement, it is only because it was too large to begin with.
                assert(otherSecondBlockWeightSum <= blockWeightSum);
            }

            if (otherGain <= 0 && gain <= 0  /* && false */ ) {
                //Oh well. None of the processors managed an improvement. No need to update data structures.

            } else {
                SCAI_REGION_START( "LocalRefinement.distributedFMStep.loop.prepareRedist" )

                gainThisRound = std::max( otherGain, gain);

                assert(gainThisRound > 0);
                gainPerRound[color] = gainThisRound;

                gainSum += gainThisRound;

                //partition must be consistent, so if gains are equal, pick one of lower index.
                bool otherWasBetter = (otherGain > gain || (otherGain == gain && partner < comm->getRank()));

                //swap result of local FM
                IndexType resultSwap[borderRegionIDs.size()];
                std::copy(assignedToSecondBlock.begin(), assignedToSecondBlock.end(), resultSwap);
                comm->swap(resultSwap, borderRegionIDs.size(), partner);

                //keep best solution. Since the two processes used different offsets, we can't copy them directly
                if (otherWasBetter) {
                    for (IndexType i = 0; i < lastRoundMarker; i++) {
                        assignedToSecondBlock[i] = !(bool(resultSwap[i+otherLastRoundMarker]));//got bool array from partner, need to invert everything.
                    }
                    for (IndexType i = lastRoundMarker; i < borderRegionSize; i++) {
                        assignedToSecondBlock[i] = !(bool(resultSwap[i-lastRoundMarker]));//got bool array from partner, need to invert everything.
                    }
                }

                std::set<IndexType> borderCandidates(nodesWithNonLocalNeighbors.begin(), nodesWithNonLocalNeighbors.end());
                std::vector<IndexType> deletedNodes;

                /*
                 * remove nodes
                 */
                for (IndexType i = 0; i < lastRoundMarker; i++) {
                    if (assignedToSecondBlock[i]) {
                        auto deleteIterator = std::lower_bound(myGlobalIndices.begin(), myGlobalIndices.end(), interfaceNodes[i]);
                        assert(*deleteIterator == interfaceNodes[i]);
                        myGlobalIndices.erase(deleteIterator);
                        deletedNodes.push_back(interfaceNodes[i]);
                    }
                }

                /*
                 * add new nodes
                 */
                for (IndexType i = 0; i < otherLastRoundMarker; i++) {
                    if (!assignedToSecondBlock[lastRoundMarker + i]) {
                        assert(requiredHaloIndices[i] == borderRegionIDs[lastRoundMarker + i]);
                        myGlobalIndices.push_back(requiredHaloIndices[i]);
                        borderCandidates.insert(requiredHaloIndices[i]);
                    }
                }

                {
                    //add neighbors of removed node to borderCandidates
                    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
                    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
                    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
                    for (IndexType globalI : deletedNodes) {
                        IndexType localI = inputDist->global2Local(globalI);
                        for (IndexType j = ia[localI]; j < ia[localI+1]; j++) {
                            borderCandidates.insert(ja[j]);
                        }
                    }
                }

                //sort indices
                std::sort(myGlobalIndices.begin(), myGlobalIndices.end());
                SCAI_REGION_END( "LocalRefinement.distributedFMStep.loop.prepareRedist" )

                SCAI_REGION_START( "LocalRefinement.distributedFMStep.loop.redistribute" )

                SCAI_REGION_START( "LocalRefinement.distributedFMStep.loop.redistribute.generalDistribution" )
                HArray<IndexType> indexTransport(myGlobalIndices.size(), myGlobalIndices.data());

                auto newDistribution = scai::dmemo::generalDistributionUnchecked(globalN, indexTransport, comm);
                SCAI_REGION_END( "LocalRefinement.distributedFMStep.loop.redistribute.generalDistribution" )

                {
                    SCAI_REGION( "LocalRefinement.distributedFMStep.loop.redistribute.updateDataStructures" )

                    redistributeFromHalo(input, newDistribution, graphHalo, haloMatrix);
                    part = scai::lama::fill<DenseVector<IndexType>>(newDistribution, localBlockID);
                    if (nodesWeighted) {
                        redistributeFromHalo<ValueType>(nodeWeights, newDistribution, graphHalo, nodeWeightHaloData);
                    }
                    redistributeFromHalo(origin, newDistribution, graphHalo, originData);
                }
                assert(input.getRowDistributionPtr()->isEqual(*part.getDistributionPtr()));
                SCAI_REGION_END( "LocalRefinement.distributedFMStep.loop.redistribute" )
                /*
                 * update local border. This could probably be optimized by only updating the part that could have changed in the last round.
                 */
                {
                    SCAI_REGION( "LocalRefinement.distributedFMStep.loop.updateLocalBorder" )
                    nodesWithNonLocalNeighbors = GraphUtils<IndexType, ValueType>::getNodesWithNonLocalNeighbors(input, borderCandidates);
                }

                /*
                 * update coordinates and block distances
                 */
                if (settings.useGeometricTieBreaking)
                {
                    SCAI_REGION( "LocalRefinement.distributedFMStep.loop.updateBlockDistances" )

                    for (IndexType dim = 0; dim < coordinates.size(); dim++) {
                        HArray<ValueType>& localCoords = coordinates[dim].getLocalValues();
                        HArray<ValueType> haloData;
                        graphHalo.updateHalo( haloData, localCoords, *comm );
                        redistributeFromHalo<ValueType>(coordinates[dim], newDistribution, graphHalo, haloData);
                    }

                    distances = LocalRefinement<IndexType, ValueType>::distancesFromBlockCenter(coordinates);
                }
            }
        } // if (partner != comm->getRank())
        if(settings.debugMode){
            std::chrono::duration<double> colorElapTime = std::chrono::steady_clock::now() - startColor;
            std::cout << "PE " << comm->getRank() << " finished round " << color << " with gain " << gainThisRound \
                << " in time " << colorElapTime.count() <<std::endl;
        }
    }//for color

    comm->synchronize();

    scai::dmemo::DistributionPtr sameDist = scai::dmemo::generalDistributionUnchecked(globalN, input.getRowDistributionPtr()->ownedGlobalIndexes(), comm);
    input = CSRSparseMatrix<ValueType>(sameDist, input.getLocalStorage());
    part.swap(part.getLocalValues(), sameDist);
    origin.swap(origin.getLocalValues(), sameDist);

    if (settings.useGeometricTieBreaking) {
        for (IndexType d = 0; d < settings.dimensions; d++) {
            coordinates[d].swap(coordinates[d].getLocalValues(), sameDist);
        }
    }

    if (nodesWeighted) {
        nodeWeights.swap(nodeWeights.getLocalValues(), sameDist);
    }

    for (IndexType color = 0; color < gainPerRound.size(); color++) {
        gainPerRound[color] = comm->sum(gainPerRound[color]) / 2;
    }

    return gainPerRound;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType ITI::LocalRefinement<IndexType, ValueType>::twoWayLocalFM(
    const CSRSparseMatrix<ValueType> &input,
    const CSRStorage<ValueType> &haloStorage,
    const scai::dmemo::HaloExchangePlan &matrixHalo,
    const std::vector<IndexType>& borderRegionIDs,
    const std::vector<ValueType>& nodeWeights,
    std::vector<bool>& assignedToSecondBlock,
    const std::pair<IndexType, IndexType> blockCapacities,
    std::pair<IndexType, IndexType>& blockSizes,
    const std::vector<ValueType>& tieBreakingKeys,
    Settings settings) {

    SCAI_REGION( "LocalRefinement.twoWayLocalFM" )

    IndexType magicStoppingAfterNoGainRounds;
    if (settings.stopAfterNoGainRounds > 0) {
        magicStoppingAfterNoGainRounds = settings.stopAfterNoGainRounds;
    } else {
        magicStoppingAfterNoGainRounds = borderRegionIDs.size();
    }

    assert(blockCapacities.first == blockCapacities.second);
    const bool nodesWeighted = (nodeWeights.size() != 0);
    //const bool edgesWeighted = nodesWeighted;//TODO: adapt this, change interface
    const bool edgesWeighted = ( scai::utilskernel::HArrayUtils::max(input.getLocalStorage().getValues()) !=1 );

    if (edgesWeighted) {
        ValueType maxWeight = scai::utilskernel::HArrayUtils::max(input.getLocalStorage().getValues());
        if (maxWeight == 0) {
            throw std::runtime_error("Edges were given as weighted, but maximum weight is zero.");
        }
    }

    const bool gainOverBalance = settings.gainOverBalance;

    if (blockSizes.first >= blockCapacities.first && blockSizes.second >= blockCapacities.second) {
        //cannot move any nodes, all blocks are overloaded already.
        return 0;
    }

    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();

    //the size of this border region
    const IndexType veryLocalN = borderRegionIDs.size();
    assert(tieBreakingKeys.size() == veryLocalN);
    if (nodesWeighted) {
        assert(nodeWeights.size() == veryLocalN);
    }

    //TODO: not used variable
    //const IndexType firstBlockSize = std::distance(assignedToSecondBlock.begin(), std::lower_bound(assignedToSecondBlock.begin(), assignedToSecondBlock.end(), 1));

    //this map provides an index from 0 to b-1 for each of the b indices in borderRegionIDs
    //globalToVeryLocal[borderRegionIDs[i]] = i
    std::map<IndexType, IndexType> globalToVeryLocal;

    for (IndexType i = 0; i < veryLocalN; i++) {
        IndexType globalIndex = borderRegionIDs[i];
        globalToVeryLocal[globalIndex] = i;
    }

    assert(globalToVeryLocal.size() == veryLocalN);

    auto isInBorderRegion = [&](IndexType globalID) {
        return globalToVeryLocal.count(globalID) > 0;
    };

    /*
     * This lambda computes the initial gain of each node.
     * Inlining to reduce the overhead of read access locks didn't give any performance benefit.
     */
    auto computeInitialGain = [&](IndexType veryLocalID) {
        SCAI_REGION( "LocalRefinement.twoWayLocalFM.computeGain" )
        ValueType result = 0;
        IndexType globalID = borderRegionIDs[veryLocalID];
        IndexType isInSecondBlock = assignedToSecondBlock[veryLocalID];
        /*
         * neighborhood information is either in local matrix or halo.
         */
        const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
        const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2Local(globalID) : matrixHalo.global2Halo(globalID);
        assert(localID != scai::invalidIndex);

        //get locks
        const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
        const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());
        const scai::hmemo::ReadAccess<ValueType> values(storage.getValues());

        //get indices for CSR structure
        const IndexType beginCols = localIa[localID];
        const IndexType endCols = localIa[localID+1];

        for (IndexType j = beginCols; j < endCols; j++) {
            IndexType globalNeighbor = localJa[j];
            if (globalNeighbor == globalID) {
                //self-loop, not counted
                continue;
            }

            const ValueType weight = edgesWeighted ? values[j] : 1;

            if (inputDist->isLocal(globalNeighbor)) {
                //neighbor is in local block,
                result += isInSecondBlock ? weight : -weight;
            } else if (matrixHalo.global2Halo(globalNeighbor) != scai::invalidIndex) {
                //neighbor is in partner block
                result += !isInSecondBlock ? weight : -weight;
            } else {
                //neighbor is from somewhere else, no effect on gain.
            }
        }

        return result;
    };

    /*
     * construct and fill gain table and priority queues. Since only one target block is possible, gain table is one-dimensional.
     * One could probably optimize this by choosing the PrioQueueForInts, but it only supports positive keys and requires some adaptations
     */
    PrioQueue<std::pair<IndexType, ValueType>, IndexType> firstQueue(veryLocalN);
    PrioQueue<std::pair<IndexType, ValueType>, IndexType> secondQueue(veryLocalN);

    std::vector<ValueType> gain(veryLocalN);

    for (IndexType i = 0; i < veryLocalN; i++) {
        gain[i] = computeInitialGain(i);
        const ValueType tieBreakingKey = tieBreakingKeys[i];
        if (assignedToSecondBlock[i]) {
            //the queues only support extractMin, since we want the maximum gain each round, we multiply it with -1
            secondQueue.insert(std::make_pair(-gain[i], tieBreakingKey), i);
        } else {
            firstQueue.insert(std::make_pair(-gain[i], tieBreakingKey), i);
        }
    }

    //whether a node was already moved
    std::vector<bool> moved(veryLocalN, false);
    //which node was transfered in each round
    std::vector<IndexType> transfers;
    transfers.reserve(veryLocalN);

    ValueType gainSum = 0;
    std::vector<ValueType> gainSumList, sizeList;
    gainSumList.reserve(veryLocalN);
    sizeList.reserve(veryLocalN);

    IndexType iter = 0;
    IndexType iterWithoutGain = 0;
    while (firstQueue.size() + secondQueue.size() > 0 && iterWithoutGain < magicStoppingAfterNoGainRounds) {
        SCAI_REGION( "LocalRefinement.twoWayLocalFM.queueloop" )
        IndexType bestQueueIndex = -1;

        assert(firstQueue.size() + secondQueue.size() == veryLocalN - iter);
        /*
         * first check break situations
         */
        if ((firstQueue.size() == 0 && blockSizes.first >= blockCapacities.first)
                ||	(secondQueue.size() == 0 && blockSizes.second >= blockCapacities.second)) {
            //cannot move any nodes
            break;
        }

        /*
         * now check situations where we have no choice, for example because one queue is empty or one block is already full
         */
        if (firstQueue.size() == 0) {
            assert(blockSizes.first < blockCapacities.first);
            bestQueueIndex = 1;
        } else if (secondQueue.size() == 0) {
            assert(blockSizes.second < blockCapacities.second);
            bestQueueIndex = 0;
        } else if (blockSizes.first >= blockCapacities.first) {
            bestQueueIndex = 1;
        } else if (blockSizes.second >= blockCapacities.second) {
            bestQueueIndex = 0;
        }
        else {
            /*
             * now the rest
             */
            SCAI_REGION( "LocalRefinement.twoWayLocalFM.queueloop.queueselection" )
            std::pair<IndexType, IndexType> firstComparisonPair;
            std::pair<IndexType, IndexType> secondComparisonPair;

            if (gainOverBalance) {
                firstComparisonPair = {-firstQueue.inspectMin().first.first, blockSizes.first};
                firstComparisonPair = {-secondQueue.inspectMin().first.first, blockSizes.second};
            } else {
                firstComparisonPair = {blockSizes.first, -firstQueue.inspectMin().first.first};
                secondComparisonPair = {blockSizes.second, -secondQueue.inspectMin().first.first};
            }

            if (firstComparisonPair > secondComparisonPair) {
                bestQueueIndex = 0;
            } else if (secondComparisonPair > firstComparisonPair) {
                bestQueueIndex = 1;
            } else {
                //tie, break randomly
                bestQueueIndex = (double(rand()) / RAND_MAX < 0.5);
            }

            assert(bestQueueIndex == 0 || bestQueueIndex == 1);
        }

        PrioQueue<std::pair<IndexType, ValueType>, IndexType>& currentQueue = bestQueueIndex == 0 ? firstQueue : secondQueue;

        //Now, we have selected a Queue. Get best vertex and gain
        IndexType veryLocalID;
        std::tie(std::ignore, veryLocalID) = currentQueue.extractMin();
        ValueType topGain = gain[veryLocalID];
        IndexType topVertex = borderRegionIDs[veryLocalID];

        //here one could assert some consistency

        if (topGain > 0) iterWithoutGain = 0;
        else iterWithoutGain++;

        //move node
        transfers.push_back(veryLocalID);
        assignedToSecondBlock[veryLocalID] = !bestQueueIndex;
        moved[veryLocalID] = true;

        //update gain sum
        gainSum += topGain;
        gainSumList.push_back(gainSum);

        //update sizes
        const ValueType nodeWeight = nodesWeighted ? nodeWeights[veryLocalID] : 1;
        blockSizes.first += bestQueueIndex == 0 ? -nodeWeight : nodeWeight;
        blockSizes.second += bestQueueIndex == 0 ? nodeWeight : -nodeWeight;
        sizeList.push_back(std::max(blockSizes.first, blockSizes.second));

        /*
         * update gains of neighbors
         */
        SCAI_REGION_START("LocalRefinement.twoWayLocalFM.queueloop.acquireLocks")
        const CSRStorage<ValueType>& storage = inputDist->isLocal(topVertex) ? input.getLocalStorage() : haloStorage;
        const IndexType localID = inputDist->isLocal(topVertex) ? inputDist->global2Local(topVertex) : matrixHalo.global2Halo(topVertex);
        assert(localID != scai::invalidIndex);

        const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
        const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());
        const scai::hmemo::ReadAccess<ValueType> localValues(storage.getValues());

        const IndexType beginCols = localIa[localID];
        const IndexType endCols = localIa[localID+1];
        SCAI_REGION_END("LocalRefinement.twoWayLocalFM.queueloop.acquireLocks")

        for (IndexType j = beginCols; j < endCols; j++) {
            SCAI_REGION( "LocalRefinement.twoWayLocalFM.queueloop.gainupdate" )
            IndexType neighbor = localJa[j];
            //here we only need to update gain of neighbors in border regions
            if (isInBorderRegion(neighbor)) {
                IndexType veryLocalNeighborID = globalToVeryLocal.at(neighbor);
                if (moved[veryLocalNeighborID]) {
                    continue;
                }
                bool wasInSameBlock = (bestQueueIndex == assignedToSecondBlock[veryLocalNeighborID]);
                const ValueType edgeWeight = edgesWeighted ? localValues[j] : 1;
                const ValueType oldGain = gain[veryLocalNeighborID];
                //gain change is twice the value of the affected edge. Direction depends on block assignment.
                gain[veryLocalNeighborID] = oldGain + 2*(2*wasInSameBlock - 1)*edgeWeight;

                const ValueType tieBreakingKey = tieBreakingKeys[veryLocalNeighborID];
                const std::pair<IndexType, ValueType> oldKey = std::make_pair(-oldGain, tieBreakingKey);
                const std::pair<IndexType, ValueType> newKey = std::make_pair(-gain[veryLocalNeighborID], tieBreakingKey);

                if (assignedToSecondBlock[veryLocalNeighborID]) {
                    secondQueue.updateKey(oldKey, newKey, veryLocalNeighborID);
                } else {
                    firstQueue.updateKey(oldKey, newKey, veryLocalNeighborID);
                }
            }
        }
        iter++;
    }

    /*
    * now find best partition among those tested
    */
    ValueType maxGain = 0;
    const IndexType testedNodes = gainSumList.size();
    if (testedNodes == 0) return 0;

    SCAI_REGION_START( "LocalRefinement.twoWayLocalFM.recoverBestCut" )
    IndexType maxIndex = -1;

    for (IndexType i = 0; i < testedNodes; i++) {
        if (gainSumList[i] > maxGain && sizeList[i] <= blockCapacities.first) {
            maxIndex = i;
            maxGain = gainSumList[i];
        }
    }
    assert(testedNodes >= maxIndex);
    assert(testedNodes-1 < transfers.size());

    /*
     * apply partition modifications in reverse until best is recovered
     */
    [[maybe_unused]] const IndexType globalN = inputDist->getGlobalSize();

    for (int i = testedNodes-1; i > maxIndex; i--) {
        assert(transfers[i] < globalN);
        IndexType veryLocalID = transfers[i];
        bool previousBlock = !assignedToSecondBlock[veryLocalID];

        //apply movement in reverse
        assignedToSecondBlock[veryLocalID] = previousBlock;
        const ValueType nodeWeight = nodesWeighted ? nodeWeights[veryLocalID] : 1;
        blockSizes.first += previousBlock == 0 ? nodeWeight : -nodeWeight;
        blockSizes.second += previousBlock == 0 ? -nodeWeight : nodeWeight;

    }
    SCAI_REGION_END( "LocalRefinement.twoWayLocalFM.recoverBestCut" )

    return maxGain;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> ITI::LocalRefinement<IndexType, ValueType>::twoWayLocalDiffusion(
    const CSRSparseMatrix<ValueType> &input,
    const CSRStorage<ValueType> &haloStorage,
    const scai::dmemo::HaloExchangePlan &matrixHalo,
    const std::vector<IndexType>& borderRegionIDs,
    std::pair<IndexType,
    IndexType> secondRoundMarkers,
    const std::vector<bool>& assignedToSecondBlock,
    Settings settings) {

    SCAI_REGION( "LocalRefinement.twoWayLocalDiffusion" )
    //settings and constants
    const IndexType magicNumberDiffusionSteps = settings.diffusionRounds;
    //const ValueType degreeEstimate = ValueType(haloStorage.getNumValues()) / matrixHalo.getHaloSize();

    const ValueType magicNumberDiffusionLoad = 1;
    const IndexType veryLocalN = borderRegionIDs.size();

    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

    const IndexType firstBlockSize = std::distance(assignedToSecondBlock.begin(), std::lower_bound(assignedToSecondBlock.begin(), assignedToSecondBlock.end(), 1));
    const IndexType secondBlockSize = veryLocalN - firstBlockSize;
    const ValueType initialLoadPerNodeInFirstBlock = magicNumberDiffusionLoad / firstBlockSize;
    const ValueType initialLoadPerNodeInSecondBlock = -magicNumberDiffusionLoad / secondBlockSize;

    //assign initial diffusion load
    std::vector<ValueType> result(veryLocalN);
    for (IndexType i = 0; i < veryLocalN; i++) {
        result[i] = assignedToSecondBlock[i] ? initialLoadPerNodeInSecondBlock : initialLoadPerNodeInFirstBlock;
    }

    //mark nodes in border as active, rest as inactive
    std::vector<bool> active(veryLocalN, false);
    for (IndexType i = 0; i < secondRoundMarkers.first; i++) {
        active[i] = true;
    }
    for (IndexType i = firstBlockSize; i < firstBlockSize+secondRoundMarkers.second; i++) {
        active[i] = true;
    }

    //this map provides an index from 0 to b-1 for each of the b indices in borderRegionIDs
    //globalToVeryLocal[borderRegionIDs[i]] = i
    std::map<IndexType, IndexType> globalToVeryLocal;

    for (IndexType i = 0; i < veryLocalN; i++) {
        IndexType globalIndex = borderRegionIDs[i];
        globalToVeryLocal[globalIndex] = i;
    }

    IndexType maxDegree = 0;
    {
        const scai::hmemo::ReadAccess<IndexType> localIa(input.getLocalStorage().getIA());
        for (IndexType i = 0; i < firstBlockSize; i++) {
            const IndexType localI = inputDist->global2Local(borderRegionIDs[i]);
            if (localIa[localI+1]-localIa[localI] > maxDegree) maxDegree = localIa[localI+1]-localIa[localI];
        }
    }
    {
        const scai::hmemo::ReadAccess<IndexType> localIa(haloStorage.getIA());
        for (IndexType i = 0; i < localIa.size()-1; i++) {
            if (localIa[i+1]-localIa[i] > maxDegree) maxDegree = localIa[i+1]-localIa[i];
        }
    }


    const ValueType magicNumberAlpha = 1.0/(maxDegree+1);

    //assert that all indices were unique
    assert(globalToVeryLocal.size() == veryLocalN);

    auto isInBorderRegion = [&](IndexType globalID) {
        return globalToVeryLocal.count(globalID) > 0;
    };

    //perform diffusion
    for (IndexType round = 0; round < magicNumberDiffusionSteps; round++) {
        std::vector<ValueType> nextDiffusionValues(result);
        std::vector<bool> nextActive(veryLocalN, false);

        //diffusion update
        for (IndexType i = 0; i < veryLocalN; i++) {
            if (!active[i]) {
                continue;
            }
            nextActive[i] = active[i];

            const ValueType oldDiffusionValue = result[i];
            const IndexType globalID = borderRegionIDs[i];
            const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
            const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2Local(globalID) : matrixHalo.global2Halo(globalID);
            assert(localID != scai::invalidIndex);

            const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
            const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());

            const IndexType beginCols = localIa[localID];
            const IndexType endCols = localIa[localID+1];

            double delta = 0.0;
            for (IndexType j = beginCols; j < endCols; j++) {
                const IndexType neighbor = localJa[j];
                if (isInBorderRegion(neighbor)) {
                    const IndexType veryLocalNeighbor = globalToVeryLocal.at(neighbor);
                    const ValueType difference = result[veryLocalNeighbor] - oldDiffusionValue;
                    delta += difference;
                    if (difference != 0 && !active[veryLocalNeighbor]) {
                        throw std::logic_error("Round " + std::to_string(round) + ": load["+std::to_string(i)+"]="+std::to_string(oldDiffusionValue)
                                               +", load["+std::to_string(veryLocalNeighbor)+"]="+std::to_string(result[veryLocalNeighbor])
                                               +", but "+std::to_string(veryLocalNeighbor)+" marked as inactive.");
                    }
                    nextActive[veryLocalNeighbor] = true;
                }
            }

            nextDiffusionValues[i] = oldDiffusionValue + delta * magicNumberAlpha;
            assert (std::abs(nextDiffusionValues[i]) <= magicNumberDiffusionLoad);
        }

        result.swap(nextDiffusionValues);
        active.swap(nextActive);
    }

    return result;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
void ITI::LocalRefinement<IndexType, ValueType>::redistributeFromHalo(
    DenseVector<T>& input, scai::dmemo::DistributionPtr newDist,
    const scai::dmemo::HaloExchangePlan& halo,
    const scai::hmemo::HArray<T>& haloData) {

    SCAI_REGION( "LocalRefinement.redistributeFromHalo.Vector" )

    scai::dmemo::DistributionPtr oldDist = input.getDistributionPtr();
    const IndexType newLocalN = newDist->getLocalSize();
    const IndexType oldLocalN = oldDist->getLocalSize();

    scai::hmemo::HArray<IndexType> oldIndices;
    oldDist->getOwnedIndexes(oldIndices);

    scai::hmemo::HArray<IndexType> newIndices;
    newDist->getOwnedIndexes(newIndices);

    scai::hmemo::ReadAccess<IndexType> oldAcc(oldIndices);
    scai::hmemo::ReadAccess<IndexType> newAcc(newIndices);

    IndexType oldLocalI = 0;

    HArray<T> newLocalValues(newLocalN);

    {
        scai::hmemo::ReadAccess<T> rOldLocalValues(input.getLocalValues());
        scai::hmemo::ReadAccess<T> rHaloData(haloData);

        scai::hmemo::WriteOnlyAccess<T> wNewLocalValues(newLocalValues);
        for (IndexType i = 0; i < newLocalN; i++) {
            const IndexType globalI = newAcc[i];
            while(oldLocalI < oldLocalN && oldAcc[oldLocalI] < globalI) oldLocalI++;
            //const IndexType oldLocalI = oldDist->global2Local(globalI);
            if (oldLocalI < oldLocalN && oldAcc[oldLocalI] == globalI) {
                wNewLocalValues[i] = rOldLocalValues[oldLocalI];
            } else {
                const IndexType localI = halo.global2Halo(globalI);
                assert(localI != scai::invalidIndex);
                wNewLocalValues[i] = rHaloData[localI];
            }
        }
    }

    input.swap(newLocalValues, newDist);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void ITI::LocalRefinement<IndexType, ValueType>::redistributeFromHalo(
    CSRSparseMatrix<ValueType>& matrix,
    scai::dmemo::DistributionPtr newDist,
    const scai::dmemo::HaloExchangePlan& halo,
    const CSRStorage<ValueType>& haloStorage) {

    SCAI_REGION( "LocalRefinement.redistributeFromHalo" )

    scai::dmemo::DistributionPtr oldDist = matrix.getRowDistributionPtr();

    const IndexType sourceNumRows = oldDist->getLocalSize();
    const IndexType targetNumRows = newDist->getLocalSize();

    const IndexType globalN = oldDist->getGlobalSize();
    if (newDist->getGlobalSize() != globalN) {
        throw std::runtime_error("Old Distribution has " + std::to_string(globalN) + " values, new distribution has " + std::to_string(newDist->getGlobalSize()));
    }

    scai::hmemo::HArray<IndexType> targetIA(targetNumRows+1);
    scai::hmemo::HArray<IndexType> targetJA;
    scai::hmemo::HArray<ValueType> targetValues;

    const CSRStorage<ValueType>& localStorage = matrix.getLocalStorage();

    // ILLEGAL: matrix.setDistributionPtr(newDist);

    scai::hmemo::HArray<IndexType> sourceSizes;
    {
        SCAI_REGION( "LocalRefinement.redistributeFromHalo.sourceSizes" )
        scai::hmemo::ReadAccess<IndexType> sourceIA(localStorage.getIA());
        scai::hmemo::WriteOnlyAccess<IndexType> wSourceSizes( sourceSizes, sourceNumRows );
        scai::sparsekernel::OpenMPCSRUtils::offsets2sizes( wSourceSizes.get(), sourceIA.get(), sourceNumRows );
    }

    scai::hmemo::HArray<IndexType> haloSizes;
    {
        SCAI_REGION( "LocalRefinement.redistributeFromHalo.haloSizes" )
        scai::hmemo::WriteOnlyAccess<IndexType> wHaloSizes( haloSizes, halo.getHaloSize() );
        scai::hmemo::ReadAccess<IndexType> rHaloIA( haloStorage.getIA() );
        scai::sparsekernel::OpenMPCSRUtils::offsets2sizes( wHaloSizes.get(), rHaloIA.get(), halo.getHaloSize() );
    }

    std::vector<IndexType> localTargetIndices;
    std::vector<IndexType> localSourceIndices;
    std::vector<IndexType> localHaloIndices;
    std::vector<IndexType> additionalLocalNodes;
    localTargetIndices.reserve(std::min(targetNumRows, sourceNumRows));
    localSourceIndices.reserve(std::min(targetNumRows, sourceNumRows));
    localHaloIndices.reserve(std::min(targetNumRows, halo.getHaloSize()));
    additionalLocalNodes.reserve(std::min(targetNumRows, halo.getHaloSize()));
    IndexType numValues = 0;
    {
        SCAI_REGION( "LocalRefinement.redistributeFromHalo.targetIA" )
        scai::hmemo::ReadAccess<IndexType> rSourceSizes(sourceSizes);
        scai::hmemo::ReadAccess<IndexType> rHaloSizes(haloSizes);
        scai::hmemo::WriteAccess<IndexType> wTargetIA( targetIA );


        scai::hmemo::HArray<IndexType> oldIndices;
        oldDist->getOwnedIndexes(oldIndices);

        scai::hmemo::HArray<IndexType> newIndices;
        newDist->getOwnedIndexes(newIndices);

        scai::hmemo::ReadAccess<IndexType> oldAcc(oldIndices);
        scai::hmemo::ReadAccess<IndexType> newAcc(newIndices);

        IndexType oldLocalIndex = 0;

        for (IndexType i = 0; i < targetNumRows; i++) {
            IndexType newGlobalIndex = newAcc[i];
            while(oldLocalIndex < sourceNumRows && oldAcc[oldLocalIndex] < newGlobalIndex) oldLocalIndex++;
            IndexType size;
            if (oldLocalIndex < sourceNumRows && oldAcc[oldLocalIndex] == newGlobalIndex) {
                localTargetIndices.push_back(i);
                localSourceIndices.push_back(oldLocalIndex);
                size = rSourceSizes[oldLocalIndex];
            } else {
                additionalLocalNodes.push_back(i);
                const IndexType haloIndex = halo.global2Halo(newGlobalIndex);
                assert(haloIndex != scai::invalidIndex);
                localHaloIndices.push_back(haloIndex);
                size = rHaloSizes[haloIndex];
            }
            wTargetIA[i] = size;
            numValues += size;
        }
        {
            SCAI_REGION( "LocalRefinement.redistributeFromHalo.targetIA.sizes2offsets" )
            scai::sparsekernel::OpenMPCSRUtils::sizes2offsets( wTargetIA.get(), targetNumRows );

            //allocate
            scai::hmemo::WriteOnlyAccess<IndexType> wTargetJA( targetJA, numValues );
            scai::hmemo::WriteOnlyAccess<ValueType> wTargetValues( targetValues, numValues );
        }
    }

    scai::hmemo::ReadAccess<IndexType> rTargetIA(targetIA);
    assert(rTargetIA.size() == targetNumRows + 1);
    IndexType numLocalIndices = localTargetIndices.size();
    IndexType numHaloIndices = localHaloIndices.size();

    for (IndexType i = 0; i < targetNumRows; i++) {
        assert(rTargetIA[i] <= rTargetIA[i+1]);
        assert(rTargetIA[i] <= numValues);
        //WARNING: the assertion was as below (added '=') but it failed when the last row was empty
        // and rTargetIA[i] = rTargetIA[i+1]
        //assert(rTargetIA[i] < numValues);
    }
    rTargetIA.release();
    {
        SCAI_REGION( "LocalRefinement.redistributeFromHalo.copy" )
        //copying JA array from local matrix and halo
        scai::utilskernel::TransferUtils::copyV( targetJA, targetIA, HArray<IndexType>(numLocalIndices, localTargetIndices.data()), localStorage.getJA(), localStorage.getIA(), HArray<IndexType>(numLocalIndices, localSourceIndices.data()) );
        scai::utilskernel::TransferUtils::copyV( targetJA, targetIA, HArray<IndexType>(additionalLocalNodes.size(), additionalLocalNodes.data()), haloStorage.getJA(), haloStorage.getIA(), HArray<IndexType>(numHaloIndices, localHaloIndices.data()) );

        //copying Values array from local matrix and halo
        scai::utilskernel::TransferUtils::copyV( targetValues, targetIA, HArray<IndexType>(numLocalIndices, localTargetIndices.data()), localStorage.getValues(), localStorage.getIA(), HArray<IndexType>(numLocalIndices, localSourceIndices.data()) );
        scai::utilskernel::TransferUtils::copyV( targetValues, targetIA, HArray<IndexType>(additionalLocalNodes.size(), additionalLocalNodes.data()), haloStorage.getValues(), haloStorage.getIA(), HArray<IndexType>(numHaloIndices, localHaloIndices.data()) );
    }

    {
        SCAI_REGION( "LocalRefinement.redistributeFromHalo.setCSRData" )
        //setting CSR data
        //matrix.getLocalStorage().setCSRDataSwap(targetNumRows, globalN, numValues, targetIA, targetJA, targetValues, scai::hmemo::ContextPtr());
        // ThomasBrandes: optimize by std::move(targetIA), std::move(targetJA), std::move(targetValues) ??
        // ThomasBrandes: col distribution is replicated !!!
        matrix = CSRSparseMatrix<ValueType>(
                     newDist,
                     CSRStorage<ValueType>( targetNumRows, globalN, std::move(targetIA), std::move(targetJA), std::move(targetValues) )
                 );
    }
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>, std::vector<IndexType>> ITI::LocalRefinement<IndexType, ValueType>::getInterfaceNodes(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const std::vector<IndexType>& nodesWithNonLocalNeighbors, IndexType otherBlock, IndexType minBorderNodes) {

    SCAI_REGION( "LocalRefinement.getInterfaceNodes" )
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

    //const IndexType n = inputDist->getGlobalSize();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType thisBlock = comm->getRank();

    if (partDist->getLocalSize() != localN) {
        throw std::runtime_error("Partition has " + std::to_string(partDist->getLocalSize()) + " local nodes, but matrix has " + std::to_string(localN) + ".");
    }

    if (otherBlock > comm->getSize()) {
        throw std::runtime_error("Currently only implemented with one block per process, block " + std::to_string(thisBlock) + " invalid for " + std::to_string(comm->getSize()) + " processes.");
    }

    if (thisBlock == otherBlock) {
        throw std::runtime_error("Block IDs must be different.");
    }

    if (minBorderNodes <= 0) {
        throw std::runtime_error("Minimum number of nodes must be positive");
    }

    scai::hmemo::HArray<IndexType> localData = part.getLocalValues();

    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    /*
     * send nodes with non-local neighbors to partner process.
     * here we assume a 1-to-1-mapping of blocks to processes and a symmetric matrix
     */
    std::unordered_set<IndexType> foreignNodes;
    {
        SCAI_REGION( "LocalRefinement.getInterfaceNodes.communication" )
        IndexType swapField[1];
        {
            SCAI_REGION( "LocalRefinement.getInterfaceNodes.communication.syncswap" );
            //swap number of local border nodes
            swapField[0] = nodesWithNonLocalNeighbors.size();
            comm->swap(swapField, 1, otherBlock);
        }
        const IndexType otherSize = swapField[0];
        const IndexType swapLength = std::max(otherSize, IndexType(nodesWithNonLocalNeighbors.size()));
        IndexType swapList[swapLength];
        for (IndexType i = 0; i < nodesWithNonLocalNeighbors.size(); i++) {
            swapList[i] = nodesWithNonLocalNeighbors[i];
        }
        comm->swap(swapList, swapLength, otherBlock);

        foreignNodes.reserve(otherSize);

        //the swapList array is only partially filled, the number of received nodes is found in swapField[0]
        for (IndexType i = 0; i < otherSize; i++) {
            foreignNodes.insert(swapList[i]);
        }
    }

    /*
     * check which of the neighbors of our local border nodes are actually the partner's border nodes
     */

    std::vector<IndexType> sourceNodes;

    for (IndexType node : nodesWithNonLocalNeighbors) {
        SCAI_REGION( "LocalRefinement.getInterfaceNodes.getBorderToPartner" )
        IndexType localI = inputDist->global2Local(node);
        assert(localI != scai::invalidIndex);

        for (IndexType j = ia[localI]; j < ia[localI+1]; j++) {
            if (foreignNodes.count(ja[j])> 0) {
                sourceNodes.push_back(node);
                break;
            }
        }
    }

    assert(sourceNodes.size() <= localN);

    return ITI::GraphUtils<IndexType,ValueType>::localMultiSourceBFSWithRoundMarkers( input, sourceNodes, minBorderNodes );
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
IndexType LocalRefinement<IndexType, ValueType>::localBlockSize(const DenseVector<IndexType> &part, IndexType blockID) {
    SCAI_REGION( "LocalRefinement.localBlockSize" )
    IndexType result = 0;
    scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());

    //possibly shorten with std::count(localPart.get(), localPart.get()+localPart.size(), blockID);
    for (IndexType i = 0; i < localPart.size(); i++) {
        if (localPart[i] == blockID) {
            result++;
        }
    }

    return result;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
IndexType ITI::LocalRefinement<IndexType, ValueType>::getDegreeSum(const CSRSparseMatrix<ValueType> &input, const std::vector<IndexType>& nodes) {
    SCAI_REGION( "LocalRefinement.getDegreeSum" )
    const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> localIa(localStorage.getIA());

    IndexType result = 0;

    for (IndexType node : nodes) {
        IndexType localID = input.getRowDistributionPtr()->global2Local(node);
        result += localIa[localID+1] - localIa[localID];
    }
    return result;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> ITI::LocalRefinement<IndexType, ValueType>::distancesFromBlockCenter(const std::vector<DenseVector<ValueType>> &coordinates) {
    SCAI_REGION("ParcoRepart.distanceFromBlockCenter");

    const IndexType localN = coordinates[0].getDistributionPtr()->getLocalSize();
    const IndexType dimensions = coordinates.size();

    std::vector<ValueType> geometricCenter(dimensions);
    for (IndexType dim = 0; dim < dimensions; dim++) {
        const HArray<ValueType>& localValues = coordinates[dim].getLocalValues();
        assert(localValues.size() == localN);
        geometricCenter[dim] = scai::utilskernel::HArrayUtils::sum(localValues) / localN;
    }

    std::vector<ValueType> result(localN);
    for (IndexType i = 0; i < localN; i++) {
        ValueType distanceSquared = 0;
        for (IndexType dim = 0; dim < dimensions; dim++) {
            const ValueType diff = coordinates[dim].getLocalValues()[i] - geometricCenter[dim];
            distanceSquared += diff*diff;
        }
        result[i] = pow(distanceSquared, 0.5);
    }
    return result;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
IndexType ITI::LocalRefinement<IndexType,ValueType>::rebalance(
    const CSRSparseMatrix<ValueType> &graph,
    const std::vector<DenseVector<ValueType>> &coordinates,
    const std::vector<DenseVector<ValueType>> &nodeWeights,
    const std::vector<std::vector<ValueType>> &targetBlockWeights,
    DenseVector<IndexType>& partition,
    const Settings settings,
    const ValueType pointPerCent){

    SCAI_REGION("LocalRefinement.rebalance");
    const scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const IndexType numWeights = nodeWeights.size();
    assert( targetBlockWeights.size()==numWeights);
    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    const IndexType localN = coordinates[0].getLocalValues().size();
    const IndexType globalN = inputDist->getGlobalSize();    
    SCAI_ASSERT_EQ_ERROR( localN, inputDist->getLocalSize(), "Possible distribution mismatch" );
    SCAI_ASSERT_EQ_ERROR( inputDist, partition.getDistributionPtr(), "Distribution mismatch" );

    //
    // convert weights to vector<vector> and get the block sizes
    //

    std::vector<std::vector<ValueType>> nodeWeightsV( numWeights );
    for(IndexType w=0; w<numWeights; w++){
        scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights[w].getLocalValues());
        nodeWeightsV[w] = std::vector<ValueType>(rWeights.get(), rWeights.get()+localN);
    }
    const IndexType numBlocks = settings.numBlocks;
    SCAI_ASSERT_EQ_ERROR( partition.max()+1, numBlocks, "Given number of blocks and partition.max mismatch" );
    assert( targetBlockWeights[0].size()==numBlocks );

    std::vector<std::pair<double,IndexType>> maxImbalancePerBlock = 
        ITI::GraphUtils<IndexType,ValueType>::getMaxImbalancePerBlock(nodeWeightsV, targetBlockWeights, partition);

    //the global weight of each block for each weight
    std::vector<std::vector<ValueType>> blockWeights = ITI::GraphUtils<IndexType,ValueType>::getGlobalBlockWeight( nodeWeightsV, partition );
    assert( blockWeights.size()==numWeights );
    assert( blockWeights[0].size()==numBlocks );

    //
    //get vertices that are on the boundaries up to some depth
    //

    std::vector<IndexType> interfaceNodes;
    std::vector<IndexType> roundMarkers;
    const IndexType minBorderNodes = localN*pointPerCent;

    {
        DenseVector<IndexType> borderNodeFlags = GraphUtils<IndexType, ValueType>::getBorderNodes( graph, partition );
        const scai::hmemo::ReadAccess<IndexType> localBorderFlags (borderNodeFlags.getLocalValues() );
        assert( localBorderFlags.size()==localN );
        std::vector<IndexType> borderNodes;
        //initialize the BFS queue with the border nodes
        for(IndexType i=0; i<localN; i++){
            if(localBorderFlags[i]==1){
                //it is a border node, put it in the queue
                borderNodes.push_back( inputDist->local2Global(i) );
            }
        }

        std::tie( interfaceNodes, roundMarkers ) = 
            ITI::GraphUtils<IndexType,ValueType>::localMultiSourceBFSWithRoundMarkers( graph, borderNodes, minBorderNodes );
    }
PRINT( interfaceNodes.size() << " __ " << roundMarkers.size() << " +_+ " << roundMarkers.back() );

    assert(interfaceNodes.size() <= localN);
    assert( interfaceNodes.size() >= minBorderNodes ||
            interfaceNodes.size() == localN ||
            roundMarkers[roundMarkers.size()-2] == roundMarkers.back()
    );

    //const IndexType numEligibleNodes = interfaceNodes.size();

    //interfaceNodes contain the global IDs of all nodes that are eligible to move to another block
    //roundMarkers has size as many as the BFS rounds and roundMarkers[i] is the size of the i-th round

    //mark eligible to move nodes
    std::vector<bool> isEligible(localN, false);

    for( IndexType node : interfaceNodes){
        const IndexType localI = inputDist->global2Local(node);
        assert(localI<localN);
        isEligible[localI] = true;
    }

//For now consider all sampled nodes; leave similar ideas for later optimizations
/*
//get only the vertices of the first round and sort them

const IndexType firstRoundSize = roundMarkers[1];
std::vector<IndexType> firstRoundNodes(firstRoundSize);
for( IndexType i=0; i<firstRoundSize; i++){
    firstRoundNodes[i] = interfaceNodes[i];
}
*/
//PRINT(comm->getRank() << ": localN " << localN << ", firstRoundSize " << firstRoundSize);

    //local partition
    scai::hmemo::ReadAccess<IndexType> rPart(partition.getLocalValues());
    assert(rPart.size()==localN);
    std::vector<IndexType> localPart( rPart.get(), rPart.get()+localN );
    rPart.release();

    //sort lexicographically: first by the imbalance of the block this point belongs to.
    //  if they are in the same block or the imbalance is the same, sort the worst weight
    auto lexSort = [&](int i, int j)->bool{
        const IndexType blockI = localPart[i];
        const IndexType blockJ = localPart[j];
        //if in the same block or blocks have the same imbalance
        if( blockI==blockJ or maxImbalancePerBlock[blockI].first==maxImbalancePerBlock[blockJ].first ){
            //get the weight that causes the imbalance
            const IndexType badWeight = maxImbalancePerBlock[blockI].second;
            //sort by which vertex is heavier in this weight
            return nodeWeightsV[badWeight][i]>nodeWeightsV[badWeight][j];
        }
        if( maxImbalancePerBlock[blockI].first>maxImbalancePerBlock[blockJ].first){
            return true;
        }else{
            return false;
        }
    };

PRINT( comm->getRank() << ": " << *std::max_element( localPart.begin(), localPart.end() ) );
for( int b=0; b<numBlocks; b++){
     PRINT0( b <<" ** " << maxImbalancePerBlock[b].first << ", for w " << maxImbalancePerBlock[b].second );
}


    //TODO: this sorts ALL local nodes and then ignores non-eligible to move nodes
    //maybe we could only consider the eligible to move nodes from here
    std::vector<IndexType> localIndices( localN );
    std::iota(localIndices.begin(), localIndices.end(), 0);
    std::sort( localIndices.begin(), localIndices.end(), lexSort );

    //get the halo of the partition
    scai::dmemo::HaloExchangePlan partHalo = ITI::GraphUtils<IndexType,ValueType>::buildNeighborHalo(graph);
    scai::hmemo::HArray<IndexType> haloData;
    partHalo.updateHalo( haloData, partition.getLocalValues() , inputDist->getCommunicator() );

    //get read accesses to the matrix data
    const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    //here we store the difference to the block weight caused by each move
    std::vector<std::vector<ValueType>> blockWeightDifference(numWeights, std::vector<ValueType>(numBlocks, 0.0));

    //determine every how many nodes we do a global sum
    const IndexType myBatchSize = localN*settings.batchPercent + 1;
    //pick min across all processors;  this is needed so they all do the global sum together
    IndexType batchSize = comm->min(myBatchSize);
    //this has to be the same for all PE otherwise the global reduce step are not synchronized
    const IndexType numPointsToCheck = comm->min(localN);
PRINT( comm->getRank() << ": numPointsToCheck " << numPointsToCheck << " , batchSize " << batchSize );

    std::vector<bool> hasMoved(localN, false);
    bool meDone = false;
    bool allDone = false;
    IndexType localI = 0;
    IndexType numMoves = 0;

//TODO: if the minimum number of points to check is agreed, can this be turned into a for loop?
// even if we restart? Well, in the end we should use a priority queue
    while( not allDone ){
        const IndexType thisInd = localIndices[localI];
        const IndexType myBlock = localPart[thisInd];
        assert(thisInd<localN);

        //for certain reasons, moving this point is not desirable
        bool tryToMove = true;
        //if this is not in the interface nodes
        if( not isEligible[thisInd] ){
            tryToMove = false;
        }
        //if my blocks is too light do not remove
//TODO: not sure how to handle this case
        if( maxImbalancePerBlock[myBlock].first< -0.08 ){
            tryToMove = false;
        }
        if(hasMoved[thisInd]){//this point has already been moved, do not move it again
            tryToMove = false;
        }

        std::vector<ValueType> myWeights(numWeights);
        for (IndexType w=0; w<numWeights; w++) {
            myWeights[w] = nodeWeightsV[w][thisInd];
        }

        //the effect that the removal of this point will have to its current block
        //these are this block's (the block this point belongs to) weight and imbalance
        //after the removal of this point

        std::vector<double> thisBlockNewImbalances(numWeights);
        std::pair<double,IndexType> thisBlockNewMaxImbalance = std::make_pair( std::numeric_limits<double>::lowest(), -1);

        for (IndexType w=0; w<numWeights; w++) {
            ValueType optWeight = targetBlockWeights[w][myBlock];
            //thisBlockNewImbalances[w] = imbalancesPerBlock[w][myBlock] - myWeights[w]/optWeight;
            ValueType thisBlockNewWeight = blockWeights[w][myBlock] - myWeights[w];
            thisBlockNewImbalances[w] = (thisBlockNewWeight - optWeight)/optWeight;
ValueType thisOldImbalance = (blockWeights[w][myBlock] - optWeight)/optWeight;
SCAI_ASSERT_LE_ERROR( thisBlockNewImbalances[w], thisOldImbalance , " ha?");
            if( thisBlockNewImbalances[w]>thisBlockNewMaxImbalance.first ){
                thisBlockNewMaxImbalance.first = thisBlockNewImbalances[w];
                thisBlockNewMaxImbalance.second = w;
            }
        }
        assert( thisBlockNewMaxImbalance.second!=-1 );
        //if the max imbalance is realized by the same weight as before, then it should be lower
        if( thisBlockNewMaxImbalance.second==maxImbalancePerBlock[myBlock].second ){
            SCAI_ASSERT_LE_ERROR( thisBlockNewMaxImbalance.first, maxImbalancePerBlock[myBlock].first, "Since we remove, imbalance value should be reduced");
        }

        //Get the blocks of all neighbors of this vertex. These are possible blocks
        //to move this vertex to

        std::set<IndexType> possibleBlocks;
        if( tryToMove ){
PRINT0("will try to move point " << thisInd << " from block " << myBlock );
for (IndexType w=0; w<numWeights; w++) {
    PRINT0(myWeights[w]);
}
            const IndexType beginCols = ia[thisInd];
            const IndexType endCols = ia[thisInd+1];
            assert(ja.size() >= endCols);

            for (IndexType j = beginCols; j < endCols; j++) {
                IndexType neighbor = ja[j];
                assert(neighbor >= 0);
                assert(neighbor < globalN);

                IndexType neighborBlock;
                if (inputDist->isLocal(neighbor)) {
                    neighborBlock = localPart[inputDist->global2Local(neighbor)];
                } else {
                    neighborBlock = haloData[partHalo.global2Halo(neighbor)];
                }
                assert(neighborBlock<numBlocks);

                if (neighborBlock != myBlock) {
                    possibleBlocks.insert( neighborBlock);
                }
            }
if( possibleBlocks.size()==0 ) PRINT0( "all neighbors of " << thisInd << " are in the same block" );
        }

        IndexType bestBlock = myBlock;
        std::pair<double,IndexType> bestBlockMaxNewImbalance = std::make_pair( std::numeric_limits<double>::lowest(), -1);
        std::vector<double> bestBlockNewImbalances;

        for( IndexType candidateBlock : possibleBlocks ){
            //
            //if( myBlock==candidateBlock) continue;
            assert( myBlock!=candidateBlock );
            //if candidate block is already too imbalanced
if( maxImbalancePerBlock[candidateBlock].first>settings.epsilon ) continue;
PRINT0("myBlock= " <<  myBlock << ", checking candidate block " << candidateBlock );
            //calculate block weight and imbalance of the new candidate block if we add this point
            std::vector<double> newBlockImbalances(numWeights);
            std::pair<double,IndexType> maxNewImbalanceNewBlock = std::make_pair( std::numeric_limits<double>::lowest(), -1);

            for (IndexType w=0; w<numWeights; w++) {
                ValueType optWeight = targetBlockWeights[w][candidateBlock];
                //will (possibly) add this point to the block, so add its weights
                ValueType candidateBlockNewWeight = blockWeights[w][candidateBlock] + myWeights[w];
                newBlockImbalances[w] = (candidateBlockNewWeight - optWeight)/optWeight;

                if(newBlockImbalances[w]>maxNewImbalanceNewBlock.first){
                    maxNewImbalanceNewBlock.first = newBlockImbalances[w];
                    maxNewImbalanceNewBlock.second = w;
                }
            }

            //the max imbalance of the new block is larger than the previous max imbalance of the same block 
            // since we added a point (with positive weight)
            SCAI_ASSERT_GE_ERROR( maxNewImbalanceNewBlock.first,  maxImbalancePerBlock[candidateBlock].first, "??") ;

            //
            //evaluate if the move was beneficial
            //

            //if this candidate block offers a better imbalance
            if( maxNewImbalanceNewBlock.first>bestBlockMaxNewImbalance.first ){
                //from all the moves that improve the imbalance, keep the one that improves it the most
                //check also if the change in this possible block is less
                bestBlockMaxNewImbalance = maxNewImbalanceNewBlock;
                bestBlock = candidateBlock;
                bestBlockNewImbalances = newBlockImbalances;
            }
        }//for( IndexType candidateBlock : possibleBlocks)
        

        if( bestBlock!=myBlock ){
PRINT0("myBlock= " <<  myBlock << ", best block " << bestBlock );
            assert( bestBlockMaxNewImbalance.second!=-1 );
            //If the best move is not actually good, do not do it
            //This can happen if the candidate block is overweighted in another weight
            if( thisBlockNewMaxImbalance < bestBlockMaxNewImbalance ){
                bestBlock=myBlock;
            }
        }else{
            assert( bestBlockMaxNewImbalance.second==-1 );
        }

        //if we actually found a block that improves the imbalance
        if( bestBlock!=myBlock ){
            //the new max imbalances
            assert( bestBlockNewImbalances.size()==numWeights );
            maxImbalancePerBlock[bestBlock].first = std::numeric_limits<double>::lowest();
            maxImbalancePerBlock[bestBlock].second = -1;
            for( int w=0; w<numWeights; w++ ){
                if(bestBlockNewImbalances[w]>maxImbalancePerBlock[bestBlock].first){
                    maxImbalancePerBlock[bestBlock].first = bestBlockNewImbalances[w];
                    maxImbalancePerBlock[bestBlock].second = w;
                }
            }
            maxImbalancePerBlock[myBlock] = thisBlockNewMaxImbalance;

            localPart[thisInd] = bestBlock;

            //update values of the block weights and imbalances locally
            for (IndexType w=0; w<numWeights; w++) {
                blockWeightDifference[w][myBlock] -= myWeights[w];
                blockWeightDifference[w][bestBlock] += myWeights[w];
//                imbalancesPerBlock[w][myBlock] = thisBlockNewImbalances[w];
//                imbalancesPerBlock[w][bestBlock] = bestBlockNewImbalances[w];
            }

            numMoves++;
            hasMoved[thisInd] = true;
PRINT0("moved point " << thisInd << " from " << myBlock << " to " << bestBlock );
        }
        assert( maxImbalancePerBlock[bestBlock].second!=-1 );
        //else no improvement was achieved

        //global sum needed
        if( (localI+1)%batchSize==0 or meDone ){
            //reset local block max weight imbalances
            std::fill( maxImbalancePerBlock.begin(), maxImbalancePerBlock.end(), 
                std::make_pair( std::numeric_limits<double>::lowest(), -1) );

            for (IndexType w=0; w<numWeights; w++) {
                //sum all the differences for all blocks among PEs
                comm->sumImpl(blockWeightDifference[w].data(), blockWeightDifference[w].data(), numBlocks, scai::common::TypeTraits<ValueType>::stype );
                std::transform( blockWeights[w].begin(), blockWeights[w].end(), blockWeightDifference[w].begin(), blockWeights[w].begin(), std::plus<ValueType>() );
  
                //recalculate imbalances after the new global blocks weights are summed
                for (IndexType b=0; b<numBlocks; b++) {
                    ValueType optWeight = targetBlockWeights[w][b];
                    ValueType imbalancesPerBlock = (ValueType(blockWeights[w][b] - optWeight)/optWeight);
                    if( imbalancesPerBlock>maxImbalancePerBlock[b].first ) {
                        maxImbalancePerBlock[b].first = imbalancesPerBlock;
                        maxImbalancePerBlock[b].second = w;
                    }
                }
                //reset local block weight differences
                std::fill( blockWeightDifference[w].begin(), blockWeightDifference[w].end(), 0.0);
            }
IndexType globMoves = comm->sum(numMoves);
PRINT0("**** after global sum, verticed moved "<< globMoves );
for( int b=0; b<numBlocks; b++){
    PRINT0( b <<" ** " << maxImbalancePerBlock[b].first << ", for w " << maxImbalancePerBlock[b].second );
}
            //TODO: check if resorting local points based on new global weights
            //and restarting would benefit
//Actually, resorting (or a priority queue) is needed as the most imbalanced block or weight changes
/*
            if(thisRun<maxNumRestarts){
                std::sort(indices.begin(), indices.end(), sortFunction );
                //restart
                localI=-1;
                thisRun++;
            }else{
                //increase the batch size
                batchSize = std::min( (IndexType) (batchSize*1.01), (IndexType) localN/1000);
                batchSize = comm->min(batchSize);
            }
*/
        }

        //exit condition
        if( localI<numPointsToCheck-1 ){
            localI++;
        }else{
            meDone=true;
        }

        try{
            allDone = comm->all(meDone);
        }catch(scai::dmemo::MPIException& e){
            //e.addCallStack( std::cout );
            std::cout << e.what() << std::endl <<
                "Probably some PE is in another global operation...." << std::endl;
        }

    }//while

for (IndexType i = 0; i < numWeights; i++) {
    ValueType imba = ITI::GraphUtils<IndexType, ValueType>::computeImbalance(partition, settings.numBlocks, nodeWeights[i], targetBlockWeights[i]);
    PRINT0(i<< " -- " << imba);
}
return numMoves;

}
//---------------------------------------------------------------------------------------


template class LocalRefinement<IndexType, double>;
template class LocalRefinement<IndexType, float>;

} // namespace ITI
