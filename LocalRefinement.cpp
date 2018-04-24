
#include <unordered_set>

#include "LocalRefinement.h"
#include "GraphUtils.h"

using scai::hmemo::HArray;

namespace ITI{

template<typename IndexType, typename ValueType>
std::vector<IndexType> ITI::LocalRefinement<IndexType, ValueType>::distributedFMStep(
    CSRSparseMatrix<ValueType>& input, 
    DenseVector<IndexType>& part,
    std::vector<IndexType>& nodesWithNonLocalNeighbors,
    DenseVector<ValueType> &nodeWeights, 
    std::vector<DenseVector<ValueType>> &coordinates, 
    std::vector<ValueType> &distances,
    DenseVector<IndexType> &origin,
    const std::vector<DenseVector<IndexType>>& communicationScheme,
    Settings settings) {
    
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

    IndexType gainSum = 0;
    std::vector<IndexType> gainPerRound(communicationScheme.size(), 0);

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

	//main loop, one iteration for each color of the graph coloring
	for (IndexType color = 0; color < communicationScheme.size(); color++) {
		SCAI_REGION( "LocalRefinement.distributedFMStep.loop" )

		const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
		const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
		const scai::dmemo::DistributionPtr commDist = communicationScheme[color].getDistributionPtr();

		const IndexType localN = inputDist->getLocalSize();

		if (!communicationScheme[color].getDistributionPtr()->isLocal(comm->getRank())) {
			throw std::runtime_error("Scheme value for " + std::to_string(comm->getRank()) + " must be local.");
		}
		
		scai::hmemo::ReadAccess<IndexType> commAccess(communicationScheme[color].getLocalValues());
		IndexType partner = commAccess[commDist->global2local(comm->getRank())];

		//check symmetry of communication scheme
		assert(partner < comm->getSize());
		if (commDist->isLocal(partner)) {
			IndexType partnerOfPartner = commAccess[commDist->global2local(partner)];
			if (partnerOfPartner != comm->getRank()) {
				throw std::runtime_error("Process " + std::to_string(comm->getRank()) + ": Partner " + std::to_string(partner) + " has partner "
						+ std::to_string(partnerOfPartner) + ".");
			}
		}

		/**
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
			for (IndexType node : nodesWithNonLocalNeighbors) {
				assert(inputDist->isLocal(node));
			}
		}

		IndexType gainThisRound = 0;

		scai::dmemo::Halo graphHalo;
		CSRStorage<ValueType> haloMatrix;
		HArray<ValueType> nodeWeightHaloData;

		if (partner != comm->getRank()) {
			//processor is active this round
                    
			/**
			 * get indices of border nodes with breadth-first search
			 */
			std::vector<IndexType> interfaceNodes;
			std::vector<IndexType> roundMarkers;
			std::tie(interfaceNodes, roundMarkers)= getInterfaceNodes(input, part, nodesWithNonLocalNeighbors, partner, settings.minBorderNodes);

			const IndexType lastRoundMarker = roundMarkers[roundMarkers.size()-1];
			const IndexType secondRoundMarker = roundMarkers[1];

			/**
			 * now swap indices of nodes in border region with partner processor.
			 * For this, first find out the length of the swap array.
			 */

			SCAI_REGION_START( "LocalRefinement.distributedFMStep.loop.prepareSets" )
			//swap size of border region and total block size.
			IndexType blockSize = part.getDistributionPtr()->getLocalSize();

			const ValueType blockWeightSum = scai::utilskernel::HArrayUtils::sum(nodeWeights.getLocalValues());

			IndexType swapField[5];
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
			const IndexType otherBlockWeightSum = swapField[4];

			if (interfaceNodes.size() == 0) {
				if (otherSize != 0) {
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
					distanceSwap[i] = distances[inputDist->global2local(interfaceNodes[i])];
				}
				comm->swap(distanceSwap, swapLength, partner);
			}

			/*
			 * Build Halo to cover border region of other PE.
			 * This uses a special halo builder method that doesn't require communication, since the required and provided indices are already known.
			 */
			IndexType numValues = input.getLocalStorage().getValues().size();
			{
				scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
				scai::hmemo::HArrayRef<IndexType> arrProvidedIndexes( interfaceNodes );
				scai::dmemo::HaloBuilder::build( *inputDist, arrRequiredIndexes, arrProvidedIndexes, partner, graphHalo );
			}

			//all required halo indices are in the halo
			for (IndexType node : requiredHaloIndices) {
				assert(graphHalo.global2halo(node) != scai::invalidIndex);
			}

			/**
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

			/**
			 * If nodes are weighted, exchange Halo for node weights
			 */
			std::vector<ValueType> borderNodeWeights = {};
			if (nodesWeighted) {
				const HArray<ValueType>& localWeights = nodeWeights.getLocalValues();
				assert(localWeights.size() == localN);
				comm->updateHalo( nodeWeightHaloData, localWeights, graphHalo );
				borderNodeWeights.resize(borderRegionSize,-1);
				for (IndexType i = 0; i < borderRegionSize; i++) {
					const IndexType globalI = borderRegionIDs[i];
					const IndexType localI = inputDist->global2local(globalI);
					if (localI != scai::invalidIndex) {
						borderNodeWeights[i] = localWeights[localI];
					} else {
						const IndexType localI = graphHalo.global2halo(globalI);
						assert(localI != scai::invalidIndex);
						borderNodeWeights[i] = nodeWeightHaloData[localI];
					}
					assert(borderNodeWeights[i] >= 0);
				}
			}

			//origin data, for redistribution in uncoarsening step
			scai::hmemo::HArray<IndexType> originData;
			comm->updateHalo(originData, origin.getLocalValues(), graphHalo);

			//block sizes and capacities
			std::pair<IndexType, IndexType> blockSizes = {blockWeightSum, otherBlockWeightSum};
			std::pair<IndexType, IndexType> maxBlockSizes = {maxAllowableBlockSize, maxAllowableBlockSize};

			//second round markers
			std::pair<IndexType, IndexType> secondRoundMarkers = {secondRoundMarker, otherSecondRoundMarker};

			//tie breaking keys
			std::vector<ValueType> tieBreakingKeys(borderRegionSize, 0);

			if (settings.useGeometricTieBreaking) {
				for (IndexType i = 0; i < lastRoundMarker; i++) {
					tieBreakingKeys[i] = -distances[inputDist->global2local(interfaceNodes[i])];
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

			/**
			 * execute FM locally
			 */
			IndexType gain = twoWayLocalFM(input, haloMatrix, graphHalo, borderRegionIDs, borderNodeWeights, assignedToSecondBlock, maxBlockSizes, blockSizes, tieBreakingKeys, settings);

			{
				SCAI_REGION( "LocalRefinement.distributedFMStep.loop.swapFMResults" )
				/**
				 * Communicate achieved gain.
				 * Since only two values are swapped, the tracing results measure the latency and synchronization overhead,
				 * the difference in running times between the two local FM implementations.
				 */
				swapField[0] = gain;
				swapField[1] = otherBlockWeightSum;
				comm->swap(swapField, 2, partner);
			}
			const IndexType otherGain = swapField[0];
			const IndexType otherSecondBlockWeightSum = swapField[1];

			if (otherSecondBlockWeightSum > maxBlockSizes.first) {
				//If a block is too large after the refinement, it is only because it was too large to begin with.
				assert(otherSecondBlockWeightSum <= blockWeightSum);
			}

			if (otherGain <= 0 && gain <= 0) {
				//Oh well. None of the processors managed an improvement. No need to update data structures.

			}else {
				SCAI_REGION_START( "LocalRefinement.distributedFMStep.loop.prepareRedist" )

				gainThisRound = std::max(IndexType(otherGain), IndexType(gain));

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

				/**
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

				/**
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
                        IndexType localI = inputDist->global2local(globalI);
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
				scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, indexTransport, comm));
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

				/**
				 * update local border. This could probably be optimized by only updating the part that could have changed in the last round.
				 */
				{
					SCAI_REGION( "LocalRefinement.distributedFMStep.loop.updateLocalBorder" )
					nodesWithNonLocalNeighbors = GraphUtils::getNodesWithNonLocalNeighbors<IndexType, ValueType>(input, borderCandidates);
				}

				/**
				 * update coordinates and block distances
				 */
				if (settings.useGeometricTieBreaking)
				{
					SCAI_REGION( "LocalRefinement.distributedFMStep.loop.updateBlockDistances" )

					for (IndexType dim = 0; dim < coordinates.size(); dim++) {
						HArray<ValueType>& localCoords = coordinates[dim].getLocalValues();
						HArray<ValueType> haloData;
						comm->updateHalo( haloData, localCoords, graphHalo );
						redistributeFromHalo<ValueType>(coordinates[dim], newDistribution, graphHalo, haloData);
					}

					distances = LocalRefinement<IndexType, ValueType>::distancesFromBlockCenter(coordinates);
				}
			}
		}
	}

	comm->synchronize();
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
    const scai::dmemo::Halo &matrixHalo,
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
	const bool edgesWeighted = nodesWeighted;//TODO: adapt this, change interface

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
	const IndexType globalN = inputDist->getGlobalSize();
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

	auto isInBorderRegion = [&](IndexType globalID){return globalToVeryLocal.count(globalID) > 0;};

	/**
	 * This lambda computes the initial gain of each node.
	 * Inlining to reduce the overhead of read access locks didn't give any performance benefit.
	 */
	auto computeInitialGain = [&](IndexType veryLocalID){
		SCAI_REGION( "LocalRefinement.twoWayLocalFM.computeGain" )
		ValueType result = 0;
		IndexType globalID = borderRegionIDs[veryLocalID];
		IndexType isInSecondBlock = assignedToSecondBlock[veryLocalID];
		/**
		 * neighborhood information is either in local matrix or halo.
		 */
		const CSRStorage<ValueType>& storage = inputDist->isLocal(globalID) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2local(globalID) : matrixHalo.global2halo(globalID);
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

			const IndexType weight = edgesWeighted ? values[j] : 1;

			if (inputDist->isLocal(globalNeighbor)) {
				//neighbor is in local block,
				result += isInSecondBlock ? weight : -weight;
			} else if (matrixHalo.global2halo(globalNeighbor) != scai::invalidIndex) {
				//neighbor is in partner block
				result += !isInSecondBlock ? weight : -weight;
			} else {
				//neighbor is from somewhere else, no effect on gain.
			}
		}

		return result;
	};

	/**
	 * construct and fill gain table and priority queues. Since only one target block is possible, gain table is one-dimensional.
	 * One could probably optimize this by choosing the PrioQueueForInts, but it only supports positive keys and requires some adaptations
	 */
	PrioQueue<std::pair<IndexType, ValueType>, IndexType> firstQueue(veryLocalN);
	PrioQueue<std::pair<IndexType, ValueType>, IndexType> secondQueue(veryLocalN);

	std::vector<IndexType> gain(veryLocalN);

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
	std::vector<IndexType> gainSumList, sizeList;
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
			/**
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

		/**
		 * update gains of neighbors
		 */
		SCAI_REGION_START("LocalRefinement.twoWayLocalFM.queueloop.acquireLocks")
		const CSRStorage<ValueType>& storage = inputDist->isLocal(topVertex) ? input.getLocalStorage() : haloStorage;
		const IndexType localID = inputDist->isLocal(topVertex) ? inputDist->global2local(topVertex) : matrixHalo.global2halo(topVertex);
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

	/**
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

	/**
	 * apply partition modifications in reverse until best is recovered
	 */
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
std::vector<ValueType> ITI::LocalRefinement<IndexType, ValueType>::twoWayLocalDiffusion(const CSRSparseMatrix<ValueType> &input, const CSRStorage<ValueType> &haloStorage,
		const scai::dmemo::Halo &matrixHalo, const std::vector<IndexType>& borderRegionIDs, std::pair<IndexType, IndexType> secondRoundMarkers,
		const std::vector<bool>& assignedToSecondBlock, Settings settings) {

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
			const IndexType localI = inputDist->global2local(borderRegionIDs[i]);
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

	auto isInBorderRegion = [&](IndexType globalID){return globalToVeryLocal.count(globalID) > 0;};

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
			const IndexType localID = inputDist->isLocal(globalID) ? inputDist->global2local(globalID) : matrixHalo.global2halo(globalID);
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
void ITI::LocalRefinement<IndexType, ValueType>::redistributeFromHalo(DenseVector<T>& input, scai::dmemo::DistributionPtr newDist, const scai::dmemo::Halo& halo, const scai::hmemo::HArray<T>& haloData) {
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
			//const IndexType oldLocalI = oldDist->global2local(globalI);
			if (oldLocalI < oldLocalN && oldAcc[oldLocalI] == globalI) {
				wNewLocalValues[i] = rOldLocalValues[oldLocalI];
			} else {
				const IndexType localI = halo.global2halo(globalI);
				assert(localI != scai::invalidIndex);
				wNewLocalValues[i] = rHaloData[localI];
			}
		}
	}

	input.swap(newLocalValues, newDist);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void ITI::LocalRefinement<IndexType, ValueType>::redistributeFromHalo(CSRSparseMatrix<ValueType>& matrix, scai::dmemo::DistributionPtr newDist, const scai::dmemo::Halo& halo, const CSRStorage<ValueType>& haloStorage) {
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
				const IndexType haloIndex = halo.global2halo(newGlobalIndex);
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
		// matrix.getLocalStorage().setCSRDataSwap(targetNumRows, globalN, numValues, targetIA, targetJA, targetValues, scai::hmemo::ContextPtr());
                // ThomasBrandes: optimize by std::move(targetIA), std::move(targetJA), std::move(targetValues) ??
                // ThomasBrandes: col distribution is replicated !!!
                matrix = CSRSparseMatrix<ValueType>( newDist, CSRStorage<ValueType>( targetNumRows, globalN, targetIA, targetJA, targetValues ) );
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
	scai::hmemo::ReadAccess<IndexType> partAccess(localData);
	
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	/**
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

	/**
	 * check which of the neighbors of our local border nodes are actually the partner's border nodes
	 */
	std::vector<IndexType> interfaceNodes;

	for (IndexType node : nodesWithNonLocalNeighbors) {
		SCAI_REGION( "LocalRefinement.getInterfaceNodes.getBorderToPartner" )
		IndexType localI = inputDist->global2local(node);
		assert(localI != scai::invalidIndex);

		for (IndexType j = ia[localI]; j < ia[localI+1]; j++) {
			if (foreignNodes.count(ja[j])> 0) {
				interfaceNodes.push_back(node);
				break;
			}
		}
	}

	assert(interfaceNodes.size() <= localN);

	//keep track of which nodes were added at each BFS round
	std::vector<IndexType> roundMarkers({0});

	/**
	 * now gather buffer zone with breadth-first search
	 */
	{
		SCAI_REGION( "LocalRefinement.getInterfaceNodes.breadthFirstSearch" )
		std::vector<bool> touched(localN, false);

		std::queue<IndexType> bfsQueue;
		for (IndexType node : interfaceNodes) {
			bfsQueue.push(node);
			const IndexType localID = inputDist->global2local(node);
			assert(localID != scai::invalidIndex);
			touched[localID] = true;
		}
		assert(bfsQueue.size() == interfaceNodes.size());
		bool active = true;

		while (active) {
			//if the target number is reached, complete this round and then stop
			if (interfaceNodes.size() >= minBorderNodes || interfaceNodes.size() == roundMarkers.back()) active = false;
			roundMarkers.push_back(interfaceNodes.size());
			std::queue<IndexType> nextQueue;
			while (!bfsQueue.empty()) {
				IndexType nextNode = bfsQueue.front();
				bfsQueue.pop();
				const IndexType localI = inputDist->global2local(nextNode);
				assert(localI != scai::invalidIndex);
				assert(touched[localI]);
				const IndexType beginCols = ia[localI];
				const IndexType endCols = ia[localI+1];

				for (IndexType j = beginCols; j < endCols; j++) {
					IndexType neighbor = ja[j];
					IndexType localNeighbor = inputDist->global2local(neighbor);
					//assume k=p
					if (localNeighbor != scai::invalidIndex && !touched[localNeighbor]) {
						nextQueue.push(neighbor);
						interfaceNodes.push_back(neighbor);
						touched[localNeighbor] = true;
					}
				}
			}
			bfsQueue = nextQueue;
		}
	}

	assert(interfaceNodes.size() <= localN);
	assert(interfaceNodes.size() >= minBorderNodes || interfaceNodes.size() == localN || roundMarkers[roundMarkers.size()-2] == roundMarkers.back());
	return {interfaceNodes, roundMarkers};
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
IndexType LocalRefinement<IndexType, ValueType>::localBlockSize(const DenseVector<IndexType> &part, IndexType blockID) {
	SCAI_REGION( "LocalRefinement.localBlockSize" )
	IndexType result = 0;
	scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());

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
		IndexType localID = input.getRowDistributionPtr()->global2local(node);
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

template class LocalRefinement<IndexType, ValueType>;

} // namespace ITI
