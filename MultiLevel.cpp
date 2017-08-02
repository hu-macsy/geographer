
#include "MultiLevel.h"
#include "GraphUtils.h"

using scai::lama::Scalar;

namespace ITI{
    
template<typename IndexType, typename ValueType>
IndexType ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, DenseVector<ValueType> &nodeWeights, std::vector<DenseVector<ValueType>> &coordinates, Settings settings) {
	
   SCAI_REGION( "MultiLevel.multiLevelStep" );
	scai::dmemo::CommunicatorPtr comm = input.getRowDistributionPtr()->getCommunicatorPtr();
	const IndexType globalN = input.getRowDistributionPtr()->getGlobalSize();

	if (coordinates.size() != settings.dimensions){
		throw std::runtime_error("Dimensions do not agree: vector.size()= " + std::to_string(coordinates.size())  + " != settings.dimensions= " + std::to_string(settings.dimensions) );
	}

	if (!input.getRowDistributionPtr()->isReplicated()) {
		//check whether distributions agree
		const scai::dmemo::Distribution &inputDist = input.getRowDistribution();
		SCAI_ASSERT(  part.getDistributionPtr()->isEqual(inputDist), "distribution mismatch" );
		SCAI_ASSERT(  nodeWeights.getDistributionPtr()->isEqual(inputDist), "distribution mismatch" );
		if (settings.useGeometricTieBreaking) {
			for (IndexType dim = 0; dim < settings.dimensions; dim++) {
				SCAI_ASSERT(  coordinates[dim].getDistributionPtr()->isEqual(inputDist), "distribution mismatch in dimension " << dim );
			}
		}

		//check whether partition agrees with distribution
		scai::hmemo::ReadAccess<IndexType> rLocal(part.getLocalValues());
		for (IndexType i = 0; i < inputDist.getLocalSize(); i++) {
			SCAI_ASSERT(rLocal[i] == comm->getRank(), "block ID " << rLocal[i] << " found on process " << comm->getRank());
		}
	}
        
	if (settings.multiLevelRounds > 0) {
		SCAI_REGION( "MultiLevel.multiLevelStep.recursiveCall" )
		CSRSparseMatrix<ValueType> coarseGraph;
		DenseVector<IndexType> fineToCoarseMap;
		if (comm->getRank() == 0) {
			std::cout << "Beginning coarsening, still " << settings.multiLevelRounds << " levels to go." << std::endl;
		}
		MultiLevel<IndexType, ValueType>::coarsen(input,nodeWeights, coarseGraph, fineToCoarseMap);
		if (comm->getRank() == 0) {
			std::cout << "Coarse graph has " << coarseGraph.getNumRows() << " nodes." << std::endl;
		}

		//project coordinates and partition
		std::vector<DenseVector<ValueType> > coarseCoords(settings.dimensions);
		if (settings.useGeometricTieBreaking) {
			for (IndexType i = 0; i < settings.dimensions; i++) {
				coarseCoords[i] = projectToCoarse(coordinates[i], fineToCoarseMap);
			}
		}

		DenseVector<IndexType> coarsePart = DenseVector<IndexType>(coarseGraph.getRowDistributionPtr(), comm->getRank());

		DenseVector<ValueType> coarseWeights = sumToCoarse(nodeWeights, fineToCoarseMap);

		assert(coarseWeights.sum().Scalar::getValue<IndexType>() == nodeWeights.sum().Scalar::getValue<IndexType>());

		Settings settingscopy(settings);
		settingscopy.multiLevelRounds--;
		// recursive call
		multiLevelStep(coarseGraph, coarsePart, coarseWeights, coarseCoords, settingscopy);

		// uncoarsening/refinement
		scai::dmemo::DistributionPtr projectedFineDist = projectToFine(coarseGraph.getRowDistributionPtr(), fineToCoarseMap);
		assert(projectedFineDist->getGlobalSize() == globalN);
		part = DenseVector<IndexType>(projectedFineDist, comm->getRank());

		if (settings.useGeometricTieBreaking) {
			for (IndexType dim = 0; dim < settings.dimensions; dim++) {
				coordinates[dim].redistribute(projectedFineDist);
			}
		}

		input.redistribute(projectedFineDist, input.getColDistributionPtr());

		nodeWeights.redistribute(projectedFineDist);
	}
 
        // do local refinement
	if (settings.multiLevelRounds % settings.coarseningStepsBetweenRefinement == 0) {
		SCAI_REGION( "MultiLevel.multiLevelStep.localRefinement" )
		scai::lama::CSRSparseMatrix<ValueType> processGraph = GraphUtils::getPEGraph<IndexType, ValueType>(input);

		std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(processGraph);

		std::vector<IndexType> nodesWithNonLocalNeighbors = GraphUtils::getNodesWithNonLocalNeighbors<IndexType, ValueType>(input);

		std::vector<ValueType> distances;
		if (settings.useGeometricTieBreaking) {
			distances = LocalRefinement<IndexType, ValueType>::distancesFromBlockCenter(coordinates);
		}

		IndexType numRefinementRounds = 0;
		IndexType oldCut = 0;

		ValueType gain = 0;
		while (numRefinementRounds == 0 || gain >= settings.minGainForNextRound) {
			std::vector<IndexType> gainPerRound = LocalRefinement<IndexType, ValueType>::distributedFMStep(input, part, nodesWithNonLocalNeighbors, nodeWeights, communicationScheme, coordinates, distances, settings);
			gain = 0;
			for (IndexType roundGain : gainPerRound) gain += roundGain;

			if (settings.skipNoGainColors) {
				IndexType i = 0;
				while (i < gainPerRound.size()) {
					if (gainPerRound[i] == 0) {
						//remove this color from future rounds. This is not terribly efficient, since it causes multiple copies, but all vectors are small and this method isn't called too often.
						communicationScheme.erase(communicationScheme.begin()+i);
						gainPerRound.erase(gainPerRound.begin()+i);
					} else {
						i++;
					}
				}
			}

			ValueType cut = comm->getSize() == 1 ? GraphUtils::computeCut(input, part) : comm->sum(ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(input, true)) / 2;
			SCAI_ASSERT(comm->sum(cut) == comm->getSize()*cut, "Cut sum inconsistency.");
			if (numRefinementRounds > 0) {
				SCAI_ASSERT(gain == oldCut - cut, "Old cut is " << oldCut << ", new cut is " << cut << ", but gain is " << gain);
				assert(gain >= 0);
			}
			if (comm->getRank() == 0) {
				std::cout << "Multilevel round "<< settings.multiLevelRounds <<": After " << numRefinementRounds + 1 << " refinement rounds, cut is " << cut << std::endl;
			}
			oldCut = cut;
			numRefinementRounds++;
		}
	}
}
//--------------------------------------------------------------------------------------- 
 
template<typename IndexType, typename ValueType>
void MultiLevel<IndexType, ValueType>::coarsen(const CSRSparseMatrix<ValueType>& adjM, const DenseVector<ValueType> &nodeWeights,  CSRSparseMatrix<ValueType>& coarseGraph, DenseVector<IndexType>& fineToCoarse, IndexType iterations) {
	SCAI_REGION("MultiLevel.coarsen");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    
    // get local data of the adjacency matrix
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia( localStorage.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localStorage.getJA() );
    scai::hmemo::ReadAccess<ValueType> values( localStorage.getValues() );

    // localN= number of local nodes
    IndexType localN= adjM.getLocalNumRows();
    IndexType globalN = adjM.getNumColumns();
    
    // ia must have size localN+1
    assert(ia.size()-1 == localN );
     
    //get a matching, the returned indices are from 0 to localN
    std::vector<std::pair<IndexType,IndexType>> matching = MultiLevel<IndexType, ValueType>::maxLocalMatching( adjM, nodeWeights );
    
    std::vector<IndexType> localMatchingPartner(localN, -1);

    // number of new local nodes
    IndexType newLocalN = localN - matching.size();

    //sort the matching according to its first element
    std::sort(matching.begin(), matching.end());

    //get new global indices by computing a prefix sum over the preserved nodes
    DenseVector<IndexType> preserved(distPtr, 1);
    {
        scai::hmemo::WriteAccess<IndexType> localPreserved(preserved.getLocalValues());
        
        for (IndexType i = 0; i < matching.size(); i++) {
            assert(matching[i].first != matching[i].second);
            assert(matching[i].first < localN);
            assert(matching[i].second < localN);
            assert(matching[i].first >= 0);
            assert(matching[i].second >= 0);
            localMatchingPartner[matching[i].first] = matching[i].second;
            localMatchingPartner[matching[i].second] = matching[i].first;
            if (matching[i].first < matching[i].second) {
                localPreserved[matching[i].second] = 0;
            } else if (matching[i].second < matching[i].first) {
                localPreserved[matching[i].first] = 0;
            }
        }
    }

    //fill gaps in index list. This might be expensive.
    scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(globalN, comm));
    preserved.redistribute(blockDist);
    fineToCoarse = computeGlobalPrefixSum(preserved, -1);
    const IndexType newGlobalN = fineToCoarse.max().Scalar::getValue<IndexType>() + 1;
    fineToCoarse.redistribute(distPtr);

    //set new indices for contracted nodes
    {
        scai::hmemo::WriteAccess<IndexType> wFineToCoarse(fineToCoarse.getLocalValues());
        assert(wFineToCoarse.size() == localN);
        IndexType nonMatched = 0;
        
        for (IndexType i = 0; i < localN; i++) {
            if (localMatchingPartner[i] < 0) {
                localMatchingPartner[i] = i;
                nonMatched++;
            }
            
            if (localMatchingPartner[i] < i) {
                wFineToCoarse[i] = wFineToCoarse[localMatchingPartner[i]];
            }
        }
        
        for (IndexType i = 0; i < localN; i++) {
            //assert(distPtr->isLocal(wFineToCoarse[i]));
            //assert(distPtr->global2local(wFineToCoarse[i]) < newLocalN);
            assert(localMatchingPartner[i] < localN);
        }
        assert(nonMatched == localN - matching.size()*2);
    }
    assert(fineToCoarse.max().Scalar::getValue<IndexType>() + 1 == newGlobalN);
    assert(newGlobalN <= globalN);
    assert(newGlobalN == comm->sum(newLocalN));

    //build halo of new global indices
    Halo halo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(adjM);
    scai::utilskernel::LArray<IndexType> haloData;
    comm->updateHalo(haloData, fineToCoarse.getLocalValues(), halo);
    
    scai::hmemo::ReadAccess<IndexType> localFineToCoarse(fineToCoarse.getLocalValues());

    //create new coarsened CSR matrix
    scai::hmemo::HArray<IndexType> newIA;
    scai::hmemo::HArray<IndexType> newJA;
    scai::hmemo::HArray<ValueType> newValues;
    
    // this is larger, need to resize afterwards
    IndexType nnzValues = values.size() - matching.size();

    {
        SCAI_REGION("MultiLevel.coarsen.getCSRMatrix");
        // this is larger, need to resize afterwards
        scai::hmemo::WriteOnlyAccess<IndexType> newIAWrite( newIA, newLocalN +1 );
        scai::hmemo::WriteOnlyAccess<IndexType> newJAWrite( newJA, nnzValues);
        scai::hmemo::WriteOnlyAccess<ValueType> newValuesWrite( newValues, nnzValues);
        newIAWrite[0] = 0;
        IndexType iaIndex = 1;
        IndexType jaIndex = 0;
        
        //for all rows before the coarsening
        for(IndexType i=0; i<localN; i++){
            IndexType matchingPartner = localMatchingPartner[i];
            //duplicate code is evil. Maybe use a lambda instead?
            if (matchingPartner >= i) {
            	std::map<IndexType, ValueType> outgoingEdges;

            	//add edges for this node
            	for (IndexType j = ia[i]; j < ia[i+1]; j++) {
            		IndexType oldNeighbor = ja[j];
            		IndexType newGlobalNeighbor;
            		IndexType localID = distPtr->global2local(oldNeighbor);
            		if (localID == matchingPartner) {
            			continue;//no self loops!
            		}
            		if (localID != nIndex) {
            			newGlobalNeighbor = localFineToCoarse[localID];
            		} else {
            			IndexType haloID = halo.global2halo(oldNeighbor);
            			assert(haloID != nIndex);
            			newGlobalNeighbor = haloData[haloID];
            		}
            		//make sure entry exists
            		if (outgoingEdges.count(newGlobalNeighbor) == 0) {
            			outgoingEdges[newGlobalNeighbor] = 0;
            		}
            		outgoingEdges[newGlobalNeighbor] += values[j];
            	}

            	//add edges for matching partner
            	if (matchingPartner > i) {
                    for (IndexType j = ia[matchingPartner]; j < ia[matchingPartner+1]; j++) {
                        IndexType oldNeighbor = ja[j];
                        IndexType localID = distPtr->global2local(oldNeighbor);
                        IndexType newGlobalNeighbor;
                        if (localID == i) {
                            continue;//no self loops!
                        }
                        if (localID != nIndex) {
                            newGlobalNeighbor = localFineToCoarse[localID];
                        } else {
                            IndexType haloID = halo.global2halo(oldNeighbor);
                            assert(haloID != nIndex);
                            newGlobalNeighbor = haloData[haloID];
                        }
                        //make sure entry exists
                        if (outgoingEdges.count(newGlobalNeighbor) == 0) {
                            outgoingEdges[newGlobalNeighbor] = 0;
                        }
                        outgoingEdges[newGlobalNeighbor] += values[j];
                    }
                }
                
                //write new matrix entries. Since entries in std::map are sorted by their keys, we can just iterate over them
                for (std::pair<IndexType, ValueType> entry : outgoingEdges) {
                    assert(jaIndex < newJAWrite.size());
                    newJAWrite[jaIndex] = entry.first;
                    newValuesWrite[jaIndex] = entry.second;
                    jaIndex++;
                }
                assert(iaIndex < newIAWrite.size());
                newIAWrite[iaIndex] = jaIndex;
                iaIndex++;
            }
        }
        
        newJA.resize(jaIndex);
        newValues.resize(jaIndex);
        nnzValues = jaIndex;
    }
    
    //create distribution object for coarse graph
    scai::utilskernel::LArray<IndexType> myGlobalIndices(fineToCoarse.getLocalValues());
    scai::hmemo::WriteAccess<IndexType> wIndices(myGlobalIndices);
    assert(wIndices.size() == localN);
    std::sort(wIndices.get(), wIndices.get() + localN);
    auto newEnd = std::unique(wIndices.get(), wIndices.get() + localN);
    wIndices.resize(std::distance(wIndices.get(), newEnd));
    assert(wIndices.size() == newLocalN);
    wIndices.release();


    const scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(newGlobalN, myGlobalIndices, comm));
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(newGlobalN));

    CSRStorage<ValueType> storage;
    storage.setCSRDataSwap(newLocalN, newGlobalN, nnzValues, newIA, newJA, newValues, scai::hmemo::ContextPtr());
    coarseGraph = CSRSparseMatrix<ValueType>(newDist, noDist);
    coarseGraph.swapLocalStorage(storage);
 }
//---------------------------------------------------------------------------------------
 
template<typename IndexType, typename ValueType>
template<typename T>
DenseVector<T> MultiLevel<IndexType, ValueType>::computeGlobalPrefixSum(const DenseVector<T> &input, T globalOffset) {
	SCAI_REGION("MultiLevel.computeGlobalPrefixSum");
	scai::dmemo::CommunicatorPtr comm = input.getDistributionPtr()->getCommunicatorPtr();

	const IndexType p = comm->getSize();

	//first, check that the input is some block distribution
	const IndexType localN = input.getDistributionPtr()->getBlockDistributionSize();
    if (localN == nIndex) {
    	throw std::logic_error("Global Prefix sum only implemented for block distribution.");
    }

    //get local prefix sum
    scai::hmemo::ReadAccess<T> localValues(input.getLocalValues());
    std::vector<T> localPrefixSum(localN);
    assert(localN == localValues.size());
    std::partial_sum(localValues.get(), localValues.get()+localN, localPrefixSum.begin());

    T localSum[1] = {0};
    if (localN > 0) {
        localSum[0] = localPrefixSum[localN-1];
    }

    //communicate local sums
    T allOffsets[p];
    comm->gather(allOffsets, 1, 0, localSum);

    //compute prefix sum of offsets.
    std::vector<T> offsetPrefixSum(p+1, 0);
    if (comm->getRank() == 0) {
    	//shift begin of output by one, since the first offset is 0
    	std::partial_sum(allOffsets, allOffsets+p, offsetPrefixSum.begin()+1);
    }

    //remove last value, since it would be the offset for the p+1th processor
    offsetPrefixSum.resize(p);

    //communicate offsets
    T myOffset[1];
    comm->scatter(myOffset, 1, 0, offsetPrefixSum.data());

    //get results by adding local sums and offsets
    DenseVector<T> result(input.getDistributionPtr());
    scai::hmemo::WriteOnlyAccess<T> wResult(result.getLocalValues(), localN);
    for (IndexType i = 0; i < localN; i++) {
    	wResult[i] = localPrefixSum[i] + myOffset[0] + globalOffset;
    }

    return result;
}
//-------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr MultiLevel<IndexType, ValueType>::projectToFine(scai::dmemo::DistributionPtr dist, const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("MultiLevel.projectToFine");
	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();
	scai::dmemo::DistributionPtr coarseDist = projectToCoarse(fineToCoarse);
	IndexType coarseLocalN = coarseDist->getLocalSize();
	scai::dmemo::CommunicatorPtr comm = fineDist->getCommunicatorPtr();

	scai::utilskernel::LArray<IndexType> myCoarseGlobalIndices;
	coarseDist->getOwnedIndexes(myCoarseGlobalIndices);

	scai::utilskernel::LArray<IndexType> owners(coarseLocalN);
	dist->computeOwners(owners, myCoarseGlobalIndices);

	//get send quantities and array
	std::vector<IndexType> quantities(comm->getSize(), 0);
	std::vector<std::vector<IndexType> > sendIndices(comm->getSize());
	{
		scai::hmemo::ReadAccess<IndexType> rOwners(owners);
		scai::hmemo::ReadAccess<IndexType> rFineToCoarse(fineToCoarse.getLocalValues());
		for (IndexType i = 0; i < fineLocalN; i++) {
			IndexType targetRank = rOwners[coarseDist->global2local(rFineToCoarse[i])];
			assert(targetRank < comm->getSize());
			sendIndices[targetRank].push_back(fineDist->local2global(i));
			quantities[targetRank]++;
		}
	}

	std::vector<IndexType> flatIndexVector;
	for (IndexType i = 0; i < sendIndices.size(); i++) {
		std::copy(sendIndices[i].begin(), sendIndices[i].end(), std::back_inserter(flatIndexVector));
	}

	assert(flatIndexVector.size() == fineLocalN);
        
        scai::dmemo::CommunicationPlan sendPlan;
        
        sendPlan.allocate( quantities.data(), comm->getSize() );
        
        assert(sendPlan.totalQuantity() == fineLocalN);
        
        scai::dmemo::CommunicationPlan recvPlan;
        
	recvPlan.allocateTranspose( sendPlan, *comm );

	scai::utilskernel::LArray<IndexType> newValues;

	IndexType newLocalSize = recvPlan.totalQuantity();

	{
            scai::hmemo::WriteOnlyAccess<IndexType> recvVals( newValues, newLocalSize );
            comm->exchangeByPlan( recvVals.get(), recvPlan, flatIndexVector.data(), sendPlan );
	}
	assert(comm->sum(newLocalSize) == fineDist->getGlobalSize());

	{
            scai::hmemo::WriteAccess<IndexType> wValues(newValues);
            std::sort(wValues.get(), wValues.get()+newLocalSize);
	}

	scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(fineDist->getGlobalSize(), newValues, comm));
	return newDist;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr MultiLevel<IndexType, ValueType>::projectToCoarse(const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("MultiLevel.projectToCoarse.distribution");
	const IndexType newGlobalN = fineToCoarse.max().Scalar::getValue<IndexType>() +1;
	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();

	//get set of coarse local indices, without repetitions
	scai::utilskernel::LArray<IndexType> myCoarseGlobalIndices(fineToCoarse.getLocalValues());
	scai::hmemo::WriteAccess<IndexType> wIndices(myCoarseGlobalIndices);
	assert(wIndices.size() == fineLocalN);
	std::sort(wIndices.get(), wIndices.get() + fineLocalN);
	auto newEnd = std::unique(wIndices.get(), wIndices.get() + fineLocalN);
	wIndices.resize(std::distance(wIndices.get(), newEnd));
	IndexType coarseLocalN = wIndices.size();
	wIndices.release();

	scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(newGlobalN, myCoarseGlobalIndices, fineToCoarse.getDistributionPtr()->getCommunicatorPtr()));
	return newDist;
}   
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
DenseVector<ValueType> MultiLevel<IndexType, ValueType>::projectToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("MultiLevel.projectToCoarse.interpolate");
	const scai::dmemo::DistributionPtr inputDist = input.getDistributionPtr();
        
	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();
	assert(inputDist->getLocalSize() == fineLocalN);
	scai::dmemo::DistributionPtr coarseDist = projectToCoarse(fineToCoarse);
	IndexType coarseLocalN = coarseDist->getLocalSize();

	//add values in preparation for interpolation
	std::vector<ValueType> sum(coarseLocalN, 0);
	std::vector<IndexType> numFineNodes(coarseLocalN, 0);
	{
		scai::hmemo::ReadAccess<ValueType> rInput(input.getLocalValues());
		scai::hmemo::ReadAccess<IndexType> rFineToCoarse(fineToCoarse.getLocalValues());
		for (IndexType i = 0; i < fineLocalN; i++) {
			const IndexType coarseTarget = coarseDist->global2local(rFineToCoarse[i]);
			sum[coarseTarget] += rInput[i];
			numFineNodes[coarseTarget] += 1;
		}
	}

	DenseVector<ValueType> result(coarseDist, 0);
	scai::hmemo::WriteAccess<ValueType> wResult(result.getLocalValues());
	for (IndexType i = 0; i < coarseLocalN; i++) {
		assert(numFineNodes[i] > 0);
		wResult[i] = sum[i] / numFineNodes[i];
	}
	wResult.release();
	return result;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
DenseVector<ValueType> MultiLevel<IndexType, ValueType>::sumToCoarse(const DenseVector<ValueType>& input, const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("MultiLevel.sumToCoarse");
	const scai::dmemo::DistributionPtr inputDist = input.getDistributionPtr();

	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();
	scai::dmemo::DistributionPtr coarseDist = projectToCoarse(fineToCoarse);
	IndexType coarseLocalN = coarseDist->getLocalSize();
	assert(inputDist->getLocalSize() == fineLocalN);

	DenseVector<ValueType> result(coarseDist, 0);
	scai::hmemo::WriteAccess<ValueType> wResult(result.getLocalValues());
	{
		scai::hmemo::ReadAccess<ValueType> rInput(input.getLocalValues());
		scai::hmemo::ReadAccess<IndexType> rFineToCoarse(fineToCoarse.getLocalValues());
		for (IndexType i = 0; i < fineLocalN; i++) {
			const IndexType coarseTarget = coarseDist->global2local(rFineToCoarse[i]);
			assert(coarseTarget < coarseLocalN);
			wResult[coarseTarget] += rInput[i];
		}
	}

	wResult.release();
	return result;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::pair<IndexType,IndexType>> MultiLevel<IndexType, ValueType>::maxLocalMatching(const scai::lama::CSRSparseMatrix<ValueType>& adjM, const DenseVector<ValueType> &nodeWeights){
	SCAI_REGION("MultiLevel.maxLocalMatching");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    
    // get local data of the adjacency matrix
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    scai::hmemo::ReadAccess<IndexType> ia( localStorage.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localStorage.getJA() );
    scai::hmemo::ReadAccess<ValueType> values( localStorage.getValues() );

    // localN= number of local nodes
    const IndexType localN= adjM.getLocalNumRows();
    
    // ia must have size localN+1
    assert(ia.size()-1 == localN );
    
    //mainly for debugging reasons
    IndexType totalNbrs= 0;
    
    // the vector<vector> to return
    // matching[0][i]-matching[1][i] are the endopoints of an edge that is matched
    std::vector<std::pair<IndexType,IndexType>> matching;
    
    // keep track of which nodes are already matched
    std::vector<bool> matched(localN, false);

    // get local part of node weights
    scai::utilskernel::LArray<ValueType> localNodeWeights = nodeWeights.getLocalValues();
    scai::hmemo::ReadAccess<ValueType> rLocalNodeWeights( localNodeWeights );
    
    // localNode is the local index of a node
    for(IndexType localNode=0; localNode<localN; localNode++){
        // if the node is already matched go to the next one;
        if(matched[localNode]){
            continue;
        }
        
        IndexType bestTarget = -1;
	ValueType maxEdgeRating = -1;
        const IndexType endCols = ia[localNode+1];
        for (IndexType j = ia[localNode]; j < endCols; j++) {
        	IndexType localNeighbor = distPtr->global2local(ja[j]);
        	if (localNeighbor != nIndex && localNeighbor != localNode && !matched[localNeighbor]) {
        		//neighbor is local and unmatched, possible partner
			ValueType thisEdgeRating = values[j]*values[j]/(rLocalNodeWeights[localNode]*rLocalNodeWeights[localNeighbor]);
        		if (bestTarget < 0 ||  thisEdgeRating > maxEdgeRating) {
        			//either we haven't found any target yet, or the current one is better
        			bestTarget = j;
				maxEdgeRating = thisEdgeRating;
        		}
        	}
        }

        if (bestTarget > 0) {
        	IndexType globalNgbr = ja[bestTarget];

			// at this point -globalNgbr- is the local node with the heaviest edge
			// and should be matched with -localNode-.
			// So, actually, globalNgbr is also local....
			assert( distPtr->isLocal(globalNgbr));
			IndexType localNgbr = distPtr->global2local(globalNgbr);
                        //TODO: search neighbors for the heaviest edge
			matching.push_back( std::pair<IndexType,IndexType> (localNode, localNgbr) );

			// mark nodes as matched
			matched[localNode]= true;
			matched[localNgbr]= true;
        }
    }
    
    assert(ia[ia.size()-1] >= totalNbrs);
    
    return matching;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> MultiLevel<IndexType, ValueType>::pixeledCoarsen (
    const scai::lama::CSRSparseMatrix<ValueType>& adjM,
    std::vector<DenseVector<ValueType>> &coordinates,
    DenseVector<ValueType> &nodeWeights,
    Settings settings){
    SCAI_REGION( "MultiLevel.pixeledCoarsen" )

    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr inputDist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
    
    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    DenseVector<IndexType> result(inputDist, 0);
    
    //TODO: if we know maximum from the input we could save that although is not too costly
    
    /**
     * get maximum of local coordinates
     */
    for (IndexType dim = 0; dim < dimensions; dim++) {
        //get local parts of coordinates
        scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[dim].getLocalValues();
        for (IndexType i = 0; i < localN; i++) {
            ValueType coord = localPartOfCoords[i];
            if (coord > maxCoords[dim]) maxCoords[dim] = coord;
        }
    }
    
    /**
     * communicate to get global  max
     */
    for (IndexType dim = 0; dim < dimensions; dim++) {
        maxCoords[dim] = comm->max(maxCoords[dim]);
    }
   
    // measure density with rounding
    // have to handle 2D and 3D cases seperately
    //const unsigned int detailLvl = settings.pixeledDetailLevel;
    //const unsigned long sideLen = std::pow(2,detailLvl);
    const unsigned long sideLen = settings.pixeledSideLen;
    const unsigned long cubeSize = std::pow(sideLen, dimensions);
    
    if(cubeSize > globalN){
        std::cout<< "Warning, in pixeledCoarsen, pixeled graph size bigger than input size. Not actually a coarsening" << std::endl;
    }
    
    //TODO: generalise this to arbitrary dimensions, do not handle 2D and 3D differently
    // a 2D or 3D arrays as a one dimensional vector
    // [i][j] is in position: i*sideLen + j
    // [i][j][k] is in: i*sideLen*sideLen + j*sideLen + k
    
    scai::hmemo::HArray<IndexType> density( cubeSize, 0);

    // initialize pixelGraph
    scai::hmemo::HArray<IndexType> pixelIA;
    scai::hmemo::HArray<IndexType> pixelJA;
    scai::hmemo::HArray<ValueType> pixelValues;
    
    // here we assume that all edges exist in the pixeled graph. That might not be true. After we add the edges that
    // do exist, we add all mising edges with a small weight.
    IndexType nnzValues= 2*dimensions*(std::pow(sideLen, dimensions) - std::pow(sideLen, dimensions-1) );
    {
        scai::hmemo::WriteOnlyAccess<IndexType> wPixelIA( pixelIA, cubeSize+1 );
        scai::hmemo::WriteOnlyAccess<IndexType> wPixelJA( pixelJA, nnzValues);
        scai::hmemo::WriteOnlyAccess<ValueType> wPixelValues( pixelValues, nnzValues);
        wPixelIA[0] = 0;
        IndexType nnzCounter =0;
        
        for(IndexType i=0; i<cubeSize; i++){
            // the indices of the neighbouring pixels
            std::vector<IndexType> ngbrPixels = ParcoRepart<IndexType, ValueType>::neighbourPixels(i, sideLen, dimensions);
            wPixelIA[i+1] = wPixelIA[i] + ngbrPixels.size();
            SCAI_ASSERT(ngbrPixels.size() <= 2*dimensions, "Too many neighbouring pixels.");
            
            for(IndexType s=0; s<ngbrPixels.size(); s++){
                SCAI_ASSERT( nnzCounter < nnzValues, "Non-zero values for CSRSparseMatrix: "<< nnzValues << " not calculated correctly.");            
                wPixelJA[nnzCounter]= ngbrPixels[s];
                wPixelValues[nnzCounter] = 0.0;
                ++nnzCounter;
            }
        }

        SCAI_ASSERT( nnzCounter == wPixelValues.size() , "Wrong values size for CSR matrix: " << wPixelValues.size() );
        SCAI_ASSERT( nnzCounter == wPixelJA.size() , "Wrong ja size for CSR matrix: " << wPixelJA.size());
        SCAI_ASSERT( wPixelIA[cubeSize] == nnzCounter, "Wrong ia for CSR matrix." );
    }
    
    // get halo for the non-local coordinates
    scai::dmemo::Halo coordHalo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(adjM);
    std::vector<scai::utilskernel::LArray<ValueType>> coordHaloData(dimensions);
    for(int d=0; d<dimensions; d++){        
        comm->updateHalo( coordHaloData[d], coordinates[d].getLocalValues(), coordHalo );
    }
  
    IndexType notCountedPixelEdges = 0; //edges between diagonal pixels are not counted
  
    if(dimensions==2){
        SCAI_REGION( "MultiLevel.pixeledCoarsen.localDensity" )
        scai::hmemo::WriteAccess<IndexType> wDensity(density);
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
    
	const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
        
        IndexType scaledX, scaledY;
        //the +1 is needed
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;

        for(IndexType i=0; i<localN; i++){
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            IndexType thisPixel = scaledX*sideLen + scaledY;      
            SCAI_ASSERT( thisPixel < wDensity.size(), "Index too big: "<< std::to_string(thisPixel) );
            
            ++wDensity[thisPixel];
            
            scai::hmemo::WriteAccess<IndexType> wPixelIA( pixelIA );
            scai::hmemo::WriteAccess<IndexType> wPixelJA( pixelJA );
            scai::hmemo::WriteAccess<ValueType> wPixelValues( pixelValues );
            
            // check the neighbours to fix the pixeledEdge weights
            const IndexType beginCols = ia[i];
            const IndexType endCols = ia[i+1];
            assert(ja.size() >= endCols);
            
            for (IndexType j = beginCols; j < endCols; j++) {
                IndexType neighbor = ja[j];
                
                // find the neighbor's pixel
                ValueType ngbrX, ngbrY;
                if( coordDist->isLocal(neighbor) ){
                    ngbrX = coordAccess0[ coordDist->global2local(neighbor) ];
                    ngbrY = coordAccess1[ coordDist->global2local(neighbor) ];
                }else{
                    ngbrX = coordHaloData[0][ coordHalo.global2halo(neighbor) ];
                    ngbrY = coordHaloData[1][ coordHalo.global2halo(neighbor) ];
                }

                IndexType ngbrPixelIndex = sideLen*(IndexType (sideLen*ngbrX/maxX))  + sideLen*ngbrY/maxY;
           
                SCAI_ASSERT( ngbrPixelIndex < cubeSize, "Index too big: "<< ngbrPixelIndex <<". Should be less than: "<< cubeSize);

                if( ngbrPixelIndex != thisPixel ){ // neighbor not in the same pixel, find the correct pixel
                    const IndexType pixelBeginCols = wPixelIA[thisPixel];
                    const IndexType pixelEndCols = wPixelIA[thisPixel+1];           
                    bool ngbrNotFound = true;
                    
                    for(IndexType p= pixelBeginCols; p<pixelEndCols; p++){
                        IndexType thisPixelOtherNeighbor = wPixelJA[p];
                        if( thisPixelOtherNeighbor == ngbrPixelIndex ){   // add in edge weights
                            SCAI_ASSERT(ngbrPixelIndex < cubeSize, "Index too big." << ngbrPixelIndex );
                            ++wPixelValues[ p ];
                            ngbrNotFound = false;
                            break;
                        }
                    }
                    // somehow got a pixel as neighbour that is either far (not a mesh?) or share only
                    // a corner with thisPixel, not a cube facet
                    if( ngbrNotFound ){ 
                        ++notCountedPixelEdges;
                    }
                }
            }          
        } 
    }else if(dimensions==3){
        SCAI_REGION( "MultiLevel.pixeledCoarsen.localDensity" )
        
        scai::hmemo::WriteAccess<IndexType> wDensity(density);
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );
        
        const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
        
        IndexType scaledX, scaledY, scaledZ;
        
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;
        IndexType maxZ = maxCoords[2]+1;
        
        for(IndexType i=0; i<localN; i++){
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            scaledZ = coordAccess2[i]/maxZ * sideLen;
            IndexType thisPixel = scaledX*sideLen*sideLen + scaledY*sideLen + scaledZ;
            SCAI_ASSERT( thisPixel < wDensity.size(), "Index too big: "<< thisPixel );
             
            ++wDensity[thisPixel];
            
            // check the neighbours to fix the pixeledEdge weights
            const IndexType beginCols = ia[i];
            const IndexType endCols = ia[i+1];
            assert(ja.size() >= endCols);
            
            scai::hmemo::WriteAccess<IndexType> wPixelIA( pixelIA );
            scai::hmemo::WriteAccess<IndexType> wPixelJA( pixelJA );
            scai::hmemo::WriteAccess<ValueType> wPixelValues( pixelValues );
            
            for (IndexType j = beginCols; j < endCols; j++) {
                IndexType neighbor = ja[j];
                
                // find the neighbor's pixel
                IndexType ngbrX, ngbrY, ngbrZ;
                if( coordDist->isLocal(neighbor) ){
                    ngbrX = coordAccess0[ coordDist->global2local(neighbor) ];
                    ngbrY = coordAccess1[ coordDist->global2local(neighbor) ];
                    ngbrZ = coordAccess2[ coordDist->global2local(neighbor) ];
                }else{
                    ngbrX = coordHaloData[0][ coordHalo.global2halo(neighbor) ];
                    ngbrY = coordHaloData[1][ coordHalo.global2halo(neighbor) ];
                    ngbrZ = coordHaloData[2][ coordHalo.global2halo(neighbor) ];
                }

                IndexType ngbrPixelIndex = sideLen*sideLen*(int (sideLen*ngbrX/maxX))  + sideLen*(int(sideLen*ngbrY/maxY)) + sideLen*ngbrZ/maxZ;
                SCAI_ASSERT( ngbrPixelIndex < cubeSize, "Index too big: "<< ngbrPixelIndex <<". Should be less than: "<< cubeSize);
                
                if( ngbrPixelIndex != thisPixel ){ // neighbor not in the same pixel
                    const IndexType pixelBeginCols = wPixelIA[thisPixel];
                    const IndexType pixelEndCols = wPixelIA[thisPixel+1];
                    bool ngbrNotFound = true;
                    
                    for(IndexType p= pixelBeginCols; p<pixelEndCols; p++){
                        IndexType thisPixelOtherNeighbor = wPixelJA[p];
                        if( thisPixelOtherNeighbor == ngbrPixelIndex ){   // add in edge weights
                            SCAI_ASSERT(ngbrPixelIndex < wPixelValues.size(), "Index too big." << ngbrPixelIndex );
                            ++wPixelValues[ngbrPixelIndex];
                            ngbrNotFound = false;
                            //break;
                        }
                    }
                    // somehow got a pixel as neighbour that is either far (not a mesh?) or share only
                    // a corner with thisPixel, not a cube edge
                    if( ngbrNotFound ){ 
                        ++notCountedPixelEdges;
                    }
                }
            }
        }
    }else{
        throw std::runtime_error("Available only for 2D and 3D. Data given have dimension:" + std::to_string(dimensions) );
    } 
    
    //PRINT(notCountedPixelEdges);
    IndexType sumMissingEdges = comm->sum(notCountedPixelEdges);
    
    PRINT0("not counted pixel edges= " << sumMissingEdges );
    
    // sum node weights of the pixeled graph
    SCAI_ASSERT( nodeWeights.getDistributionPtr()->isReplicated() == true, "Node weights of the pixeled graph should be replicated (at least for now).");
    nodeWeights.allocate(density.size());
    
    {
        SCAI_REGION( "Multilevel.pixeledCoarsen.sumDensity" )
        comm->sumArray( density );
    }

    scai::hmemo::WriteAccess<IndexType> wDensity(density);
    for(int i=0; i<density.size(); i++){
        nodeWeights.getLocalValues()[i] = wDensity[i];
    }   
    wDensity.release();
    
    {
        SCAI_REGION( "Multilevel.pixeledCoarsen.sumValues" )
        // sum the values from all PEs
        comm->sumArray( pixelValues);
    }
    
    // add a lightweight edge to isolated pixels. Hope this does not affect the spectral partition 
    // or any other usage.
    //PRINT(*comm << ": " << nnzValues);
    for(int i=0; i<pixelValues.size(); i++){    
        scai::hmemo::WriteAccess<ValueType> wPixelValues( pixelValues );    
        if(wPixelValues[i]==0){
            //PRINT(*comm<< ": " << i );
            wPixelValues[i]=0.01;
        }
    }
    
    scai::lama::CSRStorage<ValueType> pixelStorage;
    pixelStorage.setCSRData( cubeSize, cubeSize, nnzValues, pixelIA, pixelJA, pixelValues);
    

    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( cubeSize) );
    scai::lama::CSRSparseMatrix<ValueType> pixelGraph( pixelStorage, noDistPointer, noDistPointer);

    SCAI_ASSERT_DEBUG( pixelGraph.isConsistent(), pixelGraph << ": matrix is not consistent." )
 
    return pixelGraph;
    
}
//---------------------------------------------------------------------------------------

template int MultiLevel<int, double>::multiLevelStep(CSRSparseMatrix<double> &input, DenseVector<int> &part, DenseVector<double> &nodeWeights, std::vector<DenseVector<double>> &coordinates, Settings settings);

template std::vector<std::pair<int,int>> MultiLevel<int, double>::maxLocalMatching(const scai::lama::CSRSparseMatrix<double>& graph, const DenseVector<double> &nodeWeights);

template void MultiLevel<int, double>::coarsen(const CSRSparseMatrix<double>& inputGraph, const DenseVector<double> &nodeWeights, CSRSparseMatrix<double>& coarseGraph, DenseVector<int>& fineToCoarse, int iterations);

template DenseVector<int> MultiLevel<int, double>::computeGlobalPrefixSum(const DenseVector<int> &input, int offset);

template scai::lama::CSRSparseMatrix<double> MultiLevel<int, double>::pixeledCoarsen (const CSRSparseMatrix<double>& adjM, std::vector<DenseVector<double>> &coordinates, DenseVector<double> &nodeWeights, Settings settings);
    
} // namespace ITI
