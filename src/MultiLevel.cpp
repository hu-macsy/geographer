#include <scai/dmemo/GenBlockDistribution.hpp>

#include "MultiLevel.h"
#include "GraphUtils.h"
#include "HaloPlanFns.h"
#include "ParcoRepart.h"

//TODO: needed monstly(only?) for debugging, to store the PE graph
#include "FileIO.h"

using scai::hmemo::HArray;

namespace ITI{
    
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, DenseVector<ValueType> &nodeWeights, std::vector<DenseVector<ValueType>> &coordinates, const HaloExchangePlan& halo, Settings settings, Metrics& metrics) {
	
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

	//only needed to store local refinement specific metrics
	settings.thisRound++;

	auto origin = scai::lama::fill<DenseVector<IndexType>>(input.getRowDistributionPtr(), comm->getRank());//to track node movements through the hierarchies
        
	if (settings.multiLevelRounds > 0) {
		SCAI_REGION_START( "MultiLevel.multiLevelStep.prepareRecursiveCall" )
		CSRSparseMatrix<ValueType> coarseGraph;
		DenseVector<IndexType> fineToCoarseMap;
		std::chrono::time_point<std::chrono::system_clock> beforeCoarse =  std::chrono::system_clock::now();

		if (comm->getRank() == 0) {
			std::cout << "Beginning coarsening, still " << settings.multiLevelRounds << " levels to go." << std::endl;
		}
		MultiLevel<IndexType, ValueType>::coarsen(input, nodeWeights, halo, coarseGraph, fineToCoarseMap, settings.coarseningStepsBetweenRefinement);

		scai::dmemo::DistributionPtr oldCoarseDist = input.getRowDistributionPtr();
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

		DenseVector<IndexType> coarsePart =  scai::lama::fill<DenseVector<IndexType>>(coarseGraph.getRowDistributionPtr(), comm->getRank());

		DenseVector<ValueType> coarseWeights = sumToCoarse(nodeWeights, fineToCoarseMap);

		scai::hmemo::HArray<IndexType> haloData;
        halo.updateHalo(haloData, fineToCoarseMap.getLocalValues(), *comm);

        HaloExchangePlan coarseHalo = coarsenHalo(coarseGraph.getRowDistribution(), halo, fineToCoarseMap.getLocalValues(), haloData);

		assert(coarseWeights.sum() == nodeWeights.sum());

		std::chrono::duration<double> coarseningTime =  std::chrono::system_clock::now() - beforeCoarse;
		ValueType timeForCoarse = ValueType ( comm->max(coarseningTime.count() ));
		if (comm->getRank() == 0) std::cout << "Time for coarsening:" << timeForCoarse << std::endl;


		Settings settingscopy(settings);
		settingscopy.multiLevelRounds -= settings.coarseningStepsBetweenRefinement;
		SCAI_REGION_END( "MultiLevel.multiLevelStep.prepareRecursiveCall" )
		// recursive call
		DenseVector<IndexType> coarseOrigin = multiLevelStep(coarseGraph, coarsePart, coarseWeights, coarseCoords, coarseHalo, settingscopy, metrics);
		SCAI_ASSERT_DEBUG(coarseOrigin.getDistribution().isEqual(coarseGraph.getRowDistribution()), "Distributions inconsistent.");
		//SCAI_ASSERT_DEBUG(scai::dmemo::Redistributor(coarseOrigin.getLocalValues(), coarseOrigin.getDistributionPtr()).getTargetDistributionPtr()->isEqual(*oldCoarseDist), "coarseOrigin invalid");

		{
			SCAI_REGION( "MultiLevel.multiLevelStep.uncoarsen" )
			// uncoarsening/refinement
			std::chrono::time_point<std::chrono::system_clock> beforeUnCoarse =  std::chrono::system_clock::now();
			DenseVector<IndexType> fineTargets = getFineTargets(coarseOrigin, fineToCoarseMap);
			auto redistributor = scai::dmemo::redistributePlanByNewOwners( fineTargets.getLocalValues(), fineTargets.getDistributionPtr());
			scai::dmemo::DistributionPtr projectedFineDist = redistributor.getTargetDistributionPtr();

			assert(projectedFineDist->getGlobalSize() == globalN);

			part.setSameValue(projectedFineDist, comm->getRank());

			if (settings.useGeometricTieBreaking) {
				for (IndexType dim = 0; dim < settings.dimensions; dim++) {
					coordinates[dim].redistribute(redistributor);
				}
			}

			input.redistribute(redistributor, input.getColDistributionPtr());

			nodeWeights.redistribute(redistributor);

			origin.redistribute(redistributor);

			std::chrono::duration<double> uncoarseningTime =  std::chrono::system_clock::now() - beforeUnCoarse;
			ValueType time = ValueType ( comm->max(uncoarseningTime.count() ));
			if (comm->getRank() == 0) std::cout << "Time for uncoarsening:" << time << std::endl;
		}
	}

	// do local refinement
	{
		SCAI_REGION( "MultiLevel.multiLevelStep.localRefinement" )
		scai::lama::CSRSparseMatrix<ValueType> processGraph = GraphUtils<IndexType, ValueType>::getPEGraph(input);
		
		//TODO: remove from final version?
		// write the PE graph for further experiments
		if(settings.writePEgraph){//write PE graph for further experiments
			//TODO: this workos only for 12 rounds
			PRINT0( "thisRound= " << settings.thisRound << " , multiLevelRounds= " << settings.multiLevelRounds);
			if( settings.thisRound==0){
				std::chrono::time_point<std::chrono::system_clock> before =  std::chrono::system_clock::now();
				std::string filename = settings.fileName+ "_k"+ std::to_string(settings.numBlocks)+ ".PEgraph";
				if( not FileIO<IndexType,ValueType>::fileExists(filename) ){
					FileIO<IndexType,ValueType>::writeGraph(processGraph, filename, 1);
				}
				std::chrono::duration<double> elapTime = std::chrono::system_clock::now() - before;
				PRINT0("time to write PE graph: :" << elapTime.count() );
			}
		}

		std::chrono::time_point<std::chrono::system_clock> before =  std::chrono::system_clock::now();
		
		std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(processGraph, settings);

		std::vector<IndexType> nodesWithNonLocalNeighbors = GraphUtils<IndexType, ValueType>::getNodesWithNonLocalNeighbors(input);

		std::chrono::duration<double> elapTime = std::chrono::system_clock::now() - before;
		ValueType maxTime = comm->max( elapTime.count() );
		ValueType minTime = comm->min( elapTime.count() );
		if (settings.verbose) PRINT0("getCommPairs and border nodes: time " << minTime << " -- " << maxTime );
		
		std::vector<ValueType> distances;
		if (settings.useGeometricTieBreaking) {
			distances = LocalRefinement<IndexType, ValueType>::distancesFromBlockCenter(coordinates);
		}

		IndexType numRefinementRounds = 0;
		//IndexType oldCut = 0;

		ValueType gain = 0;
		while (numRefinementRounds == 0 || gain >= settings.minGainForNextRound) {

			std::chrono::time_point<std::chrono::system_clock> beforeFMStep =  std::chrono::system_clock::now();

			/* TODO: if getting the graph is fast, maybe doing it in every step might help
			// get graph before every step			
			processGraph = GraphUtils::getPEGraph<IndexType, ValueType>(input);			
			communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(processGraph, settings);
			nodesWithNonLocalNeighbors = GraphUtils::getNodesWithNonLocalNeighbors<IndexType, ValueType>(input);
			elapTime = std::chrono::system_clock::now() - beforeFMStep;
			maxTime = comm->max( elapTime.count() );
			minTime = comm->min( elapTime.count() );

			if(settings.verbose){
				PRINT0("getCommPairs and border nodes: time " << minTime << " -- " << maxTime );
			}
			*/

			std::vector<ValueType> gainPerRound = LocalRefinement<IndexType, ValueType>::distributedFMStep(input, part, nodesWithNonLocalNeighbors, nodeWeights, coordinates, distances, origin, communicationScheme, settings);
			gain = 0;
			for (ValueType roundGain : gainPerRound) gain += roundGain;

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

			std::chrono::duration<double> elapTimeFMStep = std::chrono::system_clock::now() - beforeFMStep;
			ValueType FMStepTime = comm->max( elapTimeFMStep.count() );
			//PRINT0(" one FM step time: " << FMStepTime );

			if (numRefinementRounds > 0) {
				assert(gain >= 0);
			}
			if (comm->getRank() == 0) {
				std::cout << "Multilevel round "<< settings.multiLevelRounds <<": In refinement round " << numRefinementRounds << ", gain was " << gain << " in time " << FMStepTime << std::endl;
			}
			numRefinementRounds++;

			//int thisRound= settings.multiLevelRounds + settings.coarseningStepsBetweenRefinement;
			//WARNING: this may case problems... Check also Metrics constructor
			//Hopefully, numRefinementRound will ALWAYS be less than 50
			//update: well... In some case it stucks in a period, like: 
			//gain 10, gain 30, gain 25, gain 10, gain 30, gain 25, gain 10, ...
			SCAI_ASSERT_LT_ERROR( settings.thisRound, metrics.localRefDetails.size(), "Metrics structure not allocated?" );
			if(numRefinementRounds>=50){
				metrics.localRefDetails[settings.thisRound].push_back( std::make_pair(gain, FMStepTime) );
				PRINT0("*** WARNING: numRefinementRound more than 50.\nWill force break from while loop");
				break;
			}else{
				SCAI_ASSERT_LE_ERROR( numRefinementRounds, metrics.localRefDetails[settings.thisRound].size(), "Metrics structure not allocated?" );
				metrics.localRefDetails[settings.thisRound][numRefinementRounds] = std::make_pair(gain, FMStepTime) ;
			}

		}
		std::chrono::duration<double> elapTime2 = std::chrono::system_clock::now() - before;
		ValueType refineTime = comm->max( elapTime2.count() );
		//PRINT0("local refinement time: " << refineTime );
	}

	if( settings.debugMode and settings.outFile.size() > 0 ){
		std::string filename = "mlRound_"+ std::to_string(settings.thisRound)+".mtx";
		part.writeToFile( filename );
	}

	return origin;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
DenseVector<IndexType> MultiLevel<IndexType, ValueType>::getFineTargets(const DenseVector<IndexType> &coarseOrigin, const DenseVector<IndexType> &fineToCoarseMap) {
	SCAI_REGION("MultiLevel.getFineTargets");

	const scai::dmemo::DistributionPtr coarseDist = coarseOrigin.getDistributionPtr();
	const scai::dmemo::DistributionPtr oldFineDist = fineToCoarseMap.getDistributionPtr();

	//get coarse reverse redistributor
	auto coarseReverseRedist = scai::dmemo::redistributePlanByNewOwners(coarseOrigin.getLocalValues(), coarseDist);
	const scai::dmemo::DistributionPtr oldCoarseDist = coarseReverseRedist.getTargetDistributionPtr();
	SCAI_ASSERT_EQ_ERROR(oldCoarseDist->getGlobalSize(), coarseOrigin.size(), "Old coarse distribution has wrong size.");

	//use it to inform source PEs where their elements went in the coarser local refinement step
	auto targets = fill<DenseVector<IndexType>>(coarseDist, coarseDist->getCommunicatorPtr()->getRank());
	targets.redistribute(coarseReverseRedist);//targets now have old coarse dist

	//build fine target array by checking in fineToCoarseMap
	auto result = fill<DenseVector<IndexType>>(oldFineDist, scai::invalidIndex);
	{
		scai::hmemo::ReadAccess<IndexType> rMap(fineToCoarseMap.getLocalValues());
		scai::hmemo::ReadAccess<IndexType> rTargets(targets.getLocalValues());
		scai::hmemo::WriteAccess<IndexType> wResult(result.getLocalValues());

		const IndexType oldFineLocalN = oldFineDist->getLocalSize();

		for (IndexType i = 0; i < oldFineLocalN; i++) {
			IndexType oldLocalCoarse =  oldCoarseDist->global2Local(rMap[i]);//TODO: optimize this
			SCAI_ASSERT_DEBUG(oldLocalCoarse != scai::invalidIndex, "Index " << rMap[i] << " maybe not local after all?");
			SCAI_ASSERT_DEBUG(oldLocalCoarse < rTargets.size(), "Index " << oldLocalCoarse << " does not fit in " << rTargets.size());
			wResult[i] = rTargets[oldLocalCoarse];
		}
	}
	return result;
}
 
template<typename IndexType, typename ValueType>
void MultiLevel<IndexType, ValueType>::coarsen(const CSRSparseMatrix<ValueType>& adjM, const DenseVector<ValueType> &nodeWeights,  const HaloExchangePlan& halo, CSRSparseMatrix<ValueType>& coarseGraph, DenseVector<IndexType>& fineToCoarse, IndexType iterations) {
	SCAI_REGION("MultiLevel.coarsen");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    

    // localN= number of local nodes
    IndexType localN= adjM.getLocalNumRows();
    IndexType globalN = adjM.getNumColumns();

    SCAI_REGION_START("MultiLevel.coarsen.localCopy")

    DenseVector<ValueType> localWeightCopy(nodeWeights);

    scai::hmemo::HArray<IndexType> preserved(localN, 1);

    std::vector<IndexType> localFineToCoarse(localN);

    scai::hmemo::HArray<IndexType> globalIndices;
    distPtr->getOwnedIndexes(globalIndices);
    scai::hmemo::ReadAccess<IndexType> rIndex(globalIndices);

    scai::lama::CSRSparseMatrix<ValueType> graph = adjM;
    SCAI_REGION_END("MultiLevel.coarsen.localCopy")
     
    for (IndexType i = 0; i < iterations; i++) {
    	SCAI_REGION("MultiLevel.coarsen.localLoop");
        // get local data of the adjacency matrix
        const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
        scai::hmemo::ReadAccess<IndexType> ia( localStorage.getIA() );
        scai::hmemo::ReadAccess<IndexType> ja( localStorage.getJA() );
        scai::hmemo::ReadAccess<ValueType> values( localStorage.getValues() );

        // ia must have size localN+1
        assert(ia.size()-1 == localN );

        //get a matching, the returned indices are from 0 to localN
        std::vector<std::pair<IndexType,IndexType>> matching = MultiLevel<IndexType, ValueType>::maxLocalMatching( graph, localWeightCopy );

        std::vector<IndexType> localMatchingPartner(localN, -1);

        //sort the matching according to its first element
        std::sort(matching.begin(), matching.end());
        
        {
            scai::hmemo::WriteAccess<IndexType> localPreserved(preserved);

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

        //create edge list and fine to coarse mapping of locally coarsened graph
        std::vector<std::map<IndexType, ValueType>> outgoingEdges(localN);
        std::vector<IndexType> newLocalFineToCoarse(localN);
        IndexType newLocalN = 0;
        scai::hmemo::ReadAccess<IndexType> localPreserved(preserved);

        scai::hmemo::WriteAccess<ValueType> wWeights(localWeightCopy.getLocalValues());

        {
        	SCAI_REGION("MultiLevel.coarsen.localLoop.rewireEdges");
			for (IndexType i = 0; i < localN; i++) {
				IndexType coarseNode;

				if (localPreserved[i]) {
					coarseNode = i;
					newLocalFineToCoarse[i] = i;
					newLocalN++;
				} else {
					coarseNode = localMatchingPartner[i];
					assert(coarseNode < i);
					if (coarseNode == -1) {//node was already eliminated in previous round
						IndexType oldCoarseNode = localFineToCoarse[i];
						newLocalFineToCoarse[i] = newLocalFineToCoarse[oldCoarseNode];
					} else {
						wWeights[coarseNode] += wWeights[i];
						newLocalFineToCoarse[i] = newLocalFineToCoarse[coarseNode];
					}
				}

				if (coarseNode >= 0) {
					for (IndexType j = ia[i]; j < ia[i+1]; j++) {
						IndexType edgeTarget = ja[j];
						IndexType localTarget = distPtr->global2Local(edgeTarget);//TODO: maybe optimize this
						if (localTarget != scai::invalidIndex && !localPreserved[localTarget]) {
							localTarget = localMatchingPartner[localTarget];
							edgeTarget = rIndex[localTarget];
						}
						if (outgoingEdges[coarseNode].count(edgeTarget) == 0) {
							outgoingEdges[coarseNode][edgeTarget] = 0;
						}
						outgoingEdges[coarseNode][edgeTarget] += values[j];
					}
				}
			}
        }

        localFineToCoarse.swap(newLocalFineToCoarse);

        {
        	SCAI_REGION("MultiLevel.coarsen.localLoop.getLocalCSRMatrix");
            //create CSR matrix out of edge list
            scai::hmemo::HArray<IndexType> newIA(localN+1);
            scai::hmemo::WriteAccess<IndexType> wIA(newIA);
            std::vector<IndexType> newJA;
            std::vector<ValueType> newValues;

            wIA[0] = 0;

            for (IndexType i = 0; i < localN; i++) {
                //assert(localPreserved[i] == outgoingEdges[i].size() > 0 );
                //SCAI_ASSERT_EQ_ERROR(localPreserved[i], outgoingEdges[i].size(), "Outgoing edges mismatch");
                wIA[i+1] = wIA[i] + outgoingEdges[i].size();
                IndexType jaIndex = wIA[i];
                for (std::pair<IndexType, ValueType> edge : outgoingEdges[i]) {
                    newJA.push_back(edge.first);
                    newValues.push_back(edge.second);
                    jaIndex++;
                }
                assert(jaIndex == wIA[i+1]);
                assert(jaIndex == newJA.size());
            }

            wIA.release();
            ia.release();
            ja.release();
            values.release();

            //scai::lama::CSRStorage<ValueType> localStorage(1, comm->getSize(), numNeighbors, ia, ja, values);;
            HArray<IndexType> lJA(newJA.size(), newJA.data());
            HArray<ValueType> lValues(newValues.size(), newValues.data());
            graph.getLocalStorage() = CSRStorage<ValueType>(localN, globalN, std::move( newIA ), std::move( lJA ), std::move( lValues ));
        }
    }

    SCAI_REGION_START("MultiLevel.coarsen.newGlobalIndices")
    //get new global indices by computing a prefix sum over the preserved nodes
    //fill gaps in index list. To avoid redistribution, we assign a block distribution and live with the implicit reindexing
    auto blockDist = scai::dmemo::genBlockDistributionBySize(globalN, localN, comm);
    DenseVector<IndexType> distPreserved(blockDist, preserved);
    DenseVector<IndexType> blockFineToCoarse = computeGlobalPrefixSum(distPreserved, IndexType(-1) );
    const IndexType newGlobalN = blockFineToCoarse.max() + 1;
    fineToCoarse = DenseVector<IndexType>(distPtr, blockFineToCoarse.getLocalValues());
    const IndexType newLocalN = scai::utilskernel::HArrayUtils::sum(preserved);

    //set new indices for contracted nodes
    {
        scai::hmemo::ReadAccess<IndexType> localPreserved(preserved);
        scai::hmemo::WriteAccess<IndexType> wFineToCoarse(fineToCoarse.getLocalValues());
        for (IndexType i = 0; i < localN; i++) {
            assert((localFineToCoarse[i] == i) == localPreserved[i]);
            wFineToCoarse[i] = wFineToCoarse[localFineToCoarse[i]];
        }
    }

    assert(fineToCoarse.max() + 1 == newGlobalN);
    assert(newGlobalN <= globalN);
    assert(newGlobalN == comm->sum(newLocalN));
    SCAI_REGION_END("MultiLevel.coarsen.newGlobalIndices")

    //build halo of new global indices
    HArray<IndexType> haloData;
    halo.updateHalo(haloData, fineToCoarse.getLocalValues(), *comm);

    //create new coarsened CSR matrix
    scai::hmemo::HArray<IndexType> newIA(newLocalN + 1);
    std::vector<IndexType> newJA;
    std::vector<ValueType> newValues;
    
    {
        SCAI_REGION("MultiLevel.coarsen.getCSRMatrix");
        const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
		scai::hmemo::ReadAccess<IndexType> ia( localStorage.getIA() );
		scai::hmemo::ReadAccess<IndexType> ja( localStorage.getJA() );
		scai::hmemo::ReadAccess<ValueType> values( localStorage.getValues() );

        scai::hmemo::ReadAccess<IndexType> localPreserved(preserved);
        scai::hmemo::ReadAccess<IndexType> rHalo(haloData);
        scai::hmemo::ReadAccess<IndexType> rFineToCoarse(fineToCoarse.getLocalValues());

        scai::hmemo::WriteAccess<IndexType> newIAWrite( newIA, newLocalN +1 );

        newIAWrite[0] = 0;
        IndexType iaIndex = 0;
        IndexType jaIndex = 0;
        
        //for all fine rows
        for (IndexType i = 0; i < localN; i++) {
            std::map<IndexType, ValueType> outgoingEdges;

            if (localPreserved[i]) {
            	assert(jaIndex == newIAWrite[iaIndex]);
                for (IndexType j = ia[i]; j < ia[i+1]; j++) {
                    //only need to reroute nonlocal edges
                    IndexType localNeighbor = distPtr->global2Local(ja[j]);

                    if (localNeighbor != scai::invalidIndex) {
                        assert(outgoingEdges.count(rFineToCoarse[localNeighbor]) == 0);
                        outgoingEdges[rFineToCoarse[localNeighbor]] = values[j];
                    } else {
                        IndexType haloIndex = halo.global2Halo(ja[j]);
                        assert(haloIndex != scai::invalidIndex);
                        if (outgoingEdges.count(rHalo[haloIndex]) == 0)  outgoingEdges[rHalo[haloIndex]] = 0;
                        outgoingEdges[rHalo[haloIndex]] += values[j];
                    }
                }

                newIAWrite[iaIndex+1] = newIAWrite[iaIndex] + outgoingEdges.size();

                for (std::pair<IndexType, ValueType> edge : outgoingEdges) {
                    newJA.push_back(edge.first);
                    newValues.push_back(edge.second);
                    jaIndex++;
                }

                assert(jaIndex == newIAWrite[iaIndex+1]);
                assert(jaIndex == newJA.size());
                iaIndex++;
            }
        }
    }
    
    scai::hmemo::HArray<IndexType> csrJA(newJA.size(), newJA.data());
    scai::hmemo::HArray<ValueType> csrValues(newValues.size(), newValues.data());

    //create distribution object for coarse graph
    HArray<IndexType> myGlobalIndices(fineToCoarse.getLocalValues());
    scai::hmemo::WriteAccess<IndexType> wIndices(myGlobalIndices);
    assert(wIndices.size() == localN);
    std::sort(wIndices.get(), wIndices.get() + localN);
    auto newEnd = std::unique(wIndices.get(), wIndices.get() + localN);
    wIndices.resize(std::distance(wIndices.get(), newEnd));
    assert(wIndices.size() == newLocalN);
    wIndices.release();

    const IndexType localEdgeCount = newJA.size();

    const auto newDist = scai::dmemo::generalDistributionUnchecked(newGlobalN, myGlobalIndices, comm);
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(newGlobalN));

    CSRStorage<ValueType> storage( newLocalN, newGlobalN, std::move(newIA), std::move(csrJA), std::move(csrValues) );
    coarseGraph = CSRSparseMatrix<ValueType>( newDist, std::move( storage ) );

 }//---------------------------------------------------------------------------------------
 
template<typename IndexType, typename ValueType>
template<typename T>
DenseVector<T> MultiLevel<IndexType, ValueType>::computeGlobalPrefixSum(const DenseVector<T> &input, T globalOffset) {
	SCAI_REGION("MultiLevel.computeGlobalPrefixSum");
	scai::dmemo::CommunicatorPtr comm = input.getDistributionPtr()->getCommunicatorPtr();

	const IndexType p = comm->getSize();

	//first, check that the input is some block distribution
	const IndexType localN = input.getDistributionPtr()->getBlockDistributionSize();
    if (localN == scai::invalidIndex) {
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
    HArray<T> localResult;
    {
        scai::hmemo::WriteOnlyAccess<T> wResult(localResult, localN);
        for (IndexType i = 0; i < localN; i++) {
            wResult[i] = localPrefixSum[i] + myOffset[0] + globalOffset;
        }
    }
    DenseVector<T> result(input.getDistributionPtr(), std::move(localResult));

    return result;
}
//-------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::dmemo::DistributionPtr MultiLevel<IndexType, ValueType>::projectToCoarse(const DenseVector<IndexType>& fineToCoarse) {
	SCAI_REGION("MultiLevel.projectToCoarse.distribution");
	const IndexType newGlobalN = fineToCoarse.max() + 1;
	scai::dmemo::DistributionPtr fineDist = fineToCoarse.getDistributionPtr();
	const IndexType fineLocalN = fineDist->getLocalSize();

	//get set of coarse local indices, without repetitions
	HArray<IndexType> myCoarseGlobalIndices(fineToCoarse.getLocalValues());
	scai::hmemo::WriteAccess<IndexType> wIndices(myCoarseGlobalIndices);
	assert(wIndices.size() == fineLocalN);
	std::sort(wIndices.get(), wIndices.get() + fineLocalN);
	auto newEnd = std::unique(wIndices.get(), wIndices.get() + fineLocalN);
	wIndices.resize(std::distance(wIndices.get(), newEnd));
	wIndices.release();

	auto newDist = scai::dmemo::generalDistributionUnchecked(newGlobalN, myCoarseGlobalIndices, 
                                                             fineToCoarse.getDistributionPtr()->getCommunicatorPtr());
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
			const IndexType coarseTarget = coarseDist->global2Local(rFineToCoarse[i]);
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
			const IndexType coarseTarget = coarseDist->global2Local(rFineToCoarse[i]);
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
    // matching[0][i]-matching[1][i] are the endpoints of an edge that is matched
    std::vector<std::pair<IndexType,IndexType>> matching;
    
    // keep track of which nodes are already matched
    std::vector<bool> matched(localN, false);

    // get local part of node weights
    scai::hmemo::ReadAccess<ValueType> rLocalNodeWeights( nodeWeights.getLocalValues() );
    
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
        	IndexType localNeighbor = distPtr->global2Local(ja[j]);
        	if (localNeighbor != scai::invalidIndex && localNeighbor != localNode && !matched[localNeighbor]) {
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
			IndexType localNgbr = distPtr->global2Local(globalNgbr);
			assert(localNgbr != scai::invalidIndex);
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
    const std::vector<DenseVector<ValueType>> &coordinates,
    DenseVector<ValueType> &nodeWeights,
    Settings settings){
    SCAI_REGION( "MultiLevel.pixeledCoarsen" )

    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr inputDist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
    
    //IndexType k = settings.numBlocks;
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
        const HArray<ValueType>& localPartOfCoords = coordinates[dim].getLocalValues();
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
    
    scai::hmemo::HArray<IndexType> density( cubeSize, IndexType(0) );

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
    scai::dmemo::HaloExchangePlan coordHalo = GraphUtils<IndexType, ValueType>::buildNeighborHalo(adjM);
    std::vector<HArray<ValueType>> coordHaloData(dimensions);
    for(int d=0; d<dimensions; d++){        
        coordHalo.updateHalo( coordHaloData[d], coordinates[d].getLocalValues(), *comm );
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
                    ngbrX = coordAccess0[ coordDist->global2Local(neighbor) ];
                    ngbrY = coordAccess1[ coordDist->global2Local(neighbor) ];
                }else{
                    ngbrX = coordHaloData[0][ coordHalo.global2Halo(neighbor) ];
                    ngbrY = coordHaloData[1][ coordHalo.global2Halo(neighbor) ];
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
                    ngbrX = coordAccess0[ coordDist->global2Local(neighbor) ];
                    ngbrY = coordAccess1[ coordDist->global2Local(neighbor) ];
                    ngbrZ = coordAccess2[ coordDist->global2Local(neighbor) ];
                }else{
                    ngbrX = coordHaloData[0][ coordHalo.global2Halo(neighbor) ];
                    ngbrY = coordHaloData[1][ coordHalo.global2Halo(neighbor) ];
                    ngbrZ = coordHaloData[2][ coordHalo.global2Halo(neighbor) ];
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
                    // a corner with thisPixel, not a cube facet
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

    pixelStorage.setCSRData( cubeSize, cubeSize, pixelIA, pixelJA, pixelValues);
    
    scai::lama::CSRSparseMatrix<ValueType> pixelGraph( std::move( pixelStorage ) );

    SCAI_ASSERT_DEBUG( pixelGraph.isConsistent(), pixelGraph << ": matrix is not consistent." )
 
    return pixelGraph;
    
}
//---------------------------------------------------------------------------------------

template class MultiLevel<IndexType, ValueType>;

} // namespace ITI
