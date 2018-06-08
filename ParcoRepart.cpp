/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include <scai/tracing.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <queue>
#include <string>
#include <unordered_set>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <set>
#include <iostream>
#include <iomanip> 

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "HilbertCurve.h"
#include "MultiLevel.h"
#include "SpectralPartition.h"
#include "KMeans.h"
#include "AuxiliaryFunctions.h"
#include "MultiSection.h"
#include "GraphUtils.h"

//  #include "RBC/Sort/SQuick.hpp"

namespace ITI {
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings, struct Metrics& metrics)
{
	auto uniformWeights = fill<DenseVector<ValueType>>(input.getRowDistributionPtr(), 1);
	return partitionGraph(input, coordinates, uniformWeights, settings, metrics);
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings) {
    
    struct Metrics metrics(settings.numBlocks);
    
    assert(settings.storeInfo == false); // Cannot return timing information. Better throw an error than silently drop it.
    
    DenseVector<IndexType> previous;
    assert(!settings.repartition);
    return partitionGraph(input, coordinates, nodeWeights, previous, settings, metrics);
    
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, struct Settings settings){
    
    struct Metrics metrics(settings.numBlocks);
    assert(settings.storeInfo == false); // Cannot return timing information. Better throw an error than silently drop it.
    
    auto uniformWeights = fill<DenseVector<ValueType>>(input.getRowDistributionPtr(), 1);
    return partitionGraph(input, coordinates, uniformWeights, settings, metrics);
}

// overloaded version with metrics
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings, struct Metrics& metrics) {
        
	DenseVector<IndexType> previous;
	assert(!settings.repartition);
	
    return partitionGraph(input, coordinates, nodeWeights, previous, settings, metrics);

}

/* wrapper for input in metis-like format

*   vtxDist, size=numPEs,  is a replicated array, it is the prefix sum of the number of nodes per PE
        eg: [0, 15, 25, 50], PE0 has 15 vertices, PE1 10 and PE2 25
*   xadj, size=localN+1, (= IA array of the CSR sparse matrix format), is the prefix sum of the degrees
        of the local nodes, ie, how many non-zero values the row has.
*   adjncy, size=localM (number of local edges = the JA array), contains numbers >0 and <N, each
        number is the global id of the neighboring vertex
*/
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(
    IndexType *vtxDist, IndexType *xadj, IndexType *adjncy, IndexType localM,
    IndexType *vwgt, IndexType dimensions, ValueType *xyz,
    Settings  settings, Metrics metrics ){

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType numPEs = comm->getSize();
    const IndexType thisPE = comm->getRank();

   //SCAI_ASSERT_EQ_ERROR(numPEs+1, sizeof(vtxDist)/sizeof(IndexType), "wrong size for array vtxDist" );
    // ^^ this is wrong,  sizeof(vtxDist)=size of a pointer. How to check if size is correct?
    const IndexType N = vtxDist[numPEs];

    // how to check if array has the correct size?
    const IndexType localN = vtxDist[thisPE+1]-vtxDist[thisPE];
    SCAI_ASSERT_GT_ERROR( localN, 0, "Wrong value for localN for PE " << thisPE << ". Probably wrong vtxDist array");
    SCAI_ASSERT_EQ_ERROR( N, comm->sum(localN), "Global number of vertices mismatch");

    PRINT0("N= " << N);

    // contains the size of each part
    std::vector<IndexType> partSize( numPEs );
    for( int i=0; i<numPEs; i++){
        partSize[i] = vtxDist[i+1]-vtxDist[i];
    }

    // pointer to the general block distribution created using the vtxDist array
    const scai::dmemo::DistributionPtr genBlockDistPtr = scai::dmemo::DistributionPtr ( new scai::dmemo::GenBlockDistribution(N, partSize, comm ) );

    //-----------------------------------------------------
    //
    // convert to scai data types
    //

    //
    // graph
    //

    scai::hmemo::HArray<IndexType> localIA(localN+1, xadj);
    scai::hmemo::HArray<IndexType> localJA(localM, adjncy);
    scai::hmemo::HArray<ValueType> localValues(localM, 1.0);      //TODO: weight 1.0=> no edge weights, change/generalize

    scai::lama::CSRStorage<ValueType> graphLocalStorage( localN, N, localIA, localJA, localValues);
    scai::lama::CSRSparseMatrix<ValueType> graph (genBlockDistPtr, graphLocalStorage);

    SCAI_ASSERT_EQ_ERROR( graph.getLocalNumRows(), localN, "Local size mismatch");
    SCAI_ASSERT_EQ_ERROR( genBlockDistPtr->getLocalSize(), localN, "Local size mismatch");
    
    //
    // coordinates
    //

    std::vector<std::vector<ValueType>> localCoords(dimensions);

    for (IndexType dim = 0; dim < dimensions; dim++) {
        localCoords[dim].resize(localN);
        for( int i=0; i<localN; i++){
            localCoords[dim][i] = xyz[dimensions*i+dim];
        }
    }

    std::vector<scai::lama::DenseVector<ValueType>> coordinates(dimensions);
    for (IndexType dim = 0; dim < dimensions; dim++) {
        coordinates[dim] = scai::lama::DenseVector<ValueType>(genBlockDistPtr, scai::hmemo::HArray<ValueType>(localN, localCoords[dim].data()) );
    }

    //
    // node weights
    //
    scai::lama::DenseVector<ValueType> nodeWeights(genBlockDistPtr, scai::hmemo::HArray<ValueType>(localN, *vwgt));

    return partitionGraph( graph, coordinates, nodeWeights, settings, metrics);
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void ParcoRepart<IndexType, ValueType>::hilbertRedistribution(std::vector<DenseVector<ValueType> >& coordinates, DenseVector<ValueType>& nodeWeights, Settings settings, struct Metrics& metrics) {
    SCAI_REGION_START("ParcoRepart.hilbertRedistribution.sfc")
    scai::dmemo::DistributionPtr inputDist = coordinates[0].getDistributionPtr();
    scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    const IndexType rank = comm->getRank();

    std::chrono::time_point<std::chrono::system_clock> beforeInitPart =  std::chrono::system_clock::now();

    bool nodesUnweighted = (nodeWeights.max() == nodeWeights.min());

    std::chrono::duration<double> migrationCalculation, migrationTime;

    std::vector<ValueType> hilbertIndices = HilbertCurve<IndexType, ValueType>::getHilbertIndexVector(coordinates, settings.sfcResolution, settings.dimensions);
    SCAI_REGION_END("ParcoRepart.hilbertRedistribution.sfc")
    SCAI_REGION_START("ParcoRepart.hilbertRedistribution.sort")
    /**
     * fill sort pair
     */
    scai::hmemo::HArray<IndexType> myGlobalIndices(localN, IndexType(0));
    inputDist->getOwnedIndexes(myGlobalIndices);
    std::vector<sort_pair> localPairs(localN);
    {
        scai::hmemo::ReadAccess<IndexType> rIndices(myGlobalIndices);
        for (IndexType i = 0; i < localN; i++) {
            localPairs[i].value = hilbertIndices[i];
            localPairs[i].index = rIndices[i];
        }
    }

    MPI_Comm mpi_comm = MPI_COMM_WORLD; //maybe cast the communicator ptr to a MPI communicator and get getMPIComm()?
    SQuick::sort<sort_pair>(mpi_comm, localPairs, -1); //could also do this with just the hilbert index - as a valueType
    //IndexType newLocalN = localPairs.size();
    migrationCalculation = std::chrono::system_clock::now() - beforeInitPart;
    metrics.timeMigrationAlgo[rank] = migrationCalculation.count();
    std::chrono::time_point < std::chrono::system_clock > beforeMigration = std::chrono::system_clock::now();
    assert(localPairs.size() > 0);
    SCAI_REGION_END("ParcoRepart.hilbertRedistribution.sort")

    sort_pair minLocalIndex = localPairs[0];
    std::vector<ValueType> sendThresholds(comm->getSize(), minLocalIndex.value);
    std::vector<ValueType> recvThresholds(comm->getSize());

    MPI_Datatype MPI_ValueType = MPI_DOUBLE; //TODO: properly template this
    MPI_Alltoall(sendThresholds.data(), 1, MPI_ValueType, recvThresholds.data(),
            1, MPI_ValueType, mpi_comm); //TODO: replace this monstrosity with a proper call to LAMA
    //comm->all2all(recvThresholds.data(), sendTresholds.data());//TODO: maybe speed up with hypercube
    SCAI_ASSERT_LT_ERROR(recvThresholds[comm->getSize() - 1], 1, "invalid hilbert index");
    // merge to get quantities //Problem: nodes are not sorted according to their hilbert indices, so accesses are not aligned.
    // Need to sort before and after communication
    assert(std::is_sorted(recvThresholds.begin(), recvThresholds.end()));
    std::vector<IndexType> permutation(localN);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](IndexType i, IndexType j){return hilbertIndices[i] < hilbertIndices[j];});

    //now sorting hilbert indices themselves
    std::sort(hilbertIndices.begin(), hilbertIndices.end());
    std::vector<IndexType> quantities(comm->getSize(), 0);
    {
        IndexType p = 0;
        for (IndexType i = 0; i < localN; i++) {
            //increase target block counter if threshold is reached. Skip empty blocks if necessary.
            while (p + 1 < comm->getSize()
                    && recvThresholds[p + 1] <= hilbertIndices[i]) {
                p++;
            }
            assert(p < comm->getSize());

            quantities[p]++;
        }
    }

    SCAI_REGION_START("ParcoRepart.hilbertRedistribution.communicationPlan")
    // allocate sendPlan
    scai::dmemo::CommunicationPlan sendPlan(quantities.data(), comm->getSize());
    SCAI_ASSERT_EQ_ERROR(sendPlan.totalQuantity(), localN,
            "wrong size of send plan")
    // allocate recvPlan - either with allocateTranspose, or directly
    scai::dmemo::CommunicationPlan recvPlan;
    recvPlan.allocateTranspose(sendPlan, *comm);
    IndexType newLocalN = recvPlan.totalQuantity();
    SCAI_REGION_END("ParcoRepart.hilbertRedistribution.communicationPlan")

    if (settings.verbose) {
        PRINT0(std::to_string(localN) + " old local values "
                        + std::to_string(newLocalN) + " new ones.");
    }
    //transmit indices, allowing for resorting of the received values
    std::vector<IndexType> sendIndices(localN);
    {
        SCAI_REGION("ParcoRepart.hilbertRedistribution.permute");
        scai::hmemo::ReadAccess<IndexType> rIndices(myGlobalIndices);
        for (IndexType i = 0; i < localN; i++) {
            assert(permutation[i] < localN);
            assert(permutation[i] >= 0);
            sendIndices[i] = rIndices[permutation[i]];
        }
    }
    std::vector<IndexType> recvIndices(newLocalN);
    comm->exchangeByPlan(recvIndices.data(), recvPlan, sendIndices.data(),
            sendPlan);
    //get new distribution
    scai::hmemo::HArray<IndexType> indexTransport(newLocalN,
            recvIndices.data());
    scai::dmemo::DistributionPtr newDist(
            new scai::dmemo::GeneralDistribution(globalN, indexTransport,
                    comm));
    SCAI_ASSERT_EQUAL(newDist->getLocalSize(), newLocalN,
            "wrong size of new distribution");
    for (IndexType i = 0; i < newLocalN; i++) {
        SCAI_ASSERT_VALID_INDEX_DEBUG(recvIndices[i], globalN, "invalid index");
    }

    {
        SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute");
        // for each dimension: define DenseVector with new distribution, get write access to local values, call exchangeByPlan
        std::vector<ValueType> sendBuffer(localN);
        std::vector<ValueType> recvBuffer(newLocalN);

        for (IndexType d = 0; d < settings.dimensions; d++) {
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::ReadAccess<ValueType> rCoords(coordinates[d].getLocalValues());
                for (IndexType i = 0; i < localN; i++) { //TODO:maybe extract into lambda?
                    sendBuffer[i] = rCoords[permutation[i]]; //TODO: how to make this more cache-friendly? (Probably by using pairs and sorting them.)
                }
            }

            comm->exchangeByPlan(recvBuffer.data(), recvPlan, sendBuffer.data(), sendPlan);
            coordinates[d] = DenseVector<ValueType>(newDist, 0);
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::WriteAccess<ValueType> wCoords(coordinates[d].getLocalValues());
                assert(wCoords.size() == newLocalN);
                for (IndexType i = 0; i < newLocalN; i++) {
                    wCoords[newDist->global2local(recvIndices[i])] =
                            recvBuffer[i];
                }
            }
        }
        // same for node weights
        if (nodesUnweighted) {
            nodeWeights = DenseVector<ValueType>(newDist, 1);
        }
        else {
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
                for (IndexType i = 0; i < localN; i++) {
                    sendBuffer[i] = rWeights[permutation[i]]; //TODO: how to make this more cache-friendly? (Probably by using pairs and sorting them.)
                }
            }
            comm->exchangeByPlan(recvBuffer.data(), recvPlan, sendBuffer.data(), sendPlan);
            nodeWeights = DenseVector<ValueType>(newDist, 0);
            {
                SCAI_REGION("ParcoRepart.hilbertRedistribution.redistribute.permute");
                scai::hmemo::WriteAccess<ValueType> wWeights(nodeWeights.getLocalValues());
                for (IndexType i = 0; i < newLocalN; i++) {
                    wWeights[newDist->global2local(recvIndices[i])] = recvBuffer[i];
                }
            }
        }
    }
    migrationTime = std::chrono::system_clock::now() - beforeMigration;
    metrics.timeFirstDistribution[rank] = migrationTime.count();
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, DenseVector<IndexType>& previous, Settings settings,struct Metrics& metrics)
{
	IndexType k = settings.numBlocks;
	ValueType epsilon = settings.epsilon;
    const IndexType dimensions = coordinates.size();

	SCAI_REGION( "ParcoRepart.partitionGraph" )

	std::chrono::time_point<std::chrono::steady_clock> start, afterSFC, round;
	start = std::chrono::steady_clock::now();

	SCAI_REGION_START("ParcoRepart.partitionGraph.inputCheck")
	/**
	* check input arguments for sanity
	*/
	IndexType n = input.getNumRows();
    for( int d=0; d<dimensions; d++){
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
	const IndexType rank = comm->getRank();


	if( !coordDist->isEqual( *inputDist) ){
		throw std::runtime_error( "Distributions should be equal.");
	}

	bool nodesUnweighted;
    if (nodeWeights.size() == 0) {
        nodeWeights = DenseVector<ValueType>(inputDist, 1);
        nodesUnweighted = true;
    } else {
        nodesUnweighted = (nodeWeights.max() == nodeWeights.min());
    }

    SCAI_REGION_END("ParcoRepart.partitionGraph.inputCheck")
	{
		SCAI_REGION("ParcoRepart.synchronize")
		comm->synchronize();
	}
	
	SCAI_REGION_START("ParcoRepart.partitionGraph.initialPartition")
	// get an initial partition
	DenseVector<IndexType> result;
	
	assert(nodeWeights.getDistribution().isEqual(*inputDist));
	
	
	//-------------------------
	//
	// timing info
	
	std::chrono::duration<double> kMeansTime = std::chrono::duration<double>(0.0);
	std::chrono::duration<double> migrationCalculation = std::chrono::duration<double> (0.0);
	std::chrono::duration<double> migrationTime= std::chrono::duration<double>(0.0);
	std::chrono::duration<double> secondRedistributionTime = std::chrono::duration<double>(0.0) ;
	std::chrono::duration<double> partitionTime= std::chrono::duration<double>(0.0);
	
	std::chrono::time_point<std::chrono::system_clock> beforeInitPart =  std::chrono::system_clock::now();
	
	if( settings.initialPartition==InitialPartitioningMethods::SFC) {
		PRINT0("Initial partition with SFCs");
		result= ParcoRepart<IndexType, ValueType>::hilbertPartition(coordinates, settings);
		std::chrono::duration<double> sfcTime = std::chrono::system_clock::now() - beforeInitPart;
		if ( settings.verbose ) {
			ValueType totSFCTime = ValueType(comm->max(sfcTime.count()) );
			if(comm->getRank() == 0)
				std::cout << "SFC Time:" << totSFCTime << std::endl;
		}
	} else if ( settings.initialPartition==InitialPartitioningMethods::Pixel) {
		PRINT0("Initial partition with pixels.");
		result = ParcoRepart<IndexType, ValueType>::pixelPartition(coordinates, settings);
	} else if ( settings.initialPartition == InitialPartitioningMethods::Spectral) {
		PRINT0("Initial partition with spectral");
		result = ITI::SpectralPartition<IndexType, ValueType>::getPartition(input, coordinates, settings);
	} else if (settings.initialPartition == InitialPartitioningMethods::KMeans) {
	    if (comm->getRank() == 0) {
	        std::cout << "Initial partition with K-Means" << std::endl;
	    }

		//prepare coordinates for k-means
		std::vector<DenseVector<ValueType> > coordinateCopy = coordinates;
		DenseVector<ValueType> nodeWeightCopy = nodeWeights;
		if (comm->getSize() > 1 && (settings.dimensions == 2 || settings.dimensions == 3)) {
			SCAI_REGION("ParcoRepart.partitionGraph.initialPartition.prepareForKMeans")
			Settings migrationSettings = settings;
			migrationSettings.numBlocks = comm->getSize();
			migrationSettings.epsilon = settings.epsilon;
			//migrationSettings.bisect = true;
			
			// the distribution for the initial migration   
			scai::dmemo::DistributionPtr initMigrationPtr;
			
			if (!settings.repartition || comm->getSize() != settings.numBlocks) {
				
				// the distribution for the initial migration   
				scai::dmemo::DistributionPtr initMigrationPtr;
				
				if (settings.initialMigration == InitialPartitioningMethods::SFC) {
					
					hilbertRedistribution(coordinateCopy, nodeWeightCopy, settings, metrics);
				}else {
					DenseVector<ValueType> convertedWeights(nodeWeights);
					DenseVector<IndexType> tempResult;				
					if (settings.initialMigration == InitialPartitioningMethods::Multisection) {
						tempResult  = ITI::MultiSection<IndexType, ValueType>::getPartitionNonUniform(input, coordinates, convertedWeights, migrationSettings);
					} else if (settings.initialMigration == InitialPartitioningMethods::KMeans) {
						std::vector<IndexType> migrationBlockSizes( migrationSettings.numBlocks, n/migrationSettings.numBlocks );
						tempResult = ITI::KMeans::computePartition(coordinates, convertedWeights, migrationBlockSizes, migrationSettings);
					}
					
					initMigrationPtr = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( tempResult.getDistribution(), tempResult.getLocalValues() ) );
					
					if (settings.initialMigration == InitialPartitioningMethods::None) {
						//nothing to do
						initMigrationPtr = inputDist;
					} else {
						throw std::logic_error("Method not yet supported for preparing for K-Means");
					}
					
					migrationCalculation = std::chrono::system_clock::now() - beforeInitPart;
					metrics.timeMigrationAlgo[rank]  = migrationCalculation.count();
					
					std::chrono::time_point<std::chrono::system_clock> beforeMigration =  std::chrono::system_clock::now();
					
					scai::dmemo::Redistributor prepareRedist(initMigrationPtr, nodeWeights.getDistributionPtr());
					
					std::chrono::time_point<std::chrono::system_clock> afterRedistConstruction =  std::chrono::system_clock::now();
					
					std::chrono::duration<double> redist = (afterRedistConstruction - beforeMigration);
					metrics.timeConstructRedistributor[rank] = redist.count();
					
					if (nodesUnweighted) {
						nodeWeightCopy = DenseVector<ValueType>(initMigrationPtr, nodeWeights.getLocalValues()[0]);
					} else {
						nodeWeightCopy.redistribute(prepareRedist);
					}
					
					for (IndexType d = 0; d < dimensions; d++) {
						coordinateCopy[d].redistribute(prepareRedist);
					}
					
					if (settings.repartition) {
						previous.redistribute(prepareRedist);
					}
					
					migrationTime = std::chrono::system_clock::now() - beforeMigration;
					metrics.timeFirstDistribution[rank]  = migrationTime.count();
				}
			}
			
		}
		
		const ValueType weightSum = nodeWeights.sum();
		
		// vector of size k, each element represents the size of one block
		std::vector<IndexType> blockSizes;
		if( settings.blockSizes.empty() ){
			blockSizes.assign( settings.numBlocks, weightSum/settings.numBlocks );
		}else{
			blockSizes = settings.blockSizes;
		}
		SCAI_ASSERT( blockSizes.size()==settings.numBlocks , "Wrong size of blockSizes vector: " << blockSizes.size() );
		
		std::chrono::time_point<std::chrono::system_clock> beforeKMeans =  std::chrono::system_clock::now();
		if (settings.repartition) {
			result = ITI::KMeans::computeRepartition(coordinateCopy, nodeWeightCopy, blockSizes, previous, settings);
		} else {
			result = ITI::KMeans::computePartition(coordinateCopy, nodeWeightCopy, blockSizes, settings);
		}
		
		kMeansTime = std::chrono::system_clock::now() - beforeKMeans;
		metrics.timeKmeans[rank] = kMeansTime.count();
		//timeForKmeans = ValueType ( comm->max(kMeansTime.count() ));
            assert(scai::utilskernel::HArrayUtils::min(result.getLocalValues()) >= 0);
            assert(scai::utilskernel::HArrayUtils::max(result.getLocalValues()) < k);
		
		if (settings.verbose) {
			ValueType totKMeansTime = ValueType( comm->max(kMeansTime.count()) );
			if(comm->getRank() == 0)
				std::cout << "K-Means, Time:" << totKMeansTime << std::endl;
		}
		
            assert(result.max() == settings.numBlocks -1);
            assert(result.min() == 0);
		
	} else if (settings.initialPartition == InitialPartitioningMethods::Multisection) {// multisection
		PRINT0("Initial partition with multisection");
		DenseVector<ValueType> convertedWeights(nodeWeights);
		result = ITI::MultiSection<IndexType, ValueType>::getPartitionNonUniform(input, coordinates, convertedWeights, settings);
		std::chrono::duration<double> msTime = std::chrono::system_clock::now() - beforeInitPart;
		
		if ( settings.verbose ) {
			ValueType totMsTime = ValueType ( comm->max(msTime.count()) );
			if(comm->getRank() == 0)
				std::cout << "MS Time:" << totMsTime << std::endl;
		}
	} else if (settings.initialPartition == InitialPartitioningMethods::None) {
		//no need to explicitly check for repartitioning mode or not.
		assert(comm->getSize() == settings.numBlocks);
		result = DenseVector<IndexType>(input.getRowDistributionPtr(), comm->getRank());
	}
	else {
		throw std::runtime_error("Initial Partitioning mode undefined.");
	}
	
	SCAI_REGION_END("ParcoRepart.partitionGraph.initialPartition")
	
	
	if( settings.outFile!="-" and settings.writeInFile ){
		FileIO<IndexType, ValueType>::writePartitionParallel( result, settings.outFile+"_initPart.partition" );
	}
	
	//-----------------------------------------------------------
	//
	// At this point we have the initial, geometric partition.
	//
	
	// if noRefinement then these are the times, if we do refinement they will be overwritten
	partitionTime =  std::chrono::system_clock::now() - beforeInitPart;
	metrics.timePreliminary[rank] = partitionTime.count();
	
	if (comm->getSize() == k) {
		//WARNING: the result  is not redistributed. must redistribute afterwards
		if(  !settings.noRefinement ) {
			SCAI_REGION("ParcoRepart.partitionGraph.initialRedistribution")
			/**
			 * redistribute to prepare for local refinement
			 */
			std::chrono::time_point<std::chrono::system_clock> beforeSecondRedistributiom =  std::chrono::system_clock::now();
			
			scai::dmemo::Redistributor resultRedist(result.getLocalValues(), result.getDistributionPtr());//TODO: Wouldn't it be faster to use a GeneralDistribution here?
			result = DenseVector<IndexType>(resultRedist.getTargetDistributionPtr(), comm->getRank());
			
			scai::dmemo::Redistributor redistributor(resultRedist.getTargetDistributionPtr(), input.getRowDistributionPtr());
			input.redistribute(redistributor, noDist);
			if (settings.useGeometricTieBreaking) {
				for (IndexType d = 0; d < dimensions; d++) {
					coordinates[d].redistribute(redistributor);
				}
			}
			nodeWeights.redistribute(redistributor);
			
			secondRedistributionTime =  std::chrono::system_clock::now() - beforeSecondRedistributiom;
			//ValueType timeForSecondRedistr = ValueType ( comm->max(secondRedistributionTime.count() ));
			
			partitionTime =  std::chrono::system_clock::now() - beforeInitPart;
			//ValueType timeForInitPart = ValueType ( comm->max(partitionTime.count() ));
			ValueType cut = comm->sum(ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(input, true)) / 2;//TODO: this assumes that the graph is unweighted
			ValueType imbalance = GraphUtils::computeImbalance<IndexType, ValueType>(result, k, nodeWeights);
			
			
			//-----------------------------------------------------------
			//
			// output: in std and file
			//
			
			if (settings.verbose ) {
				ValueType timeToCalcInitMigration = comm->max(migrationCalculation.count()) ;   
				ValueType timeForFirstRedistribution = comm->max( migrationTime.count() );
				ValueType timeForKmeans = comm->max( kMeansTime.count() );
				ValueType timeForSecondRedistr = comm->max( secondRedistributionTime.count() );
				ValueType timeForInitPart = comm->max( partitionTime.count() );
				
				if(comm->getRank() == 0 ){
					std::cout<< std::endl << "\033[1;32mTiming: migration algo: "<< timeToCalcInitMigration << ", 1st redistr: " << timeForFirstRedistribution << ", only k-means: " << timeForKmeans <<", only 2nd redistr: "<< timeForSecondRedistr <<", total:" << timeForInitPart << std::endl;
					std::cout << "# of cut edges:" << cut << ", imbalance:" << imbalance<< " \033[0m" <<std::endl << std::endl;
				}
			}
			
			metrics.timeSecondDistribution[rank] = secondRedistributionTime.count();
			metrics.timePreliminary[rank] = partitionTime.count();
			
			metrics.preliminaryCut = cut;
			metrics.preliminaryImbalance = imbalance;
			
			//IndexType numRefinementRounds = 0;
			
			SCAI_REGION_START("ParcoRepart.partitionGraph.multiLevelStep")
			scai::dmemo::Halo halo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(input);
			ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(input, result, nodeWeights, coordinates, halo, settings);
			SCAI_REGION_END("ParcoRepart.partitionGraph.multiLevelStep")
		}
	} else {
		result.redistribute(inputDist);
		if (comm->getRank() == 0 && !settings.noRefinement) {
			std::cout << "Local refinement only implemented for one block per process. Called with " << comm->getSize() << " processes and " << k << " blocks." << std::endl;
		}
	}
	
	return result;
}
//--------------------------------------------------------------------------------------- 

//TODO: take node weights into account
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings){

    auto uniformWeights = fill<DenseVector<ValueType>>(coordinates[0].getDistributionPtr(), 1);
    return hilbertPartition( coordinates, settings);
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings){
    SCAI_REGION( "ParcoRepart.hilbertPartition" )
    	
    std::chrono::time_point<std::chrono::steady_clock> start, afterSFC;
    start = std::chrono::steady_clock::now();
    
    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
    
    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    assert(dimensions == settings.dimensions);
    const IndexType localN = coordDist->getLocalSize();
    const IndexType globalN = coordDist->getGlobalSize();
    
    if (k != comm->getSize() && comm->getRank() == 0) {
    	throw std::logic_error("Hilbert curve partition only implemented for same number of blocks and processes.");
    }
 
 
    //
    // vector of size k, each element represents the size of each block
    //
    std::vector<IndexType> blockSizes;
	//TODO: for nowm assume uniform nodeweights
    IndexType weightSum = globalN;// = nodeWeights.sum();
    if( settings.blockSizes.empty() ){
        blockSizes.assign( settings.numBlocks, weightSum/settings.numBlocks );
    }else{
        blockSizes = settings.blockSizes;
    }
    SCAI_ASSERT( blockSizes.size()==settings.numBlocks , "Wrong size of blockSizes vector: " << blockSizes.size() );
    
    /**
     * Several possibilities exist for choosing the recursion depth.
     * Either by user choice, or by the maximum fitting into the datatype, or by the minimum distance between adjacent points.
     */
    const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(globalN), double(21));
    
    /**
     *	create space filling curve indices.
     */
    
    scai::lama::DenseVector<ValueType> hilbertIndices(coordDist, 0);
	std::vector<ValueType> localHilberIndices = HilbertCurve<IndexType,ValueType>::getHilbertIndexVector(coordinates, recursionDepth, dimensions);
	hilbertIndices.assign( scai::hmemo::HArray<ValueType>( localHilberIndices.size(), localHilberIndices.data()) , coordDist);

    //TODO: use the blockSizes vector
    //TODO: take into account node weights: just sorting will create imbalanced blocks, not so much in number of node but in the total weight of each block
    
    /**
     * now sort the global indices by where they are on the space-filling curve.
     */

    std::vector<IndexType> newLocalIndices;

    {
        SCAI_REGION( "ParcoRepart.hilbertPartition.sorting" );
        //TODO: maybe call getSortedHilbertIndices here?
        int typesize;
        MPI_Type_size(SortingDatatype<sort_pair>::getMPIDatatype(), &typesize);
        //assert(typesize == sizeof(sort_pair)); //not valid for int_double, presumably due to padding
        
        std::vector<sort_pair> localPairs(localN);

        //fill with local values
        long indexSum = 0;//for sanity checks
        scai::hmemo::ReadAccess<ValueType> localIndices(hilbertIndices.getLocalValues());//Segfault happening here, likely due to stack overflow. TODO: fix
        for (IndexType i = 0; i < localN; i++) {
        	localPairs[i].value = localIndices[i];
        	localPairs[i].index = coordDist->local2global(i);
        	indexSum += localPairs[i].index;
        }

        //create checksum
        const long checkSum = comm->sum(indexSum);
        //TODO: int overflow?
        SCAI_ASSERT_EQ_ERROR(checkSum , (long(globalN)*(long(globalN)-1))/2, "Sorting checksum is wrong (possible IndexType overflow?).");

        //call distributed sort
        //MPI_Comm mpi_comm, std::vector<value_type> &data, long long global_elements = -1, Compare comp = Compare()
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        SQuick::sort<sort_pair>(mpi_comm, localPairs, -1);

        //copy indices into array
        const IndexType newLocalN = localPairs.size();
        newLocalIndices.resize(newLocalN);

        for (IndexType i = 0; i < newLocalN; i++) {
        	newLocalIndices[i] = localPairs[i].index;
        }

        //sort local indices for general distribution
        std::sort(newLocalIndices.begin(), newLocalIndices.end());

        //check size and sanity
        SCAI_ASSERT_LT_ERROR( *std::max_element(newLocalIndices.begin(), newLocalIndices.end()) , globalN, "Too large index (possible IndexType overflow?).");
        SCAI_ASSERT_EQ_ERROR( comm->sum(newLocalIndices.size()), globalN, "distribution mismatch");

        //check checksum
        long indexSumAfter = 0;
        for (IndexType i = 0; i < newLocalN; i++) {
        	indexSumAfter += newLocalIndices[i];
        }

        const long newCheckSum = comm->sum(indexSumAfter);
        SCAI_ASSERT( newCheckSum == checkSum, "Old checksum: " << checkSum << ", new checksum: " << newCheckSum );

        //possible optimization: remove dummy values during first copy, then directly copy into HArray and sort with pointers. Would save one copy.
    }
    
    DenseVector<IndexType> result;
	
    {
    	assert(!coordDist->isReplicated() && comm->getSize() == k);
        SCAI_REGION( "ParcoRepart.hilbertPartition.createDistribution" );

        scai::hmemo::HArray<IndexType> indexTransport(newLocalIndices.size(), newLocalIndices.data());
        assert(comm->sum(indexTransport.size()) == globalN);
        scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, indexTransport, comm));
        
        if (comm->getRank() == 0) std::cout << "Created distribution." << std::endl;
        result = fill<DenseVector<IndexType>>(newDistribution, comm->getRank());
        if (comm->getRank() == 0) std::cout << "Created initial partition." << std::endl;
    }

    return result;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::pixelPartition(const std::vector<DenseVector<ValueType>> &coordinates, Settings settings){
    SCAI_REGION( "ParcoRepart.pixelPartition" )
    	
    SCAI_REGION_START("ParcoRepart.pixelPartition.initialise")
    std::chrono::time_point<std::chrono::steady_clock> start, round;
    start = std::chrono::steady_clock::now();
    
    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
    
    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    const IndexType localN = coordDist->getLocalSize();
    const IndexType globalN = coordDist->getGlobalSize();
    
    if (k != comm->getSize() && comm->getRank() == 0) {
    	throw std::logic_error("Pixel partition only implemented for same number of blocks and processes.");
    }

    std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    DenseVector<IndexType> result(coordDist, 0);
    
    //TODO: probably minimum is not needed
    //TODO: if we know maximum from the input we could save that although is not too costly
    
    /**
     * get minimum / maximum of local coordinates
     */
    for (IndexType dim = 0; dim < dimensions; dim++) {
        //get local parts of coordinates
        scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[dim].getLocalValues() );
        for (IndexType i = 0; i < localN; i++) {
            ValueType coord = localPartOfCoords[i];
            if (coord < minCoords[dim]) minCoords[dim] = coord;
            if (coord > maxCoords[dim]) maxCoords[dim] = coord;
        }
    }
    
    /**
     * communicate to get global min / max
     */
    for (IndexType dim = 0; dim < dimensions; dim++) {
        minCoords[dim] = comm->min(minCoords[dim]);
        maxCoords[dim] = comm->max(maxCoords[dim]);
    }
   
    // measure density with rounding
    // have to handle 2D and 3D cases seperately
    const IndexType sideLen = settings.pixeledSideLen;
    const IndexType cubeSize = std::pow(sideLen, dimensions);
    
    //TODO: generalize this to arbitrary dimensions, do not handle 2D and 3D differently
    //TODO: by a  for(int d=0; d<dimension; d++){ ... }
    // a 2D or 3D arrays as a one dimensional vector
    // [i][j] is in position: i*sideLen + j
    // [i][j][k] is in: i*sideLen*sideLen + j*sideLen + k
    
    //std::vector<IndexType> density( cubeSize ,0);
    scai::hmemo::HArray<IndexType> density( cubeSize, IndexType(0) );
    scai::hmemo::WriteAccess<IndexType> wDensity(density);

    SCAI_REGION_END("ParcoRepart.pixelPartition.initialise")
    
    if(dimensions==2){
        SCAI_REGION( "ParcoRepart.pixelPartition.localDensity" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );

        IndexType scaledX, scaledY;
        //the +1 is needed
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;
        
        for(IndexType i=0; i<localN; i++){
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            IndexType pixelInd = scaledX*sideLen + scaledY;      
            SCAI_ASSERT( pixelInd < wDensity.size(), "Index too big: "<< std::to_string(pixelInd) );
            ++wDensity[pixelInd];
        }
    }else if(dimensions==3){
        SCAI_REGION( "ParcoRepart.pixelPartition.localDensity" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );
        
        IndexType scaledX, scaledY, scaledZ;
        
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;
        IndexType maxZ = maxCoords[2]+1;
        
        for(IndexType i=0; i<localN; i++){
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            scaledZ = coordAccess2[i]/maxZ * sideLen;
            IndexType pixelInd = scaledX*sideLen*sideLen + scaledY*sideLen + scaledZ;
            
            SCAI_ASSERT( pixelInd < wDensity.size(), "Index too big: "<< std::to_string(pixelInd) );  
            ++wDensity[pixelInd];
        }
    }else{
        throw std::runtime_error("Available only for 2D and 3D. Data given have dimension:" + std::to_string(dimensions) );
    }
    wDensity.release();

    // sum density from all PEs 
    {
        SCAI_REGION( "ParcoRepart.pixelPartition.sumDensity" )
        comm->sumArray( density );
    }
    
    //TODO: is that needed? we just can overwrite density array.
    // use the summed density as a Dense vector
    scai::lama::DenseVector<IndexType> sumDensity( density );
    /*
    if(comm->getRank()==0){
        ITI::aux::writeHeatLike_local_2D(density, sideLen, dimensions, "heat_"+settings.fileName+".plt");
    }
    */
    //
    //using the summed density get an initial pixeled partition
    
    std::vector<IndexType> pixeledPartition( density.size() , -1);
    
    IndexType pointsLeft= globalN;
    IndexType pixelsLeft= cubeSize;
    IndexType maxBlockSize = globalN/k * 1.02; // allowing some imbalance
    PRINT0("max allowed block size: " << maxBlockSize );         
    IndexType thisBlockSize;
    
    //for all the blocks
    for(IndexType block=0; block<k; block++){
        SCAI_REGION( "ParcoRepart.pixelPartition.localPixelGrowing")
           
        ValueType averagePointsPerPixel = ValueType(pointsLeft)/pixelsLeft;
        // a factor to force the block to spread more
        ValueType spreadFactor;
        // make a block spread towards the borders (and corners) of our input space 
        ValueType geomSpread;
        // to measure the distance from the first, center pixel
        ValueType pixelDistance;
        
        // start from the densest pixel
        //IndexType maxDensityPixel = std::distance( sumDensity.begin(), std::max_element(sumDensity.begin(), sumDensity.end()) );
        
        //TODO: sumDensity is local/not distributed. No need for that, just to avoid getValue.
        scai::hmemo::WriteAccess<IndexType> localSumDens( sumDensity.getLocalValues() );
        
        //TODO: bad way to do that. linear time for every block. maybe sort or use a priority queue
        IndexType maxDensityPixel=-1;
        IndexType maxDensity=-1;
        for(IndexType ii=0; ii<sumDensity.size(); ii++){
            if(localSumDens[ii]>maxDensity){
                maxDensityPixel = ii;
                maxDensity= localSumDens[ii];
            }
        }

        if(maxDensityPixel<0){
            PRINT0("Max density pixel id = -1. Should not happen(?) or pixels are finished. For block "<< block<< " and k= " << k);
            break;
        }
        
        SCAI_ASSERT(maxDensityPixel < sumDensity.size(), "Too big index: " + std::to_string(maxDensityPixel));
        SCAI_ASSERT(maxDensityPixel >= 0, "Negative index: " + std::to_string(maxDensityPixel));
        spreadFactor = averagePointsPerPixel/localSumDens[ maxDensityPixel ];

        //TODO: change to more appropriate data type
        // insert all the neighbouring pixels
        std::vector<std::pair<IndexType, ValueType>> border; 
        std::vector<IndexType> neighbours = ParcoRepart<IndexType, ValueType>::neighbourPixels( maxDensityPixel, sideLen, dimensions);

        // insert in border if not already picked
        for(IndexType j=0; j<neighbours.size(); j++){
            // make sure this neighbour does not belong to another block
            if(localSumDens[ neighbours[j]] != -1 ){
                std::pair<IndexType, ValueType> toInsert;
                toInsert.first = neighbours[j];
                SCAI_ASSERT(neighbours[j] < sumDensity.size(), "Too big index: " + std::to_string(neighbours[j]));
                SCAI_ASSERT(neighbours[j] >= 0, "Negative index: " + std::to_string(neighbours[j]));
                geomSpread = 1 + 1/std::log2(sideLen)*( std::abs(sideLen/2 - neighbours[j]/sideLen)/(0.8*sideLen/2) + std::abs(sideLen/2 - neighbours[j]%sideLen)/(0.8*sideLen/2) );
                //PRINT0( geomSpread );            
                // value to pick a border node
                pixelDistance = aux<IndexType, ValueType>::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);
                toInsert.second = (1/pixelDistance)* geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[maxDensityPixel], 0.5) );
                border.push_back(toInsert);
            }
        }
        thisBlockSize = localSumDens[maxDensityPixel];
        
        pixeledPartition[maxDensityPixel] = block;
        
        // set this pixel to -1 so it is not picked again
        localSumDens[maxDensityPixel] = -1;
        

        while(border.size() !=0 ){      // there are still pixels to check
            
            //TODO: different data type to avoid that
            // sort border by the value in increasing order 
            std::sort( border.begin(), border.end(),
                       [](const std::pair<IndexType, ValueType> &left, const std::pair<IndexType, ValueType> &right){
                           return left.second < right.second; });
             
            std::pair<IndexType, ValueType> bestPixel;
            IndexType bestIndex=-1;
            do{
                bestPixel = border.back();                
                border.pop_back();
                bestIndex = bestPixel.first;
                
            }while( localSumDens[ bestIndex] +thisBlockSize > maxBlockSize and border.size()>0); // this pixel is too big
            
            // picked last pixel in border but is too big
            if(localSumDens[ bestIndex] +thisBlockSize > maxBlockSize ){
                break;
            }
            SCAI_ASSERT(localSumDens[ bestIndex ] != -1, "Wrong pixel choice.");
            
            // this pixel now belongs in this block
            SCAI_ASSERT(bestIndex < sumDensity.size(), "Wrong pixel index: " + std::to_string(bestIndex));
            pixeledPartition[ bestIndex ] = block;
            thisBlockSize += localSumDens[ bestIndex ];
            --pixelsLeft;
            pointsLeft -= localSumDens[ bestIndex ];
            
            //averagePointsPerPixel = ValueType(pointsLeft)/pixelsLeft;
            //spreadFactor = localSumDens[ bestIndex ]/averagePointsPerPixel;
            //spreadFactor = (k-block)*averagePointsPerPixel/localSumDens[ bestIndex ];
            spreadFactor = averagePointsPerPixel/localSumDens[ bestIndex ];

            //get the neighbours of the new pixel
            std::vector<IndexType> neighbours = ParcoRepart<IndexType, ValueType>::neighbourPixels( bestIndex, sideLen, dimensions);
            
            //insert neighbour in border or update value if already there
            for(IndexType j=0; j<neighbours.size(); j++){

                SCAI_ASSERT(neighbours[j] < sumDensity.size(), "Too big index: " + std::to_string(neighbours[j]));
                SCAI_ASSERT(neighbours[j] >= 0, "Negative index: " + std::to_string(neighbours[j]));
                
                //geomSpread = 1 + 1.0/detailLvl*( std::abs(sideLen/2.0 - neighbours[j]/sideLen)/(0.8*sideLen/2.0) + std::abs(sideLen/2.0 - neighbours[j]%sideLen)/(0.8*sideLen/2.0) );
                IndexType ngbrX = neighbours[j]/sideLen;
                IndexType ngbrY = neighbours[j]%sideLen;

                geomSpread= 1+ (std::pow(ngbrX-sideLen/2, 2) + std::pow(ngbrY-sideLen/2, 2))*(2/std::pow(sideLen,2));
                //geomSpread = geomSpread * geomSpread;// std::pow(geomSpread, 0.5);
                //
                geomSpread = 1;
                //
                
                if( localSumDens[ neighbours[j]] == -1){ // this pixel is already picked by a block (maybe this)
                    continue;
                }else{
                    bool inBorder = false;
                    
                    for(IndexType l=0; l<border.size(); l++){                        
                        if( border[l].first == neighbours[j]){ // its already in border, update value
                            //border[l].second = 1.3*border[l].second + geomSpread * (spreadFactor*(std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5) );
                            pixelDistance = aux<IndexType, ValueType>::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);    
                            border[l].second += geomSpread*  (1/(pixelDistance*pixelDistance))* ( spreadFactor *std::pow(localSumDens[neighbours[j]], 0.5) + std::pow(localSumDens[bestIndex], 0.5) );
                            inBorder= true;
                        }
                    }
                    if(!inBorder){
                        std::pair<IndexType, ValueType> toInsert;
                        toInsert.first = neighbours[j];
                        //toInsert.second = geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5));
                        pixelDistance = aux<IndexType, ValueType>::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);    
                        //toInsert.second = (1/(pixelDistance*pixelDistance))* geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5));
                        toInsert.second = geomSpread*  (1/(pixelDistance*pixelDistance))* ( spreadFactor *(std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5) );
                        //toInsert.second = geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5))/(std::pow( std::abs( localSumDens[bestIndex] - localSumDens[neighbours[j]]),0.5));
                        border.push_back(toInsert);
                    }
                }
            }
            
            localSumDens[ bestIndex ] = -1;
        }
        //PRINT0("##### final blockSize for block "<< block << ": "<< thisBlockSize);      
    } // for(IndexType block=0; block<k; block++)
    
    // assign all orphan pixels to last block
    for(unsigned long int pp=0; pp<pixeledPartition.size(); pp++){  
        scai::hmemo::ReadAccess<IndexType> localSumDens( sumDensity.getLocalValues() );
        if(pixeledPartition[pp] == -1){
            pixeledPartition[pp] = k-1;     
            thisBlockSize += localSumDens[pp];
        }
    }   
    //PRINT0("##### final blockSize for block "<< k-1 << ": "<< thisBlockSize);

    // here all pixels should have a partition 
    
    //=========
    
    // set your local part of the partition/result
    scai::hmemo::WriteOnlyAccess<IndexType> wLocalPart ( result.getLocalValues() );
    
    if(dimensions==2){
        SCAI_REGION( "ParcoRepart.pixelPartition.setLocalPartition" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        
        IndexType scaledX, scaledY;
        //the +1 is needed
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;
     
        for(IndexType i=0; i<localN; i++){
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            IndexType densInd = scaledX*sideLen + scaledY;
            //PRINT(densInd << " # " << coordAccess0[i] << " _ " << coordAccess1[i] );            
            SCAI_ASSERT( densInd < density.size(), "Index too big: "<< std::to_string(densInd) );

            wLocalPart[i] = pixeledPartition[densInd];
            SCAI_ASSERT(wLocalPart[i] < k, " Wrong block number: " + std::to_string(wLocalPart[i] ) );
        }
    }else if(dimensions==3){
        SCAI_REGION( "ParcoRepart.pixelPartition.setLocalPartition" )
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[2].getLocalValues() );
        
        IndexType scaledX, scaledY, scaledZ;
        
        IndexType maxX = maxCoords[0]+1;
        IndexType maxY = maxCoords[1]+1;
        IndexType maxZ = maxCoords[2]+1;
        
        for(IndexType i=0; i<localN; i++){
            scaledX = coordAccess0[i]/maxX * sideLen;
            scaledY = coordAccess1[i]/maxY * sideLen;
            scaledZ = coordAccess2[i]/maxZ * sideLen;
            IndexType densInd = scaledX*sideLen*sideLen + scaledY*sideLen + scaledZ;
            
            SCAI_ASSERT( densInd < density.size(), "Index too big: "<< std::to_string(densInd) );
            wLocalPart[i] = pixeledPartition[densInd];  
            SCAI_ASSERT(wLocalPart[i] < k, " Wrong block number: " + std::to_string(wLocalPart[i] ) );
        }
    }else{
        throw std::runtime_error("Available only for 2D and 3D. Data given have dimension:" + std::to_string(dimensions) );
    }
    wLocalPart.release();
    
    return result;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(const CSRSparseMatrix<ValueType> &input, const bool weighted) {
	SCAI_REGION( "ParcoRepart.localSumOutgoingEdges" )
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	IndexType sumOutgoingEdgeWeights = 0;
	for (IndexType j = 0; j < ja.size(); j++) {
		if (!input.getRowDistributionPtr()->isLocal(ja[j])) sumOutgoingEdgeWeights += weighted ? values[j] : 1;
	}

	return sumOutgoingEdgeWeights;
}
//--------------------------------------------------------------------------------------- 
 
template<typename IndexType, typename ValueType>
IndexType ParcoRepart<IndexType, ValueType>::localBlockSize(const DenseVector<IndexType> &part, IndexType blockID) {
	SCAI_REGION( "ParcoRepart.localBlockSize" )
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
void ITI::ParcoRepart<IndexType, ValueType>::checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input) {
	SCAI_REGION( "ParcoRepart.checkLocalDegreeSymmetry" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType localN = inputDist->getLocalSize();

	const CSRStorage<ValueType>& storage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> localIa(storage.getIA());
	const scai::hmemo::ReadAccess<IndexType> localJa(storage.getJA());

	std::vector<IndexType> inDegree(localN, 0);
	std::vector<IndexType> outDegree(localN, 0);
	for (IndexType i = 0; i < localN; i++) {
		IndexType globalI = inputDist->local2global(i);
		const IndexType beginCols = localIa[i];
		const IndexType endCols = localIa[i+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType globalNeighbor = localJa[j];

			if (globalNeighbor != globalI && inputDist->isLocal(globalNeighbor)) {
				IndexType localNeighbor = inputDist->global2local(globalNeighbor);
				outDegree[i]++;
				inDegree[localNeighbor]++;
			}
		}
	}

	for (IndexType i = 0; i < localN; i++) {
		if (inDegree[i] != outDegree[i]) {
			//now check in detail:
			IndexType globalI = inputDist->local2global(i);
			for (IndexType j = localIa[i]; j < localIa[i+1]; j++) {
				IndexType globalNeighbor = localJa[j];
				if (inputDist->isLocal(globalNeighbor)) {
					IndexType localNeighbor = inputDist->global2local(globalNeighbor);
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
}
//-----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector< std::vector<IndexType>> ParcoRepart<IndexType, ValueType>::getGraphEdgeColoring_local(CSRSparseMatrix<ValueType> &adjM, IndexType &colors) {
    SCAI_REGION("ParcoRepart.coloring");
    using namespace boost;
    IndexType N= adjM.getNumRows();
    assert( N== adjM.getNumColumns() ); // numRows = numColumns
    
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));
    if (!adjM.getRowDistributionPtr()->isReplicated()) {
    	adjM.redistribute(noDist, noDist);
    	//throw std::runtime_error("Input matrix must be replicated.");
    }

    // use boost::Graph and boost::edge_coloring()
    typedef adjacency_list<vecS, vecS, undirectedS, no_property, size_t, no_property> Graph;
    //typedef std::pair<std::size_t, std::size_t> Pair;
    Graph G(N);
    
    // retG[0][i] the first node, retG[1][i] the second node, retG[2][i] the color of the edge
    std::vector< std::vector<IndexType>> retG(3);
    
	const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    // create graph G from the input adjacency matrix
    for(IndexType i=0; i<N; i++){
    	//we replicated the matrix, so global indices are local indices
    	const IndexType globalI = i;
    	for (IndexType j = ia[i]; j < ia[i+1]; j++) {
    		if (globalI < ja[j]) {
				boost::add_edge(globalI, ja[j], G);
				retG[0].push_back(globalI);
				retG[1].push_back(ja[j]);
    		}
    	}
    }
    
    colors = boost::edge_coloring(G, boost::get( boost::edge_bundle, G));
    
    for (size_t i = 0; i <retG[0].size(); i++) {
        retG[2].push_back( G[ boost::edge( retG[0][i],  retG[1][i], G).first] );
    }
    
    return retG;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<DenseVector<IndexType>> ParcoRepart<IndexType, ValueType>::getCommunicationPairs_local( CSRSparseMatrix<ValueType> &adjM) {
    IndexType N= adjM.getNumRows();
    SCAI_REGION("ParcoRepart.getCommunicationPairs_local");
    // coloring.size()=3: coloring(i,j,c) means that edge with endpoints i and j is colored with color c.
    // and coloring[i].size()= number of edges in input graph

    assert(adjM.getNumColumns() == adjM.getNumRows() );

    IndexType colors;
    std::vector<std::vector<IndexType>> coloring = getGraphEdgeColoring_local( adjM, colors );
    std::vector<DenseVector<IndexType>> retG(colors);
    
    if (adjM.getNumRows()==2) {
    	assert(colors<=1);
    	assert(coloring[0].size()<=1);
    }
    
    for(IndexType i=0; i<colors; i++){        
        retG[i].allocate(N);
        // TODO: although not distributed maybe try to avoid setValue, change to std::vector ?
        // initialize so retG[i][j]= j instead of -1
        for( IndexType j=0; j<N; j++){
            retG[i].setValue( j, j );
        }
    }
    
    // for all the edges:
    // coloring[0][i] = the first block , coloring[1][i] = the second block,
    // coloring[2][i]= the color/round in which the two blocks shall communicate
    for(IndexType i=0; i<coloring[0].size(); i++){
        IndexType color = coloring[2][i]; // the color/round of this edge
        assert(color<colors);
        IndexType firstBlock = coloring[0][i];
        IndexType secondBlock = coloring[1][i];
        retG[color].setValue( firstBlock, secondBlock);
        retG[color].setValue( secondBlock, firstBlock );
    }
    
    return retG;
}
//---------------------------------------------------------------------------------------

/* A 2D or 3D matrix given as a 1D array of size sideLen^dimesion
 * */
template<typename IndexType, typename ValueType>
std::vector<IndexType> ParcoRepart<IndexType, ValueType>::neighbourPixels(const IndexType thisPixel, const IndexType sideLen, const IndexType dimension){
    SCAI_REGION("ParcoRepart.neighbourPixels");
   
    SCAI_ASSERT(thisPixel>=0, "Negative pixel value: " << std::to_string(thisPixel));
    SCAI_ASSERT(sideLen> 0, "Negative or zero side length: " << std::to_string(sideLen));
    SCAI_ASSERT(sideLen> 0, "Negative or zero dimension: " << std::to_string(dimension));
    
    IndexType totalSize = std::pow(sideLen ,dimension);    
    SCAI_ASSERT( thisPixel < totalSize , "Wrong side length or dimension, sideLen=" + std::to_string(sideLen)+ " and dimension= " + std::to_string(dimension) );
    
    std::vector<IndexType> result;
    
    //calculate the index of the neighbouring pixels
    for(IndexType i=0; i<dimension; i++){
        for( int j :{-1, 1} ){
            // possible neighbour
            IndexType ngbrIndex = thisPixel + j*std::pow(sideLen,i );
            // index is within bounds
            if( ngbrIndex < 0 or ngbrIndex >=totalSize){
                continue;
            }
            if(dimension==2){
                IndexType xCoord = thisPixel/sideLen;
                IndexType yCoord = thisPixel%sideLen;
                if( ngbrIndex/sideLen == xCoord or ngbrIndex%sideLen == yCoord){
                    result.push_back(ngbrIndex);
                }
            }else if(dimension==3){
                IndexType planeSize= sideLen*sideLen;
                IndexType xCoord = thisPixel/planeSize;
                IndexType yCoord = (thisPixel%planeSize) /  sideLen;
                IndexType zCoord = (thisPixel%planeSize) % sideLen;
                IndexType ngbrX = ngbrIndex/planeSize;
                IndexType ngbrY = (ngbrIndex%planeSize)/sideLen;
                IndexType ngbrZ = (ngbrIndex%planeSize)%sideLen;
                if( ngbrX == xCoord and  ngbrY == yCoord ){
                    result.push_back(ngbrIndex);
                }else if(ngbrX == xCoord and  ngbrZ == zCoord){
                    result.push_back(ngbrIndex);
                }else if(ngbrY == yCoord and  ngbrZ == zCoord){
                    result.push_back(ngbrIndex);
                }
            }else{
                throw std::runtime_error("Implemented only for 2D and 3D. Dimension given: " + std::to_string(dimension) );
            }
        }
    }
    return result;
}
//---------------------------------------------------------------------------------------

//to force instantiation
template class ParcoRepart<IndexType, ValueType>;

}
