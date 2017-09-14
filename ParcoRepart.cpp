/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include <scai/dmemo/HaloBuilder.hpp>
#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/GenBlockDistribution.hpp>
#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>
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

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "HilbertCurve.h"
#include "MultiLevel.h"
#include "SpectralPartition.h"
#include "KMeans.h"
#include "AuxiliaryFunctions.h"
#include "MultiSection.h"
#include "GraphUtils.h"

#include "schizoQuicksort/src/sort/SchizoQS.hpp"

using scai::lama::Scalar;

namespace ITI {
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings)
{
	DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(input.getRowDistributionPtr(), 1);
	return partitionGraph(input, coordinates, uniformWeights, settings);
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings)
{
	IndexType k = settings.numBlocks;
	ValueType epsilon = settings.epsilon;
    
	SCAI_REGION( "ParcoRepart.partitionGraph" )

	std::chrono::time_point<std::chrono::steady_clock> start, afterSFC, round;
	start = std::chrono::steady_clock::now();

	SCAI_REGION_START("ParcoRepart.partitionGraph.inputCheck")
	/**
	* check input arguments for sanity
	*/
	IndexType n = input.getNumRows();
	if (n != coordinates[0].size()) {
		throw std::runtime_error("Matrix has " + std::to_string(n) + " rows, but " + std::to_string(coordinates[0].size())
		 + " coordinates are given.");
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

	const IndexType dimensions = coordinates.size();
        
	const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(n));
	const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();

	const IndexType localN = inputDist->getLocalSize();
	const IndexType globalN = inputDist->getGlobalSize();

	if( !coordDist->isEqual( *inputDist) ){
		throw std::runtime_error( "Distributions should be equal.");
	}

	if (nodeWeights.size() != 0)

	SCAI_REGION_END("ParcoRepart.partitionGraph.inputCheck")
	{
		SCAI_REGION("ParcoRepart.synchronize")
		comm->synchronize();
	}
	
        SCAI_REGION_START("ParcoRepart.partitionGraph.initialPartition")
        // get an initial partition
        DenseVector<IndexType> result;
        if (nodeWeights.size() == 0) {
        	nodeWeights = DenseVector<ValueType>(inputDist, 1);
        }
        
        assert(nodeWeights.getDistribution().isEqual(*inputDist));

        std::chrono::time_point<std::chrono::system_clock> beforeInitPart =  std::chrono::system_clock::now();

        if( settings.initialPartition==InitialPartitioningMethods::SFC) {
            PRINT0("Initial partition with SFCs");
            result= ParcoRepart<IndexType, ValueType>::hilbertPartition(coordinates, settings);
            std::chrono::duration<double> sfcTime = std::chrono::system_clock::now() - beforeInitPart;
            ValueType timeForSfcPart = ValueType ( comm->max(sfcTime.count() ));
            if (comm->getRank() == 0) {
                std::cout << "SFC Time:" << timeForSfcPart << std::endl;
            }
        } else if ( settings.initialPartition==InitialPartitioningMethods::Pixel) {
            PRINT0("Initial partition with pixels.");
            result = ParcoRepart<IndexType, ValueType>::pixelPartition(coordinates, settings);
        } else if ( settings.initialPartition == InitialPartitioningMethods::Spectral) {
            PRINT0("Initial partition with spectral");
            result = ITI::SpectralPartition<IndexType, ValueType>::getPartition(input, coordinates, settings);
        } else if (settings.initialPartition == InitialPartitioningMethods::KMeans) {
            PRINT0("Initial partition with K-Means");
            //prepare coordinates for k-means
            std::vector<DenseVector<ValueType> > coordinateCopy;
            DenseVector<ValueType> nodeWeightCopy;
            if (settings.dimensions == 2 || settings.dimensions == 3) {
                Settings sfcSettings = settings;
                sfcSettings.numBlocks = comm->getSize();
                DenseVector<IndexType> tempResult = ParcoRepart<IndexType, ValueType>::hilbertPartition(coordinates, sfcSettings);
                nodeWeightCopy = DenseVector<ValueType>(nodeWeights, tempResult.getDistributionPtr());
                coordinateCopy.resize(dimensions);
                for (IndexType d = 0; d < dimensions; d++) {
                    coordinateCopy[d] = DenseVector<ValueType>(coordinates[d], tempResult.getDistributionPtr());
                }
            } else {
                coordinateCopy = coordinates;
                nodeWeightCopy = nodeWeights;
            }
            
            const IndexType weightSum = nodeWeights.sum().Scalar::getValue<IndexType>();
            
            // vector of size k, each element representsthe size of each block
            std::vector<IndexType> blockSizes;
            if( settings.blockSizes.empty() ){
                blockSizes.assign( settings.numBlocks, weightSum/settings.numBlocks );
            }else{
                blockSizes = settings.blockSizes;
            }
            SCAI_ASSERT( blockSizes.size()==settings.numBlocks , "Wrong size of blockSizes vector: " << blockSizes.size() );
            
            std::chrono::time_point<std::chrono::system_clock> beforeKMeans =  std::chrono::system_clock::now();
            result = ITI::KMeans::computePartition(coordinateCopy, settings.numBlocks, nodeWeightCopy, blockSizes, settings.epsilon);
            std::chrono::duration<double> kMeansTime = std::chrono::system_clock::now() - beforeKMeans;
            ValueType timeForInitPart = ValueType ( comm->max(kMeansTime.count() ));
            assert(result.getLocalValues().min() >= 0);
            assert(result.getLocalValues().max() < k);

            if (comm->getRank() == 0) {
                std::cout << "K-Means, Time:" << timeForInitPart << std::endl;
            }
            assert(result.max().Scalar::getValue<IndexType>() == settings.numBlocks -1);
            assert(result.min().Scalar::getValue<IndexType>() == 0);

        } else if (settings.initialPartition == InitialPartitioningMethods::Multisection) {// multisection
            PRINT0("Initial partition with multisection");
            DenseVector<ValueType> convertedWeights(nodeWeights);
            result = ITI::MultiSection<IndexType, ValueType>::getPartitionNonUniform(input, coordinates, convertedWeights, settings);
            std::chrono::duration<double> msTime = std::chrono::system_clock::now() - beforeInitPart;
            ValueType timeForMsPart = ValueType ( comm->max(msTime.count() ));
            if (comm->getRank() == 0) {
                std::cout << "MS Time:" << timeForMsPart << std::endl;
            }
        }
        else {
            throw std::runtime_error("Initial Partitioning mode undefined.");
        }

        SCAI_REGION_END("ParcoRepart.partitionGraph.initialPartition")

        if (comm->getSize() == k) {
        	SCAI_REGION("ParcoRepart.partitionGraph.initialRedistribution")
			/**
			 * redistribute to prepare for local refinement
			 */
			scai::dmemo::Redistributor resultRedist(result.getLocalValues(), result.getDistributionPtr());
			result.redistribute(resultRedist);

			scai::dmemo::Redistributor redistributor(resultRedist.getTargetDistributionPtr(), input.getRowDistributionPtr());
			input.redistribute(redistributor, noDist);
			if (settings.useGeometricTieBreaking) {
				for (IndexType d = 0; d < dimensions; d++) {
					coordinates[d].redistribute(redistributor);
				}
			}
			nodeWeights.redistribute(redistributor);

			std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforeInitPart;
			ValueType timeForInitPart = ValueType ( comm->max(partitionTime.count() ));
			ValueType cut = comm->sum(ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(input, true)) / 2;
			ValueType imbalance = GraphUtils::computeImbalance<IndexType, ValueType>(result, k, nodeWeights);

			if (comm->getRank() == 0) {
				std::cout<< std::endl << "\033[1;32mTime for initial partition and redistribution:" << timeForInitPart << std::endl;
				std::cout << "Cut:" << cut << ", imbalance:" << imbalance<< " \033[0m" <<std::endl << std::endl;
			}

			IndexType numRefinementRounds = 0;

			SCAI_REGION_START("ParcoRepart.partitionGraph.multiLevelStep")
			scai::dmemo::Halo halo = GraphUtils::buildNeighborHalo<IndexType, ValueType>(input);
			ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(input, result, nodeWeights, coordinates, halo, settings);
			SCAI_REGION_END("ParcoRepart.partitionGraph.multiLevelStep")
	} else {
		result.redistribute(inputDist);
		if (comm->getRank() == 0) {
			std::cout << "Local refinement only implemented for one block per process. Called with " << comm->getSize() << " processes and " << k << " blocks." << std::endl;
		}
	}

	return result;
}
//--------------------------------------------------------------------------------------- 

//TODO: take node weights into account
template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::hilbertPartition(const std::vector<DenseVector<ValueType>> &coordinates, DenseVector<ValueType> &nodeWeights, Settings settings){
    
    DenseVector<ValueType> uniformWeights = DenseVector<ValueType>(coordinates[0].getDistributionPtr(), 1);
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

    std::vector<ValueType> minCoords(dimensions);
    std::vector<ValueType> maxCoords(dimensions);
    DenseVector<IndexType> result;
    
    /**
     * get minimum / maximum of coordinates
     */
    {
		SCAI_REGION( "ParcoRepart.hilbertPartition.minMax" )
		for (IndexType dim = 0; dim < dimensions; dim++) {
			minCoords[dim] = coordinates[dim].min().Scalar::getValue<ValueType>();
			maxCoords[dim] = coordinates[dim].max().Scalar::getValue<ValueType>();
			assert(std::isfinite(minCoords[dim]));
			assert(std::isfinite(maxCoords[dim]));
			SCAI_ASSERT(maxCoords[dim] > minCoords[dim], "Wrong coordinates.");
		}
    }
    
    /**
     * Several possibilities exist for choosing the recursion depth.
     * Either by user choice, or by the maximum fitting into the datatype, or by the minimum distance between adjacent points.
     */
    const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(globalN), double(21));
    
    /**
     *	create space filling curve indices.
     */
    
    scai::lama::DenseVector<ValueType> hilbertIndices(coordDist);
    
    {
        SCAI_REGION("ParcoRepart.hilbertPartition.spaceFillingCurve");
        // get local part of hilbert indices
        scai::hmemo::WriteOnlyAccess<ValueType> hilbertIndicesLocal(hilbertIndices.getLocalValues());
        assert(hilbertIndicesLocal.size() == localN);
        // get read access to the local part of the coordinates
        // TODO: should be coordAccess[dimension] but I don't know how ... maybe HArray::acquireReadAccess? (harry)
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        // this is faulty, if dimensions=2 coordAccess2 is equal to coordAccess1
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[dimensions-1].getLocalValues() );
        
        ValueType point[dimensions];
        for (IndexType i = 0; i < localN; i++) {
            coordAccess0.getValue(point[0], i);
            coordAccess1.getValue(point[1], i);
            // TODO change how I treat different dimensions
            if(dimensions == 3){
                coordAccess2.getValue(point[2], i);
            }
            ValueType globalHilbertIndex = HilbertCurve<IndexType, ValueType>::getHilbertIndex( point, dimensions, recursionDepth, minCoords, maxCoords);
            hilbertIndicesLocal[i] = globalHilbertIndex;
        }
    }

    //
    // vector of size k, each element represents the size of each block
    //
    std::vector<IndexType> blockSizes;
    IndexType weightSum;// = nodeWeights.sum().Scalar::getValue<IndexType>();
    if( settings.blockSizes.empty() ){
        blockSizes.assign( settings.numBlocks, weightSum/settings.numBlocks );
    }else{
        blockSizes = settings.blockSizes;
    }
    SCAI_ASSERT( blockSizes.size()==settings.numBlocks , "Wrong size of blockSizes vector: " << blockSizes.size() );
    
    //TODO: use the blockSizes vector
    //TODO: take into account node weights: just sorting will create imbalanced blocks, not so much in number of node but in the total weight of each block
    
    /**
     * now sort the global indices by where they are on the space-filling curve.
     */
    std::vector<IndexType> newLocalIndices;
    {
        SCAI_REGION( "ParcoRepart.hilbertPartition.sorting" );
        
        int typesize;
        MPI_Type_size(SortingDatatype<sort_pair>::getMPIDatatype(), &typesize);
        assert(typesize == sizeof(sort_pair));
        
        const IndexType maxLocalN = comm->max(localN);
        sort_pair localPairs[maxLocalN];

        //fill with local values
        long indexSum = 0;//for sanity checks
        scai::hmemo::ReadAccess<ValueType> localIndices(hilbertIndices.getLocalValues());
        for (IndexType i = 0; i < localN; i++) {
        	localPairs[i].value = localIndices[i];
        	localPairs[i].index = coordDist->local2global(i);
        	indexSum += localPairs[i].index;
        }

        //create checksum
        const long checkSum = comm->sum(indexSum);
        assert(checkSum == (long(globalN)*(long(globalN)-1))/2);

        //fill up with dummy values to ensure equal size
        for (IndexType i = localN; i < maxLocalN; i++) {
        	localPairs[i].value = std::numeric_limits<decltype(sort_pair::value)>::max();
        	localPairs[i].index = std::numeric_limits<decltype(sort_pair::index)>::max();
        }

        //call distributed sort
        SchizoQS::sort<sort_pair>(localPairs, maxLocalN);

        //copy indices into array
        IndexType newLocalN = 0;
        newLocalIndices.resize(maxLocalN);
        for (IndexType i = 0; i < maxLocalN; i++) {
        	newLocalIndices[i] = localPairs[i].index;
        	if (newLocalIndices[i] != std::numeric_limits<decltype(sort_pair::index)>::max()) newLocalN++;
        }

        //sort local indices for general distribution
        std::sort(newLocalIndices.begin(), newLocalIndices.end());

        //remove dummy values
        auto startOfDummyValues = std::lower_bound(newLocalIndices.begin(), newLocalIndices.end(), std::numeric_limits<decltype(sort_pair::index)>::max());
        assert(std::all_of(startOfDummyValues, newLocalIndices.end(), [](IndexType index){return index == std::numeric_limits<decltype(sort_pair::index)>::max();}));
        newLocalIndices.resize(std::distance(newLocalIndices.begin(), startOfDummyValues));

        //check size and sanity
        assert(newLocalN == newLocalIndices.size());
        assert( *std::max_element(newLocalIndices.begin(), newLocalIndices.end()) < globalN);
        assert( comm->sum(newLocalIndices.size()) == globalN);

        //check checksum
        long indexSumAfter = 0;
        for (IndexType i = 0; i < newLocalN; i++) {
        	indexSumAfter += newLocalIndices[i];
        }

        const long newCheckSum = comm->sum(indexSumAfter);
        SCAI_ASSERT( newCheckSum == checkSum, "Old checksum: " << checkSum << ", new checksum: " << newCheckSum );

        //possible optimization: remove dummy values during first copy, then directly copy into HArray and sort with pointers. Would save one copy.
    }
    
    {
    	assert(!coordDist->isReplicated() && comm->getSize() == k);
        SCAI_REGION( "ParcoRepart.hilbertPartition.createDistribution" );

        scai::utilskernel::LArray<IndexType> indexTransport(newLocalIndices.size(), newLocalIndices.data());
        assert(comm->sum(indexTransport.size()) == globalN);
        scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, indexTransport, comm));
        
        if (comm->getRank() == 0) std::cout << "Created distribution." << std::endl;
        result = DenseVector<IndexType>(newDistribution, comm->getRank());
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
    scai::hmemo::HArray<IndexType> density( cubeSize, 0);
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
    
    if(comm->getRank()==0){
        ITI::aux::writeHeatLike_local_2D(density, sideLen, dimensions, "heat_"+settings.fileName+".plt");
    }
  
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
                pixelDistance = aux::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);
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
                            pixelDistance = aux::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);    
                            border[l].second += geomSpread*  (1/(pixelDistance*pixelDistance))* ( spreadFactor *std::pow(localSumDens[neighbours[j]], 0.5) + std::pow(localSumDens[bestIndex], 0.5) );
                            inBorder= true;
                        }
                    }
                    if(!inBorder){
                        std::pair<IndexType, ValueType> toInsert;
                        toInsert.first = neighbours[j];
                        //toInsert.second = geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5));
                        pixelDistance = aux::pixelL2Distance2D( maxDensityPixel, neighbours[j], sideLen);    
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
    for(int pp=0; pp<pixeledPartition.size(); pp++){  
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
    
    //scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

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

template DenseVector<int> ParcoRepart<int, double>::partitionGraph(CSRSparseMatrix<double> &input, std::vector<DenseVector<double>> &coordinates, DenseVector<double> &nodeWeights, struct Settings);

template DenseVector<int> ParcoRepart<int, double>::partitionGraph(CSRSparseMatrix<double> &input, std::vector<DenseVector<double>> &coordinates, struct Settings);

template DenseVector<int> ParcoRepart<int, double>::hilbertPartition(const std::vector<DenseVector<double>> &coordinates, DenseVector<double> &nodeWeights, Settings settings);
    
template DenseVector<int> ParcoRepart<int, double>::pixelPartition(const std::vector<DenseVector<double>> &coordinates, Settings settings);

template void ParcoRepart<int, double>::checkLocalDegreeSymmetry(const CSRSparseMatrix<double> &input);

template std::vector< std::vector<int>>  ParcoRepart<int, double>::getGraphEdgeColoring_local( CSRSparseMatrix<double> &adjM, int& colors);

template std::vector<DenseVector<int>> ParcoRepart<int, double>::getCommunicationPairs_local( CSRSparseMatrix<double> &adjM);

template std::vector<int> ParcoRepart<int, double>::neighbourPixels(const int thisPixel, const int sideLen, const int dimension);

}
