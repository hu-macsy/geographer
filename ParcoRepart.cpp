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
#include "AuxiliaryFunctions.h"

using scai::lama::Scalar;

namespace ITI {

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings)
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
	SCAI_REGION_END("ParcoRepart.partitionGraph.inputCheck")
	{
		SCAI_REGION("ParcoRepart.synchronize")
		comm->synchronize();
	}
	
        SCAI_REGION_START("ParcoRepart.partitionGraph.initialPartition")
        // get an initial partition
        DenseVector<IndexType> result;
        
        if( settings.initialPartition==0 ){ //sfc
            result= ParcoRepart<IndexType, ValueType>::hilbertPartition(input, coordinates, settings);
        }else if( settings.initialPartition==1 ){ // pixel
            result = ParcoRepart<IndexType, ValueType>::pixelPartition(input, coordinates, settings);
        }else{ // spectral
            result = ITI::SpectralPartition<IndexType, ValueType>::getPartition(input, coordinates, settings);
        }
        SCAI_REGION_END("ParcoRepart.partitionGraph.initialPartition")
        
	IndexType numRefinementRounds = 0;

        SCAI_REGION_START("ParcoRepart.partitionGraph.multiLevelStep")
	if (comm->getSize() == 1 || comm->getSize() == k) {
		ValueType gain = settings.minGainForNextRound;
		ValueType cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;

		DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(result.getDistributionPtr(), 1);

		ITI::MultiLevel<IndexType, ValueType>::multiLevelStep(input, result, uniformWeights, coordinates, settings);

	} else {
		std::cout << "Local refinement only implemented sequentially and for one block per process. Called with " << comm->getSize() << " processes and " << k << " blocks." << std::endl;
	}
	SCAI_REGION_END("ParcoRepart.partitionGraph.multiLevelStep")
	return result;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::hilbertPartition(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings){    
    SCAI_REGION( "ParcoRepart.hilbertPartition" )
    	
    std::chrono::time_point<std::chrono::steady_clock> start, afterSFC;
    start = std::chrono::steady_clock::now();
    
    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
    
    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    assert(dimensions == settings.dimensions);
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    
    std::vector<ValueType> minCoords(dimensions);
    std::vector<ValueType> maxCoords(dimensions);
    DenseVector<IndexType> result;
    
    if( ! inputDist->isEqual(*coordDist) ){
        throw std::runtime_error("Matrix and coordinates should have the same distribution");
    }
    
    /**
     * get minimum / maximum of coordinates
     */
    {
		SCAI_REGION( "ParcoRepart.initialPartition.minMax" )
		for (IndexType dim = 0; dim < dimensions; dim++) {
			minCoords[dim] = coordinates[dim].min().Scalar::getValue<ValueType>();
			maxCoords[dim] = coordinates[dim].max().Scalar::getValue<ValueType>();
			assert(std::isfinite(minCoords[dim]));
			assert(std::isfinite(maxCoords[dim]));
			assert(maxCoords[dim] > minCoords[dim]);
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
    
    scai::lama::DenseVector<ValueType> hilbertIndices(inputDist);
    
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
    
    
    /**
     * now sort the global indices by where they are on the space-filling curve.
     */
    scai::lama::DenseVector<IndexType> permutation;
    {
        SCAI_REGION( "ParcoRepart.hilbertPartition.sorting" )
        hilbertIndices.sort(permutation, true);
    }
    
    if (!inputDist->isReplicated() && comm->getSize() == k) {
        SCAI_REGION( "ParcoRepart.hilbertPartition.redistribute" )
        
        scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(globalN, comm));
        permutation.redistribute(blockDist);
        scai::hmemo::WriteAccess<IndexType> wPermutation( permutation.getLocalValues() );
        std::sort(wPermutation.get(), wPermutation.get()+wPermutation.size());
        wPermutation.release();
        
        scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, permutation.getLocalValues(), comm));
        
        input.redistribute(newDistribution, input.getColDistributionPtr());
        result = DenseVector<IndexType>(newDistribution, comm->getRank());
        
        if (settings.useGeometricTieBreaking) {
            for (IndexType dim = 0; dim < dimensions; dim++) {
                coordinates[dim].redistribute(newDistribution);
            }
        }
        
    } else {
        scai::lama::DenseVector<IndexType> inversePermutation;
        DenseVector<IndexType> tmpPerm = permutation;
        tmpPerm.sort( inversePermutation, true);
        
        scai::hmemo::WriteOnlyAccess<IndexType> wResult(result.getLocalValues(), localN);
        
        for (IndexType i = 0; i < localN; i++) {
        	wResult[i] = static_cast<IndexType>(inversePermutation.getLocalValues()[i] *k/globalN);
        }
    }
    
    ValueType cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;
    ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(result, k);
    if (comm->getRank() == 0) {
        afterSFC = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = afterSFC-start;
        std::cout << "\033[1;31mWith SFC (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
        std::cout<< "and imbalance= "<< imbalance << "\033[0m" << std::endl;
    }
    return result;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::pixelPartition(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings){    
    SCAI_REGION( "ParcoRepart.pixelPartition" )
    	
    SCAI_REGION_START("ParcoRepart.pixelPartition.initialise")
    std::chrono::time_point<std::chrono::steady_clock> start, round;
    start = std::chrono::steady_clock::now();
    
    const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
    
    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    
    std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    DenseVector<IndexType> result(inputDist, 0);
    
    //TODO: probably minimum is not needed
    //TODO: if we know maximum from the input we could save that although is not too costly
    
    /**
     * get minimum / maximum of local coordinates
     */
    for (IndexType dim = 0; dim < dimensions; dim++) {
        //get local parts of coordinates
        scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[dim].getLocalValues();
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
    const IndexType detailLvl = settings.pixeledDetailLevel;
    const IndexType sideLen = std::pow(2,detailLvl);
    const IndexType cubeSize = std::pow(sideLen, dimensions);
    
    //TODO: generalize this to arbitrary dimensions, do not handle 2D and 3D differently
    // a 2D or 3D arrays as a one dimensional vector
    // [i][j] is in position: i*sideLen + j
    // [i][j][k] is in: i*sideLen*sideLen + j*sideLen + k
    
    //std::vector<IndexType> density( cubeSize ,0);
    scai::hmemo::HArray<IndexType> density( cubeSize, 0);
    scai::hmemo::WriteAccess<IndexType> wDensity(density);
    //std::cout<< "detailLvl= " << detailLvl <<", sideLen= " << sideLen << ", " << "density.size= " << density.size() << std::endl;
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
    
    //TODO; is that needed. we just can overwrite density array
    // use the summed density as a Dense vector
    scai::lama::DenseVector<IndexType> sumDensity( density );
    
    if(comm->getRank()==0){
        ITI::aux::writeHeatLike_local_2D(density, sideLen, dimensions, "heat_"+settings.fileName+".plt");
    }
  
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
                geomSpread = 1 + 1/detailLvl*( std::abs(sideLen/2 - neighbours[j]/sideLen)/(0.8*sideLen/2) + std::abs(sideLen/2 - neighbours[j]%sideLen)/(0.8*sideLen/2) );
                //PRINT0( geomSpread );            
                // value to pick a border node
                pixelDistance = aux::pixell2Distance2D( maxDensityPixel, neighbours[j], sideLen);
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
                            pixelDistance = aux::pixell2Distance2D( maxDensityPixel, neighbours[j], sideLen);    
                            border[l].second += geomSpread*  (1/(pixelDistance*pixelDistance))* ( spreadFactor *std::pow(localSumDens[neighbours[j]], 0.5) + std::pow(localSumDens[bestIndex], 0.5) );
                            inBorder= true;
                        }
                    }
                    if(!inBorder){
                        std::pair<IndexType, ValueType> toInsert;
                        toInsert.first = neighbours[j];
                        //toInsert.second = geomSpread * (spreadFactor* (std::pow(localSumDens[neighbours[j]], 0.5)) + std::pow(localSumDens[bestIndex], 0.5));
                        pixelDistance = aux::pixell2Distance2D( maxDensityPixel, neighbours[j], sideLen);    
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
    
    SCAI_REGION_START("ParcoRepart.pixelPartition.newDistribution")
    //get new distribution
    scai::dmemo::DistributionPtr newDist( new scai::dmemo::GeneralDistribution ( *inputDist, result.getLocalValues() ) );
    SCAI_REGION_END("ParcoRepart.pixelPartition.newDistribution")
    
    SCAI_REGION_START("ParcoRepart.pixelPartition.finalRedistribute")
    //TODO: not sure if this is needed...
    result.redistribute( newDist);

    input.redistribute(newDist, input.getColDistributionPtr());
    
    // redistibute coordinates
    for (IndexType dim = 0; dim < dimensions; dim++) {
          coordinates[dim].redistribute( newDist );
    }
    // check coordinates size
    for (IndexType dim = 0; dim < dimensions; dim++) {
        assert( coordinates[dim].size() == globalN);
        assert( coordinates[dim].getLocalValues().size() == newDist->getLocalSize() );
    }
   
    ValueType cut = comm->getSize() == 1 ? computeCut(input, result) : comm->sum(localSumOutgoingEdges(input, false)) / 2;
    ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(result, k);
    if (comm->getRank() == 0) {
        std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() -start;
        std::cout << "\033[1;35mWith pixel detail level= "<< detailLvl<<" (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
        std::cout<< "and imbalance= " << imbalance << "\033[0m"<< std::endl;
    }
    SCAI_REGION_END("ParcoRepart.pixelPartition.finalRedistribute")
    
    return result;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, const bool weighted) {
	SCAI_REGION( "ParcoRepart.computeCut" )
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();

	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();
	const Scalar maxBlockScalar = part.max();
	const IndexType maxBlockID = maxBlockScalar.getValue<IndexType>();

	if (partDist->getLocalSize() != localN) {
		throw std::runtime_error("partition has " + std::to_string(partDist->getLocalSize()) + " local values, but matrix has " + std::to_string(localN));
	}

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	scai::hmemo::HArray<IndexType> localData = part.getLocalValues();
	scai::hmemo::ReadAccess<IndexType> partAccess(localData);

	scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	scai::dmemo::Halo partHalo = buildNeighborHalo(input);
	scai::utilskernel::LArray<IndexType> haloData;
	partDist->getCommunicatorPtr()->updateHalo( haloData, localData, partHalo );

	ValueType result = 0;
	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];
		assert(ja.size() >= endCols);

		const IndexType globalI = inputDist->local2global(i);
		assert(partDist->isLocal(globalI));
		IndexType thisBlock = partAccess[i];
		
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			assert(neighbor >= 0);
			assert(neighbor < n);

			IndexType neighborBlock;
			if (partDist->isLocal(neighbor)) {
				neighborBlock = partAccess[partDist->global2local(neighbor)];
			} else {
				neighborBlock = haloData[partHalo.global2halo(neighbor)];
			}

			if (neighborBlock != thisBlock) {
				if (weighted) {
					result += values[j];
				} else {
					result++;
                                }
			}
		}
	}

	if (!inputDist->isReplicated()) {
            //sum values over all processes
            result = inputDist->getCommunicatorPtr()->sum(result);
        }

  return result / 2; //counted each edge from both sides
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
ValueType ParcoRepart<IndexType, ValueType>::computeImbalance(const DenseVector<IndexType> &part, IndexType k, const DenseVector<IndexType> &nodeWeights) {
	SCAI_REGION( "ParcoRepart.computeImbalance" )
	const IndexType globalN = part.getDistributionPtr()->getGlobalSize();
	const IndexType localN = part.getDistributionPtr()->getLocalSize();
	const IndexType weightsSize = nodeWeights.getDistributionPtr()->getGlobalSize();
	const bool weighted = (weightsSize != 0);

	IndexType minWeight, maxWeight;
	if (weighted) {
		assert(weightsSize == globalN);
		assert(nodeWeights.getDistributionPtr()->getLocalSize() == localN);
		minWeight = nodeWeights.min().Scalar::getValue<IndexType>();
		maxWeight = nodeWeights.max().Scalar::getValue<IndexType>();
	} else {
		minWeight = 1;
		maxWeight = 1;
	}

	if (maxWeight <= 0) {
		throw std::runtime_error("Node weight vector given, but all weights non-positive.");
	}

	if (minWeight < 0) {
		throw std::runtime_error("Negative node weights not supported.");
	}

	std::vector<IndexType> subsetSizes(k, 0);
	const IndexType minK = part.min().Scalar::getValue<IndexType>();
	const IndexType maxK = part.max().Scalar::getValue<IndexType>();

	if (minK < 0) {
		throw std::runtime_error("Block id " + std::to_string(minK) + " found in partition with supposedly" + std::to_string(k) + " blocks.");
	}

	if (maxK >= k) {
		throw std::runtime_error("Block id " + std::to_string(maxK) + " found in partition with supposedly" + std::to_string(k) + " blocks.");
	}

	scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());
	scai::hmemo::ReadAccess<IndexType> localWeight(nodeWeights.getLocalValues());
	assert(localPart.size() == localN);
 	
	IndexType weightSum = 0;
	for (IndexType i = 0; i < localN; i++) {
		IndexType partID = localPart[i];
		IndexType weight = weighted ? localWeight[i] : 1;
		subsetSizes[partID] += weight;
		weightSum += weight;
	}

	IndexType optSize;
	scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
	if (weighted) {
		//get global weight sum
		weightSum = comm->sum(weightSum);
                //PRINT(weightSum);                
                //TODO: why not just weightSum/k ?
                // changed for now so that the test cases can agree
		//optSize = std::ceil(weightSum / k + (maxWeight - minWeight));
                optSize = std::ceil(weightSum / k );
	} else {
		optSize = std::ceil(globalN / k);
	}

	if (!part.getDistribution().isReplicated()) {
	  //sum block sizes over all processes
	  for (IndexType partID = 0; partID < k; partID++) {
	    subsetSizes[partID] = comm->sum(subsetSizes[partID]);
	  }
	}
	
	IndexType maxBlockSize = *std::max_element(subsetSizes.begin(), subsetSizes.end());
	if (!weighted) {
		assert(maxBlockSize >= optSize);
	}
	return (ValueType(maxBlockSize - optSize)/ optSize);
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
std::vector<IndexType> ITI::ParcoRepart<IndexType, ValueType>::nonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
	SCAI_REGION( "ParcoRepart.nonLocalNeighbors" )
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	std::set<IndexType> neighborSet;

	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			assert(neighbor >= 0);
			assert(neighbor < n);

			if (!inputDist->isLocal(neighbor)) {
				neighborSet.insert(neighbor);
			}
		}
	}
	return std::vector<IndexType>(neighborSet.begin(), neighborSet.end()) ;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
std::vector<ValueType> ITI::ParcoRepart<IndexType, ValueType>::distancesFromBlockCenter(const std::vector<DenseVector<ValueType>> &coordinates) {
	SCAI_REGION("ParcoRepart.distanceFromBlockCenter");

	const IndexType localN = coordinates[0].getDistributionPtr()->getLocalSize();
	const IndexType dimensions = coordinates.size();

	std::vector<ValueType> geometricCenter(dimensions);
	for (IndexType dim = 0; dim < dimensions; dim++) {
		const scai::utilskernel::LArray<ValueType>& localValues = coordinates[dim].getLocalValues();
		assert(localValues.size() == localN);
		geometricCenter[dim] = localValues.sum() / localN;
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
scai::dmemo::Halo ITI::ParcoRepart<IndexType, ValueType>::buildNeighborHalo(const CSRSparseMatrix<ValueType>& input) {

	SCAI_REGION( "ParcoRepart.buildPartHalo" )

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	std::vector<IndexType> requiredHaloIndices = nonLocalNeighbors(input);

	scai::dmemo::Halo Halo;
	{
		scai::hmemo::HArrayRef<IndexType> arrRequiredIndexes( requiredHaloIndices );
		scai::dmemo::HaloBuilder::build( *inputDist, arrRequiredIndexes, Halo );
	}

	return Halo;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
inline bool ITI::ParcoRepart<IndexType, ValueType>::hasNonLocalNeighbors(const CSRSparseMatrix<ValueType> &input, IndexType globalID) {
	SCAI_REGION( "ParcoRepart.hasNonLocalNeighbors" )
	/**
	 * this could be inlined physically to reduce the overhead of creating read access locks
	 */
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

	const IndexType localID = inputDist->global2local(globalID);
	assert(localID != nIndex);

	const IndexType beginCols = ia[localID];
	const IndexType endCols = ia[localID+1];

	for (IndexType j = beginCols; j < endCols; j++) {
		if (!inputDist->isLocal(ja[j])) {
			return true;
		}
	}
	return false;
}
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
std::vector<IndexType> ITI::ParcoRepart<IndexType, ValueType>::getNodesWithNonLocalNeighbors(const CSRSparseMatrix<ValueType>& input) {
	SCAI_REGION( "ParcoRepart.getNodesWithNonLocalNeighbors" )
	std::vector<IndexType> result;

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	if (inputDist->isReplicated()) {
		//everything is local
		return result;
	}

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const IndexType localN = inputDist->getLocalSize();

	//iterate over all nodes
	for (IndexType localI = 0; localI < localN; localI++) {
		const IndexType beginCols = ia[localI];
		const IndexType endCols = ia[localI+1];

		//over all edges
		for (IndexType j = beginCols; j < endCols; j++) {
			if (!inputDist->isLocal(ja[j])) {
				IndexType globalI = inputDist->local2global(localI);
				result.push_back(globalI);
				break;
			}
		}
	}

	//nodes should have been sorted to begin with, so a subset of them will be sorted as well
	assert(std::is_sorted(result.begin(), result.end()));
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
//--------------------------------------------------------------------------------------- 

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    const scai::utilskernel::LArray<IndexType>& localPart= part.getLocalValues();
    DenseVector<IndexType> border(dist,0);
    scai::utilskernel::LArray<IndexType>& localBorder= border.getLocalValues();
    
    IndexType globalN = dist->getGlobalSize();
    IndexType max = part.max().Scalar::getValue<IndexType>();
    
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }

    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);

	scai::dmemo::Halo partHalo = buildNeighborHalo(adjM);
	scai::utilskernel::LArray<IndexType> haloData;
	dist->getCommunicatorPtr()->updateHalo( haloData, localPart, partHalo );

    for(IndexType i=0; i<localN; i++){    // for all local nodes
    	IndexType thisBlock = localPart[i];
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){                   // for all the edges of a node
    		IndexType neighbor = ja[j];
    		IndexType neighborBlock;
			if (dist->isLocal(neighbor)) {
				neighborBlock = partAccess[dist->global2local(neighbor)];
			} else {
				neighborBlock = haloData[partHalo.global2halo(neighbor)];
			}
			assert( neighborBlock < max +1 );
			if (thisBlock != neighborBlock) {
				localBorder[i] = 1;
				break;
			}
    	}
    }

    assert(border.getDistributionPtr()->getLocalSize() == localN);
    return border;
}

//----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> ParcoRepart<IndexType, ValueType>::getPEGraph( const CSRSparseMatrix<ValueType> &adjM) {
    SCAI_REGION("ParcoRepart.getPEGraph");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr(); 
    const IndexType numPEs = comm->getSize();
    
    const std::vector<IndexType> nonLocalIndices = nonLocalNeighbors(adjM);
    
    SCAI_REGION_START("ParcoRepart.getPEGraph.getOwners");
    scai::utilskernel::LArray<IndexType> indexTransport(nonLocalIndices.size(), nonLocalIndices.data());
    // find the PEs that own every non-local index
    scai::hmemo::HArray<IndexType> owners(nonLocalIndices.size() , -1);
    dist->computeOwners( owners, indexTransport);
    SCAI_REGION_END("ParcoRepart.getPEGraph.getOwners");
    
    scai::hmemo::ReadAccess<IndexType> rOwners(owners);
    std::vector<IndexType> neighborPEs(rOwners.get(), rOwners.get()+rOwners.size());
    rOwners.release();
    std::sort(neighborPEs.begin(), neighborPEs.end());
    //remove duplicates
    neighborPEs.erase(std::unique(neighborPEs.begin(), neighborPEs.end()), neighborPEs.end());
    const IndexType numNeighbors = neighborPEs.size();

    // create the PE adjacency matrix to be returned
    scai::dmemo::DistributionPtr distPEs ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, numPEs) );
    assert(distPEs->getLocalSize() == 1);
    scai::dmemo::DistributionPtr noDistPEs (new scai::dmemo::NoDistribution( numPEs ));

    SCAI_REGION_START("ParcoRepart.getPEGraph.buildMatrix");
    scai::utilskernel::LArray<IndexType> ia(2, 0, numNeighbors);
    scai::utilskernel::LArray<IndexType> ja(numNeighbors, neighborPEs.data());
    scai::utilskernel::LArray<ValueType> values(numNeighbors, 1);
    scai::lama::CSRStorage<ValueType> myStorage(1, numPEs, neighborPEs.size(), ia, ja, values);
    SCAI_REGION_END("ParcoRepart.getPEGraph.buildMatrix");
    
    //could be optimized with move semantics
    scai::lama::CSRSparseMatrix<ValueType> PEgraph(myStorage, distPEs, noDistPEs);

    return PEgraph;
}
//-----------------------------------------------------------------------------------------

//return: there is an edge in the block graph between blocks ret[0][i]-ret[1][i]
template<typename IndexType, typename ValueType>
std::vector<std::vector<IndexType>> ParcoRepart<IndexType, ValueType>::getLocalBlockGraphEdges( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {
    SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges");
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.initialise");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::utilskernel::LArray<IndexType>& localPart= part.getLocalValues();
    IndexType N = adjM.getNumColumns();
    IndexType max = part.max().Scalar::getValue<IndexType>();
   
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.initialise");
    
    
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.addLocalEdge_newVersion");
    
    scai::hmemo::HArray<IndexType> nonLocalIndices( dist->getLocalSize() ); 
    scai::hmemo::WriteAccess<IndexType> writeNLI(nonLocalIndices, dist->getLocalSize() );
    IndexType actualNeighbours = 0;

    const CSRStorage<ValueType> localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());
    
    // we do not know the size of the non-local indices that is why we use an std::vector
    // with push_back, then convert that to a DenseVector in order to call DenseVector::gather
    // TODO: skip the std::vector to DenseVector conversion. maybe use HArray or LArray
    std::vector< std::vector<IndexType> > edges(2);
    std::vector<IndexType> localInd, nonLocalInd;

    for(IndexType i=0; i<dist->getLocalSize(); i++){ 
        for(IndexType j=ia[i]; j<ia[i+1]; j++){ 
            if( dist->isLocal(ja[j]) ){ 
                IndexType u = localPart[i];         // partition(i)
                IndexType v = localPart[dist->global2local(ja[j])]; // partition(j), 0<j<N so take the local index of j
                assert( u < max +1);
                assert( v < max +1);
                if( u != v){    // the nodes belong to different blocks                  
                        bool add_edge = true;
                        for(IndexType k=0; k<edges[0].size(); k++){ //check that this edge is not already in
                            if( edges[0][k]==u && edges[1][k]==v ){
                                add_edge= false;
                                break;      // the edge (u,v) already exists
                            }
                        }
                        if( add_edge== true){       //if this edge does not exist, add it
                            edges[0].push_back(u);
                            edges[1].push_back(v);
                        }
                }
            } else{  // if(dist->isLocal(j)) 
                // there is an edge between i and j but index j is not local in the partition so we cannot get part[j].
                localInd.push_back(i);
                nonLocalInd.push_back(ja[j]);
            }
            
        }
    }
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.addLocalEdge_newVersion");
    
    // TODO: this seems to take quite a long !
    // take care of all the non-local indices found
    assert( localInd.size() == nonLocalInd.size() );
    DenseVector<IndexType> nonLocalDV( nonLocalInd.size(), 0 );
    DenseVector<IndexType> gatheredPart( nonLocalDV.size(),0 );
    
    //get a DenseVector from a vector
    for(IndexType i=0; i<nonLocalInd.size(); i++){
        nonLocalDV.setValue(i, nonLocalInd[i]);
    }
    SCAI_REGION_START("ParcoRepart.getLocalBlockGraphEdges.gatherNonLocal")
        //gather all non-local indexes
        gatheredPart.gather(part, nonLocalDV , scai::common::binary::COPY );
    SCAI_REGION_END("ParcoRepart.getLocalBlockGraphEdges.gatherNonLocal")
    
    assert( gatheredPart.size() == nonLocalInd.size() );
    assert( gatheredPart.size() == localInd.size() );
    
    for(IndexType i=0; i<gatheredPart.size(); i++){
        SCAI_REGION("ParcoRepart.getLocalBlockGraphEdges.addNonLocalEdge");
        IndexType u = localPart[ localInd[i] ];         
        IndexType v = gatheredPart.getValue(i).Scalar::getValue<IndexType>();
        assert( u < max +1);
        assert( v < max +1);
        if( u != v){    // the nodes belong to different blocks                  
            bool add_edge = true;
            for(IndexType k=0; k<edges[0].size(); k++){ //check that this edge is not already in
                if( edges[0][k]==u && edges[1][k]==v ){
                    add_edge= false;
                    break;      // the edge (u,v) already exists
                }
            }
            if( add_edge== true){       //if this edge does not exist, add it
                edges[0].push_back(u);
                edges[1].push_back(v);
            }
        }
    }
    return edges;
}

//-----------------------------------------------------------------------------------------

// in this version the graph is an HArray with size k*k and [i,j] = i*k+j
//
// Not distributed.
//
template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> ParcoRepart<IndexType, ValueType>::getBlockGraph( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, const int k) {
    SCAI_REGION("ParcoRepart.getBlockGraph");
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    const scai::utilskernel::LArray<IndexType>& localPart= part.getLocalValues();
    
    // there are k blocks in the partition so the adjecency matrix for the block graph has dimensions [k x k]
    scai::dmemo::DistributionPtr distRowBlock ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, k) );  
    scai::dmemo::DistributionPtr distColBlock ( new scai::dmemo::NoDistribution( k ));
    
    // TODO: memory costly for big k
    IndexType size= k*k;
    // get, on each processor, the edges of the blocks that are local
    std::vector< std::vector<IndexType> > blockEdges = ParcoRepart<int, double>::getLocalBlockGraphEdges( adjM, part);
    assert(blockEdges[0].size() == blockEdges[1].size());
    
    scai::hmemo::HArray<IndexType> sendPart(size, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvPart(size);
    
    for(IndexType round=0; round<comm->getSize(); round++){
        SCAI_REGION("ParcoRepart.getBlockGraph.shiftArray");
        {   // write your part 
            scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendPart );
            for(IndexType i=0; i<blockEdges[0].size(); i++){
                IndexType u = blockEdges[0][i];
                IndexType v = blockEdges[1][i];
                sendPartWrite[ u*k + v ] = 1;
            }
        }
        comm->shiftArray(recvPart , sendPart, 1);
        sendPart.swap(recvPart);
    } 
    
    // get numEdges
    IndexType numEdges=0;
    
    scai::hmemo::ReadAccess<IndexType> recvPartRead( recvPart );
    for(IndexType i=0; i<recvPartRead.size(); i++){
        if( recvPartRead[i]>0 )
            ++numEdges;
    }
    
    //convert the k*k HArray to a [k x k] CSRSparseMatrix
    scai::lama::CSRStorage<ValueType> localMatrix;
    localMatrix.allocate( k ,k );
    
    scai::hmemo::HArray<IndexType> csrIA;
    scai::hmemo::HArray<IndexType> csrJA;
    scai::hmemo::HArray<ValueType> csrValues; 
    {
        IndexType numNZ = numEdges;     // this equals the number of edges of the graph
        scai::hmemo::WriteOnlyAccess<IndexType> ia( csrIA, k +1 );
        scai::hmemo::WriteOnlyAccess<IndexType> ja( csrJA, numNZ );
        scai::hmemo::WriteOnlyAccess<ValueType> values( csrValues, numNZ );   
        scai::hmemo::ReadAccess<IndexType> recvPartRead( recvPart );
        ia[0]= 0;
        
        IndexType rowCounter = 0; // count rows
        IndexType nnzCounter = 0; // count non-zero elements
        
        for(IndexType i=0; i<k; i++){
            IndexType rowNums=0;
            // traverse the part of the HArray that represents a row and find how many elements are in this row
            for(IndexType j=0; j<k; j++){
                if( recvPartRead[i*k+j] >0  ){
                    ++rowNums;
                }
            }
            ia[rowCounter+1] = ia[rowCounter] + rowNums;
           
            for(IndexType j=0; j<k; j++){
                if( recvPartRead[i*k +j] >0){   // there exist edge (i,j)
                    ja[nnzCounter] = j;
                    values[nnzCounter] = 1;
                    ++nnzCounter;
                }
            }
            ++rowCounter;
        }
    }
    SCAI_REGION_START("ParcoRepart.getBlockGraph.swapAndAssign");
        scai::lama::CSRSparseMatrix<ValueType> matrix;
        localMatrix.swap( csrIA, csrJA, csrValues );
        matrix.assign(localMatrix);
    SCAI_REGION_END("ParcoRepart.getBlockGraph.swapAndAssign");
    return matrix;
}

//-----------------------------------------------------------------------------------

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
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

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

template DenseVector<int> ParcoRepart<int, double>::partitionGraph(CSRSparseMatrix<double> &input, std::vector<DenseVector<double>> &coordinates, struct Settings);

template DenseVector<int> ParcoRepart<int, double>::pixelPartition(CSRSparseMatrix<double> &input, std::vector<DenseVector<double>> &coordinates, Settings settings);

template double ParcoRepart<int, double>::computeCut(const CSRSparseMatrix<double> &input, const DenseVector<int> &part, bool ignoreWeights);

template double ParcoRepart<int, double>::computeImbalance(const DenseVector<int> &partition, int k, const DenseVector<int> &nodeWeights);

template std::vector<int> ITI::ParcoRepart<int, double>::nonLocalNeighbors(const CSRSparseMatrix<double>& input);

template std::vector<double> ITI::ParcoRepart<int, double>::distancesFromBlockCenter(const std::vector<DenseVector<double>> &coordinates);

template scai::dmemo::Halo ITI::ParcoRepart<int, double>::buildNeighborHalo(const CSRSparseMatrix<double> &input);

template std::vector<int> ITI::ParcoRepart<int, double>::getNodesWithNonLocalNeighbors(const CSRSparseMatrix<double>& input);

template void ParcoRepart<int, double>::checkLocalDegreeSymmetry(const CSRSparseMatrix<double> &input);

template DenseVector<int> ParcoRepart<int, double>::getBorderNodes( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::lama::CSRSparseMatrix<double> ParcoRepart<int, double>::getPEGraph( const CSRSparseMatrix<double> &adjM);

template std::vector<std::vector<IndexType>> ParcoRepart<int, double>::getLocalBlockGraphEdges( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::lama::CSRSparseMatrix<double> ParcoRepart<int, double>::getBlockGraph( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part, const int k );

template std::vector< std::vector<int>>  ParcoRepart<int, double>::getGraphEdgeColoring_local( CSRSparseMatrix<double> &adjM, int& colors);

template std::vector<DenseVector<int>> ParcoRepart<int, double>::getCommunicationPairs_local( CSRSparseMatrix<double> &adjM);

template std::vector<int> ParcoRepart<int, double>::neighbourPixels(const int thisPixel, const int sideLen, const int dimension);

}
