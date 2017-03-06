#pragma once

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "HilbertCurve.h"
#include "Settings.h"


namespace ITI{
    
    class aux{
        
        void writeHeatLike_local(DenseVector<ValueType> input, IndexType dim, const std::string filename){
            std::ofstream f(filename);
            if(f.fail())
                throw std::runtime_error("File "+ filename+ " failed.");
            
            for(IndexType i=0; i<input.size(); i++){
                for(IndexType d=0; d<dim; d++){
                    f<< i*dim +d<< " ";
                }
            }

        }    
    
    static DenseVector<IndexType> MYpartitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, Settings settings){
        
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
	
	/**
	*	gather information for space-filling curves
	*/
	std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
	DenseVector<IndexType> result;

	{
		SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition" )

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
	

		ValueType maxExtent = 0;
		for (IndexType dim = 0; dim < dimensions; dim++) {
			if (maxCoords[dim] - minCoords[dim] > maxExtent) {
				maxExtent = maxCoords[dim] - minCoords[dim];
			}
		}

		/**
		* Several possibilities exist for choosing the recursion depth.
		* Either by user choice, or by the maximum fitting into the datatype, or by the minimum distance between adjacent points.
		*/
		const IndexType recursionDepth = settings.sfcResolution > 0 ? settings.sfcResolution : std::min(std::log2(n), double(21));
	
		/**
		*	create space filling curve indices.
		*/
		// trying the new version of getHilbertIndex
                
		scai::lama::DenseVector<ValueType> hilbertIndices(inputDist);
PRINT(*comm << ": initial local size "<<hilbertIndices.getLocalValues().size());                
                
		// get local part of hilbert indices
		scai::utilskernel::LArray<ValueType>& hilbertIndicesLocal = hilbertIndices.getLocalValues();

		{
			SCAI_REGION("ParcoRepart.partitionGraph.initialPartition.spaceFillingCurve")
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
//============	
		// care: res must be even
		IndexType res=4, numSquares =std::pow(2,res) ;
                std::vector<IndexType> density( numSquares , 0);
                for(int i=0; i<localN; i++){
                    int square = int(hilbertIndicesLocal[i]*numSquares);
                    assert(square < numSquares);
                    ++density[square];
                }

                std::vector<IndexType> sumDensity( numSquares , 0);
                for(int i=0; i<numSquares; i++){
                    sumDensity[i] = comm->sum(density[i]);
                }
                if(comm->getRank()==0){
                    for(int i=0; i<numSquares; i++){
                        std::cout<<*comm <<": "<< i << ", "<< sumDensity[i] << std::endl;
                    }
                }
                
                // from a 1D DenseVector get a 2D array
                // edge side = 2^(res/2), 
                IndexType squareSide = std::pow(2, res/2);
                std::vector<std::vector<IndexType>> heatmap(squareSide, std::vector<IndexType> (squareSide));
                for(IndexType d1=0; d1<numSquares; d1++){         
                        DenseVector<ValueType> twoDimPoint = HilbertCurve<IndexType,ValueType>::Hilbert2DIndex2Point( ValueType( d1)/numSquares, res+1);
                        SCAI_ASSERT(twoDimPoint.size()==2, "Wrong point dimension");
                        ValueType x = twoDimPoint.getLocalValues()[0];
                        ValueType y = twoDimPoint.getLocalValues()[1];
                        SCAI_ASSERT(x <squareSide, "Too big index");
                        SCAI_ASSERT(y<squareSide, "Too big index");
                        heatmap[x][y] = sumDensity[d1];
//if(comm->getRank()==0)PRINT(ValueType( d1)/numSquares << " >> "<< x << " - " << y << " as indices: "<< int( x*squareSide) << " - "<< int(y*squareSide));                                                     
                }
//^^^^^^^^^^^^^^^^^^^                
		
		/**
		* now sort the global indices by where they are on the space-filling curve.
		*/
		scai::lama::DenseVector<IndexType> permutation;
        {
			SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition.sorting" )
			hilbertIndices.sort(permutation, true);
scai::hmemo::WriteAccess<ValueType> wHilbIndex( hilbertIndices.getLocalValues() );  
ValueType* maxElem =  std::max_element(wHilbIndex.get(),wHilbIndex.get()+wHilbIndex.size());
ValueType* minElem =  std::min_element(wHilbIndex.get(),wHilbIndex.get()+wHilbIndex.size());
//PRINT(*comm << ": min hilbIndex= "<< *minElem << "  and max= "<< *maxElem);
PRINT(*comm << ": hilbertIndex size after sorting "<<hilbertIndices.getLocalValues().size());              
        }

		/**
		* initial partitioning with sfc.
		*/
		if (!inputDist->isReplicated() && comm->getSize() == k) {
			SCAI_REGION( "ParcoRepart.partitionGraph.initialPartition.redistribute" )

			scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(n, comm));
			permutation.redistribute(blockDist);
//PRINT(*comm << ": permutation size after redistribution "<<permutation.getLocalValues().size());                        
			scai::hmemo::WriteAccess<IndexType> wPermutation( permutation.getLocalValues() );
			std::sort(wPermutation.get(), wPermutation.get()+wPermutation.size());
			wPermutation.release();

			scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(n, permutation.getLocalValues(), comm));

			input.redistribute(newDistribution, input.getColDistributionPtr());
			result = DenseVector<IndexType>(newDistribution, comm->getRank());

			if (settings.useGeometricTieBreaking) {
				for (IndexType dim = 0; dim < dimensions; dim++) {
					coordinates[dim].redistribute(newDistribution);
				}
ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, n, "debugRepart");
			}

		} else {
			scai::lama::DenseVector<IndexType> inversePermutation;
			DenseVector<IndexType> tmpPerm = permutation;
			tmpPerm.sort( inversePermutation, true);

			result.allocate(inputDist);

			for (IndexType i = 0; i < localN; i++) {
				result.getLocalValues()[i] = int( inversePermutation.getLocalValues()[i] *k/n);
			}
		}
	}

	IndexType numRefinementRounds = 0;

	if (comm->getSize() == 1 || comm->getSize() == k) {
		ValueType gain = settings.minGainForNextRound;
		ValueType cut = comm->getSize() == 1 ? ParcoRepart<IndexType, ValueType>::computeCut(input, result) : comm->sum(ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(input, false)) / 2;

		/**
		scai::lama::CSRSparseMatrix<ValueType> blockGraph = ParcoRepart<IndexType, ValueType>::getPEGraph(input);

		std::vector<DenseVector<IndexType>> communicationScheme = ParcoRepart<IndexType,ValueType>::getCommunicationPairs_local(blockGraph);

		std::vector<IndexType> nodesWithNonLocalNeighbors = getNodesWithNonLocalNeighbors(input);

		std::vector<double> distances;
		if (settings.useGeometricTieBreaking) {
			distances = distancesFromBlockCenter(coordinates);
		}
*/
		DenseVector<IndexType> uniformWeights = DenseVector<IndexType>(input.getRowDistributionPtr(), 1);

		if (comm->getRank() == 0) {
			afterSFC = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsedSeconds = afterSFC-start;
			std::cout << "With SFC (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
		}

		MultiLevel<IndexType, ValueType>::multiLevelStep(input, result, uniformWeights, coordinates, settings);
		
	} else {
		std::cout << "Local refinement only implemented sequentially and for one block per process. Called with " << comm->getSize() << " processes and " << k << " blocks." << std::endl;
	}
	return result;
}
    
    
        
    }; //class aux
}// namespace ITI