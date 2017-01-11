/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include <scai/dmemo/Distribution.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/lama/storage/SparseAssemblyStorage.hpp>
#include <scai/lama/storage/CRTPMatrixStorage.hpp>

#include <assert.h>
#include <cmath>
#include <climits>
#include <queue>

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "HilbertCurve.h"

//using std::vector;

namespace ITI {

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::getMinimumNeighbourDistance(const CSRSparseMatrix<ValueType> &input, const std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions) {
	// iterate through matrix to find closest neighbours, implying necessary recursion depth for space-filling curve
	// here it can happen that the closest neighbor is not stored on this processor.

        std::vector<scai::dmemo::DistributionPtr> coordDist(dimensions);
        for(IndexType i=0; i<dimensions; i++){
            coordDist[i] = coordinates[i].getDistributionPtr(); 
        }
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const IndexType localN = inputDist->getLocalSize();
        /*
	if (coordDist[0]->getLocalSize() % int(dimensions) != 0) {
		throw std::runtime_error("Size of coordinate vector no multiple of dimension. Maybe it was split in the distribution?");
	}
        */
	if (!input.getColDistributionPtr()->isReplicated()) {
		throw std::runtime_error("Columns must be replicated.");
	}

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	//const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates.getLocalValues();
        std::vector<scai::utilskernel::LArray<ValueType>> localPartOfCoords2(dimensions);
        for(IndexType i=0; i<dimensions; i++){
            localPartOfCoords2[i] = coordinates[i].getLocalValues();
        }
        
	const scai::utilskernel::LArray<IndexType>& ia = localStorage.getIA();
    const scai::utilskernel::LArray<IndexType>& ja = localStorage.getJA();
    assert(ia.size() == localN+1);

    ValueType minDistanceSquared = std::numeric_limits<ValueType>::max();
	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];//assuming replicated columns
		assert(ja.size() >= endCols);
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];//big question: does ja give local or global indices?
			const IndexType globalI = inputDist->local2global(i);
                        // just check coordDist[0]. Is is enough? If coord[0] is here so are the others.
			if (neighbor != globalI && coordDist[0]->isLocal(neighbor)) {
				const IndexType localNeighbor = coordDist[0]->global2local(neighbor);
				ValueType distanceSquared = 0;
				for (IndexType dim = 0; dim < dimensions; dim++) {
					ValueType diff = localPartOfCoords2[dim][i] -localPartOfCoords2[dim][localNeighbor];
					distanceSquared += diff*diff;
				}
				if (distanceSquared < minDistanceSquared) minDistanceSquared = distanceSquared;
			}
		}
	}

	const ValueType minDistance = std::sqrt(minDistanceSquared);
	return minDistance;
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::partitionGraph(CSRSparseMatrix<ValueType> &input, std::vector<DenseVector<ValueType>> &coordinates, IndexType dimensions,	IndexType k,  double epsilon) 
{
	/**
	* check input arguments for sanity
	*/
	IndexType n = input.getNumRows();
	if (n != coordinates[0].size()) {
		throw std::runtime_error("Matrix has " + std::to_string(n) + " rows, but " + std::to_string(coordinates[0].size())
		 + " coordinates are given.");
	}
        
        if (dimensions != coordinates.size()){
            throw std::runtime_error("Number of dimensions given "+ std::to_string(dimensions) + "must agree with coordinates.size()=" + std::to_string(coordinates.size()) );
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
        const scai::dmemo::CommunicatorPtr comm = coordDist->getCommunicatorPtr();
        
        if( !coordDist->isEqual( *inputDist) ){ 
            std::cout<< __FILE__<< "  "<< __LINE__<< ", coordDist: " << *coordDist<< " and inputDist: "<< *inputDist<< std::endl;
            throw std::runtime_error( "Distributions: should (?) be equal.");
        }

	const IndexType localN = inputDist->getLocalSize();
        const IndexType globalN = inputDist->getGlobalSize();

        if (coordDist->getLocalSize() != localN) {
		throw std::runtime_error(std::to_string(coordDist->getLocalSize() / dimensions) + " point coordinates, "
		 + std::to_string(localN) + " rows present.");
	}	
	
	/**
	*	gather information for space-filling curves
	*/
	std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
	std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());

        /*
        std::vector<scai::utilskernel::LArray<ValueType>> localPartOfCoords2(dimensions);
        for(IndexType i=0; i<dimensions; i++){
            localPartOfCoords2[i] = coordinates[i].getLocalValues();
        }

	//Get extent of coordinates. Can probably speed this up with OpenMP by having thread-local min/max-Arrays and reducing them in the end
	
	for (IndexType i = 0; i < localPartOfCoords2[0].size() ; i++) {
		for (IndexType dim = 0; dim < dimensions; dim++) {
			ValueType coord = localPartOfCoords2[dim][i];
			if (coord < minCoords[dim]) minCoords[dim] = coord;
			if (coord > maxCoords[dim]) maxCoords[dim] = coord;
		}
	}
	*/
        //
        // the code above finds the local min and max. We want the global min and max.
	
	for (IndexType i = 0; i < globalN; i++) {
		for (IndexType dim = 0; dim < dimensions; dim++) {
			ValueType coord = coordinates[dim].getValue(i).Scalar::getValue<ValueType>();
			if (coord < minCoords[dim]) minCoords[dim] = coord;
			if (coord > maxCoords[dim]) maxCoords[dim] = coord;
		}
	}
        
	ValueType maxExtent = 0;
	for (IndexType dim = 0; dim < dimensions; dim++) {
		if (maxCoords[dim] - minCoords[dim] > maxExtent) {
			maxExtent = maxCoords[dim] - minCoords[dim];
		}
	}

	/**
	* Several possibilities exist for choosing the recursion depth. Either by user choice, or by the maximum fitting into the datatype, or by the minimum distance between adjacent points
	*/
	//getMinimumNeighbourDistance is ~5% faster if manually inlined, probably because localPartOfCoords doesn't have to be computed twice
	//const ValueType minDistance = getMinimumNeighbourDistance(input, coordinates, dimensions);
	const IndexType recursionDepth = std::log2(n);// std::ceil(std::log2(maxExtent / minDistance) / 2);

	/**
	*	create space filling curve indices.
	*/
	
	scai::lama::DenseVector<ValueType> hilbertIndices(inputDist);
	for (IndexType i = 0; i < localN; i++) {
		IndexType globalIndex = inputDist->local2global(i);
		ValueType globalHilbertIndex = HilbertCurve<IndexType, ValueType>::getHilbertIndex(coordinates, dimensions, globalIndex, recursionDepth, minCoords, maxCoords);
		hilbertIndices.setValue(globalIndex, globalHilbertIndex);              
	}

	/**
	* now sort the global indices by where they are on the space-filling curve.
	*/

	scai::lama::DenseVector<IndexType> permutation, permPerm;
	hilbertIndices.sort(permutation, true);
	//permutation.redistribute(inputDist);
        DenseVector<IndexType> tmpPerm = permutation;
        tmpPerm.sort( permPerm, true);
        //permPerm.redistribute(inputDist);
        
        /*
        for (IndexType i = 0; i <n; i++) {
            std::cout<< __FILE__<< ",  "<< __LINE__<< ", "<< i <<" __"<< *comm<< " , perm="<< permutation.getValue(i).Scalar::getValue<IndexType>() << " == permPerm="<< permPerm.getValue(i).Scalar::getValue<IndexType>() << std::endl;
        }
        */
        
        /*
        for (IndexType i = 0; i < localN; i++) {
            std::cout<< __FILE__<< ",  "<< __LINE__<< ", "<< i <<" __"<< *comm<< " , perm="<< permutation.getLocalValues()[i] << " == permPerm="<< permPerm.getLocalValues()[i] << std::endl;
        }
        */
        
        
        
        scai::utilskernel::LArray<ValueType> localPerm = permutation.getLocalValues();
        
        DenseVector<ValueType> coords_gathered( coordDist);
        coords_gathered.gather( coordinates[0], permutation, scai::utilskernel::binary::COPY);
        
	/**
	* check for uniqueness. If not unique, level of detail was insufficient.
	*/


	/**
	* initial partitioning with sfc. Upgrade to chains-on-chains-partitioning later
	*/
        DenseVector<IndexType> tmp_result(inputDist);   //values from 0 to k-1
        DenseVector<IndexType> tmp_indices(inputDist);  //the global indices
	DenseVector<IndexType> result(inputDist);
	scai::hmemo::ReadAccess<IndexType> readAccess(permutation.getLocalValues());
        
	for (IndexType i = 0; i < localN; i++) {
		IndexType targetPos;
		readAccess.getValue(targetPos, i);
                assert( targetPos==localPerm[i] );
		//original: 
                //result.setValue(inputDist->local2global(i), int(k*targetPos / n));
                //changed to:
                //result.setValue( targetPos, int(k*inputDist->local2global(i) / n));
                result.getLocalValues()[i] = int( permPerm.getLocalValues()[i] *k/n);
                //tmp_result.getLocalValues()[i] = int(k* inputDist->local2global(i) / n) ;
                //tmp_indices.getLocalValues()[i] = targetPos;
        }
        
	if (false && inputDist->isReplicated()) {
		ValueType gain = 1;
		ValueType cut = computeCut(input, result);
		while (gain > 0) {
			gain = fiducciaMattheysesRound(input, result, k, epsilon);
			ValueType oldCut = cut;
			cut = computeCut(input, result);
			assert(oldCut - gain == cut);
			std::cout << "Last FM round yielded gain of " << gain << ", for total cut of " << computeCut(input, result) << std::endl;
		}
	}   
	return result;
}

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::fiducciaMattheysesRound(const CSRSparseMatrix<ValueType> &input, DenseVector<IndexType> &part, IndexType k, ValueType epsilon, bool unweighted) {
	const IndexType n = input.getNumRows();
	/**
	* check input and throw errors
	*/
	const Scalar minPartID = part.min();
	const Scalar maxPartID = part.max();
	if (minPartID.getValue<IndexType>() != 0) {
		throw std::runtime_error("Smallest block ID is " + std::to_string(minPartID.getValue<IndexType>()) + ", should be 0");
	}

	if (maxPartID.getValue<IndexType>() != k-1) {
		throw std::runtime_error("Highest block ID is " + std::to_string(maxPartID.getValue<IndexType>()) + ", should be " + std::to_string(k-1));
	}

	if (part.size() != n) {
		throw std::runtime_error("Partition has " + std::to_string(part.size()) + " entries, but matrix has " + std::to_string(n) + ".");
	}

	if (epsilon < 0) {
		throw std::runtime_error("Epsilon must be >= 0, not " + std::to_string(epsilon));
	}

	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();

	if (!inputDist->isReplicated()) {
		throw std::runtime_error("Input matrix must be replicated, for now.");
	}

	if (!partDist->isReplicated()) {
		throw std::runtime_error("Input partition must be replicated, for now.");
	}

	if (!input.checkSymmetry()) {
		throw std::runtime_error("Only undirected graphs are supported, adjacency matrix must be symmetric.");
	}

	/**
	* allocate data structures
	*/

	//const ValueType oldCut = computeCut(input, part, unweighted);

	const IndexType optSize = ceil(double(n) / k);
	const IndexType maxAllowablePartSize = optSize*(1+epsilon);

	std::vector<IndexType> bestTargetFragment(n);
	std::vector<PrioQueue<ValueType, IndexType>> queues(k, n);

	std::vector<IndexType> gains;
	std::vector<std::pair<IndexType, IndexType> > transfers;
	std::vector<IndexType> transferedVertices;
	std::vector<double> imbalances;

	std::vector<double> fragmentSizes(k);

	double maxFragmentSize = 0;

	for (IndexType i = 0; i < n; i++) {
		Scalar partID = part.getValue(i);
		assert(partID.getValue<IndexType>() >= 0);
		assert(partID.getValue<IndexType>() < k);
		fragmentSizes[partID.getValue<IndexType>()] += 1;

		if (fragmentSizes[partID.getValue<IndexType>()] < maxFragmentSize) {
			maxFragmentSize = fragmentSizes[partID.getValue<IndexType>()];
		}
	}

	std::vector<IndexType> degrees(n);
	std::vector<std::vector<ValueType> > edgeCuts(n);

	//Using ReadAccess here didn't give a performance benefit
	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	const scai::utilskernel::LArray<IndexType>& ia = localStorage.getIA();
	const scai::utilskernel::LArray<IndexType>& ja = localStorage.getJA();
	const scai::utilskernel::LArray<IndexType>& values = localStorage.getValues();
	if (!unweighted && values.min() < 0) {
		throw std::runtime_error("Only positive edge weights are supported, " + std::to_string(values.min()) + " invalid.");
	}

	ValueType totalWeight = 0;


	for (IndexType v = 0; v < n; v++) {
		edgeCuts[v].resize(k, 0);

		const IndexType beginCols = ia[v];
		const IndexType endCols = ia[v+1];
		degrees[v] = endCols - beginCols;
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			if (neighbor == v) continue;
			Scalar partID = part.getValue(neighbor);
			edgeCuts[v][partID.getValue<IndexType>()] += unweighted ? 1 : values[j];
			totalWeight += unweighted ? 1 : values[j];
		}
	}

	//setting initial best target for each node
	for (IndexType v = 0; v < n; v++) {
		ValueType maxCut = -totalWeight;
		IndexType idAtMax = k;
		Scalar partID = part.getValue(v);

		for (IndexType fragment = 0; fragment < k; fragment++) {
			if (unweighted) {
				assert(edgeCuts[v][fragment] <= degrees[v]);
			}
			assert(edgeCuts[v][fragment] >= 0);

			if (fragment != partID.getValue<IndexType>() && edgeCuts[v][fragment] > maxCut && fragmentSizes[fragment] <= maxAllowablePartSize) {
				idAtMax = fragment;
				maxCut = edgeCuts[v][fragment];
			}
		}

		assert(idAtMax < k);
		assert(maxCut >= 0);
		if (unweighted) assert(maxCut <= degrees[v]);
		bestTargetFragment[v] = idAtMax;
		assert(partID.getValue<IndexType>() < queues.size());
		if (fragmentSizes[partID.getValue<IndexType>()] > 1) {
			ValueType key = -(maxCut-edgeCuts[v][partID.getValue<IndexType>()]);
			assert(-key <= degrees[v]);
			queues[partID.getValue<IndexType>()].insert(key, v); //negative max gain
		}
	}

	ValueType gainsum = 0;
	bool allQueuesEmpty = false;

	std::vector<bool> moved(n, false);

	while (!allQueuesEmpty) {
	allQueuesEmpty = true;

	//choose largest partition with non-empty queue.
	IndexType largestMovablePart = k;
	IndexType largestSize = 0;

	for (IndexType partID = 0; partID < k; partID++) {
		if (queues[partID].size() > 0 && fragmentSizes[partID] > largestSize) {
			largestMovablePart = partID;
			largestSize = fragmentSizes[partID];
		}
	}

	if (largestSize > 1 && largestMovablePart != k) {
		//at least one queue is not empty
		allQueuesEmpty = false;
		IndexType partID = largestMovablePart;

		assert(partID < queues.size());
		assert(queues[partID].size() > 0);

		IndexType topVertex;
		ValueType topGain;
		std::tie(topGain, topVertex) = queues[partID].extractMin();
		topGain = -topGain;//invert, since the negative gain was used as priority.
		assert(topVertex < n);
		assert(topVertex >= 0);
		if (unweighted) {
			assert(topGain <= degrees[topVertex]);
		}
		assert(!moved[topVertex]);
		Scalar partScalar = part.getValue(topVertex);
		assert(partScalar.getValue<IndexType>() == partID);

		//now get target partition.
		IndexType targetFragment = bestTargetFragment[topVertex];
		ValueType storedGain = edgeCuts[topVertex][targetFragment] - edgeCuts[topVertex][partID];
		
		assert(abs(storedGain - topGain) < 0.0001);
		assert(fragmentSizes[partID] > 1);


		//move node there
		part.setValue(topVertex, targetFragment);
		moved[topVertex] = true;

		//udpate size map
		fragmentSizes[partID] -= 1;
		fragmentSizes[targetFragment] += 1;

		//update history
		gainsum += topGain;
		gains.push_back(gainsum);
		transfers.emplace_back(partID, targetFragment);
		transferedVertices.push_back(topVertex);
		assert(transferedVertices.size() == transfers.size());
		assert(gains.size() == transfers.size());

		double imbalance = (*std::max_element(fragmentSizes.begin(), fragmentSizes.end()) - optSize) / optSize;
		imbalances.push_back(imbalance);

		//std::cout << "Moved node " << topVertex << " to block " << targetFragment << " for gain of " << topGain << ", bringing sum to " << gainsum 
		//<< " and imbalance to " << imbalance  << "." << std::endl;

		//I've tried to replace these direct accesses by ReadAccess, program was slower
		const IndexType beginCols = ia[topVertex];
		const IndexType endCols = ia[topVertex+1];

		for (IndexType j = beginCols; j < endCols; j++) {
			const IndexType neighbour = ja[j];
			if (!moved[neighbour]) {
				//update edge cuts
				Scalar neighbourBlockScalar = part.getValue(neighbour);
				IndexType neighbourBlock = neighbourBlockScalar.getValue<IndexType>();

				edgeCuts[neighbour][partID] -= unweighted ? 1 : values[j];
				assert(edgeCuts[neighbour][partID] >= 0);
				edgeCuts[neighbour][targetFragment] += unweighted ? 1 : values[j];
				assert(edgeCuts[neighbour][targetFragment] <= degrees[neighbour]);

				//find new fragment for neighbour
				ValueType maxCut = -totalWeight;
				IndexType idAtMax = k;

				for (IndexType fragment = 0; fragment < k; fragment++) {
					if (fragment != neighbourBlock && edgeCuts[neighbour][fragment] > maxCut  && fragmentSizes[fragment] <= maxAllowablePartSize) {
						idAtMax = fragment;
						maxCut = edgeCuts[neighbour][fragment];
					}
				}

				assert(maxCut >= 0);
				if (unweighted) assert(maxCut <= degrees[neighbour]);
				assert(idAtMax < k);
				bestTargetFragment[neighbour] = idAtMax;

				ValueType key = -(maxCut-edgeCuts[neighbour][neighbourBlock]);
				assert(-key == edgeCuts[neighbour][idAtMax] - edgeCuts[neighbour][neighbourBlock]);
				assert(-key <= degrees[neighbour]);

				//update prioqueue
				queues[neighbourBlock].remove(neighbour);
				queues[neighbourBlock].insert(key, neighbour);
				}
			}
		}
	}

	const IndexType testedNodes = gains.size();
	if (testedNodes == 0) return 0;
	assert(gains.size() == transfers.size());

        /**
	 * now find best partition among those tested
	 */
	IndexType maxIndex = -1;
	ValueType maxGain = 0;

	for (IndexType i = 0; i < testedNodes; i++) {
		if (gains[i] > maxGain && imbalances[i] <= epsilon) {
			maxIndex = i;
			maxGain = gains[i];
		}
	}
	assert(testedNodes >= maxIndex);
	//assert(maxIndex >= 0);
	assert(testedNodes-1 < transfers.size());

	/**
	 * apply partition modifications in reverse until best is recovered
	 */
	for (int i = testedNodes-1; i > maxIndex; i--) {
		assert(transferedVertices[i] < n);
		assert(transferedVertices[i] >= 0);
		part.setValue(transferedVertices[i], transfers[i].first);
	}
	return maxGain;
}

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::computeCut(const CSRSparseMatrix<ValueType> &input, const DenseVector<IndexType> &part, bool ignoreWeights) {
	const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
	const scai::dmemo::DistributionPtr partDist = part.getDistributionPtr();
	const IndexType n = inputDist->getGlobalSize();
	const IndexType localN = inputDist->getLocalSize();

	if (!partDist->isReplicated()) {
		throw std::runtime_error("Input partition must be replicated, for now.");
	}

	const CSRStorage<ValueType>& localStorage = input.getLocalStorage();
	scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	scai::hmemo::ReadAccess<IndexType> partAccess(part.getLocalValues());

	ValueType result = 0;
	for (IndexType i = 0; i < localN; i++) {
		const IndexType beginCols = ia[i];
		const IndexType endCols = ia[i+1];
		assert(ja.size() >= endCols);

		const IndexType globalI = inputDist->local2global(i);
		IndexType thisBlock;
		partAccess.getValue(thisBlock, globalI);
		
		for (IndexType j = beginCols; j < endCols; j++) {
			IndexType neighbor = ja[j];
			assert(neighbor >= 0);
			assert(neighbor < n);
				
			IndexType neighborBlock;
			partAccess.getValue(neighborBlock, neighbor);
			if (neighborBlock != thisBlock) {
				if (ignoreWeights) {
					result++;
				} else {
					ValueType edgeWeight;
					values.getValue(edgeWeight, j);
					result += edgeWeight;
				}
			}
		}
	}

	if (!inputDist->isReplicated()) {
    //sum block sizes over all processes
    result = inputDist->getCommunicatorPtr()->sum(result);
  }

  return result / 2; //counted each edge from both sides
}

template<typename IndexType, typename ValueType>
ValueType ParcoRepart<IndexType, ValueType>::computeImbalance(const DenseVector<IndexType> &part, IndexType k) {
	const IndexType n = part.getDistributionPtr()->getGlobalSize();
	std::vector<IndexType> subsetSizes(k, 0);
	scai::hmemo::ReadAccess<IndexType> localPart(part.getLocalValues());
	const Scalar maxK = part.max();
	if (maxK.getValue<IndexType>() >= k) {
		throw std::runtime_error("Block id " + std::to_string(maxK.getValue<IndexType>()) + " found in partition with supposedly" + std::to_string(k) + " blocks.");
	}
 	
	for (IndexType i = 0; i < localPart.size(); i++) {
		IndexType partID;
		localPart.getValue(partID, i);
		subsetSizes[partID] += 1;
	}
	IndexType optSize = std::ceil(n / k);

	//if we don't have the full partition locally, 
	scai::dmemo::CommunicatorPtr comm = part.getDistributionPtr()->getCommunicatorPtr();
	if (!part.getDistribution().isReplicated()) {
	  //sum block sizes over all processes
	  for (IndexType partID = 0; partID < k; partID++) {
	    subsetSizes[partID] = comm->sum(subsetSizes[partID]);
	  }
	}
	
	IndexType maxBlockSize = *std::max_element(subsetSizes.begin(), subsetSizes.end());
	return ((maxBlockSize - optSize)/ optSize);
}


template<typename IndexType, typename ValueType>
DenseVector<IndexType> ParcoRepart<IndexType, ValueType>::getBorderNodes( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::utilskernel::LArray<IndexType> localPart= part.getLocalValues();
    DenseVector<IndexType> border(dist,0);
    scai::utilskernel::LArray<IndexType> localBorder= border.getLocalValues();
    
    IndexType N = adjM.getNumColumns();
    IndexType max = part.max().Scalar::getValue<IndexType>();
    
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }
    //TODO: Use DenseVector from start: localV and nonLocalV sizes are not known beforehand. That is why I use a std::vector and then convert it to a DenseVector. 
    std::vector<IndexType> localV, nonLocalV;
    for(IndexType i=0; i<dist->getLocalSize(); i++){    // for all local nodes 
        scai::hmemo::HArray<ValueType> localRow;        // get local row on this processor
        adjM.getLocalRow( localRow, i);
        scai::hmemo::ReadAccess<ValueType> readLR(localRow); 

        assert(readLR.size() == adjM.getNumColumns());
        for(IndexType j=0; j<N; j++){                   // for all the edges of a node
            ValueType val;
            readLR.getValue(val, j);      
            if(val>0){                                  // i and j have an edge               
                if(dist->isLocal(j)){      
                    assert( localPart[ dist->global2local(j) ] < max +1 );
                    if( localPart[i] != localPart[ dist->global2local(j) ] ){ // i and j are in different parts
                        localBorder[i] = 1;             // then i is a border node
                        break;                          // if this is a border node then break 
                    }
                } else{     // if j is not local index in this PE, store the indices and gather later
                    localV.push_back(i);
                    nonLocalV.push_back(j);
                }
            }
        }
    }

    // take care of all the non-local indices found
    assert( localV.size() == nonLocalV.size() );
    DenseVector<IndexType> nonLocalDV( nonLocalV.size() , 0 );
    DenseVector<IndexType> gatheredPart(nonLocalDV.size() , 0);
    
    //get a DenseVector grom a vector
    for(IndexType i=0; i<nonLocalV.size(); i++){
        nonLocalDV.setValue(i, nonLocalV[i]);
    }
    //gather all non-local indexes
    gatheredPart.gather(part, nonLocalDV , scai::utilskernel::binary::COPY );

    assert( localV.size()==nonLocalDV.size() );
    assert( nonLocalDV.size()==gatheredPart.size() );
    for(IndexType i=0; i<gatheredPart.size(); i++){
        if(localPart[ localV[i]] != gatheredPart(i).Scalar::getValue<IndexType>()  ){
            localBorder[localV[i]]=1;
        }
    }
   
    border.setValues(localBorder);
    return border;
}

//----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> ParcoRepart<IndexType, ValueType>::getPEGraph( const CSRSparseMatrix<ValueType> &adjM) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr(); 
    
    //std::cout<< __FILE__<< " ,"<<__LINE__<<": matrix distribution:"<< *dist<< std::endl;
    
    scai::hmemo::HArray<IndexType> indicesH(0); 
    scai::hmemo::WriteAccess<IndexType> indicesHWrite(indicesH);

    for(IndexType i=0; i<dist->getLocalSize(); i++){    // for all local nodes 
        scai::hmemo::HArray<ValueType> localRow;        // get local row on this processor
        adjM.getLocalRow( localRow, i);
        scai::hmemo::ReadAccess<ValueType> readLR(localRow); 
        assert(readLR.size() == adjM.getNumColumns());  
        for(IndexType j=0; j<readLR.size(); j++){       // for all the edges of a node
            ValueType val;
            readLR.getValue(val, j);
            if(val>0){                                  // i and j have an edge             
                if( !dist->isLocal(j) ){                // if j is not a local node
                    indicesHWrite.resize( indicesH.size() +1);  
                    indicesHWrite[indicesHWrite.size()-1] = j;  // store the non local index
                }
            }
        }
    }
    indicesHWrite.release();
  
    // find the PEs that own every non-local index
    scai::hmemo::HArray<IndexType> owners(indicesH.size() , -1);
    dist->computeOwners( owners, indicesH);
    
    // create the PE adjacency matrix to be returned
    IndexType numPEs = comm->getSize();
    scai::dmemo::DistributionPtr distPEs ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, numPEs) );  
    scai::dmemo::DistributionPtr noDistPEs (new scai::dmemo::NoDistribution( numPEs ));
    // every PE must have one row of the matrix since we have numPes and the matrix is [numPes x numPEs]

    scai::lama::SparseAssemblyStorage<ValueType> myStorage( distPEs->getLocalSize(), numPEs);
    //scai::lama::MatrixStorage<ValueType> myStorage( distPEs->getLocalSize(), numPEs);
    scai::hmemo::ReadAccess<IndexType> readI(owners);
    for(IndexType i=0; i<readI.size(); i++){
        myStorage.setValue(0, readI[i], 1);
    }
    readI.release();
     
    scai::lama::CSRSparseMatrix<ValueType> PEgraph(myStorage, distPEs, noDistPEs);     

    return PEgraph;
}

//-----------------------------------------------------------------------------------------

//return: there is an edge is the block graph between blocks ret[0]-ret[1], ret[2]-ret[3] ... ret[2i]-ret[2i+1] 
template<typename IndexType, typename ValueType>
std::vector<std::vector<IndexType>> ParcoRepart<IndexType, ValueType>::getLocalBlockGraphEdges( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part) {
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    const scai::utilskernel::LArray<IndexType> localPart= part.getLocalValues();
    IndexType N = adjM.getNumColumns();
    IndexType max = part.max().Scalar::getValue<IndexType>();
   
    if( !dist->isEqual( part.getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and partition dist: "<< part.getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }
    
    // edges[i,j] if an edge exists between blocks i and j
    std::vector< std::vector<IndexType> > edges(2);
    std::vector<IndexType> localInd, nonLocalInd;
    
    for(IndexType i=0; i<dist->getLocalSize(); i++){    // for all local nodes 
        scai::hmemo::HArray<ValueType> localRow;        // get local row on this processor
        adjM.getLocalRow( localRow, i);
        scai::hmemo::ReadAccess<ValueType> readLR(localRow); 
        assert(readLR.size() == adjM.getNumColumns()); 
        for(IndexType j=0; j<readLR.size(); j++){       // for all the edges of a node
            ValueType val;
            readLR.getValue(val, j);
            if(val>0){                                  // i and j have an edge             
                //TODO: probably part.getValue(j) is not correct. If true the assertion should fail at some point.
                //edge (u,v) if the block graph
                //TODO: remove part.getValue(j) but now only gets an edge if both i and j are local. 
                //      when j is not local???
                if(dist->isLocal(j)){
                    IndexType u = localPart[i];         // partition(i)
                    IndexType v = localPart[dist->global2local(j)]; // partition(j), 0<j<N so take the local index of j
//std::cout<<  __FILE__<< " ,"<<__LINE__<<" == "<< i <<", " << j <<":  __"<< *comm<< " , u=" << u<< " , v="<< v << std::endl;   
                    assert( u < max +1);
                    assert( v < max +1);
                    if( u != v){    // the nodes belong to different blocks                  
                        bool add_edge = true;
                        for(IndexType k=0; k<edges[0].size(); k++){ //check that this edge is not already in
// std::cout<<  __FILE__<< " ,"<<__LINE__<<" == k="<< k <<":  __"<< *comm<< " , u=" << edges[0][k]<< " , v="<< edges[1][k] << std::endl;                            
                            if( edges[0][k]==u && edges[1][k]==v ){
                                add_edge= false;
                                break;      // the edge (u,v) already exists
                            }
                        }
                        if( add_edge== true){       //if this edge does not exist, add it
//std::cout<<  __FILE__<< " ,"<<__LINE__<< "\t __"<< *comm<<",  adding edge ("<< u<< ","<< v<< ")\n";    
                            edges[0].push_back(u);
                            edges[1].push_back(v);
                        }
                    }
                } else{  // if(dist->isLocal(j)) , what TODO when j is not local?
                // there is an edge between i and j but index j is not local in the partition so we cannot get part[j].
                    localInd.push_back(i);
                    nonLocalInd.push_back(j);
                }
            }
        }
    }//for 
    
    // take care of all the non-local indices found
    assert( localInd.size() == nonLocalInd.size() );
    DenseVector<IndexType> nonLocalDV( nonLocalInd.size() , 0 );
    DenseVector<IndexType> gatheredPart(nonLocalDV.size() , 0);
    
    //get a DenseVector grom a vector
    for(IndexType i=0; i<nonLocalInd.size(); i++){
        nonLocalDV.setValue(i, nonLocalInd[i]);
    }
    //gather all non-local indexes
    gatheredPart.gather(part, nonLocalDV , scai::utilskernel::binary::COPY );
    
    assert( gatheredPart.size() == nonLocalInd.size() );
    assert( gatheredPart.size() == localInd.size() );
    
    for(IndexType i=0; i<gatheredPart.size(); i++){
        IndexType u = localPart[ localInd[i] ];         
        IndexType v = gatheredPart.getValue(i).Scalar::getValue<IndexType>();
        //std::cout<<  __FILE__<< " ,"<<__LINE__<<" == "<< i <<", " << j <<":  __"<< *comm<< " , u=" << u<< " , v="<< v << std::endl;   
        assert( u < max +1);
        assert( v < max +1);
        if( u != v){    // the nodes belong to different blocks                  
            bool add_edge = true;
            for(IndexType k=0; k<edges[0].size(); k++){ //check that this edge is not already in
                // std::cout<<  __FILE__<< " ,"<<__LINE__<<" == k="<< k <<":  __"<< *comm<< " , u=" << edges[0][k]<< " , v="<< edges[1][k] << std::endl;                            
                if( edges[0][k]==u && edges[1][k]==v ){
                    add_edge= false;
                    break;      // the edge (u,v) already exists
                }
            }
            if( add_edge== true){       //if this edge does not exist, add it
                //std::cout<<  __FILE__<< " ,"<<__LINE__<< "\t __"<< *comm<<",  adding edge ("<< u<< ","<< v<< ")\n";    
                edges[0].push_back(u);
                edges[1].push_back(v);
            }
        }
    }
    return edges;
}

//-----------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
scai::hmemo::HArray<IndexType> ParcoRepart<IndexType, ValueType>::getBlockGraph( const CSRSparseMatrix<ValueType> &adjM, const DenseVector<IndexType> &part, const int k , const IndexType root) {

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    const scai::utilskernel::LArray<IndexType> localPart= part.getLocalValues();
    
    // IndexType k = part.max(); // if K is not given as a parameter
    // there are k blocks in the partition so the adjecency matrix for the block graph has dimensions [k x k]
    scai::dmemo::DistributionPtr distRowBlock ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, k) );  
    scai::dmemo::DistributionPtr distColBlock ( new scai::dmemo::NoDistribution( k ));
    scai::lama::CSRSparseMatrix<ValueType> blockGraph( distRowBlock, distColBlock );    // the distributed adjacency matrix of the block-graph
    
    IndexType size= k*k;
    /*
    // only root will have full size
    if( comm->getRank() == root){
        size= k*k;
    }
    */
    
    // get, on each processor, the edges of the blocks that are local
    std::vector< std::vector<IndexType> > blockEdges = ParcoRepart<int, double>::getLocalBlockGraphEdges( adjM, part);
    
    // the gather function accepts a one dimensional array
    //IndexType blockGraphMatrix[size];    
    scai::hmemo::HArray<IndexType> sendPart(size, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvPart(size);
    
    for(IndexType round=0; round<comm->getSize(); round++){
        {   // write your part 
            scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendPart );
            for(IndexType i=0; i<blockEdges[0].size(); i++){
                IndexType u = blockEdges[0][i];
                IndexType v = blockEdges[1][i];
//std::cout<<__FILE__<< "  "<< __LINE__<< " , round:"<< round<< " , __"<< *comm << " setting edge ("<< u<< ", "<< v << ") , in send-index:"<< u*k+v <<std::endl;
                sendPartWrite[ u*k + v ] = 1;
            }
        }
        comm->shiftArray(recvPart , sendPart, 1);
        sendPart.swap(recvPart);
    } 
    
        /*
    { // print
    scai::hmemo::ReadAccess<IndexType> sendPartRead( sendPart );
    std::cout<< *comm <<" , sendPart"<< std::endl;
    for(IndexType row=0; row<k; row++){
        for(IndexType col=0; col<k; col++){
            std::cout<< comm->getRank()<< ":("<< row<< ","<< col<< "):" << sendPartRead[ row*k +col] <<" - ";
        }
        std::cout<< std::endl;
    }
    
    scai::hmemo::ReadAccess<IndexType> recvPartRead( recvPart );
    std::cout<< *comm <<" , recvPart"<< std::endl;
    for(IndexType row=0; row<k; row++){
        for(IndexType col=0; col<k; col++){
            std::cout<< comm->getRank()<< ":("<< row << "," << col<< "):" << recvPartRead[ row*k +col] <<" - ";
        }
        std::cout<< std::endl;
    }
    }
    */
    
    /*
    IndexType localBlockGraph [k][k];
    
    for(IndexType i=0; i<blockEdges[0].size(); i++){
        IndexType u = blockEdges[0][i];
        IndexType v = blockEdges[1][i];
std::cout<<__FILE__<< "  "<< __LINE__<< " , __"<< *comm << " setting edge ("<< u<< ", "<< v << ")"<<std::endl;        
        localBlockGraph[u][v] = 1;
    }
    
    */
    
    return recvPart;
}


//to force instantiation
template DenseVector<int> ParcoRepart<int, double>::partitionGraph(CSRSparseMatrix<double> &input, std::vector<DenseVector<double>> &coordinates, int dimensions,	int k,  double epsilon);

template double ParcoRepart<int, double>::getMinimumNeighbourDistance(const CSRSparseMatrix<double> &input, const std::vector<DenseVector<double>> &coordinates, int dimensions);
			     
//template struct point ParcoRepart<int, double>::hilbert(double index, int level);
template double ParcoRepart<int, double>::computeImbalance(const DenseVector<int> &partition, int k);

template double ParcoRepart<int, double>::computeCut(const CSRSparseMatrix<double> &input, const DenseVector<int> &part, bool ignoreWeights);

template double ParcoRepart<int, double>::fiducciaMattheysesRound(const CSRSparseMatrix<double> &input, DenseVector<int> &part, int k, double epsilon, bool unweighted);

template DenseVector<int> ParcoRepart<int, double>::getBorderNodes( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::lama::CSRSparseMatrix<double> ParcoRepart<int, double>::getPEGraph( const CSRSparseMatrix<double> &adjM);

template std::vector<std::vector<IndexType>> ParcoRepart<int, double>::getLocalBlockGraphEdges( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part);

template scai::hmemo::HArray<int> ParcoRepart<int, double>::getBlockGraph( const CSRSparseMatrix<double> &adjM, const DenseVector<int> &part, const int k, const int root );

}
