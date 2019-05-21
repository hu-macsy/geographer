/*
 * Mapping.cpp
 *
 *  Created on: 11.10.2018
 *      Author: tzovas
 */

#include "Mapping.h"

namespace ITI {

//Tranferring code from the implementation of Roland in TiMEr/src/mapping/algorithms.cpp

//Edge weigths in the block graph represent the amount of information that needs to
//be communicated, aka the communication volume.
//Edge weights in the the processor graph represent the bandwidth/capacity of the
//edge. Higher values indicate faster connection.

template <typename IndexType, typename ValueType>
std::vector<IndexType> Mapping<IndexType, ValueType>::rolandMapping_local( 
	scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
	scai::lama::CSRSparseMatrix<ValueType>& PEGraph){

	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType N = blockGraph.getNumRows();
	SCAI_ASSERT_EQ_ERROR( N, PEGraph.getNumRows(), "The block and the processor graph must have the same number of nodes");
	SCAI_ASSERT_EQ_ERROR( N, blockGraph.getNumColumns(), "Block graph matrix must be square" );
	SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), PEGraph.getNumColumns(), "Processor graph matrix must be square" );

	//TODO: harry: not sure what is this for...
  	//unsigned int offset = P.number_of_nodes() - k;
  	//offset != 0 only for processor graphs in which there are switches

	// the mapping to be returned
	std::vector<IndexType> mapping(N,0);

	//total communication involving node i in communication graph
  	std::vector<ValueType> comFrom(N, 0);
/*	
	//compute weighted node degrees in block graph and store them in comFrom
	{
	    const scai::lama::CSRStorage<ValueType> localStorage = blockGraph.getLocalStorage();
	    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	    //const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	    const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	    for(IndexType i=0; i<N; i++){
	    	for(IndexType valuesInd=ia[i]; valuesInd<ia[i+1]; valuesInd++){
        	//for(IndexType j=ia[i]; j<ia[i+1]; j++){ 
        		comFrom[i] += values[valuesInd];
        	}
        }
	}

	//sort comFrom in decreasing order
	//store the sorted indices in permutation vector
		
	//the comFrom vector changes after nodes are mapped so sorting (maybe) does
	//not help. Just fins the max element in the vector
	
    //std::vector<IndexType> permutation(N);	
	//std::iota(permutation.begin(), permutation.end(), 0);    
    //std::sort(permutation.begin(), permutation.end(), [&](IndexType i, IndexType j){return comFrom[i] > comFrom[j];});	
	
	//choose for blockNode the node with the maximum comFrom
  	//IndexType blockNode = permutation[0];
  	typename std::vector<ValueType>::iterator blockNodeWeight = std::max_element(comFrom.begin(), comFrom.end() );
	IndexType blockNode = std::distance(comFrom, blockNodeWeight);

  	//choose processor node at random
  	//TODO: improve the random selection? 
  	//TODO: why not match the heaviest node from blockGraph with the haviest node in
  	// PEGraph?
  	srand(time(NULL));
  	IndexType procNode = rand()%N;
  	SCAI_ASSERT_GE_ERROR(procNode, 0, "Wrong node ID");

  	//reset comFrom
  	for(unsigned int i=0; i<N; i++)
  		comFrom[i]=0;

  	//sum of distances to node i in processor graph
	std::vector<int> procFrom(N, 0);
	
	//TODO: possible optimization: count the number of mapped nodes, in comFrom swap
	// the value of a mapped node with comFrom[N-numMappedNodes] and search for the
	// new maximum in [comFrom.begin(), comFrom.end()-numMappedNodes].
	// Update: not sure this is doable beacuse we mess with the numbering of nodes
	// if I map node 10 and swap the weights of node 10 with, say, node 100, then
	// node 100 is now node 10. Need to keep additional mapping...
	//IndexType numMappedNodes = 0;

  	for(IndexType i=0; i<N-1; i++){
  		mapping[blockNode] = procNode;
  		//numMappedNodes++;

  		//assigned nodes must not be assigned again
    	comFrom[blockNode] = -1;
    	procFrom[procNode] = INT_MAX;

    	//update comFrom
    	//>is the update needed? why not keep it sorted
    	//> the update below is wrong, it must exclude edges that one endpoint
    	//is already mapped(I think)

		{
		    const scai::lama::CSRStorage<ValueType> localStorage = blockGraph.getLocalStorage();
		    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
		    const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

		    for(IndexType i=0; i<N; i++)
		    	for(IndexType valuesInd=ia[i]; valuesInd<ia[i+1]; valuesInd++)
	        		comFrom[i] += values[valuesInd];
		}

		//find next blockNode
		//blockNode
  	}//for i<N
*/

  	return mapping;

}
//------------------------------------------------------------------------------------

// copy and convert/reimplement code from libTopoMap,
// http://htor.inf.ethz.ch/research/mpitopo/libtopomap/,
// function TPM_Map_greedy found in file libtopomap.cpp around line 580

template <typename IndexType, typename ValueType>
std::vector<IndexType> Mapping<IndexType, ValueType>::torstenMapping_local( 
	const scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
	const scai::lama::CSRSparseMatrix<ValueType>& PEGraph){

	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType N = blockGraph.getNumRows(); //number of nodes in bot graphs

	SCAI_ASSERT_EQ_ERROR( N, PEGraph.getNumRows(), "The block and the processor graph must have the same number of nodes");
	SCAI_ASSERT_EQ_ERROR( N, blockGraph.getNumColumns(), "Block graph matrix must be square" );
	SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), PEGraph.getNumColumns(), "Processor graph matrix must be square" );

	// PEGraph is const so make local copy
	scai::lama::CSRSparseMatrix<ValueType> copyPEGraph;// = PEGraph;

	//TODO: why is the thing below needed? we ignore the edges of the original
	// PEGraph?
	//change the edge weights in the copyPEGraph
	{
		//find max weight in block graph
		const scai::lama::CSRStorage<ValueType> localStorage = blockGraph.getLocalStorage();
		const scai::hmemo::ReadAccess<ValueType> blockValues(localStorage.getValues());
		ValueType maxW = 0;
		for( unsigned int i=0; i<blockValues.size(); i++ ){
			if( blockValues[i]>maxW ){
				maxW = blockValues[i];
			}
		}

		// initValue = max edge in blockGraph * (num nodes in PEGraph)^2
		//WARNING: in original function, is initValue = maxW * PEGraph.size()^2 
		//but here we assume that the two graph have same number of nodes
		ValueType initValue = maxW * N * N;
		SCAI_ASSERT_GT(initValue, 0, "Int overflow"); // check for overflow

		//set initValue to copyPEgrah
		//TODO: find faster, shorter way to do the copy
		// with copyGraph(PEGraph) throws a "binding const reference" error
		scai::lama::CSRStorage<ValueType> tmpStorage ( PEGraph.getLocalStorage() );
		scai::hmemo::HArray< IndexType > tmpIA( tmpStorage.getIA() );
		scai::hmemo::HArray< IndexType > tmpJA( tmpStorage.getJA() );
		scai::hmemo::HArray< ValueType > tmpValues( tmpStorage.getValues().size() );
		SCAI_ASSERT_EQ_ERROR( tmpIA.size(), N+1, "Wrong ia size");
		
		for( unsigned int i=0; i<tmpValues.size(); i++ ){
			tmpValues[i] = initValue;
		}
		scai::lama::CSRStorage<ValueType> copyStorage( N, N, tmpIA, tmpJA, tmpValues);
		copyPEGraph.assign( copyStorage );

		SCAI_ASSERT_EQ_ERROR( PEGraph.getNumValues(), copyPEGraph.getNumValues(), "Error in coping matrix");
/*		
for( int i=0; i<N; i++){
	for(int j=0; j<N; j++ ){
		SCAI_ASSERT_EQ_ERROR( PEGraph.getValue(i,j), copyPEGraph.getValue(i,j), "value is not same for edge (" << i << ", " << j << ")");
		std::cout<< i << " - " << j << " ==> " << copyPEGraph.getValue(i,j) << std::endl;
	}
}
*/
	}


  	std::vector<bool> mapped( N, false);
  	std::vector<bool> discovered( N, false);
  	std::vector<bool> usedPEs( N, false); // for PEs that are already mapped

  	/********************************************************
   	* The actual mapping code                            
   	********************************************************/

  	//in the original code, this is given as input; here pick at random
  	//TODO: possible opt: as for the blockGraph, pick from PEGraph the node with
  	// the maximum weighted degree?
  	srand(time(NULL));
   	IndexType peNode = rand()%N; // the current vertex in PEGraph to be mapped to
  	SCAI_ASSERT_GE_ERROR(peNode, 0, "Wrong node ID");
  	
  	IndexType numMappedNodes = 0; // number of already mapped nodes in blockGraph

	//std::vector<ValueType> comFrom(N, 0);

	// access the data of the block graph
	const scai::lama::CSRStorage<ValueType> blockStorage = blockGraph.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(blockStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(blockStorage.getJA());
	const scai::hmemo::ReadAccess<ValueType> blockValues(blockStorage.getValues());
	//actually, the weights of the edges of the PE graph. We will update them
	//after wvery time we map a vertex
	scai::lama::CSRStorage<ValueType> PEStorage = copyPEGraph.getLocalStorage();
	scai::hmemo::ReadAccess<ValueType> PEValues( PEStorage.getValues() );
	const scai::hmemo::ReadAccess<IndexType> PEia( PEStorage.getIA() );
	// the mapping to be returned
	std::vector<IndexType> mapping(N,0);

  	//while there are unmapped nodes
  	while(numMappedNodes < N) {
  		//TODO: instead of storing all weights and then call max_element,
  		// just find the max directly
  		//compute weighted node degrees in block graph and store them in comFrom
		//std::fill( comFrom.begin(), comFrom.end(), 0);
		
		ValueType blockNodeWeight = -1;
		IndexType blockNode = -1;

	    for(IndexType i=0; i<N; i++){
	    	if(mapped[i])	// if this is already mapped 
	    		continue;
	    	ValueType weightedDegree = 0;
	    	for(IndexType valuesInd=ia[i]; valuesInd<ia[i+1]; valuesInd++){
	    		//comFrom[i] += blockValues[valuesInd];
	    		weightedDegree += blockValues[valuesInd];
	    	}
	    	if( weightedDegree>blockNodeWeight ){
				blockNodeWeight = weightedDegree;
				blockNode = i;
	    	}
	    }

		//typename std::vector<ValueType>::iterator blockNodeWeight = std::max_element(comFrom.begin(), comFrom.end() );
		//IndexType blockNode = std::distance(comFrom.begin(), blockNodeWeight);

		SCAI_ASSERT_GE_ERROR( blockNode, 0, "Wrong node ID");
		SCAI_ASSERT_LT_ERROR( blockNode, N, "Wrong node ID");
		SCAI_ASSERT_GE_ERROR( blockNodeWeight, 0, "Wrong node weight");
		PRINT0("heaviest vertex in block graph is " << blockNode << " with weight " << blockNodeWeight);

		//TODO: check, possible opt: when/where is peNode changing?? is it always picked
		// at random
		while( usedPEs[peNode] ){ 
			peNode = rand()%N;
		}

		// map blockNode to peNode
		mapping[blockNode] = peNode;	//peNode is "current" in libtopomap.cpp
PRINT0( "mapped vertex " << blockNode << " to " << peNode);
		mapped[blockNode] = true;
		discovered[blockNode] = true;
		usedPEs[peNode] = true;
		numMappedNodes++;

		/* add all edges of this vertex that lead to not-discovered vertices
     	* to prio queue */
     	// ^^ comment from libtopomap.cpp; it is actually vertex ID, right?
     	// the pair stored is <edgeWeight, nodeID> where one endpoint of the edge
     	// is "nodeID" and the other the current node "blockNode"
     	std::priority_queue<std::pair<double,int>, std::vector<std::pair<double,int> >, max_compare_func > Q;
     	// check all edges of blockNode
     	for(IndexType iaInd=ia[blockNode]; iaInd<ia[blockNode+1]; iaInd++){
     		IndexType neighbor = ja[iaInd];	// edge (blockNode, neighbor)
     		ValueType edgeWeight = blockValues[iaInd];
     		if( !discovered[neighbor] ){
     			Q.push( std::make_pair(edgeWeight, neighbor) );
     			discovered[neighbor] = true;
     		}
     	}

	    /* while Q not empty, find most expensive edge in queue */
     	while(!Q.empty()) {
			// take the most expensive edge out of the queue
     		std::pair<double,int> Qel = Q.top();
     		Q.pop();
     		IndexType neighbor = Qel.second;	// neighbor node of blockNode

     		PRINT0("heaviest edge for node " << blockNode << ": (" << blockNode << ", "<< neighbor <<"), with weight " << Qel.first);

     		/* find next vertex that is as close as possible (minimum path from
       		* current but still has available slots (vertex weight is not 0)
       		* map the target of the edge from previous step to it */

     		//TODO: possible opt, not need a full Dijkstra, when the first not
     		// used node is found stop
//WARNING: need shortest paths of longest? we need to map "heavy" nodes of the 
// blockGraph with "heavy" nodes in the PEGraph (heavy according to their weighted
// degree).
// Update: the above is true but irrelevant here: we should find a "heavy" node
// but also one that is close/near to peNode.
     		std::vector<IndexType> predecessor;
     		std::vector<ValueType> distances = GraphUtils<IndexType, ValueType>::localDijkstra( copyPEGraph, peNode, predecessor);
     		SCAI_ASSERT_EQ_ERROR( distances.size(), N, "Wrong distances size");

     		IndexType closestNode = -1;
     		ValueType nodeDist = std::numeric_limits<double>::max();
     		for( unsigned int i=0; i<N; i++){
     			/*in the original code is "and has free processors"*/
     			if( distances[i]<nodeDist and !usedPEs[i] ){ 
     				closestNode = i;
     				nodeDist = distances[i];
     			}
     		}

     		// map the neighbor to the closest node in the PEGraph
     		mapping[neighbor] = closestNode;
PRINT0( "mapped vertex " << neighbor << " to " << closestNode);     		
     		mapped[neighbor] = true;
     		discovered[neighbor] = true;
     		usedPEs[closestNode] = true;
     		numMappedNodes++;

     		// update occupied edges in processor graph
// "occupied edges" are all the edges in the shortest path from peNode to closestNode
// for these edges, we should update (reduce?) their capacity.
     		//IndexType v = predecessor[closestNode];
     		IndexType v = closestNode;
     		while( v!= peNode ){
     			IndexType predV = predecessor[v];
     			IndexType valuesInd = PEia[v];
     			SCAI_ASSERT_LT_ERROR(valuesInd, PEValues.size(), "Wrong values index" );
//WARNING: not sure at all that this is correct     			
//PEValues[valuesInd] += Qel.first/PEValues[valuesInd];
PEStorage.setValue( v, predV, Qel.first/PEStorage.getValue(v, predV) );
     			v = predecessor[v];
     		}

     	}//while(!Q.empty()) 

  	}//while(numMappedNodes < N)

  	return mapping;
}

//------------------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
bool Mapping<IndexType, ValueType>::isValid(
	const scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
	const scai::lama::CSRSparseMatrix<ValueType>& PEGraph,
	std::vector<IndexType> mapping){

	const IndexType n = blockGraph.getNumRows();
	SCAI_ASSERT_EQ_ERROR( mapping.size(), n,"Wrong mapping size");
	SCAI_ASSERT_EQ_ERROR( mapping.size(), PEGraph.getNumRows(),"Wrong mapping size");

	IndexType mapSum = std::accumulate( mapping.begin(), mapping.end(), 0);
	SCAI_ASSERT_EQ_ERROR( mapSum, n*(n-1)/2, "Wrong mapping checksum");

	return true;
}

//------------------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
std::vector<IndexType> Mapping<IndexType, ValueType>::getSfcRenumber( 
	const std::vector<scai::lama::DenseVector<ValueType>>& coordinates,
	const std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights, //TODO,check: are these needed?
	const scai::lama::DenseVector<IndexType>& partition,
	const Settings settings){

	const IndexType localN = coordinates[0].getLocalValues().size();
	const IndexType k = settings.numBlocks;
	const IndexType dim = settings.dimensions;

	//TODO: maybe this can be avoided
	std::vector<IndexType> localIndices(localN);
	std::iota(localIndices.begin(), localIndices.end(), 0);
	const typename std::vector<IndexType>::iterator firstIndex = localIndices.begin();
	const typename std::vector<IndexType>::iterator lastIndex = localIndices.end();

	//to make it more readable
	using point = std::vector<ValueType>;

	//the center for each block
	std::vector<point> blockCenters = KMeans::findCenters(coordinates, partition, k, firstIndex, lastIndex, nodeWeights[0]);

	//remember: blockCenters.size()=dim, blockCenters[i].size()=numBlocks
	SCAI_ASSERT_EQ_ERROR( blockCenters.size(), dim, "Wrong size of centers vector." );
	SCAI_ASSERT_EQ_ERROR( blockCenters[0].size(), k, "Wrong size of centers vector." )

	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	for(IndexType i=0; i<k; i++){
		PRINT0("center " << i << " in " << blockCenters[0][i] <<", " << blockCenters[1][i]);
	}

	//
	//get the sfc index of the centers
	//

	std::vector<ValueType> centerSFC;

	//convert to vector<DenseVector> in order to call getHilbertIndexVector
	{
		std::vector<scai::lama::DenseVector<ValueType>> centersDV(dim);

		for(IndexType d=0; d<dim; d++){
			//centersDV[d].resize(k);
			centersDV[d].assign( scai::hmemo::HArray<ValueType>(blockCenters[d]) );
		}

		//TODO: check if default resolution is OK or set it properly		
		centerSFC = HilbertCurve<IndexType, ValueType>::getHilbertIndexVector( centersDV, settings.sfcResolution, dim);
	}

	//the IDs to use for sorting
	std::vector<IndexType> centerIDs(k);
	std::iota( centerIDs.begin(), centerIDs.end(), 0);

	//sort center IDs according to their SFC value
	std::sort( centerIDs.begin(), centerIDs.end(),
	 	[&](IndexType a, IndexType b){
	 		return centerSFC[a]<centerSFC[b];
	 	});

	return centerIDs;

}//sfcMapping
//------------------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
std::vector<IndexType> Mapping<IndexType, ValueType>::applySfcRenumber( 
	const std::vector<scai::lama::DenseVector<ValueType>>& coordinates,
	const std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights, //TODO,check: are these needed?
	scai::lama::DenseVector<IndexType>& partition,
	const Settings settings){

	//get the reunmbering
	std::vector<IndexType> R = Mapping<IndexType,ValueType>::getSfcRenumber( coordinates, nodeWeights, partition, settings );

	SCAI_ASSERT_EQ_ERROR( R.size(), settings.numBlocks, "Size mismatch" );
	const IndexType k = settings.numBlocks;

	const IndexType maxR = *std::max_element( R.begin(), R.end() );
	SCAI_ASSERT_EQ_ERROR( maxR, k-1, "max block id is not k-1 as it should");

	//The implied renumbering is that block R[i] should be renumbered to i. To apply it easier we better reverse it.
	//Create a vector RR where RR[R[i]]=i, this way, afterwards, we can do part[of point i]= RR[prevPart]
	std::vector<IndexType> reverseR( k );
	for( IndexType i=0; i<k; i++){
		reverseR[R[i]]=i;		
	}

	//go over the local values and renumber the blocks
	const IndexType localN = partition.getDistributionPtr()->getLocalSize();
	scai::hmemo::WriteAccess<IndexType> wPart( partition.getLocalValues() );

	for( IndexType i=0; i<localN; i++){
		IndexType prevBlock = wPart[i];
		wPart[i] = reverseR[prevBlock];
	}

	return reverseR;
}//applySfcRenumber

//to force instantiation
template class Mapping<IndexType, ValueType>;
}