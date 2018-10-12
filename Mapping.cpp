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
	scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
	scai::lama::CSRSparseMatrix<ValueType>& PEGraph){

	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType N = blockGraph.getNumRows(); //number of nodes in bot graphs

	SCAI_ASSERT_EQ_ERROR( N, PEGraph.getNumRows(), "The block and the processor graph must have the same number of nodes");
	SCAI_ASSERT_EQ_ERROR( N, blockGraph.getNumColumns(), "Block graph matrix must be square" );
	SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), PEGraph.getNumColumns(), "Processor graph matrix must be square" );

	/*TODO: add the part where we initialize the edges of the PEGraph
	to force shortest paths
	*/

  	std::vector<bool> mapped( N, false);
  	std::vector<bool> discovered( N, false);
  	std::vector<bool> usedPEs( N, false); // for PEs that are already mapped

  	/********************************************************
   	* The actual mapping code                            
   	********************************************************/

  	//in the original code, this is given as input; here picj at random
  	//TODO: possible opt: as for blockGraph, pick from PEGraph the node with
  	// the maximum weighted degree?
  	srand(time(NULL));
   	IndexType peNode = rand()%N; // the current vertex in PEGraph to be mapped to
  	SCAI_ASSERT_GE_ERROR(peNode, 0, "Wrong node ID");
  	usedPEs[peNode] = true;
  	IndexType numMappedNodes = 0; // number of already mapped nodes in blockGraph

	std::vector<ValueType> comFrom(N, 0);

	// access the data of the block graph
	const scai::lama::CSRStorage<ValueType> localStorage = blockGraph.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
	const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
	const scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

	// the mapping to be returned
	std::vector<IndexType> mapping(N,0);

  	//while there are unmapped nodes
  	while(numMappedNodes < N) {

  		//TODO: instead of storing all weights and ten call max_element,
  		// just find the max directly
  		//compute weighted node degrees in block graph and store them in comFrom
	    for(IndexType i=0; i<N; i++){
	    	if(mapped[i])	// if this is already mapped 
	    		continue;
	    	for(IndexType valuesInd=ia[i]; valuesInd<ia[i+1]; valuesInd++){
	    	//for(IndexType j=ia[i]; j<ia[i+1]; j++){ 
	    		comFrom[i] += values[valuesInd];
	    	}
	    }

		typename std::vector<ValueType>::iterator blockNodeWeight = std::max_element(comFrom.begin(), comFrom.end() );
		IndexType blockNode = std::distance(comFrom.begin(), blockNodeWeight);

		SCAI_ASSERT_GE_ERROR( blockNode, 0, "Wrong node ID");
		SCAI_ASSERT_LT_ERROR( blockNode, N, "Wrong node ID");
		PRINT0("heaviest vertex in block graph is " << blockNode << " with weight " << *blockNodeWeight);

		//TODO: check, possible opt: when/where is peNode changing?? is it always picked
		// at random
		while( !usedPEs[peNode] ){ 
			peNode = rand()%N;
		}

		// map blockNode to peNode
		mapping[blockNode] = peNode;
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
     		ValueType edgeWeight = values[iaInd];
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

     		std::vector<ValueType> distances = GraphUtils::localDijkstra( blockGraph, blockNode);

     	}//while(!Q.empty()) 

  	}//while(numMappedNodes < N)

  	return mapping;
}


//to force instantiation
template class Mapping<IndexType, ValueType>;
}