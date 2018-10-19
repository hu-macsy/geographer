#include "Metrics.h"

	void Metrics::getAllMetrics(const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const scai::lama::DenseVector<ValueType> nodeWeights, struct Settings settings ){
		
		getEasyMetrics( graph, partition, nodeWeights, settings );
		
		int numIter = 100;
		getRedistRequiredMetrics( graph, partition, settings, numIter );
		
	}

	void Metrics::getMappingMetrics(
		const scai::lama::CSRSparseMatrix<ValueType> blockGraph, 
		const scai::lama::CSRSparseMatrix<ValueType> PEGraph, 
		const std::vector<IndexType> mapping){

		const IndexType N = blockGraph.getNumRows();
		const IndexType M = blockGraph.getNumValues();

		IndexType sumDilation = 0;
		IndexType maxDilation = 0;
		IndexType maxCongestion = 0;
		std::vector<IndexType> congestion( M, 0 );

		//calculate all shortest paths in PE graph
		std::vector<std::vector<ValueType>> APSP( N, std::vector<ValueType> (N, 0.0));
		//store the predecessors for all shortest paths
		std::vector<std::vector<IndexType>> predecessor(N, std::vector<IndexType> (N, 0)); 
    
		for(IndexType i=0; i<N; i++){
			APSP[i] = ITI::GraphUtils<IndexType, ValueType>::localDijkstra( PEGraph, i, predecessor[i]);
		}

		//access to the graphs
		const scai::lama::CSRStorage<ValueType> blockStorage = blockGraph.getLocalStorage();
		const scai::hmemo::ReadAccess<IndexType> ia(blockStorage.getIA());
		const scai::hmemo::ReadAccess<IndexType> ja(blockStorage.getJA());
		const scai::hmemo::ReadAccess<ValueType> blockValues(blockStorage.getValues());

		const scai::lama::CSRStorage<ValueType> PEStorage = PEGraph.getLocalStorage();
		// edges of the PE graph
		const scai::hmemo::ReadAccess<IndexType> PEia ( PEStorage.getIA() );
		const scai::hmemo::ReadAccess<IndexType> PEja ( PEStorage.getJA() );
		const scai::hmemo::ReadAccess<ValueType> PEValues ( PEStorage.getValues() );

		// calculate dilation and congestion for every edge
		for( IndexType v=0; v<N; v++){
			for(IndexType iaInd=ia[v]; iaInd<ia[v+1]; iaInd++){
				IndexType neighbor = ja[iaInd];
				ValueType thisEdgeWeight = blockValues[iaInd];
				//only one edge direction considered
				if(mapping[v] <= mapping[neighbor]){
					// this edge is (v,blockNeighbor)
					IndexType start = mapping[v];
					IndexType target = mapping[neighbor];
					IndexType currDilation = APSP[start][target]*thisEdgeWeight;
					sumDilation += currDilation;
					if( currDilation>maxDilation ){
						maxDilation = currDilation;
					}

					//update congestion
					IndexType current = target;
					IndexType next = target;
					while( current!= start ){
						current = predecessor[start][current];
						if( next>=current ){
							//for all out edges in PE graph of current node
							for(IndexType PEiaInd = PEia[current]; PEiaInd< PEia[current+1]; PEiaInd++){
								if(PEja[PEiaInd]==next){
									congestion[PEiaInd] += thisEdgeWeight;
								}
							}
						}else{
							for(IndexType PEiaInd = PEia[next]; PEiaInd< PEia[next+1]; PEiaInd++){
								if( PEja[PEiaInd]==current ){
									congestion[PEiaInd] += thisEdgeWeight;
								}
							}
						}
						next = current;
					}
				}//if
			}//for
		}//for

		for( IndexType iaInd=0; iaInd<M; iaInd++ ){
			congestion[iaInd] /= PEValues[iaInd];
			if( congestion[iaInd]>maxCongestion ){
				maxCongestion = congestion[iaInd];
			}
		}

		ValueType avgDilation = ((ValueType) sumDilation)/((ValueType) M/2);

		std::cout<< "Maximum congestion: " << maxCongestion << std::endl;
		std::cout<< "Maximum dilation: " << maxDilation << std::endl;
		std::cout << "Average dilation: " << avgDilation << std::endl;
	}//getMappingMetrics
