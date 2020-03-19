#include <scai/solver/criteria/IterationCount.hpp>
#include <scai/solver/CG.hpp>

#include <scai/solver/logger/CommonLogger.hpp>
#include <scai/solver/criteria/ResidualThreshold.hpp>
#include <scai/lama/norm/L2Norm.hpp>

#include "Metrics.h"
#include "FileIO.h"

using namespace ITI;

template<typename ValueType>
void Metrics<ValueType>::getMetrics(const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const std::vector<scai::lama::DenseVector<ValueType>> nodeWeights, struct Settings settings){

    if( settings.metricsDetail=="all" ) {
        getAllMetrics( graph, partition, nodeWeights, settings );
    }
    if( settings.metricsDetail=="easy" ) {
        getEasyMetrics( graph, partition, nodeWeights, settings );
    }
    if( settings.metricsDetail=="mapping" ) {
        const scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();

        scai::lama::CSRSparseMatrix<ValueType> PEgraph;
        std::vector<scai::lama::DenseVector<ValueType>>  PEnodeWeights;

        if( settings.PEGraphFile!="-" ){ // a processor graph is provides by the user
            PEgraph = ITI::FileIO<IndexType, ValueType>::readGraph( settings.PEGraphFile, PEnodeWeights, comm, settings.fileFormat );
        }else{// if not, create the processor graph
            if( settings.numBlocks!=comm->getSize()){
                if( comm->getRank()==0 ){
                    std::cout<< "ERROR: no processor (aka communication) graph was provided for the mapping metrics." << std::endl;
                    std::cout<< "\tAlso, the currently used processor network does not have the same number of PEs as the provided partition."<< std::endl;
                    std::cout<< "\tWill not calculate mapping metrics."<< std::endl;
                }
                return;
            }
            PEgraph = ITI::GraphUtils<IndexType, ValueType>::getPEGraph( graph );
        }
        //PE graph needs to be replicated
        PEgraph.replicate();
        getMappingMetrics( graph, partition, PEgraph );        
    }
}

//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::getAllMetrics(const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const std::vector<scai::lama::DenseVector<ValueType>> nodeWeights, struct Settings settings ) {

    Settings tmpSettings = settings;
    settings.computeDiameter=false; //diameter will be computed inside getRedistRequiredMetrics
    getEasyMetrics( graph, partition, nodeWeights, tmpSettings );

    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    if (settings.numBlocks == comm->getSize()) {
        int numIter = 100;
        getRedistRequiredMetrics( graph, partition, settings, numIter );
    }
}
//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::print( std::ostream& out) const {

    out.precision(5);
    out<<"\nMetrics:" << std::endl;
    for( auto mapIt= MM.begin(); mapIt!=MM.end(); mapIt++ ) {
        if( mapIt->second!=-1)
            out<< mapIt->first <<": " << mapIt->second << std::endl;
    }

}
//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::printHorizontal( std::ostream& out) const {

    out.precision(5);
    for( auto mapIt= MM.begin(); mapIt!=MM.end(); mapIt++ ) {
        if( mapIt->second!=-1)
            out<< mapIt->first << ", ";
    }
    out << std::endl;
    for( auto mapIt= MM.begin(); mapIt!=MM.end(); mapIt++ ) {
        if( mapIt->second!=-1)
            out<< mapIt->second << ", ";
    }
    out << std::endl;
}
//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::printHorizontal2( std::ostream& out) const {

    out.precision(5);
    for( auto mapIt= MM.begin(); mapIt!=MM.end(); mapIt++ ) {
        if( mapIt->second!=-1)
            out<< mapIt->first <<" " << mapIt->second << " , ";
    }
    out << std::endl;
}
//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::printKMeansProfiling( std::ostream& out ) const {
    out << "KMeans::assignBlocks was called " << numBalanceIter.size() << " times" << std::endl;
    out << "Average number of balance iterations: " << std::accumulate( numBalanceIter.begin(), numBalanceIter.end(), 0.0 )/numBalanceIter.size() << std::endl;
}
//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::getRedistMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const std::vector<scai::lama::DenseVector<ValueType>> nodeWeights, struct Settings settings ) {

    getAllMetrics( graph, partition, nodeWeights, settings);

    scai::dmemo::DistributionPtr newDist = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues() );
    scai::dmemo::DistributionPtr oldDist = graph.getRowDistributionPtr();

    std::tie( MM["maxRedistVol"], MM["totRedistVol"] ) = getRedistributionVol( newDist, oldDist);

}
//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::getEasyMetrics( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, const std::vector<scai::lama::DenseVector<ValueType>> nodeWeights, struct Settings settings ) {

    MM["finalCut"] = ITI::GraphUtils<IndexType, ValueType>::computeCut(graph, partition, true);

    for( unsigned int w=0; w<nodeWeights.size(); w++ ) {
        imbalances.push_back(  ITI::GraphUtils<IndexType, ValueType>::computeImbalance( partition, settings.numBlocks, nodeWeights[w]) );
        MM["finalImbalance_w"+std::to_string(w)] = imbalances.back();
    }
    MM["finalImbalance"] = *std::max_element( imbalances.begin(), imbalances.end() );

    //TODO: getting the block graph probably fails for p>5000, removed this metric since we do not use it so much
    //std::tie(maxBlockGraphDegree, totalBlockGraphEdges) = ITI::GraphUtils::computeBlockGraphComm<IndexType, ValueType>( graph, partition, settings.numBlocks );

    // communication volume

    // 3 vector each of size numBlocks
    std::vector<IndexType> commVolume;
    std::vector<IndexType> numBorderNodesPerBlock;
    std::vector<IndexType> numInnerNodesPerBlock;

    // TODO: can re returned in an auto, check if it is faster
    // it is a bit uglier but saves time
    std::tie( commVolume, numBorderNodesPerBlock, numInnerNodesPerBlock ) = \
            ITI::GraphUtils<IndexType, ValueType>::computeCommBndInner( graph, partition, settings );

    MM["maxCommVolume"] = *std::max_element( commVolume.begin(), commVolume.end() );
    MM["totalCommVolume"] = std::accumulate( commVolume.begin(), commVolume.end(), 0 );

    //TODO: are num of boundary nodes needed ????
    MM["maxBoundaryNodes"] = *std::max_element( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end() );
    MM["totalBoundaryNodes"] = std::accumulate( numBorderNodesPerBlock.begin(), numBorderNodesPerBlock.end(), 0 );

    std::vector<ValueType> percentBorderNodesPerBlock( settings.numBlocks, 0);
    SCAI_ASSERT_EQ_ERROR( settings.numBlocks, numBorderNodesPerBlock.size(), "Vector size mismatch.");
    SCAI_ASSERT_EQ_ERROR( settings.numBlocks, numInnerNodesPerBlock.size(), "Vector size mismatch.");

    for(IndexType i=0; i<settings.numBlocks; i++) {
        percentBorderNodesPerBlock[i] = (ValueType (numBorderNodesPerBlock[i]))/(numBorderNodesPerBlock[i]+numInnerNodesPerBlock[i]);
    }

    MM["maxBorderNodesPercent"] = *std::max_element( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end() );
    MM["avgBorderNodesPercent"] = std::accumulate( percentBorderNodesPerBlock.begin(), percentBorderNodesPerBlock.end(), 0.0 )/(ValueType(settings.numBlocks));

    //get diameter if possible
    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    if (settings.numBlocks == comm->getSize() && settings.computeDiameter) {
        std::tie( MM["maxBlockDiameter"], MM["harmMeanDiam"], MM["numDisconBlocks"] ) = getDiameter(graph, partition, settings);
    } else {
        PRINT0("\tWARNING: Not computing diameter, not all vertices are in same block everywhere");
    }

}
//---------------------------------------------------------------------------

template<typename ValueType>
std::tuple<IndexType,IndexType,IndexType> Metrics<ValueType>::getDiameter( const scai::lama::CSRSparseMatrix<ValueType> graph, const scai::lama::DenseVector<IndexType> partition, struct Settings settings ) {

    std::chrono::time_point<std::chrono::high_resolution_clock> diameterStart = std::chrono::high_resolution_clock::now();
    IndexType maxBlockDiameter = 0;
    //IndexType avgBlockDiameter = 0;
    IndexType numDisconBlocks = 0;
    ValueType harmMeanDiam = 0;

    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    const IndexType numPEs = comm->getSize();

    if (settings.numBlocks == numPEs && settings.computeDiameter) {
        //maybe possible to compute diameter
        bool allLocalNodesInSameBlock;
        {
            scai::hmemo::ReadAccess<IndexType> rPart(partition.getLocalValues());
            auto result = std::minmax_element(rPart.get(), rPart.get()+localN);
            allLocalNodesInSameBlock = ((*result.first) == (*result.second));
        }
        if (comm->all(allLocalNodesInSameBlock)) {
            IndexType maxRounds = settings.maxDiameterRounds;
            if (maxRounds < 0) {
                maxRounds = localN;
            }
            IndexType localDiameter = ITI::GraphUtils<IndexType, ValueType>::getLocalBlockDiameter(graph, localN/2, 0, 0, maxRounds);

            ValueType sumInverseDiam = comm->sum( 1.0/localDiameter );
            harmMeanDiam = comm->getSize()/sumInverseDiam;

            // count the number of disconnected blocks
            IndexType isDisconnected = 0;

            if( localDiameter== std::numeric_limits<IndexType>::max()) {
                isDisconnected=1;
                //set to 0 so it does not affect the max and avg diameter
                localDiameter = 0;
            }

            numDisconBlocks = comm->sum(isDisconnected);
            if (settings.verbose) {
                PRINT0("number of disconnected blocks: " << numDisconBlocks);
            }

            //PRINT(*comm << ": "<< localDiameter);
            maxBlockDiameter = comm->max(localDiameter);

        } else {
            if(comm->getRank()==0){
                std::cout<< "WARNING: Not computing diameter, not all vertices are in same block everywhere" << std::endl;
            }
        }
    }
    std::chrono::duration<ValueType,std::ratio<1>> diameterTime = std::chrono::high_resolution_clock::now() - diameterStart;
    ValueType time = comm->max( diameterTime.count() );
    if (settings.verbose) {
        PRINT0("time to get the diameter: " <<  time );
    }

    return std::make_tuple( maxBlockDiameter, harmMeanDiam, numDisconBlocks);
}
//---------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::getRedistRequiredMetrics( const scai::lama::CSRSparseMatrix<ValueType>& graph, const scai::lama::DenseVector<IndexType>& partition, struct Settings settings, const IndexType repeatTimes ) {

    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const IndexType N = graph.getNumRows();

    //get the distribution from the partition
    scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues() );

    std::chrono::time_point<std::chrono::steady_clock> beforeRedistribution = std::chrono::steady_clock::now();

    // redistribute graph according to partition distribution
    // distribute only rows for the diameter calculation

    //TODO: change NoDist with graph.getColumnDistribution() ?
    scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( graph.getNumRows() ));
    scai::lama::CSRSparseMatrix<ValueType> copyGraph( graph ); 
    copyGraph.redistribute(distFromPartition, noDistPtr);

    std::chrono::duration<ValueType> redistributionTime =  std::chrono::steady_clock::now() - beforeRedistribution;

    ValueType time = 0;
    time = comm->max( redistributionTime.count() );
    PRINT0("time to redistribute: " << time);

    const IndexType localN = distFromPartition->getLocalSize();
    SCAI_ASSERT_EQ_ERROR( localN, copyGraph.getLocalNumRows(), "Distribution mismatch")

    const IndexType maxLocalN = comm->max(localN);
    const IndexType minLocalN = comm->min(localN);
    const ValueType optSize = ValueType(N)/comm->getSize();

    ValueType imbalance = ValueType( maxLocalN - optSize)/optSize;
    PRINT0("minLocalN= "<< minLocalN <<", maxLocalN= " << maxLocalN << ", imbalance= " << imbalance);

    // diameter
    if(  settings.computeDiameter and (MM["maxBlockDiameter"]==0 or MM["harmMeanDiam"]==0)) {
        scai::lama::DenseVector<IndexType> copyPartition( partition );
        copyPartition.redistribute( distFromPartition );
        std::tie( MM["maxBlockDiameter"], MM["harmMeanDiam"], MM["numDisconBlocks"] ) = getDiameter(copyGraph, copyPartition, settings);
    }

    //get the NNZ imbalance
    {
        const IndexType localNNZ = copyGraph.getLocalNumValues();   
        const IndexType sumNNZ = comm->sum(localNNZ);
        //this fails because the cut edges are not counted; it is sumNNZ+ 2*cutedges = copyGraph.getNumValues()
        //SCAI_ASSERT_EQ_ERROR( sumNNZ, copyGraph.getNumValues(), "??" ); 
        
        const IndexType optNNZ = sumNNZ/comm->getSize();
        const IndexType maxLocalNNZ = comm->max(localNNZ);
        const IndexType minLocalNNZ = comm->min(localNNZ);

        ValueType edgeImbalance = ValueType( maxLocalNNZ - optNNZ)/optNNZ;
        PRINT0("minLocalNNZ= "<< minLocalNNZ <<", maxLocalNNZ= " << maxLocalNNZ << ", edge imbalance= " << edgeImbalance);
        MM["edgeImbalance"] = edgeImbalance;
    }

    //
    // redistribute for SpMV, linear solver and commTime
    //

    copyGraph.redistribute( distFromPartition, distFromPartition );

    MM["SpMVtime"] = getSPMVtime(copyGraph, repeatTimes);

    //TODO: take a percentage of repeatTimes; maybe all repeatTimes are too much for CG
    std::tie( MM["CGtime"], MM["CGiterations"], MM["CGresidual"] ) = getCGTime( copyGraph, 5, settings.maxCGIterations, settings.CGResidual); 

    //TODO: maybe extract this time from the actual SpMV above
    // comm time in SpMV
    {
        PRINT0("starting comm schedule...");
        const scai::dmemo::HaloExchangePlan& matrixHaloPlan = copyGraph.getHaloExchangePlan();
        const scai::dmemo::CommunicationPlan& sendPlan  = matrixHaloPlan.getLocalCommunicationPlan();
        const scai::dmemo::CommunicationPlan& recvPlan  = matrixHaloPlan.getHaloCommunicationPlan();

        //PRINT(*comm << ": " << 	sendPlan.size() << " ++ " << sendPlan << " ___ " << recvPlan);
        scai::hmemo::HArray<ValueType> sendData( sendPlan.totalQuantity(), 1.0 );
        scai::hmemo::HArray<ValueType> recvData;

        comm->synchronize();
        std::chrono::time_point<std::chrono::steady_clock> beforeCommTime = std::chrono::steady_clock::now();
        for ( IndexType i = 0; i < repeatTimes; ++i ) {
            comm->exchangeByPlan( recvData, recvPlan, sendData, sendPlan );
        }
        //comm->synchronize();
        std::chrono::duration<ValueType> commTime = std::chrono::steady_clock::now() - beforeCommTime;

        //PRINT(*comm << ": "<< sendPlan );
        time = comm->max(commTime.count());
        MM["commTime"] = time/repeatTimes;

        ValueType minTime = comm->min( commTime.count() );
        PRINT0("max time for " << repeatTimes <<" communications: " << time << " , min time " << minTime);
    }

}


/* Calculate the volume, aka the data that will be exchanged when redistributing from oldDist to newDist.
 */
template<typename ValueType>
std::pair<IndexType,IndexType> Metrics<ValueType>::getRedistributionVol( const scai::dmemo::DistributionPtr newDist, const scai::dmemo::DistributionPtr oldDist) {

    //const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::CommunicatorPtr comm = newDist->getCommunicatorPtr();

    //get the distribution from the partition
    scai::dmemo::RedistributePlan prepareRedist = scai::dmemo::redistributePlanByNewDistribution( newDist, oldDist );

    // redistribution load
    scai::hmemo::HArray<IndexType> sourceIndices = prepareRedist.getExchangeSourceIndexes();
    scai::hmemo::HArray<IndexType> targetIndices = prepareRedist.getExchangeTargetIndexes();
    IndexType thisSourceSize = prepareRedist.getExchangeSourceSize();
    IndexType thisTargetSize = prepareRedist.getExchangeTargetSize();

    IndexType globTargetSize = comm->sum( thisTargetSize);
    IndexType globSourceSize = comm->sum( thisSourceSize);
    SCAI_ASSERT_EQ_ERROR( globSourceSize, globTargetSize, "Mismatch in total migration volume.");
    //PRINT0("total migration volume= " << globSourceSize);

    IndexType maxTargetSize = comm->max( thisTargetSize);
    IndexType maxSourceSize = comm->max( thisSourceSize);
    //PRINT0("maxTargetSize= " << maxTargetSize);
    //PRINT0("maxSourceSize= " << maxSourceSize);

    return std::make_pair( std::max(maxTargetSize,maxSourceSize), globSourceSize);
}
//---------------------------------------------------------------------------------------

template<typename ValueType>
void Metrics<ValueType>::getMappingMetrics(
    const scai::lama::CSRSparseMatrix<ValueType> blockGraph,
    const scai::lama::CSRSparseMatrix<ValueType> PEGraph,
    const std::vector<IndexType> mapping) {

    const IndexType N = blockGraph.getNumRows();
    //congestion is defined for every edge of the processor graph
    const IndexType peM = PEGraph.getNumValues();

    SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), blockGraph.getNumRows(), "Block and PE graph must have the same number of nodes" );
    SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), mapping.size(), "Block and PE graphs must have the same size as mapping" );
    SCAI_ASSERT_EQ_ERROR( *std::max_element(mapping.begin(), mapping.end()), N-1, "Wrong mapping" );
    SCAI_ASSERT_EQ_ERROR( std::accumulate(mapping.begin(), mapping.end(), 0), (N*(N-1)/2), "Wrong mapping" );

    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(N));
    SCAI_ASSERT( PEGraph.getRowDistributionPtr()->isEqual(*noDist), "Function expects the graph to be replicated" );
    SCAI_ASSERT( blockGraph.getRowDistributionPtr()->isEqual(*noDist), "Function expects the graph to be replicated" );

    ValueType sumDilation = 0;
    ValueType maxDilation = 0;
    ValueType minDilation = std::numeric_limits<ValueType>::max();
    std::vector<ValueType> congestion( peM, 0 );

    //calculate all shortest paths in PE graph
    std::vector<std::vector<ValueType>> APSP( N, std::vector<ValueType> (N, 0.0));
    //store the predecessors for all shortest paths
    std::vector<std::vector<IndexType>> predecessor(N, std::vector<IndexType> (N, 0));

    for(IndexType i=0; i<N; i++) {
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

    SCAI_ASSERT_EQ_ERROR( ia.size(), N+1, "ia size mismatch" );
    SCAI_ASSERT_EQ_ERROR( ja.size(), ia[N], "ja size mismatch" );
    SCAI_ASSERT_EQ_ERROR( PEia.size(), N+1, "ia size mismatch" );
    SCAI_ASSERT_LE_ERROR( PEia[N], peM, "Too large index in PE graph" );
    SCAI_ASSERT_LE_ERROR( scai::utilskernel::HArrayUtils::max(PEStorage.getIA()), peM, "some ia value is too large");

    // calculate dilation and congestion for every edge
    for( IndexType v=0; v<N; v++) {
        for(IndexType iaInd=ia[v]; iaInd<ia[v+1]; iaInd++) {
            IndexType neighbor = ja[iaInd];
            ValueType thisEdgeWeight = blockValues[iaInd];
            //only one edge direction considered
            if(mapping[v] <= mapping[neighbor]) {
                // this edge is (v,neighbor)
                IndexType start = mapping[v];
                IndexType target = mapping[neighbor];
                ValueType currDilation = APSP[start][target]*thisEdgeWeight;
                sumDilation += currDilation;
                if( currDilation>maxDilation ) {
                    maxDilation = currDilation;
                }
                if( currDilation<minDilation ) {
                    minDilation = currDilation;
                }

                //update congestion
                IndexType current = target;
                IndexType next = target;
                while( current!= start ) {
                    current = predecessor[start][current];
                    if( next>=current ) {
                        //for all out edges in PE graph of current node
                        for(IndexType PEiaInd = PEia[current]; PEiaInd< PEia[current+1]; PEiaInd++) {
                            if(PEja[PEiaInd]==next) {
                                congestion[PEiaInd] += thisEdgeWeight;
                            }
                        }
                    } else {
                        for(IndexType PEiaInd = PEia[next]; PEiaInd< PEia[next+1]; PEiaInd++) {
                            if( PEja[PEiaInd]==current ) {
                                congestion[PEiaInd] += thisEdgeWeight;
                            }
                        }
                    }
                    next = current;
                }
            }//if
        }//for
    }//for

    ValueType maxCongestion = 0;
    ValueType minCongestion = std::numeric_limits<ValueType>::max();
    for( IndexType edge=0; edge<peM; edge++ ) {
        congestion[edge] /= PEValues[edge];
        if( congestion[edge]>maxCongestion ) {
            maxCongestion = congestion[edge];
        }
        if( congestion[edge]<minCongestion ) {
            minCongestion = congestion[edge];
        }
    }

    ValueType avgDilation = ((ValueType) sumDilation)/((ValueType) peM/2);
    //ValueType avgCongestion = std::accumulate( congestion.begin(), congestion.end(), 0.0)/peM;

    MM["maxCongestion"] = maxCongestion;
    MM["maxDilation"] = maxDilation;
    MM["avgDilation"] = avgDilation;

}//getMappingMetrics
//---------------------------------------------------------------------------------------
template<typename ValueType>
void Metrics<ValueType>::getMappingMetrics(
    const scai::lama::CSRSparseMatrix<ValueType> appGraph,
    const scai::lama::DenseVector<IndexType> partition,
    const scai::lama::CSRSparseMatrix<ValueType> PEGraph ) {

    const IndexType k = partition.max()+1;
    SCAI_ASSERT_EQ_ERROR( k, PEGraph.getNumRows(), "Max value in partition (aka, k) should be equal with the number of vertices of the PE graph." );

    scai::lama::CSRSparseMatrix<ValueType> blockGraph = ITI::GraphUtils<IndexType,ValueType>::getBlockGraph(
                appGraph, partition, k );

    SCAI_ASSERT_EQ_ERROR( PEGraph.getNumRows(), blockGraph.getNumRows(), "Block and PE graph must have the same number of nodes" );

    std::vector<IndexType> identityMapping( k, 0 );
    std::iota( identityMapping.begin(), identityMapping.end(), 0);

    getMappingMetrics( blockGraph, PEGraph, identityMapping);
}//getMappingMetrics
//---------------------------------------------------------------------------------------

template<typename ValueType>
ValueType Metrics<ValueType>::getSPMVtime(
    scai::lama::CSRSparseMatrix<ValueType> graph,
    const IndexType repeatTimes){

    scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    PRINT0("starting SpMV...");

    // vector for multiplication
    scai::lama::DenseVector<ValueType> x ( graph.getColDistributionPtr(), 1.0 );
    scai::lama::DenseVector<ValueType> y ( graph.getRowDistributionPtr(), 0.0 );
    graph.setCommunicationKind( scai::lama::SyncKind::ASYNC_COMM );
    comm->synchronize();
    
    // perform the actual multiplication
    std::chrono::time_point<std::chrono::steady_clock> beforeSpMVTime = std::chrono::steady_clock::now();
    for(IndexType r=0; r<repeatTimes; r++) {
        y = graph *x +y;
    }
    comm->synchronize();
    std::chrono::duration<ValueType> SpMVTime = std::chrono::steady_clock::now() - beforeSpMVTime;
    //PRINT(" SpMV time for PE "<< comm->getRank() << " = " << SpMVTime.count() );

    ValueType time = comm->max(SpMVTime.count());
    PRINT0("max time for " << repeatTimes <<" SpMVs: " << time );

    return time/repeatTimes;
}
//---------------------------------------------------------------------------------------

//for the initial values of solution and rhs see papers
//Parallel Conjugate Gradient: Effects of Ordering Strategies, Programming Paradigms and Architectural Platforms.
//Oliker, Xi, Heber et al., section 4
//High-performance conjugate-gradient benchmark: A new metric for ranking high-performance computing systems
// Dongarra1, Michael A Heroux2and Piotr Luszczek, section 4
template<typename ValueType>
std::tuple<ValueType,ValueType,ValueType> Metrics<ValueType>::getCGTime( 
    const scai::lama::CSRSparseMatrix<ValueType>& graph,
    const IndexType repeatTimes,
    const IndexType maxIterations,
    const ValueType residual){

    const scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const scai::dmemo::DistributionPtr rowDist = graph.getRowDistributionPtr();
    const scai::dmemo::DistributionPtr colDist = graph.getColDistributionPtr();

    //the laplacian
    const scai::lama::CSRSparseMatrix<ValueType> laplacian = GraphUtils<IndexType,ValueType>::constructLaplacian(graph);

    //this assertion fails because lama does not set up the local data of the matrix correctly
    //SCAI_ASSERT_EQ_ERROR( laplacian.l1Norm(), 2*graph.l1Norm(), "wrong l1Norm in laplacian");
    SCAI_ASSERT_EQ_ERROR( laplacian.getNumValues(), graph.getNumValues()+graph.getNumRows(), "wrong numValues in laplacian");
    SCAI_ASSERT_EQ_ERROR( laplacian.getLocalNumValues(), graph.getLocalNumValues()+graph.getLocalNumRows(), "laplacian is wrong");

    // Allocate a common logger that prints convergenceHistory
    //bool isDisabled = comm->getRank() > 0;
    //scai::solver::LoggerPtr logger( new scai::solver::CommonLogger( "CGLogger: ", scai::solver::LogLevel::convergenceHistory, scai::solver::LoggerWriteBehaviour::toConsoleOnly, isDisabled ) );

    scai::solver::CG<ValueType> solver("CGSolver");

    scai::lama::NormPtr<ValueType> norm( new scai::lama::L2Norm<ValueType>( ) );

    scai::solver::CriterionPtr<ValueType> criterion1( new scai::solver::ResidualThreshold<ValueType>( norm, residual, scai::solver::ResidualCheck::Absolute ) );
    scai::solver::CriterionPtr<ValueType> criterion2( new scai::solver::IterationCount<ValueType>( maxIterations ) );
    scai::solver::CriterionPtr<ValueType> criterion( new scai::solver::Criterion<ValueType>( criterion1, criterion2, scai::solver::BooleanOp::OR ) );

    solver.setStoppingCriterion( criterion );
    ValueType totalTime = 0.0;
    IndexType totalIterations = 0;
    ValueType retResidual = 0.0;

    for(IndexType r=0; r<repeatTimes; r++) {
        scai::lama::DenseVector<ValueType> rhs( colDist, ValueType(1.0) );

        scai::lama::DenseVector<ValueType> solution( colDist, ValueType(0.0) );
        solver.initialize( laplacian );

        std::chrono::time_point<std::chrono::steady_clock> beforeTime = std::chrono::steady_clock::now();
        solver.solve(solution, rhs);

        std::chrono::duration<ValueType> elapTime = std::chrono::steady_clock::now() - beforeTime;
        totalTime += elapTime.count();
        //number of iterations is (should be!) always the same; maybe just get them outside the loop?
        totalIterations += solver.getIterationCount();
        //PRINT(" SpMV time for PE "<< comm->getRank() << " = " << SpMVTime.count() );
        retResidual= solver.getResidual().l2Norm();
    }
    ValueType globTime = comm->max(totalTime)/repeatTimes;
    ValueType avgIterations = ((ValueType) totalIterations)/repeatTimes;
    
    PRINT0("total time for "<< repeatTimes << " calls to CG solver: " << totalTime << 
            " and " << avgIterations << " average iterations and residual reached " << retResidual );

    return std::make_tuple( globTime, avgIterations, retResidual);
}

template class Metrics<double>;
template class Metrics<float>;

