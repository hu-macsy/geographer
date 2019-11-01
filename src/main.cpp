#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>
#include <chrono>
#include <iomanip>

#include <cxxopts.hpp>

#include "Diffusion.h"
#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "Settings.h"
#include "Metrics.h"
#include "GraphUtils.h"
#include "parseArgs.h"
#include "mainHeader.h"

/**
 *  Examples of use:
 *
 *  for reading from file "fileName"
 * ./a.out --graphFile fileName --epsilon 0.05 --sfcRecursionSteps=10 --dimensions=2 --borderDepth=10  --stopAfterNoGainRounds=3 --minGainForNextGlobalRound=10
 *
 * for generating a 10x20x30 mesh
 * ./a.out --generate --numX=10 --numY=20 --numZ=30 --epsilon 0.05 --sfcRecursionSteps=10 --dimensions=3 --borderDepth=10  --stopAfterNoGainRounds=3 --minGainForNextGlobalRound=10
 *
 * ./a.out --graphFile fileName --epsilon 0.05 --initialPartition=4 --dimensions=2 --bisect=0 --numPoints=4000000 --distribution=uniform --cutsPerDim=10 13
 *
 */

//----------------------------------------------------------------------------

//void memusage(size_t *, size_t *,size_t *,size_t *,size_t *);

int main(int argc, char** argv) {

    using namespace ITI;
    typedef double ValueType;   //use double

    std::string blockSizesFile;
    //ITI::Format coordFormat;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if (comm->getType() != scai::dmemo::CommunicatorType::MPI) {
        std::cout << "The linked lama version was compiled without MPI. Only sequential partitioning is supported." << std::endl;
    }

    std::string callingCommand = "";
    for (IndexType i = 0; i < argc; i++) {
        callingCommand += std::string(argv[i]) + " ";
    }

    cxxopts::Options options = ITI::populateOptions();
    cxxopts::ParseResult vm = options.parse(argc, argv);

    if (vm.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    struct Settings settings = ITI::interpretSettings(vm);
    if( !settings.isValid )
        return -1;

    //--------------------------------------------------------
    //
    // initialize
    //

    if( comm->getRank() ==0 ) {
        std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
        std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
        std::cout << "date and time: " << std::ctime(&timeNow);
    }

    IndexType N = -1; 		// total number of points
    std::string machine = settings.machine; //machine name
    
    // timing information
    //

    std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();

    if (comm->getRank() == 0) {

        std::cout<< "commit:"<< version<< " main file: "<< __FILE__ << " machine:" << machine << " p:"<< comm->getSize();

        auto oldprecision = std::cout.precision(std::numeric_limits<double>::max_digits10);
        std::cout <<" seed:" << vm["seed"].as<double>() << std::endl;
        std::cout.precision(oldprecision);

        std::cout << "Calling command:" << std::endl;
        std::cout << callingCommand << std::endl << std::endl;

    }

    //---------------------------------------------------------
    //
    // generate or read graph and coordinates
    //

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    std::vector<scai::lama::DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights;		//the weights for each node

    N = readInput<ValueType>( vm, settings, comm, graph, coordinates, nodeWeights );

    if( settings.setAutoSettings ){
        settings = settings.setDefault( graph );
        if( !settings.isValid )
               return -1;
    }

    //---------------------------------------------------------------
    //
    // read the communication graph or the block sizes if provided
    //

    if( vm.count("PEgraphFile") and vm.count("blockSizesFile") ) {
        throw std::runtime_error("You should provide either a file for a communication graph OR a file for block sizes. Not both.");
    }

    ITI::CommTree<IndexType,ValueType> commTree;

    if(vm.count("PEgraphFile")) {
        throw std::logic_error("Reading of communication trees not yet implemented here.");
        //commTree =  FileIO<IndexType, ValueType>::readPETree( settings.PEGraphFile );
    } else if( vm.count("blockSizesFile") ) {
        //blockSizes.size()=number of weights, blockSizes[i].size()= number of blocks
        blockSizesFile = vm["blockSizesFile"].as<std::string>();
        std::vector<std::vector<ValueType>> blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        if (blockSizes.size() < nodeWeights.size()) {
            throw std::invalid_argument("Block size file " + blockSizesFile + " has " + std::to_string(blockSizes.size()) + " weights per block, "
                                        + "but nodes have " + std::to_string(nodeWeights.size()) + " weights.");
        }

        if (blockSizes.size() > nodeWeights.size()) {
            blockSizes.resize(nodeWeights.size());
            if (comm->getRank() == 0) {
                std::cout << "Block size file " + blockSizesFile + " has " + std::to_string(blockSizes.size()) + " weights per block, "
                          + "but nodes have " + std::to_string(nodeWeights.size()) + " weights. Discarding surplus block sizes." << std::endl;
            }
        }

        for (IndexType i = 0; i < nodeWeights.size(); i++) {
            const ValueType blockSizesSum  = std::accumulate( blockSizes[i].begin(), blockSizes[i].end(), 0);
            const ValueType nodeWeightsSum = nodeWeights[i].sum();
            SCAI_ASSERT_GE( blockSizesSum, nodeWeightsSum, "The block sizes provided are not enough to fit the total weight of the input" );
        }

        commTree.createFlatHeterogeneous( blockSizes );
    }else if( settings.hierLevels.size()!=0 ){
        const IndexType numWeights = nodeWeights.size();
        commTree.createFromLevels(settings.hierLevels, numWeights );
    } else {
        commTree.createFlatHomogeneous( settings.numBlocks, nodeWeights.size() );
    }

    commTree.adaptWeights( nodeWeights );


    //---------------------------------------------------------------
    //
    // get previous partition, if set
    //

    DenseVector<IndexType> previous;
    if (vm.count("previousPartition")) {
        std::string filename = vm["previousPartition"].as<std::string>();
        previous = ITI::FileIO<IndexType, ValueType>::readPartition(filename, N);
        if (previous.size() != N) {
            throw std::runtime_error("Previous partition has wrong size.");
        }
        if (previous.max() != settings.numBlocks-1) {
            throw std::runtime_error("Illegal maximum block ID in previous partition:" + std::to_string(previous.max()));
        }
        if (previous.min() != 0) {
            throw std::runtime_error("Illegal minimum block ID in previous partition:" + std::to_string(previous.min()));
        }
        settings.repartition = true;
    }

    //
    // time needed to get the input. Synchronize first to make sure that all processes are finished.
    //

    comm->synchronize();
    std::chrono::duration<double> inputTime = std::chrono::steady_clock::now() - startTime;

    assert(N > 0);

    if (settings.repartition && comm->getSize() == settings.numBlocks) {
        //redistribute according to previous partition now to simulate the setting in a dynamic repartitioning
        assert(previous.size() == N);
        auto previousRedist = scai::dmemo::redistributePlanByNewOwners(previous.getLocalValues(), previous.getDistributionPtr());
        graph.redistribute(previousRedist, graph.getColDistributionPtr());
        for (IndexType d = 0; d < settings.dimensions; d++) {
            coordinates[d].redistribute(previousRedist);
        }

        for (IndexType i = 0; i < nodeWeights.size(); i++) {
            nodeWeights[i].redistribute(previousRedist);
        }
        previous = fill<DenseVector<IndexType>>(previousRedist.getTargetDistributionPtr(), comm->getRank());

    }

    std::vector<Metrics<ValueType>> metricsVec;

    //------------------------------------------------------------
    //
    // partition the graph
    //
    IndexType repeatTimes = settings.repeatTimes;

    if( repeatTimes>0 ) {
        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
        // SCAI_ASSERT_ERROR(rowDistPtr->isEqual( new scai::dmemo::BlockDistribution(N, comm) ) , "Graph row distribution should (?) be a block distribution." );
        SCAI_ASSERT_ERROR( coordinates[0].getDistributionPtr()->isEqual( *rowDistPtr ), "rowDistribution and coordinates distribution must be equal" );
        for (IndexType i = 0; i < nodeWeights.size(); i++) {
            SCAI_ASSERT_ERROR( nodeWeights[i].getDistributionPtr()->isEqual( *rowDistPtr ), "rowDistribution and nodeWeights distribution must be equal" );
        }
    }

    //store distributions to use later
    const scai::dmemo::DistributionPtr rowDistPtr( new scai::dmemo::BlockDistribution(N, comm) );
    const scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ) );

    scai::lama::DenseVector<IndexType> partition;

    for( IndexType r=0; r<repeatTimes; r++) {

        // for the next runs the input is redistributed, so we must redistribute to the original distributions

        if (repeatTimes > 1) {
            if(comm->getRank()==0) std::cout<< std::endl<< std::endl;
            PRINT0("\t\t ----------- Starting run number " << r +1 << " -----------");
        }

        if(r>0) {
            PRINT0("Input redistribution: block distribution for graph rows, coordinates and nodeWeigts, no distribution for graph columns");

            graph.redistribute( rowDistPtr, noDistPtr );
            for(int d=0; d<settings.dimensions; d++) {
                coordinates[d].redistribute( rowDistPtr );
            }
            for (IndexType i = 0; i < nodeWeights.size(); i++) {
                nodeWeights[i].redistribute( rowDistPtr );
            }
        }

        //metricsVec.push_back( Metrics( comm->getSize()) );
        metricsVec.push_back( Metrics<ValueType>( settings ) );

        std::chrono::time_point<std::chrono::steady_clock> beforePartTime =  std::chrono::steady_clock::now();

        partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coordinates, nodeWeights, previous, commTree, comm, settings, metricsVec[r] );
        assert( partition.size() == N);
        assert( coordinates[0].size() == N);

        std::chrono::duration<double> partitionTime =  std::chrono::steady_clock::now() - beforePartTime;

        //WARNING: with the noRefinement flag the partition is not distributed
        if (!comm->all(partition.getDistribution().isEqual(graph.getRowDistribution()))) {
            partition.redistribute( graph.getRowDistributionPtr());
        }
        SCAI_ASSERT_EQ_ERROR( nodeWeights[0].getDistributionPtr()->getLocalSize(),\
                              partition.getDistributionPtr()->getLocalSize(), "Partition distribution mismatch(?)");

        //---------------------------------------------
        //
        // Get metrics
        //

        std::chrono::time_point<std::chrono::steady_clock> beforeReport = std::chrono::steady_clock::now();

        metricsVec[r].getMetrics(graph, partition, nodeWeights, settings );

        metricsVec[r].MM["inputTime"] = ValueType ( comm->max(inputTime.count() ));
        //metricsVec[r].MM["timeFinalPartition"] = ValueType (comm->max(partitionTime.count()));

        std::chrono::duration<double> reportTime =  std::chrono::steady_clock::now() - beforeReport;

        //---------------------------------------------
        //
        // Print some output
        //

        if (comm->getRank() == 0 ) {
            std::cout<< "commit:"<< version << " machine:" << machine << " input:"<< ( vm.count("graphFile") ? vm["graphFile"].as<std::string>() :"generate");
            std::cout << " p:"<< comm->getSize() << " k:"<< settings.numBlocks;
            auto oldprecision = std::cout.precision(std::numeric_limits<double>::max_digits10);
            std::cout <<" seed:" << vm["seed"].as<double>() << std::endl;
            std::cout.precision(oldprecision);
            metricsVec[r].printHorizontal2( std::cout ); //TODO: remove?
        }

        //---------------------------------------------------------------
        //
        // Reporting output to std::cout and file for this repeatition
        //

        metricsVec[r].MM["reportTime"] = ValueType (comm->max(reportTime.count()));

        if (comm->getRank() == 0 && settings.metricsDetail.compare("no") != 0) {
            metricsVec[r].print( std::cout );
        }
        if( settings.storeInfo && settings.outFile!="-" ) {
            //TODO: create a better tmp name
            std::string fileName = settings.outFile+ "_r"+ std::to_string(r);
            if( comm->getRank()==0 ) {
                std::ofstream outF( fileName, std::ios::out);
                if(outF.is_open()) {
					outF << "Calling command:" << std::endl;
					outF<< callingCommand << std::endl << std::endl;
                    settings.print( outF, comm);
                    outF << "\nMetrics" << std::endl;
                    metricsVec[r].print( outF );
                }
            }
        }

        comm->synchronize();
    }// repeat loop

    std::chrono::duration<double> totalTime =  std::chrono::steady_clock::now() - startTime;
    ValueType totalT = ValueType ( comm->max(totalTime.count() ));

    //
    // writing results in a file and std::cout
    //

    //aggregate metrics in one struct
    const Metrics<ValueType> aggrMetrics = aggregateVectorMetrics( metricsVec, comm );

    if (repeatTimes > 1) {
        if (comm->getRank() == 0) {
			std::cout<< "\nMetrics, averaged for all runs:" << std::endl;
            std::cout<<  "\033[1;36m";
            aggrMetrics.print( std::cout );
            std::cout << " \033[0m";
        }
    }


    if( settings.storeInfo && settings.outFile!="-" ) {
        if( comm->getRank()==0) {
            std::ofstream outF( settings.outFile, std::ios::out);
            if(outF.is_open()) {
                outF << "Running " << __FILE__ << std::endl;
                settings.print( outF, comm);

                outF << "\nMetrics" << std::endl;

                aggrMetrics.print( outF );

                //	profiling info for k-means
                if(settings.verbose) {
                    outF << "iter | delta | time | imbalance | balanceIter" << std::endl;
                    ValueType totTime = 0.0;
                    SCAI_ASSERT_EQ_ERROR( metricsVec[0].kmeansProfiling.size(), metricsVec[0].numBalanceIter.size(), "mismatch in kmeans profiling metrics vectors");


                    for( int i=0; i<metricsVec[0].kmeansProfiling.size(); i++) {
                        std::tuple<ValueType, ValueType, ValueType> tuple = metricsVec[0].kmeansProfiling[i];

                        outF << i << " " << std::get<0>(tuple) << " " << std::get<1>(tuple) << " " << std::get<2>(tuple) << " " <<  metricsVec[0].numBalanceIter[i] << std::endl;
                        totTime += std::get<1>(tuple);
                    }
                    outF << "totTime: " << totTime << std::endl;
                }
                //

                //printVectorMetrics( metricsVec, outF );
                std::cout<< "Output information written to file " << settings.outFile << " in total time " << totalT << std::endl;
            }	else	{
                std::cout<< "Could not open file " << settings.outFile << " information not stored"<< std::endl;
            }
        }
    }


    if( settings.outFile!="-" and settings.storePartition ) {
        std::chrono::time_point<std::chrono::steady_clock> beforePartWrite = std::chrono::steady_clock::now();
        std::string partOutFile = settings.outFile+".part";
        ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partition, partOutFile );

        std::chrono::duration<double> writePartTime =  std::chrono::steady_clock::now() - beforePartWrite;
        if( comm->getRank()==0 ) {
            std::cout << " and last partition of the series in file " << partOutFile << std::endl;
            std::cout<< " Time needed to write .part file: " << writePartTime.count() <<  std::endl;
        }
    }

    // the code below writes the output coordinates in one file per processor for visualization purposes.
    //=================

    if (settings.writeDebugCoordinates) {

        std::vector<DenseVector<ValueType> > coordinateCopy = coordinates;
        auto distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues() );
        for (IndexType dim = 0; dim < settings.dimensions; dim++) {
            assert( coordinateCopy[dim].size() == N);
            coordinateCopy[dim].redistribute( distFromPartition );
        }

        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinateCopy, settings.dimensions, "debugResult");
        comm->synchronize();
    }

    if (vm.count("callExit")) {
        //this is needed for supermuc
        std::exit(0);
    }

    return 0;
}
