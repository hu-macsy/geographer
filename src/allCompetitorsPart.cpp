/*
 * Author Charilaos "Harry" Tzovas
 *
 *
 * example of use:
 * mpirun -n #p parMetis --graphFile="meshes/hugetrace/hugetrace-00008.graph" --dimensions=2 --numBlocks=#k --geom=0
 *
 *
 * where #p is the number of PEs that the graph is distributed to and #k the number of blocks to partition to
*/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <sys/stat.h>

#include <cxxopts.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include "FileIO.h"
#include "Settings.h"
#include "Metrics.h"
#include "MeshGenerator.h"
#include "Wrappers.h"
#include "parseArgs.h"
#include "mainHeader.h"

#if PARMETIS_FOUND
#include "parmetisWrapper.h"
#include <parmetis.h>
#endif

#if ZOLTAN_FOUND
#include "zoltanWrapper.h"
#endif

#if PARHIP_FOUND
#include "parhipWrapper.h"
#endif



//---------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

    using namespace ITI;
    typedef double ValueType;   //use double

#if ZOLTAN_FOUND
    Tpetra::ScopeGuard tscope(&argc, &argv);
#endif

    // timing information
    std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();

    //global communicator
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    const int prevArgc = argc; // options.parse(argc, argv) changed argc

    cxxopts::Options options = ITI::populateOptions();
    cxxopts::ParseResult vm = options.parse(argc, argv);
    Settings settings = initialize( prevArgc, argv, vm, comm);

    if (vm.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    } 

    printInfo( std::cout, comm, settings);

    //-----------------------------------------
    //
    // read the input graph or generate
    //

    CSRSparseMatrix<ValueType> graph;
    std::vector<DenseVector<ValueType>> coords(settings.dimensions);
    std::vector<DenseVector<ValueType>> nodeWeights;	//the weights for each node

    IndexType N = readInput<ValueType>( vm, settings, comm, graph, coords, nodeWeights );

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    //---------------------------------------------------------------
    //
    // read the communication graph or the block sizes if provided
    //
    
    ITI::CommTree<IndexType,ValueType> commTree = createCommTree( vm, settings, comm, nodeWeights);
    commTree.adaptWeights( nodeWeights );

    //---------------------------------------------------------------------------------
    //
    // start main for loop for all tools
    //

    //WARNING: 1) removed parmetis sfc
    //WARNING: 2) parMetisGraph should be last because it often crashes
    std::vector<ITI::Tool> allTools = {ITI::Tool::zoltanRCB, ITI::Tool::zoltanRIB, ITI::Tool::zoltanMJ, ITI::Tool::zoltanXPulp, ITI::Tool::zoltanSFC, ITI::Tool::parMetisSFC, ITI::Tool::parMetisGeom, ITI::Tool::parMetisGraph, ITI::Tool::parhipFastMesh, ITI::Tool::parhipUltraFastMesh, ITI::Tool::parhipEcoMesh  };

    std::vector<ITI::Tool> wantedTools;
    std::vector<std::string> tools = settings.tools; //not really needed

    //if no 'tools' parameter was gives, use all tools
    if( tools.size()==0 ){
        wantedTools = allTools;
    } else {
        for( std::vector<std::string>::iterator tool=tools.begin(); tool!=tools.end(); tool++) {
            wantedTools.push_back( to_tool(*tool) );
        }
    }


    const IndexType thisPE = comm->getRank();   

    for( int t=0; t<wantedTools.size(); t++) {

        comm->synchronize();
        ITI::Tool thisTool = wantedTools[t];
        std::cout.precision(5);
        MSG0("will start partition with tool " << ITI::to_string(thisTool) );

        // if using unit weights, set flag for wrappers
        bool nodeWeightsUse = true;

        // if graph is too big, repeat less times to avoid memory and time problems
        if( N>std::pow(2,29) ) {
            settings.repeatTimes = 3;
            if( thisPE==0 ) {
                std::cout << "WARNING: because the graph is too big, we repeat only " << settings.repeatTimes << " times" << std::endl;
            }
        } 
     
        //set outFile depending if we get outDir or outFile parameter
        std::string outFile = getOutFileName(settings, ITI::to_string(thisTool), comm);

        std::ifstream f(outFile);
        if( f.good() and settings.storeInfo ) {
            comm->synchronize();	// maybe not needed
            PRINT0("\n\tWARNING: File " << outFile << " already exists. Skipping partition with " << ITI::to_string(thisTool));
            continue;
        }

        //get the partition
        ITI::Wrappers<IndexType,ValueType>* partitioner;
        if( ITI::to_string(thisTool).rfind("zoltan",0)==0 ){
#if ZOLTAN_FOUND            
            partitioner = new zoltanWrapper<IndexType,ValueType>;
#else
            throw std::runtime_error("Requested a zoltan tool but zoltan is not found. Pick another tool.\nAborting...");
#endif            
        }else if( ITI::to_string(thisTool).rfind("parMetis",0)==0 ){
#if PARMETIS_FOUND            
            partitioner = new parmetisWrapper<IndexType,ValueType>;
#else
            throw std::runtime_error("Requested a parmetis tool but parmetis is not found. Pick another tool.\nAborting...");
#endif
        }
        else if(ITI::to_string(thisTool).rfind("parhip",0)==0 ){
#if PARHIP_FOUND
            partitioner = new parhipWrapper<IndexType,ValueType>;
#else         
            throw std::runtime_error("Requested a parhip tool but parhip is not found. Pick another tool.\nAborting...");
#endif   
        }else{
            throw std::runtime_error("Provided tool: "+ ITI::to_string(thisTool) + " not supported.\nAborting..." );
        }

        // get the partition and metrics
        //
        scai::lama::DenseVector<IndexType> partition;
        scai::lama::DenseVector<IndexType> oldPartition(dist, 1);
        std::vector<Metrics<ValueType>> metricsVec;

        for( int r=0; r<settings.repeatTimes; r++){
            metricsVec.push_back( Metrics<ValueType>( settings ) );

            partition = partitioner->partition( graph, coords, nodeWeights, nodeWeightsUse, thisTool, commTree, settings, metricsVec[r]);
            
            // partition has the the same distribution as the graph rows
            SCAI_ASSERT_ERROR( partition.getDistribution().isEqual( graph.getRowDistribution() ), "Distribution mismatch.")

            //ValueType partDiff = oldPartition.maxDiffNorm(partition); // another way to check if partitions are close but is more expensive
            bool isIdentical = partition.all(scai::common::CompareOp::EQ, oldPartition);
            if( isIdentical ){
                if(comm->getRank()==0){
                    std::cout<< "Partition is identical, not calculating metrics" << std::endl;  
                }
                //keep only the run time of this run
                ValueType runTime = metricsVec[r].MM["timeTotal"];
                metricsVec[r] = metricsVec[r-1];
                metricsVec[r].MM["timeTotal"] = runTime;
            }else{
                //std::vector<std::vector<ValueType>> blockSizes = commTree.getBalanceVectors();
                metricsVec[r].getMetrics( graph, partition, nodeWeights, settings, commTree );
            }

            PRINT0("time to get the partition with " << ITI::to_string(thisTool) << ": " << metricsVec[r].MM["timeTotal"] );

            //if one run exceeds the time limit, do not execute the rest of the runs
            if( metricsVec[r].MM["timeTotal"]>ITI::HARD_TIME_LIMIT) {
                if(comm->getRank()==0){
                    std::cout<< "Stopping runs because of excessive running total running time: " << metricsVec[r].MM["timeTotal"] << std::endl;
                }
                break;
            }
            oldPartition = partition;
        }

        //aggregate metrics in one struct
        const Metrics<ValueType> aggrMetrics = aggregateVectorMetrics( metricsVec, comm );

        //---------------------------------------------------------------
        //
        // Reporting output to std::cout
        //

        if( thisPE==0 ) {
            printInfo( std::cout, comm, settings);
            std::cout << "\nFinished tool: " << ITI::to_string(thisTool) << std::endl;
            
            aggrMetrics.print( std::cout );

            // write in a file
            if( outFile!= "-" and settings.storeInfo) {
                std::ofstream outF( outFile, std::ios::out);
                if(outF.is_open()) {
                    outF << "Running " << __FILE__ << " for tool " << ITI::to_string(thisTool) << std::endl;
                    printInfo( outF, comm, settings);

                    aggrMetrics.print( outF );
                    //printMetricsShort( aggrMetrics, outF);
                    std::cout<< "Output information written to file " << outFile << std::endl;
                } else {
                    std::cout<< "\n\tWARNING: Could not open file " << outFile << " informations not stored.\n"<< std::endl;
                }
            }
            std::cout<< "###" << std::endl;
        }

        if( outFile!="-" and settings.storePartition ) {
            std::chrono::time_point<std::chrono::steady_clock> beforePartWrite = std::chrono::steady_clock::now();
            std::string partOutFile = outFile+".part";
            ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partition, partOutFile );

            std::chrono::duration<double> writePartTime =  std::chrono::steady_clock::now() - beforePartWrite;
            if( comm->getRank()==0 ) {
                std::cout << " and last partition of the series in file " << partOutFile << std::endl;
                std::cout<< " Time needed to write .partition file: " << writePartTime.count() <<  std::endl;
            }
        }

        // the code below writes the output coordinates in one file per processor for visualization purposes.
        //=================

        if (settings.writeDebugCoordinates) {

            std::vector<DenseVector<ValueType> > coordinateCopy = coords;

            //scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partition.getDistribution(), partition.getLocalValues() ) );
            scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues());
            for (IndexType dim = 0; dim < settings.dimensions; dim++) {
                assert( coordinateCopy[dim].size() == N);
                //coordinates[dim].redistribute(partition.getDistributionPtr());
                coordinateCopy[dim].redistribute( distFromPartition );
            }

            std::string destPath = "partResults/" +  ITI::to_string(thisTool) +"/blocks_" + std::to_string(settings.numBlocks) ;
            struct stat sb;
            if (stat(destPath.data(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
                ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinateCopy, settings.dimensions, destPath + "/debugResult");
                comm->synchronize();
            } else {
                std::cout<< "WARNING: directrory " << destPath << " does not exist. De buf coordinates were not stored. Create directory and re-run" << std::endl;
            }
        }
        if( thisPE==0) std::cout<< std::endl;
        comm->synchronize();    // needed when storing files 
    } // for wantedTools.size()

    std::chrono::duration<ValueType> totalTimeLocal = std::chrono::steady_clock::now() - startTime;
    ValueType totalTime = comm->max( totalTimeLocal.count() );
    if( thisPE==0 ) {
        std::cout<<"Exiting file " << __FILE__ << " , total time= " << totalTime <<  std::endl;
    }

    if (vm.count("callExit")) {
        //this is needed for supermuc
        std::exit(0);
    }

    return 0;
}

