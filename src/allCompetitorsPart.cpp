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

    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

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

    if( comm->getRank() ==0 ) {
        std::cout <<"Starting file " << __FILE__ << std::endl;

        std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
        std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
        std::cout << "date and time: " << std::ctime(&timeNow) << std::endl;
    }

    //-----------------------------------------
    //
    // read the input graph or generate
    //

    CSRSparseMatrix<ValueType> graph;
    std::vector<DenseVector<ValueType>> coords(settings.dimensions);
    std::vector<DenseVector<ValueType>> nodeWeights;	//the weights for each node

    IndexType N = readInput<ValueType>( vm, settings, comm, graph, coords, nodeWeights );

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    //---------------------------------------------------------------------------------
    //
    // start main for loop for all tools
    //

    //WARNING: 1) removed parmetis sfc
    //WARNING: 2) parMetisGraph should be last because it often crashes
    std::vector<ITI::Tool> allTools = {ITI::Tool::zoltanRCB, ITI::Tool::zoltanRIB, ITI::Tool::zoltanMJ, ITI::Tool::zoltanSFC, ITI::Tool::parMetisSFC, ITI::Tool::parMetisGeom, ITI::Tool::parMetisGraph, ITI::Tool::parhipFastMesh, ITI::Tool::parhipUltraFastMesh, ITI::Tool::parhipEcoMesh  };

    //used for printing and creating filenames
    std::map<ITI::Tool, std::string> toolName = {
        {ITI::Tool::zoltanRCB,"zoltanRcb"}, {ITI::Tool::zoltanRIB,"zoltanRib"}, {ITI::Tool::zoltanMJ,"zoltanMJ"}, {ITI::Tool::zoltanSFC,"zoltanHsfc"},
        {ITI::Tool::parMetisSFC,"parMetisSFC"}, {ITI::Tool::parMetisGeom,"parMetisGeom"}, {ITI::Tool::parMetisGraph,"parMetisGraph"},
        {ITI::Tool::parhipFastMesh,"parhipFastMesh"}, {ITI::Tool::parhipUltraFastMesh,"parhipUltraFastMesh"}, {ITI::Tool::parhipEcoMesh,"parhipEcoMesh"}
    };

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


    std::string machine = settings.machine;
    const IndexType thisPE = comm->getRank();
    

    for( int t=0; t<wantedTools.size(); t++) {

        ITI::Tool thisTool = wantedTools[t];

        // get the partition and metrics
        //
        scai::lama::DenseVector<IndexType> partition;

        // the constructor with metrics(comm->getSize()) is needed for ParcoRepart timing details
        Metrics<ValueType> metrics( settings );
        //metrics.numBlocks = settings.numBlocks;

        // if using unit weights, set flag for wrappers
        bool nodeWeightsUse = true;

        // if graph is too big, repeat less times to avoid memory and time problems
        if( N>std::pow(2,29) ) {
            settings.repeatTimes = 2;
        } 
        
        //set outFile depending if we get outDir or outFile parameter
        std::string outFile = settings.outFile;
		
        if( vm.count("outDir") and vm.count("storeInfo") ) {
            //set the graphName in order to create the outFile name
            std::string copyName;
            if( vm.count("graphFile") ){
                copyName = vm["graphFile"].as<std::string>();
            }else{     
                copyName = "generate_"+vm["numX"].as<std::string>()+"_"+vm["numY"].as<std::string>();
                //if( settings.dimensions==3 ){
                //    copyName += "_"+vm["numZ"].as<std::string>();
               // }
            }
            std::reverse( copyName.begin(), copyName.end() );
            std::vector<std::string> strs = aux<IndexType,ValueType>::split( copyName, '/' );
            std::string graphName = aux<IndexType,ValueType>::split(strs.back(), '.')[0];
            std::reverse( graphName.begin(), graphName.end() );
            //PRINT0( graphName );
            //add specific folder for each tool
            outFile = settings.outDir+ toolName[thisTool]+ "/"+ graphName+ "_k"+ std::to_string(settings.numBlocks)+ "_"+ toolName[thisTool]+ ".info";
        }

        std::ifstream f(outFile);
        if( f.good() and settings.storeInfo ) {
            comm->synchronize();	// maybe not needed
            PRINT0("\n\tWARNING: File " << outFile << " allready exists. Skipping partition with " << toolName[thisTool]);
            continue;
        }
PRINT0(toolName[thisTool]);
        //get the partition
        ITI::Wrappers<IndexType,ValueType>* partitioner;
        if( toolName[thisTool].rfind("zoltan",0)==0 ){
#if ZOLTAN_FOUND            
            partitioner = new zoltanWrapper<IndexType,ValueType>;
#else
            throw std::runtime_error("Requested a zoltan tool but zoltan is not found. Pick another tool.\nAborting...");
#endif            
        }else if( toolName[thisTool].rfind("parMetis",0)==0 ){
#if PARMETIS_FOUND            
            partitioner = new parmetisWrapper<IndexType,ValueType>;
#else
            throw std::runtime_error("Requested a parmetis tool but parmetis is not found. Pick another tool.\nAborting...");
#endif
        }
        else if(toolName[thisTool].rfind("parhip",0)==0 ){
#if PARHIP_FOUND
            partitioner = new parhipWrapper<IndexType,ValueType>;
#else         
            throw std::runtime_error("Requested a parhip tool but parhip is not found. Pick another tool.\nAborting...");
#endif   
        }else{
            throw std::runtime_error("Provided tool: "+ toolName[thisTool] + " not supported.\nAborting..." );
        }

        partition = partitioner->partition( graph, coords, nodeWeights, nodeWeightsUse, thisTool, settings, metrics);

        PRINT0("time to get the partition: " <<  metrics.MM["timeTotal"] );

        // partition has the the same distribution as the graph rows
        SCAI_ASSERT_ERROR( partition.getDistribution().isEqual( graph.getRowDistribution() ), "Distribution mismatch.")


        if( settings.metricsDetail=="all" ) {
            metrics.getAllMetrics( graph, partition, nodeWeights, settings );
        }
        if( settings.metricsDetail=="easy" ) {
            metrics.getEasyMetrics( graph, partition, nodeWeights, settings );
        }


        //---------------------------------------------------------------
        //
        // Reporting output to std::cout
        //

        if( thisPE==0 ) {
            if( vm.count("generate") ) {
                std::cout << std::endl << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon;
            } else {
                std::cout << std::endl << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon;
            }
            std::cout << "\nFinished tool" << std::endl;
            std::cout << "\033[1;36m";
            std::cout << "\n >>>> " << toolName[thisTool];
            std::cout<<  "\033[0m" << std::endl;

            metrics.print( std::cout );

            // write in a file
            if( outFile!= "-" ) {
                std::ofstream outF( outFile, std::ios::out);
                if(outF.is_open()) {
                    outF << "Running " << __FILE__ << " for tool " << toolName[thisTool] << std::endl;
                    if( vm.count("generate") ) {
                        outF << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                    } else {
                        outF << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                    }

                    metrics.print( outF );
                    //printMetricsShort( metrics, outF);
                    std::cout<< "Output information written to file " << outFile << std::endl;
                } else {
                    std::cout<< "\n\tWARNING: Could not open file " << outFile << " informations not stored.\n"<< std::endl;
                }
            }
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

            std::string destPath = "partResults/" +  toolName[thisTool] +"/blocks_" + std::to_string(settings.numBlocks) ;
            struct stat sb;
            if (stat(destPath.data(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
                ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinateCopy, settings.dimensions, destPath + "/debugResult");
                comm->synchronize();
            } else {
                std::cout<< "WARNING: directrory " << destPath << " does not exist. De buf coordinates were not stored. Create directory and re-run" << std::endl;
            }
        }

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

