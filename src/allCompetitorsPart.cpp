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


extern "C" {
    void memusage(size_t *, size_t *,size_t *,size_t *,size_t *);
}



//---------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

    using namespace ITI;
    typedef double ValueType;   //use double

    std::chrono::time_point<std::chrono::system_clock> startTime =  std::chrono::system_clock::now();

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

    const IndexType thisPE = comm->getRank();
    IndexType N;


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

    std::string graphFile;

    if (vm.count("graphFile")) {

        graphFile = vm["graphFile"].as<std::string>();
        std::string coordFile;
        if (vm.count("coordFile")) {
            coordFile = vm["coordFile"].as<std::string>();
        } else {
            coordFile = graphFile + ".xyz";
        }

        // read the graph
        if (vm.count("fileFormat")) {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, settings.fileFormat );
        } else {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights );
        }

        N = graph.getNumRows();

        SCAI_ASSERT_EQUAL( graph.getNumColumns(),  graph.getNumRows(), "matrix not square");
        SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");

        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();

        // set the node weights
        IndexType numReadNodeWeights = nodeWeights.size();
        if (numReadNodeWeights == 0) {
            nodeWeights.resize(1);
            nodeWeights[0] = fill<DenseVector<ValueType>>(rowDistPtr, 1);
        }

        if (settings.numNodeWeights > 0) {
            if (settings.numNodeWeights < nodeWeights.size()) {
                nodeWeights.resize(settings.numNodeWeights);
                if (comm->getRank() == 0) {
                    std::cout << "Read " << numReadNodeWeights << " weights per node but " << settings.numNodeWeights << " weights were specified, thus discarding "
                              << numReadNodeWeights - settings.numNodeWeights << std::endl;
                }
            } else if (settings.numNodeWeights > nodeWeights.size()) {
                nodeWeights.resize(settings.numNodeWeights);
                for (IndexType i = numReadNodeWeights; i < settings.numNodeWeights; i++) {
                    nodeWeights[i] = fill<DenseVector<ValueType>>(rowDistPtr, 1);
                }
                if (comm->getRank() == 0) {
                    std::cout << "Read " << numReadNodeWeights << " weights per node but " << settings.numNodeWeights << " weights were specified, padding with "
                              << settings.numNodeWeights - numReadNodeWeights << " uniform weights. " << std::endl;
                }
            }
        }

        //read the coordinates file
        if (vm.count("coordFormat")) {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.coordFormat);
        } else if (vm.count("fileFormat")) {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.fileFormat);
        } else {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
        }
        SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size(), coords[1].getLocalValues().size(), "coordinates not of same size" );

    } else if(vm.count("generate")) {
        if (settings.dimensions != 3) {
            if(comm->getRank() == 0) {
                std::cout << "Graph generation supported only fot 2 or 3 dimensions, not " << settings.dimensions << std::endl;
                return 127;
            }
        }

        N = settings.numX * settings.numY * settings.numZ;

        std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D

        maxCoord[0] = settings.numX;
        maxCoord[1] = settings.numY;
        maxCoord[2] = settings.numZ;

        std::vector<IndexType> numPoints(3); // number of points in each dimension, used only for 3D

        for (IndexType i = 0; i < 3; i++) {
            numPoints[i] = maxCoord[i];
        }

        if( comm->getRank()== 0) {
            std::cout<< "Generating for dim= "<< settings.dimensions << " and numPoints= "<< settings.numX << ", " << settings.numY << ", "<< settings.numZ << ", in total "<< N << " number of points" << std::endl;
            std::cout<< "\t\t and maxCoord= "<< maxCoord[0] << ", "<< maxCoord[1] << ", " << maxCoord[2]<< std::endl;
        }

        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>( rowDistPtr, noDistPtr );

        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<settings.dimensions; i++) {
            coords[i].allocate(coordDist);
            coords[i] = static_cast<ValueType>( 0 );
        }

        // create the adjacency matrix and the coordinates
        ITI::MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist( graph, coords, maxCoord, numPoints, settings.dimensions);

        IndexType nodes= graph.getNumRows();
        IndexType edges= graph.getNumValues()/2;
        if(comm->getRank()==0) {
            std::cout<< "Generated structured 3D graph with "<< nodes<< " and "<< edges << " edges."<< std::endl;
        }

    } else {
        std::cout << "Either an input file or generation parameters are needed. Call again with --graphFile, --quadTreeFile, or --generate" << std::endl;
        return 126;
    }

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    /*
    //TODO: must compile with mpicc and module mpi.ibm/1.4 and NOT mpi.ibm/1.4_gcc
    size_t total=-1,used=-1,free=-1,buffers=-1, cached=-1;
    memusage(&total, &used, &free, &buffers, &cached);
    printf("MEM: avail: %ld , used: %ld , free: %ld , buffers: %ld , file cache: %ld \n",total,used,free,buffers, cached);
    */

    //---------------------------------------------------------------------------------
    //
    // start main for loop for all tools
    //

    //WARNING: 1) removed parmetis sfc
    //WARNING: 2) parMetisGraph should be last because it often crashes
    std::vector<ITI::Tool> allTools = {ITI::Tool::zoltanRCB, ITI::Tool::zoltanRIB, ITI::Tool::zoltanMJ, ITI::Tool::zoltanSFC, ITI::Tool::parMetisSFC, ITI::Tool::parMetisGeom, ITI::Tool::parMetisGraph };

    //used for printing and creating filenames
    std::map<ITI::Tool, std::string> toolName = {
        {ITI::Tool::zoltanRCB,"zoltanRcb"}, {ITI::Tool::zoltanRIB,"zoltanRib"}, {ITI::Tool::zoltanMJ,"zoltanMJ"}, {ITI::Tool::zoltanSFC,"zoltanHsfc"},
        {ITI::Tool::parMetisSFC,"parMetisSFC"}, {ITI::Tool::parMetisGeom,"parMetisGeom"}, {ITI::Tool::parMetisGraph,"parMetisGraph"}
    };

    std::vector<ITI::Tool> wantedTools;
    std::vector<std::string> tools = settings.tools; //not really needed

    //if no 'tools' parameter was gives, use all tools
    if( tools.size()==0 )
        tools.resize( 1, "all" );

    if( tools[0] == "all") {
        wantedTools = allTools;
    } else {
        for( std::vector<std::string>::iterator tool=tools.begin(); tool!=tools.end(); tool++) {
            ITI::Tool thisTool;
            if( (*tool).substr(0,8)=="parMetis") {
                if 		( *tool=="parMetisGraph") {
                    thisTool = ITI::Tool::parMetisGraph;
                }
                else if ( *tool=="parMetisGeom") {
                    thisTool = ITI::Tool::parMetisGeom;
                }
                else if	( *tool=="parMetisSfc") {
                    thisTool = ITI::Tool::parMetisSFC;
                }
            } else if ( (*tool).substr(0,6)=="zoltan") {
                std::string algo;
                if		( *tool=="zoltanRcb") {
                    thisTool = ITI::Tool::zoltanRCB;
                }
                else if ( *tool=="zoltanRib") {
                    thisTool = ITI::Tool::zoltanRIB;
                }
                else if ( *tool=="zoltanMJ") {
                    thisTool = ITI::Tool::zoltanMJ;
                }
                else if ( *tool=="zoltanHsfc") {
                    thisTool = ITI::Tool::zoltanSFC;
                }
            } else {
                std::cout<< "Tool "<< *tool <<" not supported.\nAborting..."<<std::endl;
                return -1;
            }
            wantedTools.push_back( thisTool );
        }
    }


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
        } else {
            settings.repeatTimes = 5;
        }
        int parMetisGeom=0;

        std::string outFile = "-";

        if( vm.count("outDir") and vm.count("storeInfo") ) {
            //set the graphName in order to create the outFile name
            std::string copyName = graphFile;
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

        //get the partition
        partition = ITI::Wrappers<IndexType,ValueType>::partition ( graph, coords, nodeWeights, nodeWeightsUse, thisTool, settings, metrics);

        PRINT0("time to get the partition: " <<  metrics.MM["timeFinalPartition"] );

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

        std::string machine = settings.machine;

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

            //printMetricsShort( metrics, std::cout);

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
            std::chrono::time_point<std::chrono::system_clock> beforePartWrite = std::chrono::system_clock::now();
            std::string partOutFile = outFile+".part";
            ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partition, partOutFile );

            std::chrono::duration<double> writePartTime =  std::chrono::system_clock::now() - beforePartWrite;
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

    std::chrono::duration<ValueType> totalTimeLocal = std::chrono::system_clock::now() - startTime;
    ValueType totalTime = comm->max( totalTimeLocal.count() );
    if( thisPE==0 ) {
        std::cout<<"Exiting file " << __FILE__ << " , total time= " << totalTime <<  std::endl;
    }

    //this is needed for supermuc
    std::exit(0);

    return 0;
}

