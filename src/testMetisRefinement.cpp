/*
 * @author Charilaos "Harry" Tzovas
 * @date 10/09/2019
 *
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
#include "ParcoRepart.h"
#include "parseArgs.h"

#include "mainHeader.h"

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

    if( comm->getRank() ==0 ) {
        std::cout <<"Starting file " << __FILE__ << std::endl;

        std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
        std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
        std::cout << "date and time: " << std::ctime(&timeNow) << std::endl;
    }


    //-----------------------------------------
    //
    // read the input graph 
    //


    CSRSparseMatrix<ValueType> graph;
    std::vector<DenseVector<ValueType>> coords(settings.dimensions);
    std::vector<DenseVector<ValueType>> nodeWeights;    //the weights for each node

    std::string graphFile =  vm["graphFile"].as<std::string>();

    IndexType N = readInput<ValueType>( vm, settings, comm, graph, coords, nodeWeights );

    //-----------------------------------------
    //
    //partition input using some tool
    
    Metrics<ValueType> metricsBefore(settings);

    DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coords, nodeWeights, settings, metricsBefore );

    metricsBefore.getAllMetrics( graph, partition, nodeWeights, settings );
    if (comm->getRank() == 0 && settings.metricsDetail.compare("no") != 0) {
        metricsBefore.print( std::cout );
/*
        if( settings.storeInfo && settings.outFile!="-" ){
            std::ofstream outF( settings.outFile, std::ios::out);
            if(outF.is_open()) {
                outF << "Running " << __FILE__ << std::endl;
                settings.print( outF, comm);

                metricsBefore.print( outF );
                std::cout<< "Output information written to file " << settings.outFile << std::endl;
            }else{
                std::cout<< "Could not open file " << settings.outFile << " information not stored"<< std::endl;
            }
        }
*/        
    }

    //-----------------------------------------
    //
    // if distribution do not agree, redistribute

    const scai::dmemo::DistributionPtr graphDist = graph.getRowDistributionPtr();

    bool willRedistribute = false;

    if( not coords[0].getDistributionPtr()->isEqual(*graphDist) ){
        PRINT0("Coordinate and graph distribution do not agree; will redistribute input");
        willRedistribute = true;
    }
    if( not nodeWeights[0].getDistributionPtr()->isEqual(*graphDist) ){
        PRINT0("nodeWeights and graph distribution do not agree; will redistribute input");
        willRedistribute = true;        
    }
    if( not partition.getDistributionPtr()->isEqual(*graphDist) ){
        PRINT0("nodeWeights and graph distribution do not agree; will redistribute input");
        willRedistribute = true;        
    }

comm->synchronize();
for( int p=0; p<comm->getSize(); p++){
    if( comm->getRank()==p)
        PRINT( *graphDist );
}


    if( willRedistribute ){
        bool renumberPEs = true;
        
PRINT0("will redistribute input");

        //redistribute
        scai::dmemo::DistributionPtr distFromPart = aux<IndexType,ValueType>::redistributeFromPartition(
            partition, graph, coords, nodeWeights, settings, false, renumberPEs);        

        //TODO?: can also redistribute everything based on a block or genBlock distribution
        //const  scai::dmemo::DistributionPtr newGenBlockDist = GraphUtils<IndexType, ValueType>::genBlockRedist(graph);
        //partition.redistribute( newGenBlockDist );
    }
    //scai::dmemo::DistributionPtr distFromPart = aux<IndexType,ValueType>::redistributeFromPartition( partition, graph , coords, nodeWeights, settings, false, true); 

const scai::dmemo::DistributionPtr newGraphDist = graph.getRowDistributionPtr();
comm->synchronize();
for( int p=0; p<comm->getSize(); p++){
    if( comm->getRank()==p)
        PRINT( *newGraphDist );
}

    PRINT0("\tStarting metis refinement\n");

    //-----------------------------------------
    //
    //refine partition using metisRefine

    DenseVector<IndexType> refinedPartition = Wrappers<IndexType,ValueType>::refine( graph, coords, nodeWeights, partition, settings );

    Metrics<ValueType> metricsRefined(settings);
    metricsRefined.getAllMetrics( graph, refinedPartition, nodeWeights, settings );
    if (comm->getRank() == 0 && settings.metricsDetail.compare("no") != 0) {
        metricsRefined.print( std::cout );
    }

    //this is needed for supermuc
    std::exit(0);

    return 0;    

}