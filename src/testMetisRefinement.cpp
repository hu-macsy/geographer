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
    using ValueType = double;   //use double

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

    readInput<ValueType>( vm, settings, comm, graph, coords, nodeWeights );


    //-----------------------------------------
    //
    //partition input using some tool
    
    Metrics<ValueType> metricsBefore(settings);

    DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coords, nodeWeights, settings, metricsBefore );


    metricsBefore.getMetrics( graph, partition, nodeWeights, settings );
    if (comm->getRank() == 0 && settings.metricsDetail.compare("no") != 0) {
        metricsBefore.print( std::cout );

        if( settings.storeInfo && settings.outFile!="-" ){
            std::ofstream outF( settings.outFile, std::ios::out);
            if(outF.is_open()) {
                outF << "Running " << __FILE__ << std::endl;
                settings.print( outF, comm);
                outF << "Metrics after the first partition with " << settings.initialPartition << std::endl;

                metricsBefore.print( outF );
                std::cout<< "Output information written to file " << settings.outFile << std::endl;
            }else{
                std::cout<< "Could not open file " << settings.outFile << " information not stored"<< std::endl;
            }
        }        
    }

    //-----------------------------------------
    //
    //refine partition using metisRefine

    Metrics<ValueType> metrics(settings);

    DenseVector<IndexType> refinedPartition = Wrappers<IndexType,ValueType>::refine( graph, coords, nodeWeights, partition, settings, metrics );
    
    PRINT0("\tFinished metis refinement\n");

    //-----------------------------------------
    //
    // gather and print metrics

    metrics.getMetrics( graph, refinedPartition, nodeWeights, settings );
    if (comm->getRank() == 0 && settings.metricsDetail.compare("no") != 0) {
        metrics.print( std::cout );

        if( settings.storeInfo && settings.outFile!="-" ){
            std::ofstream outF( settings.outFile, std::ios::app);
            if(outF.is_open()) {
                outF << "\n### Metrics after the metis local refinement\n" << std::endl;
                metrics.print( outF );
                std::cout<< "Output information written to file " << settings.outFile << std::endl;
            }else{
                std::cout<< "Could not open file " << settings.outFile << " information not stored"<< std::endl;
            }
        }        
    }

    if (vm.count("callExit")) {
        //this is needed for supermuc
        std::exit(0);
    }

    return 0;    

}