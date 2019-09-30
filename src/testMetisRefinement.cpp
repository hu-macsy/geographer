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
#include "parseArgs.h"


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

    std::string graphFile;

    if (vm.count("graphFile")) {

    }else{
        std::cout << "No input file was given. Call again with --graphFile, --quadTreeFile" << std::endl;
        return 126;        
    }

    //partition input using some tool

    //refine partition using metisRefine

}