/**
 * @file analyzePartition.cpp
 *
 * A standalone executable to analyze an existing partition stored in a file.
 */

#include <cxxopts.hpp>

#include "../src/FileIO.h"
#include "../src/Metrics.h"
#include "../src/GraphUtils.h"
#include "../src/Settings.h"

using ITI::Settings;
using ITI::IndexType;
using ITI::version;
using ITI::Metrics;
using scai::lama::DenseVector;

using ValueType = double; //hardcode to double


int main(int argc, char** argv) {
    using namespace cxxopts;
    cxxopts::Options options("analyze", "Analyzing existing partitions");

    struct Settings settings;

    std::string blockSizesFile;

    options.add_options()
    ("help", "display options")
    ("version", "show version")
    //input and coordinates
    ("graphFile", "read graph from file", value<std::string>())
    ("fileFormat", "The format of the file to read, available are AUTO, METIS, ADCRIC and MatrixMarket format. See FileIO.h for more details.", value<ITI::Format>())
    ("dimensions", "Number of dimensions of generated graph", value<IndexType>()->default_value(std::to_string(settings.dimensions)))
    ("partition", "file of partition", value<std::string>())
    ("blockSizesFile", " file to read the block sizes for every block", value<std::string>() )
    ("computeDiameter", "Compute Diameter of resulting block files.")
    ("maxDiameterRounds", "abort diameter algorithm after that many BFS rounds", value<IndexType>()->default_value(std::to_string(settings.maxDiameterRounds)))
    ;


    cxxopts::ParseResult vm = options.parse(argc, argv);

    if (vm.count("help")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    if (vm.count("version")) {
        std::cout << "Git commit " << ITI::version << std::endl;
        return 0;
    }

    if (!vm.count("graphFile")) {
        std::cout << "Graph file needed." << std::endl;
        return 126;
    }

    if (!vm.count("partition")) {
        std::cout << "Partition needed." << std::endl;
        return 126;
    }

    settings.computeDiameter = vm.count("computeDiameter");

    std::string graphFile = vm["graphFile"].as<std::string>();
    settings.fileName = graphFile;
    std::string coordFile;
    if (vm.count("coordFile")) {
        coordFile = vm["coordFile"].as<std::string>();
    } else {
        coordFile = graphFile + ".xyz";
    }

    if (vm.count("fileFormat")) {
        settings.fileFormat = vm["fileFormat"].as<ITI::Format>();
    }

    if (vm.count("maxDiameterRounds")) {
        settings.maxDiameterRounds = vm["maxDiameterRounds"].as<IndexType>();
    }

    if (vm.count("dimensions")) {
        settings.dimensions = vm["dimensions"].as<IndexType>();
    }

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    //std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    std::vector<DenseVector<ValueType>>  nodeWeights;
    graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, comm, settings.fileFormat );

    IndexType N = graph.getNumRows();
    scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));

    std::vector<std::vector<ValueType>> blockSizes;

    if( vm.count("blockSizesFile") ) {
        std::string blockSizesFile = vm["blockSizesFile"].as<std::string>();
        blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        for (IndexType i = 0; i < nodeWeights.size(); i++) {
            const ValueType blockSizesSum  = std::accumulate( blockSizes[i].begin(), blockSizes[i].end(), 0);
            const ValueType nodeWeightsSum = nodeWeights[i].sum();
            SCAI_ASSERT_GE( blockSizesSum, nodeWeightsSum, "The block sizes provided are not enough to fit the total weight of the input" );
        }
    }

    std::string partname = vm["partition"].as<std::string>();
    DenseVector<IndexType> part = ITI::FileIO<IndexType, ValueType>::readPartition(partname, N);
    settings.numBlocks = part.max() +1;

    if (part.min() != 0) {
        throw std::runtime_error("Illegal minimum block ID in partition:" + std::to_string(part.min()));
    }

    if (settings.computeDiameter) {
        if (comm->getSize() != settings.numBlocks) {
            if (comm->getRank() == 0) {
                std::cout << "Can only compute diameter if number of processes is equal to number of blocks." << std::endl;
            }
        } else {
            scai::dmemo::DistributionPtr newDist = scai::dmemo::generalDistributionByNewOwners(part.getDistribution(), part.getLocalValues());
            graph.redistribute(newDist, noDistPtr);
            part.redistribute(newDist);
        }
    }

    Metrics<ValueType> metrics(settings);
    metrics.getEasyMetrics( graph, part, nodeWeights, settings );

    if (comm->getRank() == 0) {
        metrics.print(std::cout);//TODO: adapt this
    }
}
