/**
 * @file analyzePartition.cpp
 *
 * A standalone executable to analyze an existing partition stored in a file. Can get
 */

#include <cxxopts.hpp>

#include "../src/FileIO.h"
#include "../src/Metrics.h"
#include "../src/GraphUtils.h"
#include "../src/Settings.h"
#include "../src/parseArgs.h"
#include "../src/AuxiliaryFunctions.h"

using ITI::Settings;
using ITI::IndexType;
using ITI::version;
using ITI::Metrics;
using scai::lama::DenseVector;


int main(int argc, char** argv) {
	typedef double ValueType;   //use double
	
    using namespace cxxopts;

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType rank = comm->getRank();

    cxxopts::Options options = ITI::populateOptions();
    options.add_options()
    ("PEgraphFormat", "the file format of the PEgraph", value<ITI::Format>());    
    cxxopts::ParseResult vm = options.parse(argc, argv);
    Settings settings = ITI::interpretSettings(vm);



    if (vm.count("help")) {
        if(rank==0){
            std::cout << options.help() << "\n";
            std::cout << "Example of usage:\n>./tools/analyze --graphFile meshes/rotation-00000.graph --partition tmp/rotation_k8.part --metricsDetail=mappingALL --PEgraphFile tmp/rotation_k8.PEgraph" << std::endl;
            std::cout << ">mpirun -np 4 ./tools/analyze --graphFile meshes/rotation-00000.graph --partition tmp/rotation_k8.part --metricsDetail=easy" << std::endl;
            std::cout << ">mpirun -np 4 ./tools/analyze --graphFile meshes/rotation-00000.graph --partition tmp/rotation_k8.part --metricsDetail=easy --PEgraphFile tmp/rotation_k8.PEgraph --PEgraphFormat METIS" << std::endl;
        }
        return 0;
    }

    if (vm.count("version")) {
        if(rank==0)
            std::cout << "Git commit " << ITI::version << std::endl;
        return 0;
    }

    if (!vm.count("graphFile")) {
        if(rank==0)
            std::cout << "ERROR: Graph file needed." << std::endl;
        return -1;
    }

    if (!vm.count("partition")) {
        if(rank==0)
            std::cout << "ERROR: Partition needed." << std::endl;
        return -1;
    }

    std::string graphFile = vm["graphFile"].as<std::string>();
    std::string coordFile;

    if (vm.count("coordFile")) {
        coordFile = vm["coordFile"].as<std::string>();
    } else {
        coordFile = graphFile + ".xyz";
    }

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    std::vector<DenseVector<ValueType>>  nodeWeights;
    graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, comm, settings.fileFormat );

    IndexType N = graph.getNumRows();
    scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));

    // set the node weights
    IndexType numReadNodeWeights = nodeWeights.size();
    if (numReadNodeWeights == 0) {
        nodeWeights.resize(1);
        nodeWeights[0] = scai::lama::fill<DenseVector<ValueType>>(rowDistPtr, 1);
    }

    std::vector<std::vector<ValueType>> blockSizes;
    std::string blockSizesFile;

    if( vm.count("blockSizesFile") ) {
        std::string blockSizesFile = vm["blockSizesFile"].as<std::string>();
        blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        for (IndexType i = 0; i < nodeWeights.size(); i++) {
            const ValueType blockSizesSum  = std::accumulate( blockSizes[i].begin(), blockSizes[i].end(), 0);
            const ValueType nodeWeightsSum = nodeWeights[i].sum();
            SCAI_ASSERT_GE( blockSizesSum, nodeWeightsSum, "The block sizes provided are not enough to fit the total weight of the input" );
        }
    }

    //read partition from file
    std::string partname = vm["partition"].as<std::string>();
    DenseVector<IndexType> part = ITI::FileIO<IndexType, ValueType>::readPartition(partname, N);
    settings.numBlocks = part.max() +1;

    SCAI_ASSERT_EQ_ERROR( part.getDistributionPtr()->getLocalSize(), nodeWeights[0].getDistributionPtr()->getLocalSize(), "Wrong distributions" );

    if( vm.count("numBlocks") ){
        std::cout << "## Warning, numBlocks option was given but is ignored. The number of blocks is inferred by the provided partition and is " << settings.numBlocks << std::endl;
    }

    if (part.min() != 0) {
        throw std::runtime_error("Illegal minimum block ID in partition:" + std::to_string(part.min()));
    }

    //calculate metrics

    Metrics<ValueType> metrics(settings);

    if (settings.numBlocks != comm->getSize() and comm->getRank() == 0) {
        std::cout<<"WARNING: the number of block in the partition and the number of mpi processes differ. Some metrics will not be calculated." << std::endl;
    }
    
    if( settings.metricsDetail=="mappingALL" ) {
        settings.metricsDetail="all";
        metrics.getMetrics(graph, part, nodeWeights, settings );
        settings.metricsDetail="mapping";
        //abuse fileFormat for the PEgraph format
        if( vm.count("PEgraphFormat"))
            settings.fileFormat= vm["PEgraphFormat"].as<ITI::Format>();

        metrics.getMetrics(graph, part, nodeWeights, settings );
        //metrics.getMappingMetrics( graph, part, PEgraph );
        //metrics.getAllMetrics( graph, part, nodeWeights, settings );
    }else{
        metrics.getMetrics(graph, part, nodeWeights, settings );
    }

    if (comm->getRank() == 0) {
        metrics.print(std::cout);//TODO: adapt this
    }
}
