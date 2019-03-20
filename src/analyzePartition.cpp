/**
 * @file analyzePartition.cpp
 *
 * A standalone executable to analyze an existing partition stored in a file.
 */

#include <boost/program_options.hpp>

#include "FileIO.h"
#include "Metrics.h"
#include "GraphUtils.h"
#include "Settings.h"

int main(int argc, char** argv) {
	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;

	std::string blockSizesFile;

	desc.add_options()
				("help", "display options")
				("version", "show version")
				//input and coordinates
				("graphFile", value<std::string>(), "read graph from file")
				//("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
				("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
				//("coordFormat", value<ITI::Format>(&coordFormat), "format of coordinate file: AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4 ")
				("dimensions", value<IndexType>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
				("partition", value<std::string>(), "file of partition")
				("blockSizesFile", value<std::string>(&blockSizesFile) , " file to read the block sizes for every block")
				("computeDiameter", "Compute Diameter of resulting block files.")
                ("maxDiameterRounds", value<IndexType>(&settings.maxDiameterRounds)->default_value(settings.maxDiameterRounds), "abort diameter algorithm after that many BFS rounds")
				;


	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).run(), vm);
	notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}

	if (vm.count("version")) {
		std::cout << "Git commit " << version << std::endl;
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

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    //std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph
    
    std::vector<DenseVector<ValueType> >  nodeWeights;
	graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, settings.fileFormat );

	IndexType N = graph.getNumRows();
    scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));

    if( vm.count("blockSizesFile") ){
        settings.blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        for (IndexType i = 0; i < nodeWeights.size(); i++) {
            const ValueType blockSizesSum  = std::accumulate( settings.blockSizes[i].begin(), settings.blockSizes[i].end(), 0);
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

    scai::dmemo::CommunicatorPtr comm = rowDistPtr->getCommunicatorPtr();
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

    Metrics metrics(settings);
    metrics.getEasyMetrics( graph, part, nodeWeights[0], settings );//TODO: adapt metrics for multiple node weights
    
    if (comm->getRank() == 0) {
        metrics.print(std::cout);//TODO: adapt this
    }
}
