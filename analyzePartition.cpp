#include <boost/program_options.hpp>

#include "FileIO.h"
#include "Metrics.h"
#include "GraphUtils.h"
#include "Settings.h"

namespace ITI {
	std::istream& operator>>(std::istream& in, Format& format)
	{
		std::string token;
		in >> token;
		if (token == "AUTO" or token == "0")
			format = ITI::Format::AUTO ;
		else if (token == "METIS" or token == "1")
			format = ITI::Format::METIS;
		else if (token == "ADCIRC" or token == "2")
			format = ITI::Format::ADCIRC;
		else if (token == "OCEAN" or token == "3")
			format = ITI::Format::OCEAN;
        else if (token == "MATRIXMARKET" or token == "4")
			format = ITI::Format::MATRIXMARKET;
		else if (token == "TEEC" or token == "5")
			format = ITI::Format::TEEC;
        else if (token == "BINARY" or token == "6")
			format = ITI::Format::BINARY;
		else
			in.setstate(std::ios_base::failbit);
		return in;
	}

	std::ostream& operator<<(std::ostream& out, Format method)
	{
		std::string token;

		if (method == ITI::Format::AUTO)
			token = "AUTO";
		else if (method == ITI::Format::METIS)
			token = "METIS";
		else if (method == ITI::Format::ADCIRC)
			token = "ADCIRC";
		else if (method == ITI::Format::OCEAN)
			token = "OCEAN";
		else if (method == ITI::Format::MATRIXMARKET)
			token = "MATRIXMARKET";
		else if (method == ITI::Format::TEEC)
			token = "TEEC";
        else if (method == ITI::Format::BINARY)
            token == "BINARY";
		out << token;
		return out;
	}
}


std::istream& operator>>(std::istream& in, InitialPartitioningMethods& method)
{
    std::string token;
    in >> token;
    if (token == "SFC" or token == "0")
        method = InitialPartitioningMethods::SFC;
    else if (token == "Pixel" or token == "1")
        method = InitialPartitioningMethods::Pixel;
    else if (token == "Spectral" or token == "2")
    	method = InitialPartitioningMethods::Spectral;
    else if (token == "KMeans" or token == "Kmeans" or token == "K-Means" or token == "K-means" or token == "3")
        method = InitialPartitioningMethods::KMeans;
    else if (token == "Multisection" or token == "MultiSection" or token == "4")
    	method = InitialPartitioningMethods::Multisection;
    else if (token == "None" or token == "5")
        	method = InitialPartitioningMethods::None;
    else
        in.setstate(std::ios_base::failbit);
    return in;
}

std::ostream& operator<<(std::ostream& out, InitialPartitioningMethods method)
{
    std::string token;

    if (method == InitialPartitioningMethods::SFC)
        token = "SFC";
    else if (method == InitialPartitioningMethods::Pixel)
    	token = "Pixel";
    else if (method == InitialPartitioningMethods::Spectral)
    	token = "Spectral";
    else if (method == InitialPartitioningMethods::KMeans)
        token = "KMeans";
    else if (method == InitialPartitioningMethods::Multisection)
    	token = "Multisection";
    else if (method == InitialPartitioningMethods::None)
        token = "None";
    out << token;
    return out;
}

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
				("nodeWeightIndex", value<int>()->default_value(0), "index of node weight")
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
    std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph
    
    DenseVector<ValueType> nodeWeights;

	std::vector<DenseVector<ValueType> > vectorOfNodeWeights;
	graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights, settings.fileFormat );

	IndexType N = graph.getNumRows();
    scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));

    if( vm.count("blockSizesFile") ){
        settings.blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        IndexType blockSizesSum  = std::accumulate( settings.blockSizes.begin(), settings.blockSizes.end(), 0);
        IndexType nodeWeightsSum = nodeWeights.sum().Scalar::getValue<IndexType>();
        SCAI_ASSERT_GE( blockSizesSum, nodeWeightsSum, "The block sizes provided are not enough to fit the total weight of the input" );
    }

    std::string partname = vm["partition"].as<std::string>();
    DenseVector<IndexType> part = ITI::FileIO<IndexType, ValueType>::readPartition(partname, N);
    settings.numBlocks = part.max().Scalar::getValue<IndexType>() +1;

    if (part.min().Scalar::getValue<IndexType>() != 0) {
    	throw std::runtime_error("Illegal minimum block ID in partition:" + std::to_string(part.min().Scalar::getValue<IndexType>()));
    }

    Metrics metrics(1);
    metrics.getMetrics( graph, part, nodeWeights, settings );

    if (settings.computeDiameter) {
    	scai::dmemo::CommunicatorPtr comm = rowDistPtr->getCommunicatorPtr();

    	if (comm->getSize() != settings.numBlocks) {
    		std::cout << "Can only compute diameter if number of processes is equal to number of blocks." << std::endl;
    	} else {
    		scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution(N, part.getLocalValues(), comm));
    		graph.redistribute(newDist, noDistPtr);
    		part.redistribute(newDist);

    		IndexType localN = newDist->getLocalSize();
    		IndexType maxRounds = settings.maxDiameterRounds;
            if (maxRounds < 0) {
                maxRounds = localN;
            }

            IndexType localDiameter = ITI::GraphUtils::getLocalBlockDiameter<IndexType, ValueType>(graph, localN/2, 0, 0, maxRounds);
            metrics.maxBlockDiameter = comm->max(localDiameter);
            metrics.avgBlockDiameter = comm->sum(localDiameter) / comm->getSize();
    	}
    }
    
    metrics.print(std::cout);//TODO: adapt this
}