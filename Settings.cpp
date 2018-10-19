#include "Settings.h"

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

boost::variables_map Settings::parseInput(){

	boost::options_description desc("Supported options");

	desc.add_options()
				("help", "display options")
				("version", "show version")
				//input and coordinates
				("graphFile", value<std::string>(), "read graph from file")
				("quadTreeFile", value<std::string>(), "read QuadTree from file")
				("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
				("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
				/*
				("coordFormat", value<ITI::Format>(&coordFormat), "format of coordinate file: AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4 ")
				("nodeWeightIndex", value<int>()->default_value(0), "index of node weight")
				("useDiffusionCoordinates", value<bool>(&settings.useDiffusionCoordinates)->default_value(settings.useDiffusionCoordinates), "Use coordinates based from diffusive systems instead of loading from file")
				("dimensions", value<IndexType>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
				("previousPartition", value<std::string>(), "file of previous partition, used for repartitioning")
				//mesh generation
				("generate", "generate random graph. Currently, only uniform meshes are supported.")
				("numX", value<IndexType>(&settings.numX), "Number of points in x dimension of generated graph")
				("numY", value<IndexType>(&settings.numY), "Number of points in y dimension of generated graph")
				("numZ", value<IndexType>(&settings.numZ), "Number of points in z dimension of generated graph")
				//general partitioning parameters
				("numBlocks", value<IndexType>(&settings.numBlocks)->default_value(comm->getSize()), "Number of blocks, default is number of processes")
				("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
				("blockSizesFile", value<std::string>(&blockSizesFile) , " file to read the block sizes for every block")
				("seed", value<double>()->default_value(time(NULL)), "random seed, default is current time")
				//multi-level and local refinement
				("initialPartition", value<InitialPartitioningMethods>(&settings.initialPartition), "Choose initial partitioning method between space-filling curves ('SFC' or 0), pixel grid coarsening ('Pixel' or 1), spectral partition ('Spectral' or 2), k-means ('K-Means' or 3) and multisection ('MultiSection' or 4). SFC, Spectral and K-Means are most stable.")
				("noRefinement", "skip local refinement steps")
				("multiLevelRounds", value<IndexType>(&settings.multiLevelRounds)->default_value(settings.multiLevelRounds), "Tuning Parameter: How many multi-level rounds with coarsening to perform")
				("minBorderNodes", value<IndexType>(&settings.minBorderNodes)->default_value(settings.minBorderNodes), "Tuning parameter: Minimum number of border nodes used in each refinement step")
				("stopAfterNoGainRounds", value<IndexType>(&settings.stopAfterNoGainRounds)->default_value(settings.stopAfterNoGainRounds), "Tuning parameter: Number of rounds without gain after which to abort localFM. A value of 0 means no stopping.")
				("minGainForNextGlobalRound", value<IndexType>(&settings.minGainForNextRound)->default_value(settings.minGainForNextRound), "Tuning parameter: Minimum Gain above which the next global FM round is started")
				("gainOverBalance", value<bool>(&settings.gainOverBalance)->default_value(settings.gainOverBalance), "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance")
				("useDiffusionTieBreaking", value<bool>(&settings.useDiffusionTieBreaking)->default_value(settings.useDiffusionTieBreaking), "Tuning Parameter: Use diffusion to break ties in Fiduccia-Mattheyes algorithm")
				("useGeometricTieBreaking", value<bool>(&settings.useGeometricTieBreaking)->default_value(settings.useGeometricTieBreaking), "Tuning Parameter: Use distances to block center for tie breaking")
				("skipNoGainColors", value<bool>(&settings.skipNoGainColors)->default_value(settings.skipNoGainColors), "Tuning Parameter: Skip Colors that didn't result in a gain in the last global round")
				("mec", "Use the MEC algorithm for the edge coloring of the PE graph instead of the classical boost algorithm" )
				//multisection
				("bisect", value<bool>(&settings.bisect)->default_value(settings.bisect), "Used for the multisection method. If set to true the algorithm perfoms bisections (not multisection) until the desired number of parts is reached")
				("cutsPerDim", value<std::vector<IndexType>>(&settings.cutsPerDim)->multitoken(), "If MultiSection is chosen, then provide d values that define the number of cuts per dimension.")
				("pixeledSideLen", value<IndexType>(&settings.pixeledSideLen)->default_value(settings.pixeledSideLen), "The resolution for the pixeled partition or the spectral")
				// K-Means
				("minSamplingNodes", value<IndexType>(&settings.minSamplingNodes)->default_value(settings.minSamplingNodes), "Tuning parameter for K-Means")
				("influenceExponent", value<double>(&settings.influenceExponent)->default_value(settings.influenceExponent), "Tuning parameter for K-Means, default is ")
				("influenceChangeCap", value<double>(&settings.influenceChangeCap)->default_value(settings.influenceChangeCap), "Tuning parameter for K-Means")
				("balanceIterations", value<IndexType>(&settings.balanceIterations)->default_value(settings.balanceIterations), "Tuning parameter for K-Means")
				("maxKMeansIterations", value<IndexType>(&settings.maxKMeansIterations)->default_value(settings.maxKMeansIterations), "Tuning parameter for K-Means")
				("tightenBounds", "Tuning parameter for K-Means")
				("erodeInfluence", "Tuning parameter for K-Means, in case of large deltas and imbalances.")
				("initialMigration", value<InitialPartitioningMethods>(&settings.initialMigration)->default_value(settings.initialMigration), "Choose a method to get the first migration, 0: SFCs, 3:k-means, 4:Multisection")
				("manhattanDistance", "Tuning parameter for K-Means")
				//output
				("outFile", value<std::string>(&settings.outFile), "write result partition into file")
				//debug
				("writeDebugCoordinates", value<bool>(&settings.writeDebugCoordinates)->default_value(settings.writeDebugCoordinates), "Write Coordinates of nodes in each block")
				("verbose", "Increase output.")
                ("storeInfo", "Store timing and other metrics in file.")
                ("writePartition", "Writes the partition in the outFile.partition file")
                // evaluation
                ("repeatTimes", value<IndexType>(&repeatTimes), "How many times we repeat the partitioning process.")
                ("noComputeDiameter", "Compute Diameter of resulting block files.")
                ("maxDiameterRounds", value<IndexType>(&settings.maxDiameterRounds)->default_value(settings.maxDiameterRounds), "abort diameter algorithm after that many BFS rounds")
				("metricsDetail", value<std::string>(&metricsDetail)->default_value("no"), "no: no metrics, easy:cut, imbalance, communication volume and diameter if possible, all: easy + SpMV time and communication time in SpMV")
				*/
				;

    //------------------------------------------------
    //
    // checks
    //
                            
    std::string s = "0.12345";
    ValueType stdDouble = std::stod( s );
    ValueType boostDouble = boost::lexical_cast<ValueType>(s);
    if( stdDouble!=boostDouble ){
        PRINT0( "\033[1;31mWARNING: std::stod and boost::lexical_cast do not agree \033[0m"  );
        PRINT0( "\033[1;31mWARNING: std::stod and boost::lexical_cast do not agree \033[0m"  );
    }

    variables_map vm;
	store(command_line_parser(argc, argv).options(desc).run(), vm);
	notify(vm);

	/*
	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}

	if (vm.count("version")) {
		std::cout << "Git commit " << version << std::endl;
		return 0;
	}

	if (vm.count("generate") + vm.count("graphFile") + vm.count("quadTreeFile") != 1) {
		std::cout << "Pick one of --graphFile, --quadTreeFile or --generate" << std::endl;
		return 126;
	}

	if (vm.count("generate") && (vm["dimensions"].as<IndexType>() != 3)) {
		std::cout << "Mesh generation currently only supported for three dimensions" << std::endl;
		return 126;
	}

	if (vm.count("coordFile") && vm.count("useDiffusionCoords")) {
		std::cout << "Cannot both load coordinates from file with --coordFile or generate them with --useDiffusionCoords." << std::endl;
		return 126;
	}

	if( vm.count("cutsPerDim") ) {
		SCAI_ASSERT( !settings.cutsPerDim.empty(), "options cutsPerDim was given but the vector is empty" );
		SCAI_ASSERT_EQ_ERROR(settings.cutsPerDim.size(), settings.dimensions, "cutsPerDime: user must specify d values for mutlisection using option --cutsPerDim. e.g.: --cutsPerDim=4,20 for a partition in 80 parts/" );
	}
        
	if( vm.count("initialMigration") ){

		if( !(settings.initialMigration==InitialPartitioningMethods::SFC
				or settings.initialMigration==InitialPartitioningMethods::KMeans
				or settings.initialMigration==InitialPartitioningMethods::Multisection
				or settings.initialMigration==InitialPartitioningMethods::None) ){
			PRINT0("Initial migration supported only for 0:SFCs, 3:k-means, 4:MultiSection or 5:None, invalid option " << settings.initialMigration << " was given");
			return 126;
		}
	}

	if (vm.count("fileFormat") && settings.fileFormat == ITI::Format::TEEC) {
		if (!vm.count("numX")) {
			std::cout << "TEEC file format does not specify graph size, please set with --numX" << std::endl;
			return 126;
		}
	}

	if (vm.count("previousPartition")) {
		settings.repartition = true;
		if (vm.count("initialPartition")) {
			if (!(settings.initialPartition == InitialPartitioningMethods::KMeans || settings.initialPartition == InitialPartitioningMethods::None)) {
				std::cout << "Method " << settings.initialPartition << " not supported for repartitioning, currently only kMeans." << std::endl;
				return 126;
			}
		} else {
			if (comm->getRank() == 0) {
				std::cout << "Setting initial partitioning method to kMeans." << std::endl;
			}
			settings.initialPartition = InitialPartitioningMethods::KMeans;
		}
	}
	
	if( settings.storeInfo && settings.outFile=="-" ) {
		if (comm->getRank() == 0) {
			std::cout<< "Option to store information used but no output file given to write to. Specify an output file using the option --outFile. Aborting." << std::endl;
		}
		return 126;
	}

	if (!vm.count("influenceExponent")) {
	    settings.influenceExponent = 1.0/settings.dimensions;
	}

	if (vm.count("manhattanDistance")) {
		throw std::logic_error("Manhattan distance not yet implemented");
	}

	if(vm.count("mec")){
		settings.mec = true;
	}
	*/

}