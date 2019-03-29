#include "Settings.h"

//#include <boost/algorithm/string.hpp>

std::istream& operator>>(std::istream& in, InitialPartitioningMethods& method){
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



using namespace boost::program_options;

variables_map Settings::parseInput(int argc, char** argv){

	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	options_description desc("Supported options");

	desc.add_options()
				("help", "display options")
				("version", "show version")
				//input and coordinates
				("graphFile", value<std::string>(), "read graph from file")
				("quadTreeFile", value<std::string>(), "read QuadTree from file")
				("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
				("fileFormat", value<ITI::Format>(&fileFormat)->default_value(fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
				("coordFormat", value<ITI::Format>(&coordFormat), "format of coordinate file: AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4 ")
				("PEgraphFile", value<std::string>(&PEGraphFile), "read communication graph from file")
				("numNodeWeights", value<IndexType>(&numNodeWeights), "Number of node weights to use. If the input graph contains more node weights, only the first ones are used.")
				("useDiffusionCoordinates", value<bool>(&useDiffusionCoordinates)->default_value(useDiffusionCoordinates), "Use coordinates based from diffusive systems instead of loading from file")
				("dimensions", value<IndexType>(&dimensions)->default_value(dimensions), "Number of dimensions of generated graph")
				("previousPartition", value<std::string>(), "file of previous partition, used for repartitioning")
				//mesh generation
				("generate", "generate random graph. Currently, only uniform meshes are supported.")
				("numX", value<IndexType>(&numX), "Number of points in x dimension of generated graph")
				("numY", value<IndexType>(&numY), "Number of points in y dimension of generated graph")
				("numZ", value<IndexType>(&numZ), "Number of points in z dimension of generated graph")
				//general partitioning parameters
				("numBlocks", value<IndexType>(&numBlocks)->default_value(comm->getSize()), "Number of blocks, default is number of processes")
				("epsilon", value<double>(&epsilon)->default_value(epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
				("blockSizesFile", value<std::string>(&blockSizesFile) , " file to read the block sizes for every block")
				("seed", value<double>()->default_value(time(NULL)), "random seed, default is current time")
				//multi-level and local refinement
				("initialPartition", value<InitialPartitioningMethods>(&initialPartition), "Choose initial partitioning method between space-filling curves ('SFC' or 0), pixel grid coarsening ('Pixel' or 1), spectral partition ('Spectral' or 2), k-means ('K-Means' or 3) and multisection ('MultiSection' or 4). SFC, Spectral and K-Means are most stable.")
				("noRefinement", "skip local refinement steps")
				("multiLevelRounds", value<IndexType>(&multiLevelRounds)->default_value(multiLevelRounds), "Tuning Parameter: How many multi-level rounds with coarsening to perform")
				("minBorderNodes", value<IndexType>(&minBorderNodes)->default_value(minBorderNodes), "Tuning parameter: Minimum number of border nodes used in each refinement step")
				("stopAfterNoGainRounds", value<IndexType>(&stopAfterNoGainRounds)->default_value(stopAfterNoGainRounds), "Tuning parameter: Number of rounds without gain after which to abort localFM. A value of 0 means no stopping.")
				("minGainForNextGlobalRound", value<IndexType>(&minGainForNextRound)->default_value(minGainForNextRound), "Tuning parameter: Minimum Gain above which the next global FM round is started")
				("gainOverBalance", value<bool>(&gainOverBalance)->default_value(gainOverBalance), "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance")
				("useDiffusionTieBreaking", value<bool>(&useDiffusionTieBreaking)->default_value(useDiffusionTieBreaking), "Tuning Parameter: Use diffusion to break ties in Fiduccia-Mattheyes algorithm")
				("useGeometricTieBreaking", value<bool>(&useGeometricTieBreaking)->default_value(useGeometricTieBreaking), "Tuning Parameter: Use distances to block center for tie breaking")
				("skipNoGainColors", value<bool>(&skipNoGainColors)->default_value(skipNoGainColors), "Tuning Parameter: Skip Colors that didn't result in a gain in the last global round")
				("mec", "Use the MEC algorithm for the edge coloring of the PE graphFile instead of the classical boost algorithm" )
				//multisection
				("bisect", value<bool>(&bisect)->default_value(bisect), "Used for the multisection method. If set to true the algorithm perfoms bisections (not multisection) until the desired number of parts is reached")
				("cutsPerDim", value<std::vector<IndexType>>(&cutsPerDim)->multitoken(), "If MultiSection is chosen, then provide d values that define the number of cuts per dimension.")
				("pixeledSideLen", value<IndexType>(&pixeledSideLen)->default_value(pixeledSideLen), "The resolution for the pixeled partition or the spectral")
				// K-Means
				("minSamplingNodes", value<IndexType>(&minSamplingNodes)->default_value(minSamplingNodes), "Tuning parameter for K-Means")
				("influenceExponent", value<double>(&influenceExponent)->default_value(influenceExponent), "Tuning parameter for K-Means, default is ")
				("influenceChangeCap", value<double>(&influenceChangeCap)->default_value(influenceChangeCap), "Tuning parameter for K-Means")
				("balanceIterations", value<IndexType>(&balanceIterations)->default_value(balanceIterations), "Tuning parameter for K-Means")
				("maxKMeansIterations", value<IndexType>(&maxKMeansIterations)->default_value(maxKMeansIterations), "Tuning parameter for K-Means")
				("tightenBounds", "Tuning parameter for K-Means")
				("erodeInfluence", "Tuning parameter for K-Means, in case of large deltas and imbalances.")
				("initialMigration", value<InitialPartitioningMethods>(&initialMigration)->default_value(initialMigration), "Choose a method to get the first migration, 0: SFCs, 3:k-means, 4:Multisection")
				//("manhattanDistance", "Tuning parameter for K-Means")
				//output
				("outFile", value<std::string>(&outFile), "write result partition into file")
				//debug
				("writeDebugCoordinates", value<bool>(&writeDebugCoordinates)->default_value(writeDebugCoordinates), "Write Coordinates of nodes in each block")
				("verbose", "Increase output.")
                ("storeInfo", "Store timing and other metrics in file.")
                ("writePartition", "Writes the partition in the outFile.partition file")
                // evaluation
                ("repeatTimes", value<IndexType>(&repeatTimes), "How many times we repeat the partitioning process.")
                ("noComputeDiameter", "Compute diameter of resulting block files.")
                ("maxDiameterRounds", value<IndexType>(&maxDiameterRounds)->default_value(maxDiameterRounds), "abort diameter algorithm after that many BFS rounds")
				("metricsDetail", value<std::string>(&metricsDetail)->default_value("no"), "no: no metrics, easy:cut, imbalance, communication volume and diameter if possible, all: easy + SpMV time and communication time in SpMV")
				//used for the competitors main
				("outDir", value<std::string>(&outDir), "write result partition into file")
				//("tool", value<std::string>(&tool), "The tool to partition with.")
				("tools", value<std::vector<std::string>>(&tools)->multitoken(), "The tool to partition with.")
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

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}

	if (vm.count("version")) {
		std::cout << "Git commit " << version << std::endl;
		return 0;
	}

	if (vm.count("generate") + vm.count("graphFile") + vm.count("quadTreeFile") != 1) {
		std::cout << "Pick one of --graphFile, --quadTreeFile or --generate. Use --help for more parameters" << std::endl;
		isValid = false;
		//return 126;
	}

	if (vm.count("generate") && (vm["dimensions"].as<IndexType>() != 3)) {
		std::cout << "Mesh generation currently only supported for three dimensions" << std::endl;
		isValid = false;
		//return 126;
	}

	if (vm.count("coordFile") && vm.count("useDiffusionCoords")) {
		std::cout << "Cannot both load coordinates from file with --coordFile or generate them with --useDiffusionCoords." << std::endl;
		isValid = false;
		//return 126;
	}

	if( vm.count("cutsPerDim") ) {
		SCAI_ASSERT( !cutsPerDim.empty(), "options cutsPerDim was given but the vector is empty" );
		SCAI_ASSERT_EQ_ERROR(cutsPerDim.size(), dimensions, "cutsPerDime: user must specify d values for mutlisection using option --cutsPerDim. e.g.: --cutsPerDim=4,20 for a partition in 80 parts/" );
	}
        
	if( vm.count("initialMigration") ){

		if( !(initialMigration==InitialPartitioningMethods::SFC
				or initialMigration==InitialPartitioningMethods::KMeans
				or initialMigration==InitialPartitioningMethods::Multisection
				or initialMigration==InitialPartitioningMethods::None) ){
			PRINT0("Initial migration supported only for 0:SFCs, 3:k-means, 4:MultiSection or 5:None, invalid option " << initialMigration << " was given");
			isValid = false;
			//return 126;
		}
	}

	if (vm.count("fileFormat") && fileFormat == ITI::Format::TEEC) {
		if (!vm.count("numX")) {
			std::cout << "TEEC file format does not specify graph size, please set with --numX" << std::endl;
			isValid = false;
			//return 126;
		}
	}

	// check if coordFormat is provided
	// if no coordFormat was given but was given a fileFormat assume they are the same
	if( !vm.count("coordFormat") and vm.count("fileFormat") ){
		coordFormat = fileFormat;
	}

	if (vm.count("previousPartition")) {
		repartition = true;
		if (vm.count("initialPartition")) {
			if (!(initialPartition == InitialPartitioningMethods::KMeans || initialPartition == InitialPartitioningMethods::None)) {
				std::cout << "Method " << initialPartition << " not supported for repartitioning, currently only kMeans." << std::endl;
				isValid = false;
				//return 126;
			}
		} else {
			if (comm->getRank() == 0) {
				std::cout << "Setting initial partitioning method to kMeans." << std::endl;
			}
			initialPartition = InitialPartitioningMethods::KMeans;
		}
	}
	
	if( storeInfo && outFile=="-" ) {
		if (comm->getRank() == 0) {
			std::cout<< "Option to store information used but no output file given to write to. Specify an output file using the option --outFile. Aborting." << std::endl;
		}
		isValid = false;
		//return 126;
	}

	if (!vm.count("influenceExponent")) {
	    influenceExponent = 1.0/dimensions;
	}

	if (vm.count("manhattanDistance")) {
		throw std::logic_error("Manhattan distance not yet implemented");
	}

	if(vm.count("mec")){
		mec = true;
	}

    if( vm.count("metricsDetail") ){
		if( not (metricsDetail=="no" or metricsDetail=="easy" or metricsDetail=="all") ){
			if(comm->getRank() ==0 ){
				std::cout<<"WARNING: wrong value for parameter metricsDetail= " << metricsDetail << ". Setting to all" <<std::endl;
				metricsDetail="all";
			}
		}
	}else{
		metricsDetail = "easy";
	}	

	if( vm.count("noComputeDiameter") ){
		computeDiameter = false;
	}else{
		computeDiameter = true;
	}
	
    if( vm.count("writePartition") ){
        writeInFile = true;
    }
   
    char machineChar[255];
    std::string machine;
    gethostname(machineChar, 255);
    
    machine = std::string(machineChar);
        
    verbose = vm.count("verbose");
    storeInfo = vm.count("storeInfo");
    erodeInfluence = vm.count("erodeInfluence");
    tightenBounds = vm.count("tightenBounds");
    manhattanDistance = vm.count("manhattanDistance");
	noRefinement = vm.count("noRefinement");

    srand(vm["seed"].as<double>());

	return vm;
}