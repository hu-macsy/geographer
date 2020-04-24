#include <scai/lama.hpp>
#include <scai/dmemo/Communicator.hpp>

#include "parseArgs.h"
#include "Settings.h"

using namespace cxxopts;

namespace ITI {

Options populateOptions() {
    cxxopts::Options options("Geographer", "Parallel geometric graph partitioner for load balancing");

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    Settings settings;

	options.add_options()
    ("help", "display options")
    ("version", "show version")
    //main arguments for daily use
    ("graphFile", "read graph from file", value<std::string>())
    ("coordFile", "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz", value<std::string>())
    ("dimensions", "Number of dimensions of input graph", value<IndexType>()->default_value(std::to_string(settings.dimensions)))
    ("numBlocks", "Number of blocks, default is number of processes", value<IndexType>())
    ("epsilon", "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.", value<double>()->default_value(std::to_string(settings.epsilon)))
    // other input specification
    ("fileFormat", "Format of graph file, available are AUTO, METIS, ADCRIC and MatrixMarket format. See Readme.md and src/Settings.h for more details.", value<ITI::Format>())
    ("coordFormat", "format of coordinate file: AUTO, METIS, ADCIRC and MATRIXMARKET. See src/Settings.h for more details.", value<ITI::Format>())
    ("numNodeWeights", "Number of node weights to use. If the input graph contains more node weights, only the first ones are used.", value<IndexType>())
    ("seed", "random seed, default is current time", value<double>()->default_value(std::to_string(time(NULL))))
    //mapping
    ("PEgraphFile", "read communication graph from file", value<std::string>())
    ("topologyFile", "read system topology  from file: a line per processor with CPU, MEM and number of cores", value<std::string>())
    ("blockSizesFile", "file to read the block sizes for every block", value<std::string>() )
    ("autoSetCpuMem", "if set, geographer will gather cpu and memory info and use them to build a heterogeneous communication tree used for partitioning")
    ("processPerNode", "the number of processes per compute node. Is used with autoSetCpuMem to determine the internal cpu/core ID within a compute node and query the cpu frequency.",  value<IndexType>())
    ("mappingRenumbering", "map blocks to PEs using the SFC index of the block's center. This works better when PUs are numbered consecutively." )
    //repartitioning
    ("previousPartition", "file of previous partition, used for repartitioning", value<std::string>())
    //multi-level and local refinement
    ("initialPartition", "Choose initial partitioning method between space-filling curves (geoSFC), balanced k-means (geoKmeans) or the hierarchical version (geoHierKM) and MultiJagged (geoMS). If parmetis or zoltan are installed, you can also choose to partition with them using for example, parMetisGraph or zoltanMJ. For more information, see src/Settings.h file.", value<std::string>())
    ("initialMigration", "The preprocessing step to distribute data before calling the partitioning algorithm", value<std::string>())
    ("noRefinement", "skip local refinement steps")
    ("multiLevelRounds", "Tuning Parameter: How many multi-level rounds with coarsening to perform", value<IndexType>()->default_value(std::to_string(settings.multiLevelRounds)))
    ("minBorderNodes", "Tuning parameter: Minimum number of border nodes used in each refinement step", value<IndexType>())
    ("minBorderNodesPercent", "Tuning parameter: Percentage of local nodes used in each refinement step. Recommended  are values around 0.05", value<double>())
    ("stopAfterNoGainRounds", "Tuning parameter: Number of rounds without gain after which to abort localFM. 0 means no stopping.", value<IndexType>())
    ("minGainForNextGlobalRound", "Tuning parameter: Minimum Gain above which the next global FM round is started", value<IndexType>())
    ("gainOverBalance", "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance", value<bool>())
    ("useDiffusionTieBreaking", "Tuning Parameter: Use diffusion to break ties in Fiduccia-Mattheyes algorithm", value<bool>())
    ("useGeometricTieBreaking", "Tuning Parameter: Use distances to block center for tie breaking", value<bool>())
    ("skipNoGainColors", "Tuning Parameter: Skip Colors that didn't result in a gain in the last global round", value<bool>())
    ("nnCoarsening", "When coarsening, pick the nearest neighbor based on the euclidean distance", value<bool>())
    ("localRefAlgo", "With which algorithm to do local refinement.", value<Tool>() )
    //multisection
    ("bisect", "Used for the multisection method. If set to true the algorithm perfoms bisections (not multisection) until the desired number of parts is reached", value<bool>())
    ("cutsPerDim", "If MultiSection is chosen, then provide d values that define the number of cuts per dimension. You must provide as many numbers as the dimensions separated with commas. For example, --cutsPerDim=3,4,10 for 3 dimensions resulting in 3*4*10=120 blocks", value<std::string>())
    ("pixeledSideLen", "The resolution for the pixeled partition or the spectral", value<IndexType>())
    //sfc
    ("sfcResolution", "The resolution depth of the hilbert space filling curve", value<IndexType>())
    // K-Means
    ("minSamplingNodes", "Tuning parameter for K-Means", value<IndexType>())
    ("influenceExponent", "Tuning parameter for K-Means, default is ", value<double>()->default_value(std::to_string(settings.influenceExponent)))
    ("influenceChangeCap", "Tuning parameter for K-Means", value<double>())
    ("balanceIterations", "Tuning parameter for K-Means", value<IndexType>())
    ("maxKMeansIterations", "Tuning parameter for K-Means", value<IndexType>())
    ("tightenBounds", "Tuning parameter for K-Means")
    ("erodeInfluence", "Tuning parameter for K-Means, in case of large deltas and imbalances.")
    // using '/' to separate the lines breaks the output message
    ("hierLevels", "The number of blocks per level. Total number of PEs (=number of leaves) is the product for all hierLevels[i] and there are hierLevels.size() hierarchy levels. Example: --hierLevels 3,4,10 there are 3 levels. In the first one, each node has 3 children, in the next one each node has 4 and in the last, each node has 10. In total 3*4*10= 120 leaves/PEs", value<std::string>())
    //output
    ("outFile", "write result partition into file", value<std::string>())
    //debug
    ("writeDebugCoordinates", "Write Coordinates of nodes in each block", value<bool>())
    ("writePEgraph", "Write the processor graph to a file", value<bool>())
    ("verbose", "Increase output.")
    ("debugMode", "Increase output and more expensive checks")
    ("storeInfo", "Store timing and other metrics in file.")
    ("storePartition", "Store the partition file.")
    ("callExit", "Call std::exit after finishing partitioning, useful in case of lingering MPI data structures.")
    // evaluation
    ("repeatTimes", "How many times we repeat the partitioning process.", value<IndexType>())
    ("noComputeDiameter", "Compute diameter of resulting block files.")
    ("maxDiameterRounds", "abort diameter algorithm after that many BFS rounds", value<IndexType>())
    ("maxCGIterations", "max number of iterations of the CG solver in metrics",  value<IndexType>())
    ("CGResidual", "solution precision of the CG solver in metrics",  value<double>())
    ("metricsDetail", "no: no metrics, easy:cut, imbalance, communication volume and diameter if possible, all: easy + SpMV time and communication time in SpMV", value<std::string>())
    ("autoSettings", "Set some settings automatically to some values possibly overwriting some user passed parameters. ", value<bool>() )
    ("partition", "file of partition (typically used by tools/analyzePartition)", value<std::string>())
    //used for the competitors main
    ("outDir", "write result partition into folder", value<std::string>())
    ("tools", "choose which supported tools to use. For multiple tool use comma to separate without spaces. See in Settings::Tools for the supported tools and how to call them.", value<std::string>() )
    //mesh generation
    ("generate", "generate uniform mesh as input graph")
    ("numX", "Number of points in x dimension of generated graph", value<IndexType>())
    ("numY", "Number of points in y dimension of generated graph", value<IndexType>())
    ("numZ", "Number of points in z dimension of generated graph", value<IndexType>())
    // exotic test cases
    ("quadTreeFile", "read QuadTree from file", value<std::string>())
    ("useDiffusionCoordinates", "Use coordinates based from diffusive systems instead of loading from file", value<bool>())
	//("myAlgoParam", "help message", value<int>())
    ;

    return options;
}

Settings interpretSettings(cxxopts::ParseResult vm) {

    Settings settings;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    srand(vm["seed"].as<double>());
    settings.seed = vm["seed"].as<double>();

    if (vm.count("version")) {
        std::cout << "Git commit " << version << std::endl;
        settings.isValid = false;
        return settings;
    }

    if (vm.count("generate") + vm.count("graphFile") + vm.count("quadTreeFile") != 1) {
        std::cout << "Call with --graphFile <input>. Use --help for more parameters." << std::endl;
        settings.isValid = false;
        //return 126;
    }

    if (vm.count("generate") && (vm["dimensions"].as<IndexType>() != 3)) {
        std::cout << "Mesh generation currently only supported for three dimensions" << std::endl;
        settings.isValid = false;
        //return 126;
    }

    if (vm.count("coordFile") && vm.count("useDiffusionCoords")) {
        std::cout << "Cannot both load coordinates from file with --coordFile or generate them with --useDiffusionCoords." << std::endl;
        settings.isValid = false;
        //return 126;
    }

    if (vm.count("fileFormat") && vm["fileFormat"].as<ITI::Format>() == ITI::Format::TEEC) {
        if (!vm.count("numX")) {
            std::cout << "TEEC file format does not specify graph size, please set with --numX" << std::endl;
            settings.isValid = false;
            //return 126;
        }
    }

    // check if coordFormat is provided
    // if no coordFormat was given but was given a fileFormat assume they are the same
    if( !vm.count("coordFormat") and vm.count("fileFormat") ) {
        settings.coordFormat = settings.fileFormat;
    }

    if (vm.count("outFile")) {
        settings.outFile = vm["outFile"].as<std::string>();
        settings.storeInfo = true;
    }

    if (vm.count("outDir")) {
        settings.outDir = vm["outDir"].as<std::string>();
        settings.storeInfo = true;
    }

    if (!vm.count("influenceExponent")) {
        settings.influenceExponent = 1.0/settings.dimensions;
    }

    if( vm.count("metricsDetail") ) {
        if( not (settings.metricsDetail=="no" or settings.metricsDetail=="easy" or settings.metricsDetail=="all" or settings.metricsDetail=="mapping") ) {
            if(comm->getRank() ==0 ) {
                std::cout<<"WARNING: wrong value for parameter metricsDetail= " << settings.metricsDetail << ". Setting to all" <<std::endl;
                settings.metricsDetail="all";
            }
        }
    }

    if( vm.count("noComputeDiameter") ) {
        settings.computeDiameter = false;
    } else {
        settings.computeDiameter = true;
    }

    using std::vector;
    settings.verbose = vm.count("verbose");
    settings.debugMode = vm.count("debugMode");
    //settings.storeInfo = vm.count("storeInfo");
    settings.storePartition = vm.count("storePartition");
    settings.erodeInfluence = vm.count("erodeInfluence");
    settings.tightenBounds = vm.count("tightenBounds");
    settings.noRefinement = vm.count("noRefinement");
    settings.useDiffusionCoordinates = vm.count("useDiffusionCoordinates");
    settings.gainOverBalance = vm.count("gainOverBalance");
    settings.useDiffusionTieBreaking = vm.count("useDiffusionTieBreaking");
    settings.useGeometricTieBreaking = vm.count("useGeometricTieBreaking");
    settings.skipNoGainColors = vm.count("skipNoGainColors");
    settings.nnCoarsening = vm.count("nnCoarsening");
    settings.bisect = vm.count("bisect");
    settings.writeDebugCoordinates = vm.count("writeDebugCoordinates");
    settings.writePEgraph = vm.count("writePEgraph");
    settings.setAutoSettings = vm.count("autoSettings");
    settings.mappingRenumbering = vm.count("mappingRenumbering");
    settings.autoSetCpuMem = vm.count("autoSetCpuMem");

    //28/11/19, deprecate storeInfo parameter. Leaving it as an option for backwards compatibility.    
    //if outFile was provided but storeInfo was not given as an argument
    if( vm.count("storeInfo") ) {
        if(comm->getRank()==0){
            std::cout << "WARNING: Option --storeInfo is deprecated and (most probably) will be ignored; metrics will be stored depending on the options --outFile and --outDir" << std::endl;
        }
        settings.storeInfo = true;
    }

    if (vm.count("graphFile")) {
        settings.fileName = vm["graphFile"].as<std::string>();
    }
    if (vm.count("fileFormat")) {
        settings.fileFormat = vm["fileFormat"].as<ITI::Format>();
    }
    if (vm.count("coordFormat")) {
        settings.coordFormat = vm["coordFormat"].as<ITI::Format>();
    }

    if (vm.count("PEgraphFile")) {
        settings.PEGraphFile = vm["PEgraphFile"].as<std::string>();
    }

    if (vm.count("numNodeWeights")) {
        settings.numNodeWeights = vm["numNodeWeights"].as<IndexType>();
    }

    if (vm.count("dimensions")) {
        settings.dimensions = vm["dimensions"].as<IndexType>();
    }
    if (vm.count("numX")) {
        settings.numX = vm["numX"].as<IndexType>();
    }
    if (vm.count("numY")) {
        settings.numY = vm["numY"].as<IndexType>();
    }
    if (vm.count("numZ")) {
        settings.numZ = vm["numZ"].as<IndexType>();
    }
    if (vm.count("numBlocks")) {
        settings.numBlocks = vm["numBlocks"].as<IndexType>();
    } else {
        settings.numBlocks = comm->getSize();
    }
    if (vm.count("sfcResolution")) {
        settings.sfcResolution = vm["sfcResolution"].as<IndexType>();
    }

    if (vm.count("epsilon")) {
        settings.epsilon = vm["epsilon"].as<double>();
    }

    if (vm.count("processPerNode")) {
        settings.processPerNode = vm["processPerNode"].as<IndexType>();
    }    
    if ( vm.count("initialMigration") ){
        std::string s = vm["initialMigration"].as<std::string>();
        settings.initialMigration = to_tool(s);        
    }
    if (vm.count("initialPartition")) {
        std::string s = vm["initialPartition"].as<std::string>();
        settings.initialPartition = to_tool(s);
    }
    if (vm.count("multiLevelRounds")) {
        settings.multiLevelRounds = vm["multiLevelRounds"].as<IndexType>();
    }
    if (vm.count("minBorderNodes")) {
        settings.minBorderNodes = vm["minBorderNodes"].as<IndexType>();
    }
    if (vm.count("minBorderNodesPercent")) {
        settings.minBorderNodesPercent = vm["minBorderNodesPercent"].as<double>();
    }
    if (vm.count("stopAfterNoGainRounds")) {
        settings.stopAfterNoGainRounds = vm["stopAfterNoGainRounds"].as<IndexType>();
    }
    if (vm.count("minGainForNextGlobalRound")) {
        settings.minGainForNextRound = vm["minGainForNextGlobalRound"].as<IndexType>();
    }
    if (vm.count("localRefAlgo")) {
        settings.localRefAlgo = vm["localRefAlgo"].as<Tool>();
    }
    //TODO: cxxopts supports parsing of multiple arguments and storing them as vectors
    //  use that and not our own parsing
    if (vm.count("cutsPerDim")) {
        std::stringstream ss( vm["cutsPerDim"].as<std::string>() );
        std::string item;
        std::vector<IndexType> cutsPerDim;
        IndexType product = 1;

        while (!std::getline(ss, item, ',').fail()) {
            IndexType cutsInDim = std::stoi(item);
            cutsPerDim.push_back(cutsInDim);
            product *= cutsInDim;
        }

        settings.cutsPerDim = cutsPerDim;

        if (!vm.count("numBlocks")) {
            settings.numBlocks = product;
        } else {
            if (vm["numBlocks"].as<IndexType>() != product) {
                throw std::invalid_argument("When giving --cutsPerDim, either omit --numBlocks or set it to the product of cutsPerDim.");
            }
        }
    }
    if (vm.count("pixeledSideLen")) {
        settings.pixeledSideLen = vm["pixeledSideLen"].as<IndexType>();
    }
    if (vm.count("minSamplingNodes")) {
        settings.minSamplingNodes = vm["minSamplingNodes"].as<IndexType>();
    }
    if (vm.count("influenceExponent")) {
        settings.influenceExponent = vm["influenceExponent"].as<double>();
    }
    if (vm.count("influenceChangeCap")) {
        settings.influenceChangeCap = vm["influenceChangeCap"].as<double>();
    }
    if (vm.count("balanceIterations")) {
        settings.balanceIterations = vm["balanceIterations"].as<IndexType>();
    }
    if (vm.count("maxKMeansIterations")) {
        settings.maxKMeansIterations = vm["maxKMeansIterations"].as<IndexType>();
    }
    if (vm.count("hierLevels")) {  
        std::stringstream ss( vm["hierLevels"].as<std::string>() );
        std::string item;
        std::vector<IndexType> hierLevels;
        IndexType product = 1;

        while (!std::getline(ss, item, ',').fail()) {
            IndexType blocksInLevel = std::stoi(item);
            hierLevels.push_back(blocksInLevel);
            product *= blocksInLevel;
        }

        settings.hierLevels = hierLevels;

        if (!vm.count("numBlocks")) {
            settings.numBlocks = product;
        } else {
            if (vm["numBlocks"].as<IndexType>() != product) {
                std::cout << vm["numBlocks"].as<IndexType>() << " " << product << std::endl;
                throw std::invalid_argument("When giving --hierLevels, either omit --numBlocks or set it to the product of level entries.");
            }
        }
    }

    if (vm.count("repeatTimes")) {
        settings.repeatTimes = vm["repeatTimes"].as<IndexType>();
    }
    if (vm.count("maxDiameterRounds")) {
        settings.maxDiameterRounds = vm["maxDiameterRounds"].as<IndexType>();
    }
    if (vm.count("maxCGIterations")) {
        settings.maxCGIterations = vm["maxCGIterations"].as<IndexType>();
    }
    if (vm.count("CGResidual")) {
        settings.CGResidual = vm["CGResidual"].as<double>();
    }
    if (vm.count("metricsDetail")) {
        settings.metricsDetail = vm["metricsDetail"].as<std::string>();
    }

    /*** consistency checks ***/
    if (vm.count("previousPartition")) {
        settings.repartition = true;
        if (vm.count("initialPartition")) {
            if (!(settings.initialPartition == Tool::geoKmeans || settings.initialPartition == Tool::none)) {
                std::cout << "Method " << settings.initialPartition << " not supported for repartitioning, currently only kMeans." << std::endl;
                settings.isValid = false;
                //return 126;
            }
        } else {
            PRINT0("Setting initial partitioning method to kMeans.");
            settings.initialPartition = Tool::geoKmeans;
        }
    }

    if ( settings.hierLevels.size() > 0 ) {
        if (!(settings.initialPartition == Tool::geoHierKM
                || settings.initialPartition == Tool::geoHierRepart)) {
            if(comm->getRank() ==0 ) {
                std::cout << " WARNING: Without using hierarchical partitioning, ";
                std::cout << "the given hierarchy levels will be ignored." << std::endl;
            }
        }

        if (!vm.count("numBlocks")) {
            IndexType numBlocks = 1;
            for (IndexType level : settings.hierLevels) {
                numBlocks *= level;
            }
            settings.numBlocks = numBlocks;
        }
    }

    //used (mainly) from allCompetitors to define which tools to use
    if (vm.count("tools")) {  
        std::stringstream ss( vm["tools"].as<std::string>() );       
        std::string item;
        std::vector<std::string> tools;

        while (!std::getline(ss, item, ',').fail()) {
            tools.push_back( item );
        }

        settings.tools = tools;
    }

    return settings;

}

}
