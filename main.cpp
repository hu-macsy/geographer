#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <memory>
#include <cstdlib>
#include <chrono>

#include <unistd.h>

#include "Diffusion.h"
#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "Settings.h"
#include "SpectralPartition.h"
#include "GraphUtils.h"

typedef double ValueType;
typedef int IndexType;


/**
 *  Examples of use:
 * 
 *  for reading from file "fileName" 
 * ./a.out --graphFile fileName --epsilon 0.05 --sfcRecursionSteps=10 --dimensions=2 --borderDepth=10  --stopAfterNoGainRounds=3 --minGainForNextGlobalRound=10 
 * 
 * for generating a 10x20x30 mesh
 * ./a.out --generate --numX=10 --numY=20 --numZ=30 --epsilon 0.05 --sfcRecursionSteps=10 --dimensions=3 --borderDepth=10  --stopAfterNoGainRounds=3 --minGainForNextGlobalRound=10
 * 
 * ./a.out --graphFile fileName --epsilon 0.05 --initialPartition=4 --dimensions=2 --bisect=0 --numPoints=4000000 --distribution=uniform --cutsPerDim=10 13
 * 
 */

//----------------------------------------------------------------------------
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
	ITI::Format coordFormat;
        
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	desc.add_options()
				("help", "display options")
				("version", "show version")
				//input and coordinates
				("graphFile", value<std::string>(), "read graph from file")
				("quadTreeFile", value<std::string>(), "read QuadTree from file")
				("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
				("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
				("coordFormat", value<ITI::Format>(&coordFormat), "format of coordinate file: AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4 ")
				("nodeWeightIndex", value<int>()->default_value(0), "index of node weight")
				("useDiffusionCoordinates", value<bool>(&settings.useDiffusionCoordinates)->default_value(settings.useDiffusionCoordinates), "Use coordinates based from diffusive systems instead of loading from file")
				("dimensions", value<int>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
				("previousPartition", value<std::string>(), "file of previous partition, used for repartitioning")
				//output
				("outFile", value<std::string>(), "write result partition into file")
				//mesh generation
				("generate", "generate random graph. Currently, only uniform meshes are supported.")
				("numX", value<int>(&settings.numX), "Number of points in x dimension of generated graph")
				("numY", value<int>(&settings.numY), "Number of points in y dimension of generated graph")
				("numZ", value<int>(&settings.numZ), "Number of points in z dimension of generated graph")
				//general partitioning parameters
				("numBlocks", value<int>(&settings.numBlocks)->default_value(comm->getSize()), "Number of blocks, default is number of processes")
				("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
				("blockSizesFile", value<std::string>(&blockSizesFile) , " file to read the block sizes for every block")
				("seed", value<double>()->default_value(time(NULL)), "random seed, default is current time")
				//multi-level and local refinement
				("initialPartition", value<InitialPartitioningMethods>(&settings.initialPartition), "Choose initial partitioning method between space-filling curves ('SFC' or 0), pixel grid coarsening ('Pixel' or 1), spectral partition ('Spectral' or 2), k-means ('K-Means' or 3) and multisection ('MultiSection' or 4). SFC, Spectral and K-Means are most stable.")
				("multiLevelRounds", value<int>(&settings.multiLevelRounds)->default_value(settings.multiLevelRounds), "Tuning Parameter: How many multi-level rounds with coarsening to perform")
				("minBorderNodes", value<int>(&settings.minBorderNodes)->default_value(settings.minBorderNodes), "Tuning parameter: Minimum number of border nodes used in each refinement step")
				("stopAfterNoGainRounds", value<int>(&settings.stopAfterNoGainRounds)->default_value(settings.stopAfterNoGainRounds), "Tuning parameter: Number of rounds without gain after which to abort localFM. A value of 0 means no stopping.")
				("minGainForNextGlobalRound", value<int>(&settings.minGainForNextRound)->default_value(settings.minGainForNextRound), "Tuning parameter: Minimum Gain above which the next global FM round is started")
				("gainOverBalance", value<bool>(&settings.gainOverBalance)->default_value(settings.gainOverBalance), "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance")
				("useDiffusionTieBreaking", value<bool>(&settings.useDiffusionTieBreaking)->default_value(settings.useDiffusionTieBreaking), "Tuning Parameter: Use diffusion to break ties in Fiduccia-Mattheyes algorithm")
				("useGeometricTieBreaking", value<bool>(&settings.useGeometricTieBreaking)->default_value(settings.useGeometricTieBreaking), "Tuning Parameter: Use distances to block center for tie breaking")
				("skipNoGainColors", value<bool>(&settings.skipNoGainColors)->default_value(settings.skipNoGainColors), "Tuning Parameter: Skip Colors that didn't result in a gain in the last global round")
				//multisection
				("bisect", value<bool>(&settings.bisect)->default_value(settings.bisect), "Used for the multisection method. If set to true the algorithm perfoms bisections (not multisection) until the desired number of parts is reached")
				("cutsPerDim", value<std::vector<IndexType>>(&settings.cutsPerDim)->multitoken(), "If MultiSection is chosen, then provide d values that define the number of cuts per dimension.")
				("pixeledSideLen", value<int>(&settings.pixeledSideLen)->default_value(settings.pixeledSideLen), "The resolution for the pixeled partition or the spectral")
				// K-Means
				("minSamplingNodes", value<int>(&settings.minSamplingNodes)->default_value(settings.minSamplingNodes), "Tuning parameter for K-Means")
				("influenceExponent", value<double>(&settings.influenceExponent)->default_value(settings.influenceExponent), "Tuning parameter for K-Means")
				("influenceChangeCap", value<double>(&settings.influenceChangeCap)->default_value(settings.influenceChangeCap), "Tuning parameter for K-Means")
				("balanceIterations", value<int>(&settings.balanceIterations)->default_value(settings.balanceIterations), "Tuning parameter for K-Means")
				("maxKMeansIterations", value<int>(&settings.maxKMeansIterations)->default_value(settings.maxKMeansIterations), "Tuning parameter for K-Means")
				("tightenBounds", "Tuning parameter for K-Means")
				("erodeInfluence", "Tuning parameter for K-Means, in case of large deltas and imbalances.")
				("initialMigration", value<InitialPartitioningMethods>(&settings.initialMigration)->default_value(settings.initialMigration), "Choose a method to get the first migration, 0: SFCs, 3:k-means, 4:Multisection")
				//debug
				("writeDebugCoordinates", value<bool>(&settings.writeDebugCoordinates)->default_value(settings.writeDebugCoordinates), "Write Coordinates of nodes in each block")
				("verbose", "Increase output.")
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

	if (vm.count("generate") + vm.count("graphFile") + vm.count("quadTreeFile") != 1) {
		std::cout << "Pick one of --graphFile, --quadTreeFile or --generate" << std::endl;
		return 126;
	}

	if (vm.count("generate") && (vm["dimensions"].as<int>() != 3)) {
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
        
        
        int i=2147483644;
        int kl=0;
        while( kl<10 ){
            PRINT(i);
            i++;
            kl++;
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

    IndexType N = -1; 		// total number of points

    char machineChar[255];
    std::string machine;
    gethostname(machineChar, 255);
    if (machineChar) {
    	machine = std::string(machineChar);
    } else {
    	std::cout << "machine char not valid" << std::endl;
    }

    settings.verbose = vm.count("verbose");
    settings.erodeInfluence = vm.count("erodeInfluence");
    settings.tightenBounds = vm.count("tightenBounds");

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph

    std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D

    DenseVector<ValueType> nodeWeights;

    srand(vm["seed"].as<double>());

    /* timing information
     */
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    if (comm->getRank() == 0)
	{
    	std::string inputstring;
    	if (vm.count("graphFile")) {
    		inputstring = vm["graphFile"].as<std::string>();
    	} else if (vm.count("quadTreeFile")) {
    		inputstring = vm["quadTreeFile"].as<std::string>();
    	} else {
    		inputstring = "generate";
    	}

        std::cout<< "commit:"<< version<< " input:"<< inputstring << std::endl;
	}

    if (vm.count("graphFile")) {
    	std::string graphFile = vm["graphFile"].as<std::string>();
    	std::string coordFile;
    	if (vm.count("coordFile")) {
	   	coordFile = vm["coordFile"].as<std::string>();
	} else {
		coordFile = graphFile + ".xyz";
	}

    	std::string coordString;
    	if (settings.useDiffusionCoordinates) {
    		coordString = "and generating coordinates with diffusive distances.";
    	} else {
    		coordString = "and \"" + coordFile + "\" for coordinates";
    	}

        if (comm->getRank() == 0)
        {
            std::cout<< "Reading from file \""<< graphFile << "\" for the graph " << coordString << std::endl;
        }

        //
        // read the adjacency matrix and the coordinates from a file
        //
        std::vector<DenseVector<ValueType> > vectorOfNodeWeights;
        if (vm.count("fileFormat")) {
        	if (settings.fileFormat == ITI::Format::TEEC) {
        		IndexType n = vm["numX"].as<IndexType>();
				scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(n, comm));
				scai::dmemo::DistributionPtr noDist( new scai::dmemo::NoDistribution(n));
				graph = scai::lama::CSRSparseMatrix<ValueType>(dist, noDist);
				ITI::FileIO<IndexType, ValueType>::readCoordsTEEC(graphFile, n, settings.dimensions, vectorOfNodeWeights);
				if (settings.verbose) {
					ValueType minWeight = vectorOfNodeWeights[0].min().Scalar::getValue<ValueType>();
					ValueType maxWeight = vectorOfNodeWeights[0].max().Scalar::getValue<ValueType>();
					if (comm->getRank() == 0) std::cout << "Min node weight:" << minWeight << ", max weight: " << maxWeight << std::endl;
				}
				coordFile = graphFile;
			} else {
				graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights, settings.fileFormat );
			}
        } else{
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights );
        }
        N = graph.getNumRows();
        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
        scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
        assert(graph.getColDistribution().isEqual(*noDistPtr));

        IndexType numNodeWeights = vectorOfNodeWeights.size();
        if (numNodeWeights == 0) {
			nodeWeights = DenseVector<ValueType>(rowDistPtr, 1);
		}
		else if (numNodeWeights == 1) {
			nodeWeights = vectorOfNodeWeights[0];
		} else {
			IndexType index = vm["nodeWeightIndex"].as<int>();
			assert(index < numNodeWeights);
			nodeWeights = vectorOfNodeWeights[index];
		}

        // for 2D we do not know the size of every dimension
        settings.numX = N;
        settings.numY = 1;
        settings.numZ = 1;

        comm->synchronize();
        if (comm->getRank() == 0) {
        	std::cout<< "Read " << N << " points." << std::endl;
        }
        
        if (settings.useDiffusionCoordinates) {
        	scai::lama::CSRSparseMatrix<ValueType> L = ITI::Diffusion<IndexType, ValueType>::constructLaplacian(graph);

        	std::vector<IndexType> nodeIndices(N);
        	std::iota(nodeIndices.begin(), nodeIndices.end(), 0);

        	ITI::GraphUtils::FisherYatesShuffle(nodeIndices.begin(), nodeIndices.end(), settings.dimensions);

        	if (comm->getRank() == 0) {
        		std::cout << "Chose diffusion sources";
        		for (IndexType i = 0; i < settings.dimensions; i++) {
        			std::cout << " " << nodeIndices[i];
        		}
        		std::cout << "." << std::endl;
        	}

        	coordinates.resize(settings.dimensions);

			for (IndexType i = 0; i < settings.dimensions; i++) {
				coordinates[i] = ITI::Diffusion<IndexType, ValueType>::potentialsFromSource(L, nodeWeights, nodeIndices[i]);
			}

        } else {
            if( vm.count("coordFormat") ) { // coordFormat given
                coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, coordFormat);
            }else if ( !vm.count("coordFormat") and vm.count("fileFormat") ) { 
                // if no coordFormat was given but was given a fileFormat assume they are the same
                coordFormat = settings.fileFormat;
                coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, coordFormat);
            } else {
                coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
            }
        }

        comm->synchronize();
        if (comm->getRank() == 0) {
        	std::cout << "Read coordinates." << std::endl;
        }

    } else if(vm.count("generate")){
    	if (settings.dimensions == 2) {
    		settings.numZ = 1;
    	}

        N = settings.numX * settings.numY * settings.numZ;
            
        maxCoord[0] = settings.numX;
        maxCoord[1] = settings.numY;
        maxCoord[2] = settings.numZ;

        std::vector<IndexType> numPoints(3); // number of points in each dimension, used only for 3D

        for (IndexType i = 0; i < 3; i++) {
        	numPoints[i] = maxCoord[i];
        }

        if( comm->getRank()== 0){
            std::cout<< "Generating for dim= "<< settings.dimensions << " and numPoints= "<< settings.numX << ", " << settings.numY << ", "<< settings.numZ << ", in total "<< N << " number of points" << std::endl;
            std::cout<< "\t\t and maxCoord= "<< maxCoord[0] << ", "<< maxCoord[1] << ", " << maxCoord[2]<< std::endl;
        }
        
        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph = scai::lama::CSRSparseMatrix<ValueType>( rowDistPtr , noDistPtr );
        
        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<settings.dimensions; i++){
            coordinates[i].allocate(coordDist);
            coordinates[i] = static_cast<ValueType>( 0 );
        }

        // create the adjacency matrix and the coordinates
        ITI::MeshGenerator<IndexType, ValueType>::createStructured3DMesh_dist( graph, coordinates, maxCoord, numPoints);
        
        IndexType nodes= graph.getNumRows();
        IndexType edges= graph.getNumValues()/2;
        if(comm->getRank()==0){
            std::cout<< "Generated random 3D graph with "<< nodes<< " and "<< edges << " edges."<< std::endl;
        }

        nodeWeights = scai::lama::DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);

	} else if (vm.count("quadTreeFile")) {
		//if (comm->getRank() == 0) {
			graph = ITI::FileIO<IndexType, ValueType>::readQuadTree(vm["quadTreeFile"].as<std::string>(), coordinates);
			N = graph.getNumRows();
		//}

		//broadcast graph size from root to initialize distributions
		//IndexType NTransport[1] = {static_cast<IndexType>(graph.getNumRows())};
		//comm->bcast( NTransport, 1, 0 );
		//N = NTransport[0];

        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph.redistribute(rowDistPtr, noDistPtr);
        for (IndexType i = 0; i < settings.dimensions; i++) {
        	coordinates[i].redistribute(rowDistPtr);
        }
        nodeWeights = scai::lama::DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);

    } else{
    	std::cout << "Either an input file or generation parameters are needed. Call again with --graphFile, --quadTreeFile, or --generate" << std::endl;
    	return 126;
    }
    
    //
    //  read block sizes from a file if it is passed as an argument
    //
    if( vm.count("blockSizesFile") ){
        settings.blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        IndexType blockSizesSum  = std::accumulate( settings.blockSizes.begin(), settings.blockSizes.end(), 0);
        IndexType nodeWeightsSum = nodeWeights.sum().Scalar::getValue<IndexType>();
        SCAI_ASSERT_GE( blockSizesSum, nodeWeightsSum, "The block sizes provided are not enough to fit the total weight of the input" );
    }
    
    // get previous partition, if set
    DenseVector<IndexType> previous;
    if (vm.count("previousPartition")) {
    	std::string filename = vm["previousPartition"].as<std::string>();
    	previous = ITI::FileIO<IndexType, ValueType>::readPartition(filename, N);
    	if (previous.size() != N) {
    		throw std::runtime_error("Previous partition has wrong size.");
    	}
    	if (previous.max().Scalar::getValue<IndexType>() != settings.numBlocks-1) {
    		throw std::runtime_error("Illegal maximum block ID in previous partition:" + std::to_string(previous.max().Scalar::getValue<IndexType>()));
    	}
    	if (previous.min().Scalar::getValue<IndexType>() != 0) {
    		throw std::runtime_error("Illegal minimum block ID in previous partition:" + std::to_string(previous.min().Scalar::getValue<IndexType>()));
    	}
    	settings.repartition = true;
    }

    // time needed to get the input. Synchronize first to make sure that all processes are finished.
    comm->synchronize();
    std::chrono::duration<double> inputTime = std::chrono::system_clock::now() - startTime;

    assert(N > 0);

    if (settings.repartition && comm->getSize() == settings.numBlocks) {
    	//redistribute according to previous partition now to simulate the setting in a dynamic repartitioning
    	assert(previous.size() == N);
    	scai::dmemo::Redistributor previousRedist(previous.getLocalValues(), previous.getDistributionPtr());
    	graph.redistribute(previousRedist, graph.getColDistributionPtr());
    	for (IndexType d = 0; d < settings.dimensions; d++) {
    		coordinates[d].redistribute(previousRedist);
    	}

    	if (nodeWeights.size() > 0) {
    		nodeWeights.redistribute(previousRedist);
    	}
    	previous = DenseVector<IndexType>(previousRedist.getTargetDistributionPtr(), comm->getRank());

    }

    if( comm->getRank() ==0 && settings.verbose){
          settings.print(std::cout);
    }
    
    std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
    
    scai::lama::DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coordinates, nodeWeights, previous, settings );
    assert( partition.size() == N);
    assert( coordinates[0].size() == N);
    
    std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforePartTime;
    
    // the code below writes the output coordinates in one file per processor for visualization purposes.
    //=================
    /*
    if (settings.writeDebugCoordinates) {
		for (IndexType dim = 0; dim < settings.dimensions; dim++) {
			assert( coordinates[dim].size() == N);
			coordinates[dim].redistribute(partition.getDistributionPtr());
		}

        std::string destPath = "partResults/main/blocks_" + std::to_string(settings.numBlocks) ;
        boost::filesystem::create_directories( destPath );   
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath + "/debugResult");
    }
    */
    std::chrono::time_point<std::chrono::system_clock> beforeReport = std::chrono::system_clock::now();
    
    ValueType cut = ITI::GraphUtils::computeCut(graph, partition, true);
    ValueType imbalance = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( partition, settings.numBlocks, nodeWeights );
    IndexType maxComm, totalComm;
    std::tie(maxComm, totalComm) = ITI::GraphUtils::computeComm<IndexType, ValueType>( graph, partition, settings.numBlocks);
    
    std::chrono::duration<double> reportTime =  std::chrono::system_clock::now() - beforeReport;
    
    if (vm.count("outFile")) {
    	ITI::FileIO<IndexType, ValueType>::writePartition(partition, vm["outFile"].as<std::string>());
    }

    // Reporting output to std::cout
    ValueType inputT = ValueType ( comm->max(inputTime.count() ));
    ValueType partT = ValueType (comm->max(partitionTime.count()));
    ValueType repT = ValueType (comm->max(reportTime.count()));

    if (comm->getRank() == 0) {
        for (IndexType i = 0; i < argc; i++) {
            std::cout << std::string(argv[i]) << " ";
        }
        std::cout << std::endl;
        std::cout<< "commit:"<< version << " machine:" << machine << " input:"<< ( vm.count("graphFile") ? vm["graphFile"].as<std::string>() :"generate");
        std::cout << " p:"<< comm->getSize() << " k:"<< settings.numBlocks;
        auto oldprecision = std::cout.precision(std::numeric_limits<double>::max_digits10);
        std::cout <<" seed:" << vm["seed"].as<double>() << std::endl;
        std::cout.precision(oldprecision);
        
        std::cout<< std::endl<< "\033[1;36mcut:"<< cut<< "   imbalance:"<< imbalance << "   maxComm= "<< maxComm << std::endl;
        std::cout<<"inputTime:" << inputT << "   partitionTime:" << partT <<"   reportTime:"<< repT << " \033[0m" << std::endl;
    }
}
