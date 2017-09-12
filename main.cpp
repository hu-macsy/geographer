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
//enum class Format {AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3};
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
		else
			in.setstate(std::ios_base::failbit);
		return in;
	}

	std::ostream& operator<<(std::ostream& out, Format& method)
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
    else
        in.setstate(std::ios_base::failbit);
    return in;
}

std::ostream& operator<<(std::ostream& out, InitialPartitioningMethods& method)
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
    out << token;
    return out;
}



int main(int argc, char** argv) {
	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	desc.add_options()
				("help", "display options")
				("version", "show version")
				("graphFile", value<std::string>(), "read graph from file")
				("quadTreeFile", value<std::string>(), "read QuadTree from file")
				("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
				("coordFormat", value<ITI::Format>(), "format of coordinate file")
				("nodeWeightIndex", value<int>()->default_value(0), "index of node weight")
				("generate", "generate random graph. Currently, only uniform meshes are supported.")
				("dimensions", value<int>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
				("numX", value<int>(&settings.numX)->default_value(settings.numX), "Number of points in x dimension of generated graph")
				("numY", value<int>(&settings.numY)->default_value(settings.numY), "Number of points in y dimension of generated graph")
				("numZ", value<int>(&settings.numZ)->default_value(settings.numZ), "Number of points in z dimension of generated graph")
				("numBlocks", value<int>(&settings.numBlocks)->default_value(comm->getSize()), "Number of blocks, default is number of processes")
				("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
				("seed", value<double>()->default_value(time(NULL)), "random seed, default is current time")
				("minBorderNodes", value<int>(&settings.minBorderNodes)->default_value(settings.minBorderNodes), "Tuning parameter: Minimum number of border nodes used in each refinement step")
				("stopAfterNoGainRounds", value<int>(&settings.stopAfterNoGainRounds)->default_value(settings.stopAfterNoGainRounds), "Tuning parameter: Number of rounds without gain after which to abort localFM. A value of 0 means no stopping.")
				("bisect", value<bool>(&settings.bisect)->default_value(settings.bisect), "Used for the multisection method. If set to true the algorithm perfoms bisections (not multisection) until the desired number of parts is reached")
				("cutsPerDim", value<std::vector<IndexType>>(&settings.cutsPerDim)->multitoken(), "If MultiSection is chosen, then provide d values that define the number of cuts per dimension.")
				("initialPartition", value<InitialPartitioningMethods>(&settings.initialPartition), "Choose initial partitioning method between space-filling curves ('SFC' or 0), pixel grid coarsening ('Pixel' or 1), spectral partition ('Spectral' or 2), k-means ('K-Means' or 3) and multisection ('MultiSection' or 4). SFC, Spectral and K-Means are most stable.")
				("pixeledSideLen", value<int>(&settings.pixeledSideLen)->default_value(settings.pixeledSideLen), "The resolution for the pixeled partition or the spectral")
				("minGainForNextGlobalRound", value<int>(&settings.minGainForNextRound)->default_value(settings.minGainForNextRound), "Tuning parameter: Minimum Gain above which the next global FM round is started")
				("gainOverBalance", value<bool>(&settings.gainOverBalance)->default_value(settings.gainOverBalance), "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance")
				("useDiffusionTieBreaking", value<bool>(&settings.useDiffusionTieBreaking)->default_value(settings.useDiffusionTieBreaking), "Tuning Parameter: Use diffusion to break ties in Fiduccia-Mattheyes algorithm")
				("useGeometricTieBreaking", value<bool>(&settings.useGeometricTieBreaking)->default_value(settings.useGeometricTieBreaking), "Tuning Parameter: Use distances to block center for tie breaking")
				("useDiffusionCoordinates", value<bool>(&settings.useDiffusionCoordinates)->default_value(settings.useDiffusionCoordinates), "Use coordinates based from diffusive systems instead of loading from file")
				("skipNoGainColors", value<bool>(&settings.skipNoGainColors)->default_value(settings.skipNoGainColors), "Tuning Parameter: Skip Colors that didn't result in a gain in the last global round")
				("writeDebugCoordinates", value<bool>(&settings.writeDebugCoordinates)->default_value(settings.writeDebugCoordinates), "Write Coordinates of nodes in each block")
				("multiLevelRounds", value<int>(&settings.multiLevelRounds)->default_value(settings.multiLevelRounds), "Tuning Parameter: How many multi-level rounds with coarsening to perform")
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
	if( vm.count("cutsPerDim") ){
            SCAI_ASSERT( !settings.cutsPerDim.empty(), "options cutsPerDim was given but the vector is empty" );
            SCAI_ASSERT_EQ_ERROR(settings.cutsPerDim.size(), settings.dimensions, "cutsPerDime: user must specify d values for mutlisection using option --cutsPerDim. e.g.: --cutsPerDim=4,20 for a partition in 80 parts/" );
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

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph

    std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D

    DenseVector<ValueType> nodeWeights;

    srand(vm["seed"].as<double>());

    /* timing information
     */
    std::chrono::time_point<std::chrono::system_clock> startTime;
     
    startTime = std::chrono::system_clock::now();
    
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

        // read the adjacency matrix and the coordinates from a file
        std::vector<DenseVector<ValueType> > vectorOfNodeWeights;
        graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights );

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
        	ITI::Format format;
        	if (vm.count("coordFormat")) {
        		format = vm["coordFormat"].as<ITI::Format>();
        		coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, format);
        	} else {
        		coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
        	}

        }

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
    
    // time needed to get the input. Synchronize first to make sure that all processes are finished.
    comm->synchronize();
    std::chrono::duration<double> inputTime = std::chrono::system_clock::now() - startTime;

    assert(N > 0);

    if( comm->getRank() ==0){
          settings.print(std::cout);
    }
    
    std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
    
    scai::lama::DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coordinates, nodeWeights, settings );
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
		ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, "debugResult");
    }
    */
    std::chrono::time_point<std::chrono::system_clock> beforeReport = std::chrono::system_clock::now();
    
    ValueType cut = ITI::GraphUtils::computeCut(graph, partition, true);
    ValueType imbalance = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( partition, settings.numBlocks, nodeWeights );
    
    std::chrono::duration<double> reportTime =  std::chrono::system_clock::now() - beforeReport;
    
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
        
        std::cout<< "cut:"<< cut<< " imbalance:"<< imbalance << std::endl;
        std::cout<<"inputTime:" << inputT << " partitionTime:" << partT <<" reportTime:"<< repT << std::endl;
    }
}
