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

#include "MeshIO.h"
#include "ParcoRepart.h"
#include "Settings.h"

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
 * !! for now, when reading a file --dimensions must be 2
 */

//----------------------------------------------------------------------------

int main(int argc, char** argv) {
	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;

	desc.add_options()
				("help", "display options")
				("version", "show version")
				("graphFile", value<std::string>(), "read graph from file")
				("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
				("generate", "generate random graph. Currently, only uniform meshes are supported.")
				("dimensions", value<int>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
				("numX", value<int>(&settings.numX)->default_value(settings.numX), "Number of points in x dimension of generated graph")
				("numY", value<int>(&settings.numY)->default_value(settings.numY), "Number of points in y dimension of generated graph")
				("numZ", value<int>(&settings.numZ)->default_value(settings.numZ), "Number of points in z dimension of generated graph")
				("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
				("borderDepth", value<int>(&settings.borderDepth)->default_value(settings.borderDepth), "Tuning parameter: Depth of border region used in each refinement step")
				("stopAfterNoGainRounds", value<int>(&settings.stopAfterNoGainRounds)->default_value(settings.stopAfterNoGainRounds), "Tuning parameter: Number of rounds without gain after which to abort localFM. A value of 0 means no stopping.")
				("sfcRecursionSteps", value<int>(&settings.sfcResolution)->default_value(settings.sfcResolution), "Tuning parameter: Recursion Level of space filling curve. A value of 0 causes the recursion level to be derived from the graph size.")
				("minGainForNextGlobalRound", value<int>(&settings.minGainForNextRound)->default_value(settings.minGainForNextRound), "Tuning parameter: Minimum Gain above which the next global FM round is started")
				("gainOverBalance", value<bool>(&settings.gainOverBalance)->default_value(settings.gainOverBalance), "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance")
				("numberOfRestarts", value<int>(&settings.numberOfRestarts)->default_value(settings.numberOfRestarts), "Tuning parameter: Restart local FM with different seeds and keep best.")
				;

	variables_map vm;
	store(command_line_parser(argc, argv).
			  options(desc).run(), vm);
	notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}

	if (vm.count("version")) {
		std::cout << "Git commit " << version << std::endl;
		return 0;
	}

	if (vm.count("generate") && vm.count("file")) {
		std::cout << "Pick one of --file or --generate" << std::endl;
		return 0;
	}

	if (vm.count("generate") && (vm["dimensions"].as<int>() != 3)) {
		std::cout << "Mesh generation currently only supported for three dimensions" << std::endl;
		return 0;
	}

    IndexType N = -1; 		// total number of points

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph

    std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    /* timing information
     */
    std::chrono::time_point<std::chrono::system_clock> startTime;
     
    startTime = std::chrono::system_clock::now();
    
    if (vm.count("graphFile")) {
    	std::string graphFile = vm["graphFile"].as<std::string>();
    	std::string coordFile;
    	if (vm.count("coordFile")) {
    		coordFile = vm["coordFile"].as<std::string>();
    	} else {
    		coordFile = graphFile + ".xyz";
    	}
    
        std::fstream f(graphFile);

        if(f.fail()){
            throw std::runtime_error("File "+ graphFile + " failed.");
        }
        
        f >> N;				// first line must have total number of nodes and edges
        
        // for 2D we do not know the size of every dimension
        settings.numX = N;
        settings.numY = 1;
        settings.numZ = 1;
        
        if (comm->getRank() == 0)
        {
			std::cout<< "Reading from file \""<< graphFile << "\" for the graph and \"" << coordFile << "\" for coordinates"<< std::endl;
			std::cout<< "Read " << N << " points." << std::endl;
        }

        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
        graph = scai::lama::CSRSparseMatrix<ValueType>( rowDistPtr , noDistPtr );

        // read the adjacency matrix and the coordinates from a file
        ITI::MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( graph , graphFile );
        graph.redistribute(rowDistPtr , noDistPtr);
        // take care, when reading from file graph needs redistribution
        
        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<settings.dimensions; i++){
            coordinates[i].allocate(coordDist);
            coordinates[i] = static_cast<ValueType>( 0 );
        }
        
        if (settings.dimensions == 2) {
        	ITI::MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, coordinates, N );
        } else if (settings.dimensions == 3){
        	ITI::MeshIO<IndexType, ValueType>::fromFile2Coords_3D(coordFile, coordinates, N );
        }

        for(IndexType i=0; i<settings.dimensions; i++){
        	coordinates[i].redistribute(coordDist);
        }

    } else if(vm.count("generate")){
    	if (settings.dimensions == 2) {
    		settings.numZ = 1;
    	}

        N = settings.numX * settings.numY * settings.numZ;
            
        maxCoord[0] = settings.numX;
        maxCoord[1] = settings.numY;
        maxCoord[2] = settings.numZ;

        std::vector<IndexType> numPoints(3); // number of poitns in each dimension, used only for 3D

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
        ITI::MeshIO<IndexType, ValueType>::createStructured3DMesh_dist( graph, coordinates, maxCoord, numPoints);

    } else{
    	std::cout << "Either an input file or generation parameters are needed. Call again with --file or --generate" << std::endl;
    	return 0;
    }
    
    // time needed to get the input
    std::chrono::duration<double> inputTime = std::chrono::system_clock::now() - startTime;

    assert(N > 0);

    if (comm->getSize() > 0) {
    	settings.numBlocks = comm->getSize();
    }

    if( comm->getRank() ==0){
        if(settings.dimensions==2){
            settings.print2D(std::cout);
        }else{
            settings.print3D(std::cout);
        }
    }
    
    std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
    
    scai::lama::DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coordinates, settings );
    
    std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforePartTime;
    
    std::chrono::time_point<std::chrono::system_clock> beforeReport = std::chrono::system_clock::now();
    
    ValueType cut = ITI::ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true); 
    ValueType imbalance = ITI::ParcoRepart<IndexType, ValueType>::computeImbalance( partition, comm->getSize() );
    
    std::chrono::duration<double> reportTime =  std::chrono::system_clock::now() - beforeReport;
    
    
    // Reporting output to std::cout
    ValueType inputT = ValueType ( comm->max(inputTime.count() ));
    ValueType partT = ValueType (comm->max(partitionTime.count()));
    ValueType repT = ValueType (comm->max(reportTime.count()));
        
    if (comm->getRank() == 0) {
        std::cout<<"commit:"<< version<< " input:"<< ( vm.count("graphFile") ? vm["graphFile"].as<std::string>() :"generate");
        std::cout<< " nodes:"<< N<< " dimensions:"<< settings.dimensions <<" k:" << settings.numBlocks;
        std::cout<< " epsilon:" << settings.epsilon << " borderDepth:"<< settings.borderDepth;
        std::cout<< " minGainForNextRound:" << settings.minGainForNextRound;
        std::cout<< " stopAfterNoGainRounds:"<< settings.stopAfterNoGainRounds << std::endl;
        
        std::cout<< "Cut is: "<< cut<< " and imbalance: "<< imbalance << std::endl;
        std::cout<<"inputTime:" << inputT << " partitionTime:" << partT <<" reportTime:"<< repT << std::endl;
    }
}
