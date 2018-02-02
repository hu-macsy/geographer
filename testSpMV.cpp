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
#include <boost/filesystem.hpp>

#include <memory>
#include <cstdlib>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <stdlib.h>

#include "MeshGenerator.h"
#include "FileIO.h"
#include "Diffusion.h"
#include "ParcoRepart.h"
#include "Settings.h"
#include "Metrics.h"
#include "MultiLevel.h"
#include "LocalRefinement.h"
#include "SpectralPartition.h"
#include "MultiSection.h"
#include "KMeans.h"
#include "GraphUtils.h"
#include "Wrappers.h"

/**
 *  Examples of use:
 * 
 *  for reading from file "fileName" 
 * ./a.out --graphFile fileName --epsilon 0.05 --minBorderNodes=10 --dimensions=2 --borderDepth=10  --stopAfterNoGainRounds=3 --minGainForNextGlobalRound=10
 * 
 * for generating a 10x20x30 mesh
 * ./a.out --generate --numX=10 --numY=20 --numZ=30 --epsilon 0.05 --sfcRecursionSteps=10 --dimensions=3 --borderDepth=10  --stopAfterNoGainRounds=3 --minGainForNextGlobalRound=10
 * 
 */

//----------------------------------------------------------------------------


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
    out << token;
    return out;
}






int main(int argc, char** argv) {
	using namespace boost::program_options;
        options_description desc("Supported options");
        
        struct Settings settings;
        //ITI::Format ff = ITI::Format::METIS;
        std::string blockSizesFile;
        bool writePartition = false;
        IndexType repeatTimes = 1;
		IndexType partAlgo = 0;
		
        desc.add_options()
            ("help", "display options")
            ("version", "show version")
            ("graphFile", value<std::string>(), "read graph from file")
            ("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
            ("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
            ("generate", "generate random graph. Currently, only uniform meshes are supported.")
            ("dimensions", value<IndexType>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
            ("numX", value<IndexType>(&settings.numX)->default_value(settings.numX), "Number of points in x dimension of generated graph")
            ("numY", value<IndexType>(&settings.numY)->default_value(settings.numY), "Number of points in y dimension of generated graph")
            ("numZ", value<IndexType>(&settings.numZ)->default_value(settings.numZ), "Number of points in z dimension of generated graph")
            ("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
            ("numBlocks", value<IndexType>(&settings.numBlocks), "Number of blocks to partition to")
            ("minBorderNodes", value<IndexType>(&settings.minBorderNodes)->default_value(settings.minBorderNodes), "Tuning parameter: Minimum number of border nodes used in each refinement step")
            ("stopAfterNoGainRounds", value<IndexType>(&settings.stopAfterNoGainRounds)->default_value(settings.stopAfterNoGainRounds), "Tuning parameter: Number of rounds without gain after which to abort localFM. A value of 0 means no stopping.")
            //("initialPartition",  value<InitialPartitioningMethods> (&settings.initialPartition), "Parameter for different initial partition: 0 or 'SFC' for the hilbert space filling curve, 1 or 'Pixel' for the pixeled method, 2 or 'Spectral' for spectral parition, 3 or 'KMeans' for Kmeans and 4 or 'MultiSection' for Multisection")
			
			("partAlgo", value<IndexType>(&partAlgo),"The algorithm to be used for partitioning")
			
            ("bisect", value<bool>(&settings.bisect)->default_value(settings.bisect), "Used for the multisection method. If set to true the algorithm perfoms bisections (not multisection) until the desired number of parts is reached")
            ("cutsPerDim", value<std::vector<IndexType>>(&settings.cutsPerDim)->multitoken(), "If msOption=2, then provide d values that define the number of cuts per dimension.")
            ("pixeledSideLen", value<IndexType>(&settings.pixeledSideLen)->default_value(settings.pixeledSideLen), "The resolution for the pixeled partition or the spectral")
            ("minGainForNextGlobalRound", value<IndexType>(&settings.minGainForNextRound)->default_value(settings.minGainForNextRound), "Tuning parameter: Minimum Gain above which the next global FM round is started")
            ("gainOverBalance", value<bool>(&settings.gainOverBalance)->default_value(settings.gainOverBalance), "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance")
            ("useDiffusionTieBreaking", value<bool>(&settings.useDiffusionTieBreaking)->default_value(settings.useDiffusionTieBreaking), "Tuning Parameter: Use diffusion to break ties in Fiduccia-Mattheyes algorithm")
            ("useGeometricTieBreaking", value<bool>(&settings.useGeometricTieBreaking)->default_value(settings.useGeometricTieBreaking), "Tuning Parameter: Use distances to block center for tie breaking")
            ("skipNoGainColors", value<bool>(&settings.skipNoGainColors)->default_value(settings.skipNoGainColors), "Tuning Parameter: Skip Colors that didn't result in a gain in the last global round")
            ("multiLevelRounds", value<IndexType>(&settings.multiLevelRounds)->default_value(settings.multiLevelRounds), "Tuning Parameter: How many multi-level rounds with coarsening to perform")
			("repeatTimes", value<IndexType>(&repeatTimes), "How many times we repeat the SpMV process.")
            ("blockSizesFile", value<std::string>(&blockSizesFile) , " file to read the block sizes for every block")
            ("writePartition", "Writes the partition in the outFile.partition file")
            ("outFile", value<std::string>(&settings.outFile), "write result partition into file")
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
        
        if (vm.count("generate") && vm.count("file")) {
            std::cout << "Pick one of --file or --generate" << std::endl;
            return 0;
        }
        
        if (vm.count("generate") && (vm["dimensions"].as<int>() != 3)) {
            std::cout << "Mesh generation currently only supported for three dimensions" << std::endl;
            return 0;
        }
        
        if( vm.count("cutsPerDim") ){
            SCAI_ASSERT( !settings.cutsPerDim.empty(), "options cutsPerDim was given but the vector is empty" );
            SCAI_ASSERT_EQ_ERROR(settings.cutsPerDim.size(), settings.dimensions, "cutsPerDime: user must specify d values for mutlisection using option --cutsPerDim. e.g.: --cutsPerDim=4,20 for a partition in 80 parts/" );
            IndexType tmpK = 1;
            for( const auto& i: settings.cutsPerDim){
                tmpK *= i;
            }
            settings.numBlocks= tmpK;
        }
            
        IndexType N = -1; 		// total number of points
        
        writePartition = vm.count("writePartition");
        
        char machineChar[255];
        std::string machine;
        gethostname(machineChar, 255);
        if (machineChar) {
            machine = std::string(machineChar);
		settings.machine = machine;
        } else {
            std::cout << "machine char not valid" << std::endl;
 		machine = "machine char not valid";
        }
        
        scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
        std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph
        scai::lama::DenseVector<ValueType> nodeWeights;     // node weights
        
        std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D
        
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		const IndexType thisPE = comm->getRank();
        
        /* timing information
         */
        std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
        
        if ( thisPE== 0){
            std::string inputstring;
            if (vm.count("graphFile")) {
				inputstring = vm["graphFile"].as<std::string>();
			} else if (vm.count("quadTreeFile")) {
				inputstring = vm["quadTreeFile"].as<std::string>();
			} else {
				inputstring = "generate";
			}
			std::cout<< "commit:"<< version<<  " input:"<< inputstring << std::endl;
		}

    std::string graphFile;
	
    if (vm.count("graphFile")) {
        if ( thisPE == 0){
            std::cout<< "input: graphFile" << std::endl;
        }
    	graphFile = vm["graphFile"].as<std::string>();
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
        
        if ( thisPE== 0)
        {
            std::cout<< "Reading from file \""<< graphFile << "\" for the graph and \"" << coordFile << "\" for coordinates"<< std::endl;
            std::cout<< "File format: " << settings.fileFormat << std::endl;
        }

        // read the adjacency matrix and the coordinates from a file  
        
        std::vector<DenseVector<ValueType> > vectorOfNodeWeights;
               
        if (vm.count("fileFormat")) {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights, settings.fileFormat );
        } else{
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights );
        }
        N = graph.getNumRows();
        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
        scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
        assert(graph.getColDistribution().isEqual(*noDistPtr));
        
        // for 2D we do not know the size of every dimension
        settings.numX = N;
        settings.numY = IndexType(1);
        settings.numZ = IndexType(1);
        
        
        if (vm.count("fileFormat")) {
            coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.fileFormat);
        } else {
            coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
        }
        PRINT0("read  graph and coordinates");        
        
        //unit weights
        scai::hmemo::HArray<ValueType> localWeights( rowDistPtr->getLocalSize(), 1 );
        nodeWeights.swap( localWeights, rowDistPtr );

        if (comm->getRank() == 0) {
            std::cout << "Read " << N << " points." << std::endl;
            std::cout << "Read coordinates." << std::endl;
            std::cout << "On average there are about " << N/comm->getSize() << " points per PE."<<std::endl;
        }

    }
    else if(vm.count("generate")){
        if (thisPE==0){
            std::cout<< "input: generate" << std::endl;
        }
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
        
        if( thisPE== 0){
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
        
    }
    else{
    	std::cout << "Only input file as input. Call again with --graphFile" << std::endl;
    	return 0;
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
    
    // time needed to get the input
    std::chrono::duration<double> inputTime = std::chrono::system_clock::now() - startTime;

    assert(N > 0);

    if( !(vm.count("numBlocks") or vm.count("cutsPerDim")) ){
        settings.numBlocks = comm->getSize();
    }
    
    //------------------------------------------------------------------------------------------
    
    scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
    
    //DenseVector<IndexType> uniformWeights;
    
    settings.minGainForNextRound = IndexType(10);
    settings.minBorderNodes = IndexType(10);
    settings.useGeometricTieBreaking = IndexType(1);
    settings.pixeledSideLen = IndexType ( std::min(settings.numBlocks, IndexType(100) ) );
    
    std::string destPath = "./partResults/testInitial/blocks_"+std::to_string(settings.numBlocks)+"/";
    boost::filesystem::create_directories( destPath );   
    
    std::size_t found= graphFile.std::string::find_last_of("/");
    std::string logFile = destPath + "results_" + graphFile.substr(found+1)+ ".log";
    std::ofstream logF(logFile);
    std::ifstream f(graphFile);
    
    
    settings.print( std::cout , comm );
        

    IndexType dimensions = settings.dimensions;
    IndexType k = settings.numBlocks;
    
    std::chrono::time_point<std::chrono::system_clock> beforeInitialTime;
    std::chrono::duration<double> partitionTime;
    std::chrono::duration<double> finalPartitionTime;
    
    
    using namespace ITI;
    
    
    if(comm->getRank()==0) std::cout <<std::endl<<std::endl;
    
    scai::lama::DenseVector<IndexType> partition;
    IndexType initialPartition = static_cast<IndexType> (settings.initialPartition);
    
    comm->synchronize();
	
	struct Metrics metrics(1);
	
	//----------------------------------------------------------------
	//
	// partition with the chosen algorithm
	//
    
    switch( partAlgo ){
        case 0:{  //------------------------------------------- hilbert/sfc
           
            beforeInitialTime =  std::chrono::system_clock::now();
            PRINT0( "Get a hilbert/sfc partition");
            
            // get a hilbertPartition
            partition = ParcoRepart<IndexType, ValueType>::hilbertPartition( coordinates, settings);
            
            partitionTime =  std::chrono::system_clock::now() - beforeInitialTime;
            
            // the hilbert partition internally sorts and thus redistributes the points
            graph.redistribute( partition.getDistributionPtr() , noDistPtr );
            rowDistPtr = graph.getRowDistributionPtr();
            
            assert( partition.size() == N);
            assert( coordinates[0].size() == N);
            
            break;
        }
        case 1:{  //------------------------------------------- pixeled
  
            beforeInitialTime =  std::chrono::system_clock::now();
            PRINT0( "Get a pixeled partition");
            
            // get a pixelPartition
            partition = ParcoRepart<IndexType, ValueType>::pixelPartition( coordinates, settings);
            
            partitionTime =  std::chrono::system_clock::now() - beforeInitialTime;
            
            assert( partition.size() == N);
            assert( coordinates[0].size() == N);
            
            break;
        }
        case 2:{  //------------------------------------------- spectral
            std::cout<< "Not included in testInitial yet, choose another option."<< std::endl;
            std::terminate();
        }
        case 3:{  //------------------------------------------- k-means
            beforeInitialTime =  std::chrono::system_clock::now();
            PRINT0( "Get a k-means partition");
            
            // get a k-means partition
			IndexType weightSum;
			bool uniformWeights = true;
				
			DenseVector<IndexType> tempResult = ParcoRepart<IndexType, ValueType>::hilbertPartition(coordinates, settings);
			
			if( not uniformWeights){	
				scai::hmemo::HArray<IndexType> localWeightsInt( rowDistPtr->getLocalSize(), 1 );
				scai::lama::DenseVector<IndexType> nodeWeightsInt;     // node weights
				nodeWeightsInt.swap( localWeightsInt, rowDistPtr );
				nodeWeightsInt.redistribute(tempResult.getDistributionPtr());
					
				weightSum = nodeWeightsInt.sum().Scalar::getValue<IndexType>();
			}else{
				// if all nodes have weight 1 then weightSum = globalN
				weightSum = N;
			}
			const std::vector<IndexType> blockSizes(settings.numBlocks, weightSum/settings.numBlocks);
				
			// WARNING: getting an error in KMeans.h, try to redistribute coordinates
			std::vector<DenseVector<ValueType> > coordinateCopy = coordinates;
			for (IndexType d = 0; d < dimensions; d++) {
				coordinateCopy[d].redistribute( tempResult.getDistributionPtr() );
			}
			//
			beforeInitialTime =  std::chrono::system_clock::now();

			partition = ITI::KMeans::computePartition(coordinateCopy, settings.numBlocks, nodeWeights, blockSizes, settings);      
				
			partitionTime =  std::chrono::system_clock::now() - beforeInitialTime;
				
			// must repartition graph according to the new partition/distribution
			//graph.redistribute( partition.getDistributionPtr() , noDistPtr );
			
            assert( partition.size() == N);
            assert( coordinates[0].size() == N);
            break;
        }
        case 4:{  //------------------------------------------- multisection
            
            if ( settings.bisect==1){
                PRINT0( "Get a partition with bisection");
            }else{
                PRINT0( "Get a partition with multisection");
            }
            
            beforeInitialTime =  std::chrono::system_clock::now();
            
            // get a multisection partition
            partition =  MultiSection<IndexType, ValueType>::getPartitionNonUniform( graph, coordinates, nodeWeights, settings);
            
            partitionTime =  std::chrono::system_clock::now() - beforeInitialTime;
            
            assert( partition.size() == N);
            assert( coordinates[0].size() == N);
            break;   
        }
		case 5 : { 	//--------------------------------------- parmetis
			
			int parMetisGeom = 0; 		//0 no geometric info, 1 partGeomKway, 2 PartGeom (only geometry)
			settings.repeatTimes = 1;
			
			beforeInitialTime =  std::chrono::system_clock::now();
			// get parMetis parition
			partition = ITI::Wrappers<IndexType,ValueType>::metisWrapper ( graph, coordinates, nodeWeights, parMetisGeom, settings, metrics);
			partitionTime =  std::chrono::system_clock::now() - beforeInitialTime;
			
			break;
		}
        default:{
            PRINT0("Value "<< initialPartition << " for option initialPartition not supported" );
            break;
        }
    }
    
    ValueType time = 0;
	
	time = comm->max( partitionTime.count() );
	PRINT0("time to get partition: " << time);

	//get the distribution from the partition
	scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partition.getDistribution(), partition.getLocalValues() ) );
	
	std::chrono::time_point<std::chrono::system_clock> beforeRedistribution;
	//
	// redistribute according to the new distribution
	//
	//WARNING: is this needed or not???
	graph.redistribute( distFromPartition, graph.getColDistributionPtr());
	partition.redistribute( distFromPartition );
	nodeWeights.redistribute( distFromPartition );
	//TODO: redistribute also coordinates?	probably not needed	
	
	std::chrono::duration<double> redistributionTime =  std::chrono::system_clock::now() - beforeRedistribution;
	
	time = comm->max( redistributionTime.count() );
	PRINT0("time to redistribute: " << time);
	
	rowDistPtr = graph.getRowDistributionPtr();
	
	
	//----------------------------------------------------------------
    //
    // perform the SpMV
    //
    
    
    // the graph is distributed here based on the algorithm we chose
    IndexType localN = rowDistPtr->getLocalSize();
	PRINT(" localN for "<< thisPE << " = " << localN);
	
	std::chrono::time_point<std::chrono::system_clock> beforeLaplacian = std::chrono::system_clock::now();
	
    // the laplacian has the same row and column distributios as the (now partitioned) graph
	
    scai::lama::CSRSparseMatrix<ValueType> laplacian = SpectralPartition<IndexType, ValueType>::getLaplacian( graph );
	//scai::lama::CSRSparseMatrix<ValueType> laplacian = Diffusion<IndexType, ValueType>::constructLaplacian( graph );

	SCAI_ASSERT( laplacian.getRowDistributionPtr()->isEqual( graph.getRowDistribution() ), "Row distributions do not agree" );
	SCAI_ASSERT( laplacian.getColDistributionPtr()->isEqual( graph.getColDistribution() ), "Column distributions do not agree" );
	
	std::chrono::duration<ValueType> laplacianTime = std::chrono::system_clock::now() - beforeLaplacian;
	time = comm->max(laplacianTime.count());
	PRINT0("time to get the laplacian: " << time );
	
		
	// vector for multiplication
	scai::lama::DenseVector<ValueType> x ( graph.getColDistributionPtr(), 3.3 );
	
	// perfom the actual multiplication
	std::chrono::time_point<std::chrono::system_clock> beforeSpMVTime = std::chrono::system_clock::now();
	for(IndexType r=0; r<repeatTimes; r++){
		DenseVector<ValueType> result( laplacian * x );
		//DenseVector<ValueType> result( graph * x );
	}
	std::chrono::duration<ValueType> SpMVTime = std::chrono::system_clock::now() - beforeSpMVTime;
	PRINT(" SpMV time for PE "<< thisPE << " = " << SpMVTime.count() );
	
	time = comm->max(SpMVTime.count());
	PRINT0("time for " << repeatTimes <<" SpMVs: " << time );
	
	/*
	for(int i=0; i<localN; i++){
		PRINT(*comm << ": " << rowDistPtr->local2global(i) );
	}
    */
	
	
	ValueType imbalance = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( partition, settings.numBlocks, nodeWeights );
	
	if( thisPE==0 )
		std::cout<<"imbalance = " << imbalance << std::endl;
	
	/*
    //
    // Get metrics
    //
    
    
    
	
	metrics.timeFinalPartition = comm->max( partitionTime.count() );
	metrics.getMetrics( graph, partition, nodeWeights, settings );
	
	//
    // Reporting output to std::cout and/or outFile
	//
	
    if (comm->getRank() == 0) {
		//metrics.print( std::cout );
		std::cout << "Running " << __FILE__ << std::endl;
		printMetricsShort( metrics, std::cout);
		// write in a file
        if( settings.outFile!="-" ){
			std::ofstream outF( settings.outFile, std::ios::out);
			if(outF.is_open()){
				outF << "Running " << __FILE__ << std::endl;
                if( vm.count("generate") ){
                    outF << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                }else{
                    outF << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                }
                settings.print( outF, comm );
                //outF << "numBlocks= " << settings.numBlocks << std::endl;
                //metrics.print( outF ); 
				printMetricsShort( metrics, outF);
                std::cout<< "Output information written to file " << settings.outFile << std::endl;
            }else{
                std::cout<< "Could not open file " << settings.outFile << " informations not stored"<< std::endl;
            } 
		}
		
    }
	*/
   
    //std::exit(0);
	return 0;
}
