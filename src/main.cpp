#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/lama/Vector.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <memory>
#include <cstdlib>
#include <chrono>
#include <iomanip> 
#include <unistd.h>

#include "Diffusion.h"
#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "Settings.h"
#include "Metrics.h"
#include "SpectralPartition.h"
#include "GraphUtils.h"


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

//void memusage(size_t *, size_t *,size_t *,size_t *,size_t *);	


int main(int argc, char** argv) {
	using namespace boost::program_options;
	
	//bool writePartition = false;
    
	std::string metricsDetail = "all";
	std::string blockSizesFile;
	//ITI::Format coordFormat;
    IndexType repeatTimes = 1;
        
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

	struct Settings settings;
	variables_map vm = settings.parseInput( argc, argv);
	if( !settings.isValid )
		return -1;

    //--------------------------------------------------------
    //
    // initialize
    //
    
    if( comm->getRank() ==0 ){
		std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
		std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
		std::cout << "date and time: " << std::ctime(&timeNow);
	}
	
    IndexType N = -1; 		// total number of points

	std::string machine = settings.machine;
    /* timing information
     */

    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    if (comm->getRank() == 0){
    	std::string inputstring;
    	if (vm.count("graphFile")) {
    		inputstring = vm["graphFile"].as<std::string>();
    	} else if (vm.count("quadTreeFile")) {
    		inputstring = vm["quadTreeFile"].as<std::string>();
    	} else {
    		inputstring = "generate";
    	}

        std::cout<< "commit:"<< version<< " main file: "<< __FILE__ << " machine:" << machine << " p:"<< comm->getSize();

        auto oldprecision = std::cout.precision(std::numeric_limits<double>::max_digits10);
        std::cout <<" seed:" << vm["seed"].as<double>() << std::endl;
        std::cout.precision(oldprecision);

        std::cout << "Calling command:" << std::endl;
        for (IndexType i = 0; i < argc; i++) {
            std::cout << std::string(argv[i]) << " ";
        }
        std::cout << std::endl << std::endl;

    }

    //---------------------------------------------------------
    //
    // generate or read graph and coordinates
    //
    
    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    std::vector<scai::lama::DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph
	std::vector<scai::lama::DenseVector<ValueType>> nodeWeights;		//the weights for each node
	
	
    if (vm.count("graphFile")) {
    	std::string graphFile = vm["graphFile"].as<std::string>();
        settings.fileName = graphFile;
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
				graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDist);
				ITI::FileIO<IndexType, ValueType>::readCoordsTEEC(graphFile, n, settings.dimensions, vectorOfNodeWeights);
				if (settings.verbose) {
					ValueType minWeight = vectorOfNodeWeights[0].min();
					ValueType maxWeight = vectorOfNodeWeights[0].max();
					if (comm->getRank() == 0) std::cout << "Min node weight:" << minWeight << ", max weight: " << maxWeight << std::endl;
				}
				coordFile = graphFile;
            }else {
				graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights, settings.fileFormat );
			}
        } else{
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfNodeWeights );
        }
        N = graph.getNumRows();
        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
        scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
        assert(graph.getColDistribution().isEqual(*noDistPtr));

        nodeWeights = vectorOfNodeWeights;
        IndexType numNodeWeights = vectorOfNodeWeights.size();
        if (numNodeWeights == 0) {
        	nodeWeights.resize(1);
			nodeWeights[0] = fill<DenseVector<ValueType>>(rowDistPtr, 1);
		}

        // for 2D we do not know the size of every dimension
        settings.numX = N;
        settings.numY = 1;
        settings.numZ = 1;

        std::chrono::duration<double> readGraphTime = std::chrono::system_clock::now() - startTime;
        ValueType timeToReadGraph = ValueType ( comm->max(readGraphTime.count()) );     
        
        comm->synchronize();
        if (comm->getRank() == 0) {
        	std::cout<< "Read " << N << " points in " << timeToReadGraph << " ms." << std::endl;
        }
        
        if (settings.useDiffusionCoordinates) {
        	scai::lama::CSRSparseMatrix<ValueType> L = ITI::GraphUtils<IndexType, ValueType>::constructLaplacian(graph);

        	std::vector<IndexType> nodeIndices(N);
        	std::iota(nodeIndices.begin(), nodeIndices.end(), 0);

        	ITI::GraphUtils<IndexType, ValueType>::FisherYatesShuffle(nodeIndices.begin(), nodeIndices.end(), settings.dimensions);

        	if (comm->getRank() == 0) {
        		std::cout << "Chose diffusion sources";
        		for (IndexType i = 0; i < settings.dimensions; i++) {
        			std::cout << " " << nodeIndices[i];
        		}
        		std::cout << "." << std::endl;
        	}

        	coordinates.resize(settings.dimensions);

			for (IndexType i = 0; i < settings.dimensions; i++) {
				coordinates[i] = ITI::Diffusion<IndexType, ValueType>::potentialsFromSource(L, nodeWeights[0], nodeIndices[i]);
			}

        } else {
            coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.coordFormat);
        }
        
        std::chrono::duration<double> readCoordsTime = std::chrono::system_clock::now() - startTime;
        ValueType timeToReadCoords = ValueType ( comm->max(readCoordsTime.count()) ) -timeToReadGraph ;     
        
        comm->synchronize();
        if (comm->getRank() == 0) {
        	std::cout << "Read coordinates in "<< timeToReadCoords << " ms." << std::endl;
        }       

    } else if(vm.count("generate")){
    	if (settings.dimensions == 2) {
    		settings.numZ = 1;
    	}

        N = settings.numX * settings.numY * settings.numZ;
            
        std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D
        maxCoord[0] = settings.numX;
        maxCoord[1] = settings.numY;
        if(settings.dimensions==3){
            maxCoord[2] = settings.numZ;
        }

        std::vector<IndexType> numPoints(3); // number of points in each dimension, used only for 3D

        for (IndexType i = 0; i < settings.dimensions; i++) {
        	numPoints[i] = maxCoord[i];
        }

        if( comm->getRank()== 0){
            std::cout<< "Generating for dim= "<< settings.dimensions << " and numPoints= "<< settings.numX << ", " << settings.numY << ", "<< settings.numZ << ", in total "<< N << " number of points" << std::endl;
            std::cout<< "\t\t and maxCoord= "; //<< maxCoord[0] << ", "<< maxCoord[1] << ", " << maxCoord[2]<< std::endl;
            for (IndexType i = 0; i < settings.dimensions; i++) {
                std::cout << maxCoord[i] << ", ";
            }
            std::cout << std::endl;
        }
        
        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>( rowDistPtr , noDistPtr );
        
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
        
        nodeWeights.resize(1);
        nodeWeights[0] = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);
        
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

        nodeWeights.resize(1);
        nodeWeights[0] = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);

    } else{
    	std::cout << "Either an input file or generation parameters are needed. Call again with --graphFile, --quadTreeFile, or --generate" << std::endl;
    	return 126;
    }

    //
    // read the communication graph or the block sizes if provided
    //

    if( vm.count("PEgraphFile") and vm.count("blockSizesFile") ){
    	throw std::runtime_error("You should provide either a file for a communication graph OR a file for block sizes. Not both.");
    }

    ITI::CommTree<IndexType,ValueType> commTree;

    if(vm.count("PEgraphFile")){
        throw std::logic_error("Reading of communication trees not yet implemented here.");
    	//commTree =  FileIO<IndexType, ValueType>::readPETree( settings.PEGraphFile );
    }else if( vm.count("blockSizesFile") ){
    	//blockSizes.size()=number of weights, blockSizes[i].size()= number of blocks
        std::vector<std::vector<ValueType>> blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        for (IndexType i = 0; i < nodeWeights.size(); i++) {
        	const ValueType blockSizesSum  = std::accumulate( blockSizes[i].begin(), blockSizes[i].end(), 0);
			const ValueType nodeWeightsSum = nodeWeights[i].sum();
			SCAI_ASSERT_GE( blockSizesSum, nodeWeightsSum, "The block sizes provided are not enough to fit the total weight of the input" );
        }

        commTree.createFlatHeterogeneous( blockSizes );
    }else{
    	commTree.createFlatHomogeneous( settings.numBlocks );
    }
    
    commTree.adaptWeights( nodeWeights );

    //---------------------------------------------------------------------
    //
    //  read block sizes from a file if it is passed as an argument
    //
    
    std::vector<std::vector<ValueType> > blockSizes;

    
    
    //---------------------------------------------------------------
    //
    // get previous partition, if set
    //
    
    DenseVector<IndexType> previous;
    if (vm.count("previousPartition")) {
    	std::string filename = vm["previousPartition"].as<std::string>();
    	previous = ITI::FileIO<IndexType, ValueType>::readPartition(filename, N);
    	if (previous.size() != N) {
    		throw std::runtime_error("Previous partition has wrong size.");
    	}
    	if (previous.max() != settings.numBlocks-1) {
    		throw std::runtime_error("Illegal maximum block ID in previous partition:" + std::to_string(previous.max()));
    	}
    	if (previous.min() != 0) {
    		throw std::runtime_error("Illegal minimum block ID in previous partition:" + std::to_string(previous.min()));
    	}
    	settings.repartition = true;
    }

    //
    // time needed to get the input. Synchronize first to make sure that all processes are finished.
    //
    
    comm->synchronize();
    std::chrono::duration<double> inputTime = std::chrono::system_clock::now() - startTime;

    assert(N > 0);

    if (settings.repartition && comm->getSize() == settings.numBlocks) {
    	//redistribute according to previous partition now to simulate the setting in a dynamic repartitioning
    	assert(previous.size() == N);
    	auto previousRedist = scai::dmemo::redistributePlanByNewOwners(previous.getLocalValues(), previous.getDistributionPtr());
    	graph.redistribute(previousRedist, graph.getColDistributionPtr());
    	for (IndexType d = 0; d < settings.dimensions; d++) {
    		coordinates[d].redistribute(previousRedist);
    	}

    	for (IndexType i = 0; i < nodeWeights.size(); i++) {
    		nodeWeights[i].redistribute(previousRedist);
    	}
    	previous = fill<DenseVector<IndexType>>(previousRedist.getTargetDistributionPtr(), comm->getRank());

    }
    
    std::vector<struct Metrics> metricsVec;
	
    //------------------------------------------------------------
    //
    // partition the graph
    //
    
    if( repeatTimes>0 ){
        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
        // SCAI_ASSERT_ERROR(rowDistPtr->isEqual( new scai::dmemo::BlockDistribution(N, comm) ) , "Graph row distribution should (?) be a block distribution." );
        SCAI_ASSERT_ERROR( coordinates[0].getDistributionPtr()->isEqual( *rowDistPtr ) , "rowDistribution and coordinates distribution must be equal" );
        for (IndexType i = 0; i < nodeWeights.size(); i++) {
        	SCAI_ASSERT_ERROR( nodeWeights[i].getDistributionPtr()->isEqual( *rowDistPtr ) , "rowDistribution and nodeWeights distribution must be equal" );
        }
    }
    
    //store distributions to use later
    const scai::dmemo::DistributionPtr rowDistPtr( new scai::dmemo::BlockDistribution(N, comm) );
    const scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ) );
    
    scai::lama::DenseVector<IndexType> partition;
    
    for( IndexType r=0; r<repeatTimes; r++){
                
        // for the next runs the input is redistributed, so we must redistribute to the original distributions
        
        if (repeatTimes > 1) {
            if(comm->getRank()==0) std::cout<< std::endl<< std::endl;
            PRINT0("\t\t ----------- Starting run number " << r +1 << " -----------");
        }
        
        if(r>0){
            PRINT0("Input redistribution: block distribution for graph rows, coordinates and nodeWeigts, no distribution for graph columns");
            
            graph.redistribute( rowDistPtr, noDistPtr );
            for(int d=0; d<settings.dimensions; d++){
                coordinates[d].redistribute( rowDistPtr );
            }
            for (IndexType i = 0; i < nodeWeights.size(); i++) {
            		nodeWeights[i].redistribute( rowDistPtr );
            }
        }
          
        //metricsVec.push_back( Metrics( comm->getSize()) );
        metricsVec.push_back( Metrics( settings ) );
            
        std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
        
        partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coordinates, nodeWeights, previous, commTree, settings, metricsVec[r] );
        assert( partition.size() == N);
        assert( coordinates[0].size() == N);
        
        std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforePartTime;
		
		//WARNING: with the noRefinement flag the partition is not distributed
        if (!comm->all(partition.getDistribution().isEqual(graph.getRowDistribution()))) {
            partition.redistribute( graph.getRowDistributionPtr());
        }
                
        //---------------------------------------------
        //
        // Get metrics
        //        
        
        std::chrono::time_point<std::chrono::system_clock> beforeReport = std::chrono::system_clock::now();
    
		if( metricsDetail=="all" ){
			metricsVec[r].getAllMetrics( graph, partition, nodeWeights[0], settings );
		}
        if( metricsDetail=="easy" ){
			metricsVec[r].getEasyMetrics( graph, partition, nodeWeights[0], settings );
		}
        
        metricsVec[r].MM["inputTime"] = ValueType ( comm->max(inputTime.count() ));
        metricsVec[r].MM["timeFinalPartition"] = ValueType (comm->max(partitionTime.count()));

		std::chrono::duration<double> reportTime =  std::chrono::system_clock::now() - beforeReport;

        //---------------------------------------------
        //
        // Print some output
        //

        if (comm->getRank() == 0 ) {
            std::cout<< "commit:"<< version << " machine:" << machine << " input:"<< ( vm.count("graphFile") ? vm["graphFile"].as<std::string>() :"generate");
            std::cout << " p:"<< comm->getSize() << " k:"<< settings.numBlocks;
            auto oldprecision = std::cout.precision(std::numeric_limits<double>::max_digits10);
            std::cout <<" seed:" << vm["seed"].as<double>() << std::endl;
            std::cout.precision(oldprecision);
            metricsVec[r].printHorizontal2( std::cout ); //TODO: remove?
        }
       
        //---------------------------------------------------------------
        //
        // Reporting output to std::cout
        //
        
        metricsVec[r].MM["reportTime"] = ValueType (comm->max(reportTime.count()));
        
        
        if (comm->getRank() == 0 && metricsDetail != "no") {
            metricsVec[r].print( std::cout );            
        }
        
        comm->synchronize();
    }// repeat loop
        
    std::chrono::duration<double> totalTime =  std::chrono::system_clock::now() - startTime;
    ValueType totalT = ValueType ( comm->max(totalTime.count() ));
            
    //
    // writing results in a file and std::cout
    //
    
    //aggregate metrics in one struct
    const struct Metrics aggrMetrics = aggregateVectorMetrics( metricsVec );

    if (repeatTimes > 1) {
        if (comm->getRank() == 0) {
            std::cout<<  "\033[1;36m";    
        	aggrMetrics.print( std::cout ); 
            std::cout << " \033[0m";
        }
    }
    

    if( settings.storeInfo && settings.outFile!="-" ) {
        if( comm->getRank()==0){
            std::ofstream outF( settings.outFile, std::ios::out);
            if(outF.is_open()){
				outF << "Running " << __FILE__ << std::endl;
				settings.print( outF, comm);
				
				aggrMetrics.print( outF ); 
			
				//	profiling info for k-means
				if(settings.verbose){
					outF << "iter | delta | time | imbalance | balanceIter" << std::endl;
					ValueType totTime = 0.0;
					SCAI_ASSERT_EQ_ERROR( metricsVec[0].kmeansProfiling.size(), metricsVec[0].numBalanceIter.size() , "mismatch in kmeans profiling metrics vectors");


					for( int i=0; i<metricsVec[0].kmeansProfiling.size(); i++){
						std::tuple<ValueType, ValueType, ValueType> tuple = metricsVec[0].kmeansProfiling[i];

						outF << i << " " << std::get<0>(tuple) << " " << std::get<1>(tuple) << " " << std::get<2>(tuple) << " " <<  metricsVec[0].numBalanceIter[i] << std::endl;
						totTime += std::get<1>(tuple);
					}
					outF << "totTime: " << totTime << std::endl;
				}	
				//

				//printVectorMetrics( metricsVec, outF ); 
                std::cout<< "Output information written to file " << settings.outFile << " in total time " << totalT << std::endl;
            }	else	{
                std::cout<< "Could not open file " << settings.outFile << " information not stored"<< std::endl;
            }            
        }
    }    
    
    
    if( settings.outFile!="-" and settings.writeInFile ){
        std::chrono::time_point<std::chrono::system_clock> beforePartWrite = std::chrono::system_clock::now();
        std::string partOutFile = settings.outFile;
		ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partition, partOutFile );

        std::chrono::duration<double> writePartTime =  std::chrono::system_clock::now() - beforePartWrite;
        if( comm->getRank()==0 ){
            std::cout << " and last partition of the series in file " << partOutFile << std::endl;
            std::cout<< " Time needed to write .partition file: " << writePartTime.count() <<  std::endl;
        }
    }    
    
    // the code below writes the output coordinates in one file per processor for visualization purposes.
    //=================
    
    if (settings.writeDebugCoordinates) {
		
		std::vector<DenseVector<ValueType> > coordinateCopy = coordinates;
		auto distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues() );
        for (IndexType dim = 0; dim < settings.dimensions; dim++) {
            assert( coordinateCopy[dim].size() == N);
			coordinateCopy[dim].redistribute( distFromPartition );
        }
        
        std::string destPath = "partResults/main/blocks_" + std::to_string(settings.numBlocks) ;
        boost::filesystem::create_directories( destPath );   
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coordinateCopy, settings.dimensions, destPath + "/debugResult");
        comm->synchronize();
        
        //TODO: use something like the code below instead of a NoDistribution
        //std::vector<IndexType> gatheredPart;
        //comm->gatherImpl( gatheredPart.data(), N, 0, partition.getLocalValues(), scai::common::TypeTraits<IndexType>::stype );
        /*
        scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
        graph.redistribute( noDistPtr, noDistPtr );
        partition.redistribute( noDistPtr );
        for (IndexType dim = 0; dim < settings.dimensions; dim++) {
            coordinates[dim].redistribute( noDistPtr );
        }
        
        //scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();     
        
        //TODO: change that in later version, where data are gathered in one PE and are not replicated
        SCAI_ASSERT_ERROR( graph.getRowDistributionPtr()->isReplicated()==1, "Adjacency should be replicated. Aborting...");
        SCAI_ASSERT_ERROR( partition.getDistributionPtr()->isReplicated()==1, "Partition should be replicated. Aborting...");
        SCAI_ASSERT_ERROR( coordinates[0].getDistributionPtr()->isReplicated()==1, "Coordinates should be replicated. Aborting...");
        
        if( comm->getRank()==0 ){
            if( settings.outFile != "-" ){
                ITI::FileIO<IndexType,ValueType>::writeVTKCentral( graph, coordinates, partition, settings.outFile+".vtk" );
            }else{
                ITI::FileIO<IndexType,ValueType>::writeVTKCentral( graph, coordinates, partition, destPath + "/debugResult.vtk" );
            }
        }
        */
    }
      	  
    //this is needed for supermuc
    std::exit(0);   
    
    return 0;
}
