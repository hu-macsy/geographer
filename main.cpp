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
	
	bool writePartition = false;
    
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
		settings.print( std::cout);
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
    std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph
	scai::lama::DenseVector<ValueType> nodeWeights;		//the weights for each node
	
	
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

        IndexType numNodeWeights = vectorOfNodeWeights.size();
        if (numNodeWeights == 0) {
			nodeWeights = fill<DenseVector<ValueType>>(rowDistPtr, 1);
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
				coordinates[i] = ITI::Diffusion<IndexType, ValueType>::potentialsFromSource(L, nodeWeights, nodeIndices[i]);
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
        
        nodeWeights = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);
        
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
        nodeWeights = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);

    } else{
    	std::cout << "Either an input file or generation parameters are needed. Call again with --graphFile, --quadTreeFile, or --generate" << std::endl;
    	return 126;
    }
    
    //---------------------------------------------------------------------
    //
    //  read block sizes from a file if it is passed as an argument
    //
    
    if( vm.count("blockSizesFile") ){
        settings.blockSizes = ITI::FileIO<IndexType, ValueType>::readBlockSizes( blockSizesFile, settings.numBlocks );
        IndexType blockSizesSum  = std::accumulate( settings.blockSizes.begin(), settings.blockSizes.end(), 0);
        IndexType nodeWeightsSum = nodeWeights.sum();
        SCAI_ASSERT_GE( blockSizesSum, nodeWeightsSum, "The block sizes provided are not enough to fit the total weight of the input" );
    }
    
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
    	scai::dmemo::Redistributor previousRedist(previous.getLocalValues(), previous.getDistributionPtr());
    	graph.redistribute(previousRedist, graph.getColDistributionPtr());
    	for (IndexType d = 0; d < settings.dimensions; d++) {
    		coordinates[d].redistribute(previousRedist);
    	}

    	if (nodeWeights.size() > 0) {
    		nodeWeights.redistribute(previousRedist);
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
        SCAI_ASSERT_ERROR( nodeWeights.getDistributionPtr()->isEqual( *rowDistPtr ) , "rowDistribution and nodeWeights distribution must be equal" ); 
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
            nodeWeights.redistribute( rowDistPtr );
        }
          
        //metricsVec.push_back( Metrics( comm->getSize()) );
        metricsVec.push_back( Metrics( settings ) );
            
        std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
        
        partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coordinates, nodeWeights, previous, settings, metricsVec[r] );
        assert( partition.size() == N);
        assert( coordinates[0].size() == N);
        
        std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforePartTime;
		
		//WARNING: with the noRefinement flag the partition is not distributed
        if (!comm->all(partition.getDistribution().isEqual(graph.getRowDistribution()))) {
            partition.redistribute( graph.getRowDistributionPtr());
        }
		
        metricsVec[r].finalCut = ITI::GraphUtils<IndexType, ValueType>::computeCut(graph, partition, true);
        metricsVec[r].finalImbalance = ITI::GraphUtils<IndexType, ValueType>::computeImbalance(partition, settings.numBlocks ,nodeWeights);
        metricsVec[r].inputTime = ValueType ( comm->max(inputTime.count() ));
        metricsVec[r].timeFinalPartition = ValueType (comm->max(partitionTime.count()));

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
            std::cout<< std::endl<< "\033[1;36mcut:"<< metricsVec[r].finalCut<< "   imbalance:"<< metricsVec[r].finalImbalance << std::endl;
            std::cout<<"inputTime:" << metricsVec[r].inputTime << "   partitionTime:" << metricsVec[r].timeFinalPartition << " \033[0m" << std::endl;
        }
                
        //---------------------------------------------
        //
        // Get metrics
        //
        
        
        std::chrono::time_point<std::chrono::system_clock> beforeReport = std::chrono::system_clock::now();
    
		if( metricsDetail=="all" ){
			metricsVec[r].getAllMetrics( graph, partition, nodeWeights, settings );
		}
        if( metricsDetail=="easy" ){
			metricsVec[r].getEasyMetrics( graph, partition, nodeWeights, settings );
		}
        std::chrono::duration<double> reportTime =  std::chrono::system_clock::now() - beforeReport;
        
        
        //---------------------------------------------------------------
        //
        // Reporting output to std::cout
        //
        
        metricsVec[r].reportTime = ValueType (comm->max(reportTime.count()));
        
        
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
    
    
    if (repeatTimes > 1) {
        if (comm->getRank() == 0) {
            std::cout<<  "\033[1;36m";
        }
        printVectorMetrics( metricsVec, std::cout );
        if (comm->getRank() == 0) {
            std::cout << " \033[0m";
        }
    }
    
 	//printVectorMetrics( metricsVec, std::cout );

    if( settings.storeInfo && settings.outFile!="-" ) {
        if( comm->getRank()==0){
            std::ofstream outF( settings.outFile, std::ios::out);
            if(outF.is_open()){
				outF << "Running " << __FILE__ << std::endl;
				settings.print( outF, comm);
				
				if( settings.noRefinement)
					printVectorMetricsShort( metricsVec, outF ); 
				else
					printVectorMetrics( metricsVec, outF ); 
			
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
            }else{
                std::cout<< "Could not open file " << settings.outFile << " information not stored"<< std::endl;
            }            
        }
    }    
    
    
    if( settings.outFile!="-" and writePartition ){
        std::chrono::time_point<std::chrono::system_clock> beforePartWrite = std::chrono::system_clock::now();
        std::string partOutFile = settings.outFile + ".partition";
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
		scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partition.getDistribution(), partition.getLocalValues() ) );
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
