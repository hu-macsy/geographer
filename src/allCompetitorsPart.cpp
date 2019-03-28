/*
 * Author Charilaos "Harry" Tzovas
 *
 *
 * example of use: 
 * mpirun -n #p parMetis --graphFile="meshes/hugetrace/hugetrace-00008.graph" --dimensions=2 --numBlocks=#k --geom=0
 * 
 * 
 * where #p is the number of PEs that the graph is distributed to and #k the number of blocks to partition to
*/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include "FileIO.h"
#include "Settings.h"
#include "Metrics.h"
#include "MeshGenerator.h"
#include "Wrappers.h"



extern "C"{
	void memusage(size_t *, size_t *,size_t *,size_t *,size_t *);	
}



//---------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

	using namespace boost::program_options;
	//options_description desc("Supported options");

	//int parMetisGeom = 0;			//0 no geometric info, 1 partGeomKway, 2 PartGeom (only geometry)
    //bool writePartition = false;
	bool storeInfo = true;
	ITI::Format coordFormat;
	std::string outPath;
	std::string graphName;
    std::string metricsDetail = "all";
	
	std::chrono::time_point<std::chrono::system_clock> startTime =  std::chrono::system_clock::now();
/*	
	desc.add_options()
		("help", "display options")
		("version", "show version")
		("graphFile", value<std::string>(), "read graph from file")
        ("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
		("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
		("coordFormat",  value<ITI::Format>(&coordFormat), "format of coordinate file")
        ("nodeWeightIndex", value<int>()->default_value(0), "index of node weight")
		
        ("generate", "generate random graph. Currently, only uniform meshes are supported.")
        ("numX", value<IndexType>(&settings.numX), "Number of points in x dimension of generated graph")
		("numY", value<IndexType>(&settings.numY), "Number of points in y dimension of generated graph")
		("numZ", value<IndexType>(&settings.numZ), "Number of points in z dimension of generated graph")        
        
		("dimensions", value<IndexType>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
		("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
        ("numBlocks", value<IndexType>(&settings.numBlocks), "Number of blocks to partition to")
        
		//TODO: parse the string to get these info automatically
		("outPath", value<std::string>(&outPath), "write result partition into file")
		("graphName", value<std::string>(&graphName), "this is needed to create the correct outFile for every tool. Must be the graphFile with the path and the ending")
		
		//("tool", value<std::string>(&tool), "The tool to partition with.")
		("tools", value<std::vector<std::string>>(&tools)->multitoken(), "The tool to partition with.")

		("computeDiameter", value<bool>(&settings.computeDiameter)->default_value(true), "Compute Diameter of resulting block files.")
		("storeInfo", "is this is false then no outFile is produced")
		("metricsDetail", value<std::string>(&metricsDetail), "no: no metrics, easy:cut, imbalance, communication volume and diamter if possible, all: easy + SpMV time and communication time in SpMV")
        //("writePartition", "Writes the partition in the outFile.partition file")
        ("writeDebugCoordinates", value<bool>(&settings.writeDebugCoordinates)->default_value(settings.writeDebugCoordinates), "Write Coordinates of nodes in each block")
		;
*/        

	struct Settings settings;
	variables_map vm = settings.parseInput( argc, argv);

	if( !settings.isValid )
		return -1;
	

	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType thisPE = comm->getRank();
    IndexType N;

/*

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}

	if (vm.count("version")) {
		std::cout << "Git commit " << version << std::endl;
		return 0;
	}

	if (! (vm.count("graphFile") or vm.count("generate")) ) {
		std::cout << "Specify input file with --graphFile or mesh generation with --generate and number of points per dimension." << std::endl; //TODO: change into positional argument
	}
           
	if( !vm.count("numBlocks") ){
        settings.numBlocks = comm->getSize();
    }

    if( (!vm.count("outPath")) and vm.count("storeInfo") ){
    	if( comm->getRank() ==0 ){
			std::cout<< "Must give parameter outPath to store metrics.\nAborting..." << std::endl;
			return -1;
		}
	}

	if( vm.count("metricsDetail") ){
		if( not (metricsDetail=="no" or metricsDetail=="easy" or metricsDetail=="all") ){
			if(comm->getRank() ==0 ){
				std::cout<<"WARNING: wrong value for parameter metricsDetail= " << metricsDetail << ". Setting to all" <<std::endl;
				metricsDetail="all";
			}
		}
	}
			
*/

    if( comm->getRank() ==0 ){
		std::cout <<"Starting file " << __FILE__ << std::endl;
		
		std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
		std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
		std::cout << "date and time: " << std::ctime(&timeNow) << std::endl;
	}
             
    //-----------------------------------------
    //
    // read the input graph or generate
    //
    
    CSRSparseMatrix<ValueType> graph;
    std::vector<DenseVector<ValueType>> coords(settings.dimensions);
    std::vector<DenseVector<ValueType>> nodeWeights;	//the weights for each node
    
    std::string graphFile;
    
    if (vm.count("graphFile")) {
        
        graphFile = vm["graphFile"].as<std::string>();
        std::string coordFile;
        if (vm.count("coordFile")) {
            coordFile = vm["coordFile"].as<std::string>();
        } else {
            coordFile = graphFile + ".xyz";
        }
              		
		// read the graph
        if (vm.count("fileFormat")) {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, settings.fileFormat );
        }else{
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights );
        }
        
        N = graph.getNumRows();
                
        SCAI_ASSERT_EQUAL( graph.getNumColumns(),  graph.getNumRows() , "matrix not square");
        SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");

        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
        		
		// set the node weigths
        IndexType numReadNodeWeights = nodeWeights.size();
        if (numReadNodeWeights == 0) {
        	nodeWeights.resize(1);
			nodeWeights[0] = fill<DenseVector<ValueType>>(rowDistPtr, 1);
		}

        if (settings.numNodeWeights > 0) {
            if (settings.numNodeWeights < nodeWeights.size()) {
                nodeWeights.resize(settings.numNodeWeights);
                if (comm->getRank() == 0) {
                    std::cout << "Read " << numReadNodeWeights << " weights per node but " << settings.numNodeWeights << " weights were specified, thus discarding "
                    << numReadNodeWeights - settings.numNodeWeights << std::endl;
                }
            } else if (settings.numNodeWeights > nodeWeights.size()) {
                nodeWeights.resize(settings.numNodeWeights);
                for (IndexType i = numReadNodeWeights; i < settings.numNodeWeights; i++) {
                    nodeWeights[i] = fill<DenseVector<ValueType>>(rowDistPtr, 1);
                }
                if (comm->getRank() == 0) {
                    std::cout << "Read " << numReadNodeWeights << " weights per node but " << settings.numNodeWeights << " weights were specified, padding with "
                    << settings.numNodeWeights - numReadNodeWeights << " uniform weights. " << std::endl;
                }
            }
        }

        //read the coordinates file
		if (vm.count("coordFormat")) {
			coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, coordFormat);
		}else if (vm.count("fileFormat")) {
			coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.fileFormat);
		} else {
			coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
		}
		SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size() , coords[1].getLocalValues().size(), "coordinates not of same size" );
        
    } else if(vm.count("generate")){
        if (settings.dimensions != 3) {
            if(comm->getRank() == 0) {
                std::cout << "Graph generation supported only fot 2 or 3 dimensions, not " << settings.dimensions << std::endl;
                return 127;
            }
        }
        
        N = settings.numX * settings.numY * settings.numZ;
        
        std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D
            
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
        graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>( rowDistPtr , noDistPtr );
                
        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<settings.dimensions; i++){
            coords[i].allocate(coordDist);
            coords[i] = static_cast<ValueType>( 0 );
        }
        
        // create the adjacency matrix and the coordinates
        ITI::MeshGenerator<IndexType, ValueType>::createStructured3DMesh_dist( graph, coords, maxCoord, numPoints);
        
        IndexType nodes= graph.getNumRows();
        IndexType edges= graph.getNumValues()/2;
        if(comm->getRank()==0){
            std::cout<< "Generated structured 3D graph with "<< nodes<< " and "<< edges << " edges."<< std::endl;
        }
        
    }else{
    	std::cout << "Either an input file or generation parameters are needed. Call again with --graphFile, --quadTreeFile, or --generate" << std::endl;
    	return 126;
    }
    
    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

	/*
	//TODO: must compile with mpicc and module mpi.ibm/1.4 and NOT mpi.ibm/1.4_gcc
	size_t total=-1,used=-1,free=-1,buffers=-1, cached=-1;
	memusage(&total, &used, &free, &buffers, &cached);	
	printf("MEM: avail: %ld , used: %ld , free: %ld , buffers: %ld , file cache: %ld \n",total,used,free,buffers, cached);
	*/

	//---------------------------------------------------------------------------------
	//
	// start main for loop for all tools
	//
	
	//WARNING: 1) removed parmetis sfc
	//WARNING: 2) parMetisGraph should be last because it often crashes
	std::vector<ITI::Tool> allTools = {ITI::Tool::zoltanRCB, ITI::Tool::zoltanRIB, ITI::Tool::zoltanMJ, ITI::Tool::zoltanSFC, ITI::Tool::parMetisSFC, ITI::Tool::parMetisGeom, ITI::Tool::parMetisGraph };

	//used for printing and creating filenames
	std::map<ITI::Tool, std::string> toolName = { 
		{ITI::Tool::zoltanRCB,"zoltanRCB"}, {ITI::Tool::zoltanRIB,"zoltanRIB"}, {ITI::Tool::zoltanMJ,"zoltanMJ"}, {ITI::Tool::zoltanSFC,"zoltanSFC"},
		{ITI::Tool::parMetisSFC,"parMetisSFC"}, {ITI::Tool::parMetisGeom,"parMetisGeom"}, {ITI::Tool::parMetisGraph,"parMetisGraph"} };
	
	std::vector<ITI::Tool> wantedTools;
	std::vector<std::string> tools = settings.tools; //not really needed

	if( tools[0] == "all"){
		wantedTools = allTools;
	}else{
		for( std::vector<std::string>::iterator tool=tools.begin(); tool!=tools.end(); tool++){
			ITI::Tool thisTool;
			if( (*tool).substr(0,8)=="parMetis"){
				if 		( *tool=="parMetisGraph"){	thisTool = ITI::Tool::parMetisGraph; }
				else if ( *tool=="parMetisGeom"){	thisTool = ITI::Tool::parMetisGeom;	}
				else if	( *tool=="parMetisSfc"){		thisTool = ITI::Tool::parMetisSFC;	}
			}else if ( (*tool).substr(0,6)=="zoltan"){
				std::string algo;
				if		( *tool=="zoltanRcb"){	thisTool = ITI::Tool::zoltanRCB;	}
				else if ( *tool=="zoltanRib"){	thisTool = ITI::Tool::zoltanRIB;	}
				else if ( *tool=="zoltanMJ"){	thisTool = ITI::Tool::zoltanMJ;		}
				else if ( *tool=="zoltanHsfc"){	thisTool = ITI::Tool::zoltanSFC;	}
			}else{
				std::cout<< "Tool "<< *tool <<" not supported.\nAborting..."<<std::endl;
				return -1;
			}
			wantedTools.push_back( thisTool );
		}
	}


	for( int t=0; t<wantedTools.size(); t++){
		
		ITI::Tool thisTool = wantedTools[t];
	
		// get the partition and metrics
		//
		scai::lama::DenseVector<IndexType> partition;
		
		// the constuctor with metrics(comm->getSize()) is needed for ParcoRepart timing details
		struct Metrics metrics( settings );
		//metrics.numBlocks = settings.numBlocks;
		
		// if usign unit weights, set flag for wrappers
		bool nodeWeightsUse = true;
		
		// if graph is too big, repeat less times to avoid memory and time problems
		if( N>std::pow(2,29) ){
			settings.repeatTimes = 2;
		}else{
			settings.repeatTimes = 5;
		}
		int parMetisGeom=0	;

		
		if( vm.count("outDir") and vm.count("storeInfo") ){
			//set the graphName in order to create the outFile name
			std::string copyName = graphFile;
			std::reverse( copyName.begin(), copyName.end() ); 
			std::vector<std::string> strs;			
			boost::split( strs, copyName, boost::is_any_of("./") );
			graphName = strs[1]; //[0] is "hparg" (graph reversed)
			std::reverse( graphName.begin(), graphName.end() );
			//PRINT0( graphName );		
			settings.outFile = settings.outDir	+ graphName+ "_k"+ std::to_string(settings.numBlocks)+ "_"+ toolName[thisTool]+ ".info";
		}else{
			settings.outFile ="-";
		}

		std::ifstream f(settings.outFile);
		if( f.good() and storeInfo ){
			comm->synchronize();	// maybe not needed
			PRINT0("\n\tWARNING: File " << settings.outFile << " allready exists. Skipping partition with " << toolName[thisTool]);
			continue;
		}
		
		//get the partition
		partition = ITI::Wrappers<IndexType,ValueType>::partition ( graph, coords, nodeWeights, nodeWeightsUse, thisTool, settings, metrics);
		
		PRINT0("time to get the partition: " <<  metrics.MM["timeFinalPartition"] );
		
		// partition has the the same distribution as the graph rows 
		SCAI_ASSERT_ERROR( partition.getDistribution().isEqual( graph.getRowDistribution() ), "Distribution mismatch.")
		
		
		if( metricsDetail=="all" ){
			metrics.getAllMetrics( graph, partition, nodeWeights, settings );
		}
        if( metricsDetail=="easy" ){
			metrics.getEasyMetrics( graph, partition, nodeWeights, settings );
		}
		
if( thisPE==0 ) metrics.printHorizontal2( std::cout );
		//---------------------------------------------------------------
		//
		// Reporting output to std::cout
		//
		
		char machineChar[255];
		std::string machine;
		gethostname(machineChar, 255);
		if (machineChar) {
			machine = std::string(machineChar);
		}
		
		if( thisPE==0 ){
			if( vm.count("generate") ){
				std::cout << std::endl << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon;
			}else{
				std::cout << std::endl << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon;
			}
			std::cout << "\nFinished tool" << std::endl;
			std::cout << "\033[1;36m";
			std::cout << "\n >>>> " << toolName[thisTool];
			std::cout<<  "\033[0m" << std::endl;

			//printMetricsShort( metrics, std::cout);

			// write in a file
			if( settings.outFile!= "-" ){
				std::ofstream outF( settings.outFile, std::ios::out);
				if(outF.is_open()){
					outF << "Running " << __FILE__ << " for tool " << toolName[thisTool] << std::endl;
					if( vm.count("generate") ){
						outF << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
					}else{
						outF << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
					}

					metrics.print( outF ); 
					//printMetricsShort( metrics, outF);
					std::cout<< "Output information written to file " << settings.outFile << std::endl;
				}else{
					std::cout<< "\n\tWARNING: Could not open file " << settings.outFile << " informations not stored.\n"<< std::endl;
				}       
			}

		    if( settings.outFile!="-" and settings.writeInFile ){
		        std::chrono::time_point<std::chrono::system_clock> beforePartWrite = std::chrono::system_clock::now();
		        std::string partOutFile = settings.outFile+".part";
				ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partition, partOutFile );

		        std::chrono::duration<double> writePartTime =  std::chrono::system_clock::now() - beforePartWrite;
		        if( comm->getRank()==0 ){
		            std::cout << " and last partition of the series in file " << partOutFile << std::endl;
		            std::cout<< " Time needed to write .partition file: " << writePartTime.count() <<  std::endl;
		        }
		    }   
		}
		
	// the code below writes the output coordinates in one file per processor for visualization purposes.
    //=================
    
    if (settings.writeDebugCoordinates) {
		
		std::vector<DenseVector<ValueType> > coordinateCopy = coords;
		
		//scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partition.getDistribution(), partition.getLocalValues() ) );		
		scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues());
        for (IndexType dim = 0; dim < settings.dimensions; dim++) {
            assert( coordinateCopy[dim].size() == N);
            //coordinates[dim].redistribute(partition.getDistributionPtr());
			coordinateCopy[dim].redistribute( distFromPartition );			
        }
        
        std::string destPath = "partResults/" +  toolName[thisTool] +"/blocks_" + std::to_string(settings.numBlocks) ;
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
            coords[dim].redistribute( noDistPtr );
        }
        */
    }
	/*		
	memusage(&total, &used, &free, &buffers, &cached);	
	printf("\nMEM: avail: %ld , used: %ld , free: %ld , buffers: %ld , file cache: %ld \n\n",total,used,free,buffers, cached);
	*/		
	} // for wantedTools.size()
     
	std::chrono::duration<ValueType> totalTimeLocal = std::chrono::system_clock::now() - startTime;
	ValueType totalTime = comm->max( totalTimeLocal.count() );
	if( thisPE==0 ){
		std::cout<<"Exiting file " << __FILE__ << " , total time= " << totalTime <<  std::endl;
	}

    //this is needed for supermuc
    std::exit(0);   
	
    return 0;
}

