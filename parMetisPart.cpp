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
 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include "FileIO.h"
//#include "GraphUtils.h"
#include "Settings.h"
#include "Metrics.h"
#include "MeshGenerator.h"
#include "Wrappers.h"

#include <parmetis.h>

/*
void printCompetitorMetrics(struct Metrics metrics, std::ostream& out){
	out << "gather" << std::endl;
	out << "timeTotal finalCut imbalance maxBnd totBnd maxCommVol totCommVol maxBndPercnt avgBndPercnt" << std::endl;
    out << metrics.timeFinalPartition<< " " \
		<< metrics.finalCut << " "\
		<< metrics.finalImbalance << " "\
		<< metrics.maxBoundaryNodes << " "\
		<< metrics.totalBoundaryNodes << " "\
		<< metrics.maxCommVolume << " "\
		<< metrics.totalCommVolume << " ";
	out << std::setprecision(6) << std::fixed;
	out <<  metrics.maxBorderNodesPercent << " " \
		<<  metrics.avgBorderNodesPercent \
		<< std::endl; 
}
*/



//---------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;
    int parMetisGeom = 0;			//0 no geometric info, 1 partGeomKway, 2 PartGeom (only geometry)
    bool writePartition = false;
    
	desc.add_options()
		("help", "display options")
		("version", "show version")
		("graphFile", value<std::string>(), "read graph from file")
        ("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
		("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
		("coordFormat", value<ITI::Format>(), "format of coordinate file")
        
        ("generate", "generate random graph. Currently, only uniform meshes are supported.")
        ("numX", value<IndexType>(&settings.numX), "Number of points in x dimension of generated graph")
		("numY", value<IndexType>(&settings.numY), "Number of points in y dimension of generated graph")
		("numZ", value<IndexType>(&settings.numZ), "Number of points in z dimension of generated graph")        
        
		("dimensions", value<IndexType>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
		("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
        ("numBlocks", value<IndexType>(&settings.numBlocks), "Number of blocks to partition to")
        ("geom", value<int>(&parMetisGeom), "use ParMetisGeomKway, with coordinates. Default is parmetisKway (no coordinates)")
        
        ("writePartition", "Writes the partition in the outFile.partition file")
        ("outFile", value<std::string>(&settings.outFile), "write result partition into file")
        ("writeDebugCoordinates", value<bool>(&settings.writeDebugCoordinates)->default_value(settings.writeDebugCoordinates), "Write Coordinates of nodes in each block")
		;
        
	variables_map vm;
	store(command_line_parser(argc, argv).
	options(desc).run(), vm);
	notify(vm);

    //parMetisGeom = vm.count("geom");
    writePartition = vm.count("writePartition");
	bool writeDebugCoordinates = settings.writeDebugCoordinates;
	
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType N;
	
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

    if( comm->getRank()==0 ){
        std::cout << "\033[1;31m";
        std::cout << "IndexType size: " << sizeof(IndexType) << " , ValueType size: "<< sizeof(ValueType) << std::endl;
        if( sizeof(IndexType)!=sizeof(idx_t) ){
            std::cout<< "WARNING: IndexType size= " << sizeof(IndexType) << " and idx_t size=" << sizeof(idx_t) << "  do not agree, this may cause problems " << std::endl;
        }
        if( sizeof(ValueType)!=sizeof(real_t) ){
            std::cout<< "WARNING: IndexType size= " << sizeof(IndexType) << " and idx_t size=" << sizeof(idx_t) << "  do not agree, this may cause problems " << std::endl;
        }
        std::cout<<"\033[0m";
    }
             
    //-----------------------------------------
    //
    // read the input graph or generate
    //
    
    CSRSparseMatrix<ValueType> graph;
    std::vector<DenseVector<ValueType>> coords(settings.dimensions);
    
    std::string graphFile;
    
    if (vm.count("graphFile")) {
        
        graphFile = vm["graphFile"].as<std::string>();
        std::string coordFile;
        if (vm.count("coordFile")) {
            coordFile = vm["coordFile"].as<std::string>();
        } else {
            coordFile = graphFile + ".xyz";
        }
        
        if (vm.count("fileFormat")) {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, settings.fileFormat );
        }else{
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile );
        }
        
        N = graph.getNumRows();
                
        SCAI_ASSERT_EQUAL( graph.getNumColumns(),  graph.getNumRows() , "matrix not square");
        SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");
        
        if(parMetisGeom!=0 or settings.writeDebugCoordinates ){
            if (vm.count("fileFormat")) {
                coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.fileFormat);
            } else {
                coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
            }
            SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size() , coords[1].getLocalValues().size(), "coordinates not of same size" );
        }
        
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
        graph = scai::lama::CSRSparseMatrix<ValueType>( rowDistPtr , noDistPtr );
                
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
        
        //nodeWeights = scai::lama::DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);
    }else{
    	std::cout << "Either an input file or generation parameters are needed. Call again with --graphFile, --quadTreeFile, or --generate" << std::endl;
    	return 126;
    }
    
    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

	//--------------------------------------------
	//
	// get the partition and metrics
	//
	
	// the constuctor with metrics(comm->getSize()) is needed for ParcoRepart timing details
	struct Metrics metrics(1);
	
    // uniform node weights
    scai::lama::DenseVector<ValueType> nodeWeights = scai::lama::DenseVector<ValueType>( graph.getRowDistributionPtr(), 1);
	
	settings.repeatTimes = 5;
	
	DenseVector<IndexType> partitionKway = ITI::Wrappers<IndexType,ValueType>::metisPartition ( graph, coords, nodeWeights, parMetisGeom, settings, metrics);
	
	//DenseVector<IndexType> partitionKway = ITI::Wrappers<IndexType,ValueType>::zoltanPartition ( graph, coords, nodeWeights, zoltanAlgo, settings, metrics);
	
	//metrics.timeFinalPartition = avgKwayTime;
	PRINT0("time for partition: " <<  metrics.timeFinalPartition );
	
	
    metrics.getMetrics( graph, partitionKway, nodeWeights, settings );
    
        
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
    
    if(comm->getRank()==0){
		std::cout << "Running " << __FILE__ << std::endl;
        if( vm.count("generate") ){
            std::cout << std::endl << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon;
        }else{
            std::cout << std::endl << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon;
        }
        std::cout << "\033[1;36m";
        
        if( parMetisGeom ){
            std::cout << std::endl << "ParMETIS_V3_PartGeomKway: "<< std::endl;
        }else{
            std::cout << std::endl << "ParMETIS_V3_PartKway: " << std::endl;
        }
        std::cout<<  " \033[0m" << std::endl;

        metrics.print( std::cout );
        
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

                //metrics.print( outF ); 
				printMetricsShort( metrics, outF);
                std::cout<< "Output information written to file " << settings.outFile << std::endl;
            }else{
                std::cout<< "Could not open file " << settings.outFile << " informations not stored"<< std::endl;
            }       
        }
    }
    
    // the code below writes the output coordinates in one file per processor for visualization purposes.
    //=================

    // WARNING: the function writePartitionCentral redistributes the coordinates
    if( writePartition ){
        if( parMetisGeom ){    
            std::cout<<" write partition" << std::endl;
            ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partitionKway, settings.outFile+"_parMetisGeom_k_"+std::to_string(settings.numBlocks)+".partition");    
        }else{
            std::cout<<" write partition" << std::endl;
            ITI::FileIO<IndexType, ValueType>::writePartitionCentral( partitionKway, settings.outFile+"_parMetisGraph_k_"+std::to_string(settings.numBlocks)+".partition");    
        }
    }

    //settings.writeDebugCoordinates = 0;
    if (settings.writeDebugCoordinates and parMetisGeom) {
        scai::dmemo::DistributionPtr metisDistributionPtr = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partitionKway.getDistribution(), partitionKway.getLocalValues() ) );
        scai::dmemo::Redistributor prepareRedist(metisDistributionPtr, coords[0].getDistributionPtr());
        
		for (IndexType dim = 0; dim < settings.dimensions; dim++) {
			SCAI_ASSERT_EQ_ERROR( coords[dim].size(), N, "Wrong coordinates size for coord "<< dim);
			coords[dim].redistribute( prepareRedist );
		}
        
        std::string destPath;
        if( parMetisGeom){
            destPath = "partResults/parMetisGeom/blocks_" + std::to_string(settings.numBlocks) ;
        }else{
            destPath = "partResults/parMetis/blocks_" + std::to_string(settings.numBlocks) ;
        }
        boost::filesystem::create_directories( destPath );   		
		ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coords, N, settings.dimensions, destPath + "/metisResult");
    }
	        
    //this is needed for supermuc
    std::exit(0);   
	
    return 0;
}

