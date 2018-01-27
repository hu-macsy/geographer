/*
 * Author Charilaos "Harry" Tzovas
 *
 *
 * example of use: 
 * mpirun -n #p pulpEXE --graphFile="meshes/hugetrace/hugetrace-00008.graph" --dimensions=2 --numBlocks=#k
 * 
 * 
 * where #p is the number of PEs that the graph is distributed to and #k the number of blocks to partition to
*/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <mpi.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include "FileIO.h"
#include "GraphUtils.h"
#include "Settings.h"
#include "Metrics.h"
#include "MeshGenerator.h"

/*
#include "comms.h"
#include "generate.h"
#include "io_pp.h"
#include "pulp_util.h"
#include "util.h"
*/
#include "fast_map.h"
#include "dist_graph.h"
#include "xtrapulp.h"

extern int procid, nprocs;



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
    else if (token == "BINARY" or token == "6")
        format = ITI::Format::BINARY;
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
    else if (method == ITI::Format::BINARY)
        token == "BINARY";
	out << token;
	return out;
}
}


//---------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;
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
        
        ("writePartition", "Writes the partition in the outFile.partition file")
        ("outFile", value<std::string>(&settings.outFile), "write result partition into file")
		;
        
		
	//--------------------------------------------------------------
	//
	// read options, input parameters and declare and initialize variables
	//
		
	variables_map vm;
	store(command_line_parser(argc, argv).
			  options(desc).run(), vm);
	notify(vm);

    writePartition = vm.count("writePartition");
	
	
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
	

	
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType N;
	IndexType numBlocks = settings.numBlocks;
	
             
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

	
    //-----------------------------------------------------
    //
    // convert to xtraPulp data types
    //
    
	// xtrapulp variables
	
	ValueType vert_balance = 1.1;
	ValueType edge_balance = 1.1;
	bool do_bfs_init = true;
	bool do_lp_init = false;
	bool do_repart = false;
	bool do_edge_balance = false;
	bool do_maxcut_balance = false;	
	int pulp_seed = rand();
		
	pulp_part_control_t ppc = {vert_balance,      edge_balance, do_lp_init,
                             do_bfs_init,       do_repart,    do_edge_balance,
                             do_maxcut_balance, false,        pulp_seed};
/*							 
	mpi_data_t pulpComm;
	init_comm_data(&pulpComm);
	
	queue_data_t q;
	init_queue_data(&g, &q);
	get_ghost_degrees(&g, &comm, &q);
	
	pulp_data_t pulp;
	init_pulp_data(&g, &pulp, numBlocks);
*/

    IndexType localN = dist->getLocalSize();
	IndexType M = graph.getNumValues()/2;
	IndexType localM = graph.getLocalNumValues()/*/2*/;
	IndexType thisPE = comm->getRank();
	IndexType numPEs = comm->getSize();

	dist_graph_t g;
	
    g.n = N /*+1*/;				// global/total number of vertices
	g.m = M;					// global/total number of edges
	g.m_local = localM;			// local number of edges
	g.n_offset = thisPE * (g.n/numPEs /*+1*/);
	g.n_local = g.n/numPEs /*+1*/;	// local number of vertices
	SCAI_ASSERT_EQ_ERROR(g.n_local, localN, "Should be equal??" );
	
	bool offset_vids = false;
	if( thisPE==numPEs-1 && !offset_vids ){
			g.n_local = g.n - g.n_offset /*+1*/;
	}
			
	g.vertex_weights = NULL;
	g.edge_weights = NULL;
	
	//WARNING: assuming that out_edges is the local ja values of the CSRSparseMatrix
	uint64_t* out_edges = new uint64_t[g.m_local];
	uint64_t* out_degree_list = new uint64_t[g.n_local+1];
	{
		scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();
		scai::hmemo::ReadAccess<IndexType> ia( localMatrix.getIA() );
		scai::hmemo::ReadAccess<IndexType> ja( localMatrix.getJA() );
		SCAI_ASSERT_EQ_ERROR( g.m_local, ja.size(), "Should be equal?");		
		
		for(int i=0; i<ja.size() ; i++){
			out_edges[i]= ja[i];
			//PRINT(*comm << ": " << i << " , " << ja[i]);
			SCAI_ASSERT( out_edges[i] >=0, "negative value for i= "<< i << " , val= "<< out_edges[i]);
		}
		for(int i=0; i<ia.size() ; i++){
			out_degree_list[i]= ia[i];
			//PRINT(*comm << ": " << i << " , " << ia[i]);
		}
		
	}
	g.out_edges = out_edges;				//ja array of CSR sparce matrix
	g.out_degree_list = out_degree_list;	//ia array of CSR sparce matrix
	
	uint64_t* local_unmap = new uint64_t[ g.n_local];	// map from local index to global
	for (uint64_t i = 0; i < g.n_local; ++i){
		local_unmap[i] = i + g.n_offset;
		//PRINT(*comm << ": " << i << ", g.local_unmap= " << i + g.n_offset << " ___ local2global= " << dist->local2global(i) );
		SCAI_ASSERT_EQ_ERROR( i + g.n_offset, dist->local2global(i), "global id should be equal ?? ");
	}
	g.local_unmap = local_unmap;
	
	//g.max_degree_vert =
	//g.max_degree	=
	
	//
	// ghost nodes
	//
	std::vector<IndexType> nonLocalNgbrs = ITI::GraphUtils::nonLocalNeighbors<IndexType,ValueType>( graph );
	//PRINT(*comm <<": "<< nonLocalNgbrs.size() );	
	
	g.n_ghost = nonLocalNgbrs.size();	//number of ghost nodes: nodes that share an edge with a local node but are not local
	g.n_total = g.n_local + g.n_ghost;	
	
	uint64_t* ghost_unmap = new uint64_t[g.n_ghost];			// global id of the ghost nodes
	for( unsigned int gh=0; gh<g.n_ghost; gh++){
		ghost_unmap[gh] = (uint64_t) nonLocalNgbrs[gh];
	}
	g.ghost_unmap = ghost_unmap;
	
	
	//g.ghost_tasks = new uint64_t[g.n_ghost];			// WARNING: owning PE of each ghost node?
	scai::utilskernel::LArray<IndexType> indexTransport(nonLocalNgbrs.size(), nonLocalNgbrs.data());
    // find the PEs that own every non-local index
    scai::hmemo::HArray<IndexType> owners(nonLocalNgbrs.size() , -1);
    dist->computeOwners( owners, indexTransport);
	
	scai::hmemo::ReadAccess<IndexType> rOwners(owners);
	std::vector<IndexType> neighborPEs(rOwners.get(), rOwners.get()+rOwners.size());
	g.ghost_tasks = (uint64_t *) neighborPEs.data();
    rOwners.release();
	
	
	g.ghost_degrees = new uint64_t[g.n_ghost];		// degree of every shost node
	
	/*
	PRINT( sizeof(g.ghost_degrees) );
		get_ghost_degrees(g, &comm, &q);
	PRINT( sizeof(g.ghost_degrees) );  
	*/
	

	// copied from xtrapulp/dist_graph.h
  
	g.map = (struct fast_map*)malloc(sizeof(struct fast_map));
	
	uint64_t cur_label = g.n_local;
	uint64_t total_edges = g.m_local + g.n_local;

	if (total_edges * 2 < g.n)
		init_map_nohash(g.map, g.n); 
	else 
		init_map(g.map, total_edges * 2);
		
		for (uint64_t i = 0; i < g.n_local; ++i) {
			uint64_t vert = g.local_unmap[i];
			set_value(g.map, vert, i);
		}

	for (uint64_t i = 0; i < g.m_local; ++i) {	
		uint64_t out = g.out_edges[i];
		uint64_t val = get_value(g.map, out);
		if (val == NULL_KEY) {
			set_value_uq(g.map, out, cur_label);
			g.out_edges[i] = cur_label++;
		} else
			g.out_edges[i] = val;
	}
	


	// PRINTS
/*	
	
	PRINT(g.m_local );
    for(unsigned int h=0; h<g.m_local; h++){
		std::cout<< g.out_edges[h] << ", ";
	}
	std::cout<< std::endl;
	PRINT(g.n_local);
	
	for(unsigned int h=0; h<g.n_local+1; h++){
		std::cout<< g.out_degree_list[h] << ", ";
	}
	std::cout<< std::endl;
	PRINT(g.n_offset);


	PRINT(*comm << ": g.n= " << g.n << ", g.m= " << g.m << ", g.m_local = " << g.m_local << ", g.n_local= " << g.n_local << \
			", g.n_offset= " << g.n_offset << ", g.ghost= " << g.n_ghost << ", g.n_total= " << g.n_total);
	
	for(unsigned int h=0; h<g.n_ghost; h++){
		PRINT(*comm << ": >>> " << g.ghost_unmap[h] << " ~~~~~ " << g.ghost_tasks[h]);// << "  ### " << g.ghost_degrees[h] );
	}
*/		


	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


	//---------------------------------------------------------
    //
    //  xtraPulp partition
    //

    double sumPulpTime = 0.0;
    int repeatTimes = 5;
    
	int* pulpPartitionArray = new int[g.n_local];
	
	std::vector<struct Metrics> metricsVec;
	scai::lama::DenseVector<ValueType> nodeWeights = scai::lama::DenseVector<ValueType>( graph.getRowDistributionPtr(), 1);
	scai::lama::DenseVector<IndexType> pulpPartitionDV(dist);
	
	int r;
	for( r=0; r<repeatTimes; r++){		
		metricsVec.push_back( Metrics( comm->getSize()) );
		
		std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
		//
		xtrapulp_run( &g, &ppc, pulpPartitionArray, numBlocks);
		//
		std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforePartTime;
		double partPulpTime= comm->max(partitionTime.count() );
		sumPulpTime += partPulpTime;
		
		
		//scai::lama::DenseVector<IndexType> pulpPartitionDV(dist);
		for(unsigned int i=0; i<localN; i++){
			pulpPartitionDV.getLocalValues()[i] = pulpPartitionArray[i];
		}
		
		metricsVec[r].finalCut = ITI::GraphUtils::computeCut(graph, pulpPartitionDV, true);
        metricsVec[r].finalImbalance = ITI::GraphUtils::computeImbalance<IndexType,ValueType>(pulpPartitionDV, settings.numBlocks ,nodeWeights);
        //metricsVec[r].inputTime = ValueType ( comm->max(inputTime.count() ));
        metricsVec[r].timeFinalPartition = ValueType (comm->max(partitionTime.count()));		
		
		// get metrics
		metricsVec[r].getMetrics( graph, pulpPartitionDV, nodeWeights, settings );
		
		if (comm->getRank() == 0 ) {
            metricsVec[r].print( std::cout );            
            std::cout<< "Running time for run number " << r << " is " << partPulpTime << std::endl;
        }
        
        if( sumPulpTime>500){
			std::cout<< "Stopping runs because of excessive running total running time: " << sumPulpTime << std::endl;
            break;
        }
    }// for(repeatTimes)
	
	if( r!=repeatTimes){		// in case it has to break before all the runs are completed
		repeatTimes = r+1;
	}
	if(comm->getRank()==0 ){
        std::cout<<"Number of runs: " << repeatTimes << std::endl;	
    }
    
    double avgPulpTime = sumPulpTime/repeatTimes;
	
	
	//
	// free arrays
	//
	
	delete[] out_edges;
	delete[] out_degree_list;
	delete[] local_unmap;
	delete[] ghost_unmap;
	
	//clear_graph( &g );
	
	//
    // convert partition to a DenseVector
    //
	/*
    scai::lama::DenseVector<IndexType> pulpPartitionDV(dist);
    for(unsigned int i=0; i<localN; i++){
        pulpPartitionDV.getLocalValues()[i] = pulpPartitionArray[i];
    }
    
	// check correct transformation to DenseVector
    for(int i=0; i<localN; i++){
        PRINT(*comm << ": "<< pulpPartitionArray[i] << " _ "<< pulpPartitionDV.getLocalValues()[i] );
        assert( pulpPartitionArray[i]== pulpPartitionDV.getLocalValues()[i]);
    }
    
    delete[] pulpPartitionArray;
    */
    
    
    //---------------------------------------------
    //
    // Get metrics
    //
    /*
    // the constuctor with metrics(comm->getSize()) is needed for ParcoRepart timing details
    struct Metrics metrics(1);
    
    metrics.timeFinalPartition = avgPulpTime;
    
    // uniform node weights
    scai::lama::DenseVector<ValueType> nodeWeights = scai::lama::DenseVector<ValueType>( graph.getRowDistributionPtr(), 1);
    metrics.getMetrics( graph, pulpPartitionDV, nodeWeights, settings );
    */
        
    //---------------------------------------------------------------
    //
    // Reporting output to std::cout and to the given outFile
    //
    
    char machineChar[255];
    std::string machine;
    gethostname(machineChar, 255);
    if (machineChar) {
        machine = std::string(machineChar);
    }
    
    //
    printVectorMetricsShort( metricsVec, std::cout );
    
    if(comm->getRank()==0){
        if( vm.count("generate") ){
            std::cout << std::endl << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon;
        }else{
            std::cout << std::endl << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon;
        }
        std::cout << "\033[1;36m";
		std::cout << std::endl << "XtraPulp: "<< std::endl;        
        std::cout<<  " \033[0m" << std::endl;

        //metrics.print( std::cout );
		//printMetricsShort( metricsVec, std::cout );
        //printVectorMetricsShort( metricsVec, std::cout ); 
		
        // write in a file
        if( settings.outFile!="-" ){
            std::ofstream outF( settings.outFile, std::ios::out);
            if(outF.is_open()){
                if( vm.count("generate") ){
                    outF << std::endl << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                }else{
                    outF << std::endl << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                }
                outF << "numBlocks= " << numBlocks << std::endl;
                //metrics.print( outF ); 
		printVectorMetricsShort( metricsVec, outF ); 
		//printMetricsShort( metrics, outF);
                std::cout<< "Output information written to file " << settings.outFile << std::endl;
            }else{
                std::cout<< "Could not open file " << settings.outFile << " informations not stored"<< std::endl;
            }       
        }
    }
    
    // WARNING: the function writePartitionCentral redistributes the coordinates
    if( writePartition ){
		if(comm->getRank()==0) std::cout<<" write partition" << std::endl;
		ITI::FileIO<IndexType, ValueType>::writePartitionCentral( pulpPartitionDV, settings.outFile+"_xtrapulp_k_"+std::to_string(numBlocks)+".partition");    
        
    }
  
		
    //this is needed for supermuc
    std::exit(0);   
	
    return 0;
}
