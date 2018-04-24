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


#include "fast_map.h"
#include "dist_graph.h"
#include "xtrapulp.h"
#include "comms.h"
#include "pulp_util.h"
#include "util.h"
#include "io_pp.h"
#include "generate.h"

#include "RBC/Sort/SQuick.hpp"


extern int procid, nprocs;

/*

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
*/

extern bool verbose;

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
	
	if( !vm.count("numBlocks") ){
        settings.numBlocks = comm->getSize();
    }
    
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
    
    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	scai::dmemo::DistributionPtr noDistPtr  = graph.getColDistributionPtr();
	
    //-----------------------------------------------------
    //
    // convert to xtraPulp data types
    //
    
	// xtrapulp variables
	
	
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	
	
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
							 
	graph_gen_data_t ggi;
	dist_graph_t g;
	
	
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
	IndexType M = graph.getNumValues()/*/2*/;
	IndexType localM = graph.getLocalNumValues()/*/2*/;
	IndexType thisPE = comm->getRank();
	IndexType numPEs = comm->getSize();

	/*
	* 	version where we create and sort an edge list	
	*/

	std::vector<sort_pair> localPairs(localM);
	IndexType e = 0;
	
	{
		scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();
		scai::hmemo::ReadAccess<IndexType> ia( localMatrix.getIA() );
		scai::hmemo::ReadAccess<IndexType> ja( localMatrix.getJA() );
		SCAI_ASSERT_EQ_ERROR( localM, ja.size(), "Should be equal?");		
		
		for( IndexType i=0; i<localN; i++){
			IndexType v = dist->local2global(i);
			const IndexType beginCols = ia[i];
			const IndexType endCols = ia[i+1];

			for (IndexType j = beginCols; j < endCols; j++) {
				//IndexType neighbor = ja[j];
				localPairs[e].value = dist->local2global(i);	//global id of thisNode
				localPairs[e].index = ja[j];
				++e;
			}
		SCAI_ASSERT_LE_ERROR( e, localM , "Edge index too large");
		}
	}
	SCAI_ASSERT_EQ_ERROR(e, localM, "Edge count mismatch");
	SCAI_ASSERT_EQ_ERROR( comm->sum(e), M, "Global edge count mismatch");
	
	{	
		// globally sort edges
		//
		
//		const MPI_Comm mpi_comm = MPI_COMM_WORLD;
		SQuick::sort<sort_pair>(MPI_COMM_WORLD, localPairs, -1);
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	
	mpi_data_t xtrapulpComm;
PRINT(*comm << ": PE: " << procid << " , all " << nprocs ); 		
	init_comm_data(&xtrapulpComm);
	
	//
	// build ggi
	//
	
	ggi.n = N;
	ggi.m = M;
	
	ggi.m_local_read = localPairs.size();	// local size of edges
	
	IndexType maxLocalVertex=0;
	
	//WARNING: the next line is wrong as after the sort the localM can be different
	//uint64_t* gen_edges = new uint64_t[ 2*localM];
	uint64_t* gen_edges = new uint64_t[ 2*ggi.m_local_read ];	// 2 values per edge
	for(IndexType i=0; i<localPairs.size(); i++){
		gen_edges[2*i] = localPairs[i].value;
		gen_edges[2*i+1] = localPairs[i].index;
		SCAI_ASSERT_LE_ERROR( 2*i+1, 2*ggi.m_local_read, "Large edge index.");
		// check only the first vertex of the edge, it is considered to be local
		if( localPairs[i].value>maxLocalVertex ){
			maxLocalVertex = localPairs[i].value;
		}
	}
	ggi.gen_edges = gen_edges;
	
	uint64_t n_global = comm->max( maxLocalVertex );
	ggi.n = n_global +1;
	ggi.n_offset = (uint64_t)procid * (ggi.n / (uint64_t)nprocs + 1);
	ggi.n_local = ggi.n / (uint64_t)nprocs + 1;
	if (procid == nprocs - 1 )
		ggi.n_local = n_global - ggi.n_offset + 1;
	
	printf("Task %d, n %lu, n_offset %lu, n_local %lu, m %lu, m_local_read %lu\n", procid, ggi.n, ggi.n_offset, ggi.n_local, ggi.m , ggi.m_local_read);
	
	verbose = true;
	
	
	if (nprocs > 1){
		exchange_edges( &ggi, &xtrapulpComm );
		create_graph( &ggi, &g );
		relabel_edges( &g );
	}
	else{
		create_graph_serial(&ggi, &g);
	}
	
	/*
	 *  version where we create the graph_gen_data_t ggi and then call exchange_edges, create_graph, relabel_edges
	 */

/*	
	ggi.n = N;
	ggi.m = M;
	//ggi.n_local = localN;
	ggi.m_local_read = localM;
		
	//uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
	uint64_t* gen_edges = new uint64_t[ 2*localM ];	// 2 values per edge
	IndexType e = 0;
	
	{
		scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();
		scai::hmemo::ReadAccess<IndexType> ia( localMatrix.getIA() );
		scai::hmemo::ReadAccess<IndexType> ja( localMatrix.getJA() );
		SCAI_ASSERT_EQ_ERROR( ggi.m_local_read, ja.size(), "Should be equal?");		
		
		for( IndexType i=0; i<localN; i++){
			IndexType v = dist->local2global(i);
			const IndexType beginCols = ia[i];
			const IndexType endCols = ia[i+1];

			for (IndexType j = beginCols; j < endCols; j++) {
				//IndexType neighbor = ja[j];
				gen_edges[e++] = (uint64_t) v;
				gen_edges[e++] = (uint64_t) ja[j];
			}
		SCAI_ASSERT_LE_ERROR( e, 2*localM , "Edge index too large");
		}
	}
	SCAI_ASSERT_EQ_ERROR(e, 2*localM, "Edge count mismatch");
	SCAI_ASSERT_EQ_ERROR( comm->sum(e), 2*M, "Global edge count mismatch");
	
	ggi.gen_edges = gen_edges;
	
	ggi.n_offset = (uint64_t)procid * (ggi.n / (uint64_t)nprocs + 1);
	ggi.n_local = ggi.n / (uint64_t)nprocs + 1;
	if (procid == nprocs - 1 ){
		ggi.n_local = N - ggi.n_offset + 1; 
	}
	
	IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, N, thisPE, numPEs );
PRINT(*comm << ": " << beginLocalRange << " - " << endLocalRange );
	
	ggi.n_offset = beginLocalRange;
	ggi.n_local = localN;
	
	printf("Task %d, n %lu, n_offset %lu, n_local %lu, m %lu, m_local_read %lu\n", procid, ggi.n, ggi.n_offset, ggi.n_local, ggi.m , ggi.m_local_read);
	
	verbose = true;
	
	if (nprocs > 1){
		exchange_edges( &ggi, &xtrapulpComm );
		create_graph( &ggi, &g );
		relabel_edges( &g );
	}
	else{
		create_graph_serial(&ggi, &g);
	}
*/	
	
	
	/* 
	 * 		version where we create directly the distributed graph
	 * 
	 */ 
	
/*	
    g.n = N; //+1;				// global/total number of vertices
	g.m = 2*M;					// global/total number of edges
	g.m_local = localM;			// local number of edges
	//g.n_offset = thisPE * (g.n/numPEs); // +1);
	g.n_offset = dist->local2global(0);
	//g.n_local = g.n/numPEs;// +1;	// local number of vertices
	
	//SCAI_ASSERT_EQ_ERROR(g.n_local, localN, "Should be equal??" );
	
	g.n_local = localN;					// local number of vertices
	PRINT(*comm << ": g.n_local= " << g.n_local <<" , localN= " << localN);
	
	
	bool offset_vids = false;
	//if( thisPE==numPEs-1 && !offset_vids ){
	//		g.n_local = g.n - g.n_offset;// +1;
	//}
			
	g.vertex_weights = NULL;
	g.edge_weights = NULL;
	
	//WARNING: assuming that out_edges is the local ja values of the CSRSparseMatrix
	uint64_t* out_edges = new uint64_t[g.m_local];
	uint64_t* out_degree_list = new uint64_t[g.n_local +1];
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
		
		SCAI_ASSERT_EQ_ERROR( ia.size(), localN+1,  *comm << ": Should they be equal ?? ");
		SCAI_ASSERT_EQ_ERROR( ia.size(), g.n_local+1, *comm << ": Should they be equal ?? "); // ia.size == out_degree_list.size ??
		
		for(int i=0; i<ia.size() ; i++){
			out_degree_list[i]= ia[i];
			//PRINT(*comm << ": " << i << " , " << ia[i]);
		}
		
	}
	g.out_edges = out_edges;				//ja array of CSR sparce matrix
	g.out_degree_list = out_degree_list;	//ia array of CSR sparce matrix
	
	uint64_t* local_unmap = new uint64_t[ g.n_local];	// map from local index to global
	for (uint64_t i = 0; i < g.n_local; ++i){
		//local_unmap[i] = i + g.n_offset;
		local_unmap[i] = dist->local2global(i);
		//PRINT(*comm << ": " << i << ", g.local_unmap= " << i + g.n_offset << " ___ local2global= " << dist->local2global(i) );
		//SCAI_ASSERT_EQ_ERROR( i + g.n_offset, dist->local2global(i), "PE " << comm->getRank() <<": for i= " << i << ", id should be equal ?? ");
	}
	g.local_unmap = local_unmap;			// local ids to global
	
	//g.max_degree_vert =
	//g.max_degree	=
	
	//
	// ghost nodes
	//
	std::vector<IndexType> nonLocalNgbrs = ITI::GraphUtils::nonLocalNeighbors<IndexType,ValueType>( graph );
	//PRINT(*comm <<": "<< nonLocalNgbrs.size() );	
	
	g.n_ghost = nonLocalNgbrs.size();	//number of ghost nodes: nodes that share an edge with a local node but are not local
	//g.n_ghost = 1;
	//if( thisPE==0){
	//	g.n_ghost = g.n-g.n_local;
	//}
	g.n_total = g.n_local + g.n_ghost;	
	
	uint64_t* ghost_unmap = new uint64_t[g.n_ghost];			// global id of the ghost nodes
	for( unsigned int gh=0; gh<g.n_ghost; gh++){
		ghost_unmap[gh] = (uint64_t) nonLocalNgbrs[gh];
	}
	g.ghost_unmap = ghost_unmap;
	
	
	g.ghost_tasks = new uint64_t[g.n_ghost];			// WARNING: owning PE of each ghost node?
	scai::hmemo::HArray<IndexType> indexTransport(nonLocalNgbrs.size(), nonLocalNgbrs.data());
    // find the PEs that own every non-local index
    scai::hmemo::HArray<IndexType> owners(nonLocalNgbrs.size() , -1);
    dist->computeOwners( owners, indexTransport);
	
	scai::hmemo::ReadAccess<IndexType> rOwners(owners);
	std::vector<IndexType> neighborPEs(rOwners.get(), rOwners.get()+rOwners.size());
	g.ghost_tasks = (uint64_t *) neighborPEs.data();
    rOwners.release();
	

	// copied from xtrapulp/dist_graph.cpp::relabel_edges
  
	g.map = (struct fast_map*)malloc(sizeof(struct fast_map));		// ?
	
	uint64_t total_edges = g.m_local + g.n_local;

	if (total_edges * 2 < g.n){
		PRINT("\n\t\t"<< *comm<< ": initializing NO hash map with " << g.n<<" values \n\n");
		init_map_nohash(g.map, g.n); 
	}else{ 
		PRINT("\n\t\t"<< *comm<< ": initializing hash map with "<< total_edges*2<< " values \n");
		init_map(g.map, total_edges * 2);
	}
		
	for (uint64_t i = 0; i < g.n_local; ++i) {
		uint64_t vert = g.local_unmap[i];		//vert= global vertex id
		//if( comm->getRank()==0)
//PRINT(*comm <<":  vert "<< vert << " -> " << i);
		set_value(g.map, vert, i);
	}
	
	uint64_t cur_label = g.n_local;
	
	for (uint64_t i = 0; i < g.m_local; ++i) {	
		uint64_t out = g.out_edges[i];			// out= global id of neighbor
		uint64_t val = get_value(g.map, out);
			
		// if val==NULL_KEY then this neighbor is not local
		if (val == NULL_KEY) {
//PRINT( *comm << ":non-local neighbor " << out << ", setting value " << cur_label);
			set_value_uq(g.map, out, cur_label);
			g.out_edges[i] = cur_label++;
		} else{
//PRINT(*comm << "local neighbor " << out <<", setting value " << val);
			g.out_edges[i] = val;
		}
	}
*/	



	// PRINTS


	PRINT(*comm << ": g.n= " << g.n << ", g.m= " << g.m << ", g.m_local = " << g.m_local << ", g.n_local= " << g.n_local << \
			", g.n_offset= " << g.n_offset << ", g.n_ghost= " << g.n_ghost << ", g.n_total= " << g.n_total);

	
g.ghost_degrees = new uint64_t[g.n_ghost];		// degree of every ghost node	
queue_data_t q;
init_queue_data(&g, &q);
get_ghost_degrees(&g, &xtrapulpComm, &q);	


pulp_data_t pulp;
init_pulp_data(&g, &pulp, numBlocks);	
	



	//---------------------------------------------------------
    //
    //  xtraPulp partition
    //

    double sumPulpTime = 0.0;
    int repeatTimes = 1;
//printf("Task %d, n %lu, n_offset %lu, n_local %lu, m %lu, m_local_read %lu\n", procid, ggi.n, ggi.n_offset, ggi.n_local, ggi.m , ggi.m_local_read);
	
	int* pulpPartitionArray = new int[g.n_local];
	

	std::vector<struct Metrics> metricsVec;
	scai::lama::DenseVector<ValueType> nodeWeights = scai::lama::DenseVector<ValueType>( graph.getRowDistributionPtr(), 1);
	//scai::lama::DenseVector<IndexType> pulpPartitionDV(dist);
	
	// create a new GeneralDistribution based on the distribution created by the sort
	//WARNING: is it assumed here that the local indices are consequtive/successive??
	scai::hmemo::HArray<IndexType> localIndices( g.n_local , -1);
	{
		// get indices from offset till offset+n_local
		scai::hmemo::WriteOnlyAccess<IndexType> wLocalIndices(localIndices);
		for( IndexType i=0; i<g.n_local; i++){
			wLocalIndices[i] = g.n_offset+i;
		}
		SCAI_ASSERT_LE_ERROR( g.n_offset+g.n_local, N, *comm <<": Too large vertex index.");
		// check number of vertices in last PE
		if( thisPE==comm->getSize()-1){
			SCAI_ASSERT_EQ_ERROR(g.n_offset+g.n_local, N, *comm <<": Vertex index mismatch on last PE.");
		}
		//checksum for small values of N
		if( N<std::pow(2,26) ){		//TODO: do not ask why, just picked a value...
			IndexType localCheckSum = g.n_local*g.n_offset + (0.5)*(g.n_local)*(g.n_local+1);
			IndexType globalCheckSum = (0.5)*N*(N+1);
PRINT(*comm << ": " << localCheckSum << " == " << globalCheckSum <<"\t"<< g.n_offset << " , " << g.n_local);
			IndexType commGlobalSum = comm->sum(localCheckSum);
			SCAI_ASSERT_EQ_ERROR( globalCheckSum, commGlobalSum, "Global sum mismatch, maybe not correct, check again");
		}
	}
	
	const scai::dmemo::DistributionPtr pulpDist(new scai::dmemo::GeneralDistribution(N, localIndices, comm));
	scai::lama::DenseVector<IndexType> pulpPartitionDV(pulpDist);
	
	SCAI_ASSERT_EQ_ERROR( pulpPartitionDV.getLocalValues().size(), g.n_local, *comm<<": Sizes mismatch");
	IndexType newLocalN = pulpDist->getLocalSize();
	
	int r;
	for( r=0; r<repeatTimes; r++){		
		metricsVec.push_back( Metrics( comm->getSize()) );
		metricsVec[r].numBlocks = settings.numBlocks;
		
		std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
		//
		xtrapulp_run( &g, &ppc, pulpPartitionArray, numBlocks);
		//xtrapulp(&g, &ppc, &xtrapulpComm, &pulp, &q);
		//
		std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforePartTime;
		double partPulpTime= comm->max(partitionTime.count() );
		sumPulpTime += partPulpTime;
		
		
		//scai::lama::DenseVector<IndexType> pulpPartitionDV(dist);
		for(unsigned int i=0; i<newLocalN; i++){
			pulpPartitionDV.getLocalValues()[i] = pulpPartitionArray[i];
		}
		{
			// vector must have same distribution as the graph to get metrics
			//pulpPartitionDV.redistribute( dist );
			graph.redistribute( pulpDist, noDistPtr);
			nodeWeights.redistribute( pulpDist );
			
			metricsVec[r].finalCut = ITI::GraphUtils::computeCut(graph, pulpPartitionDV, true);
			metricsVec[r].finalImbalance = ITI::GraphUtils::computeImbalance<IndexType,ValueType>(pulpPartitionDV, settings.numBlocks ,nodeWeights);
			metricsVec[r].timeFinalPartition = ValueType (comm->max(partitionTime.count()));		
			
			// get metrics
			metricsVec[r].getMetrics( graph, pulpPartitionDV, nodeWeights, settings );
			// redistribute back to pulp distribution
			//pulpPartitionDV.redistribute( pulpDist );
			graph.redistribute( dist, noDistPtr);
			nodeWeights.redistribute( dist );
		}
		
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
	
	//delete[] gen_edges;
	
	/*
	delete[] out_edges;
	delete[] out_degree_list;
	delete[] local_unmap;
	delete[] ghost_unmap;
	*/
	
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
    if( writePartition && settings.outFile!="-"){
		if(comm->getRank()==0) std::cout<<" write partition" << std::endl;
		ITI::FileIO<IndexType, ValueType>::writePartitionCentral( pulpPartitionDV, settings.outFile+"_xtrapulp_k_"+std::to_string(numBlocks)+".partition");    
        
    }
    {
		comm->synchronize();
		// vector must have same distribution as the graph to get metrics
		//pulpPartitionDV.redistribute( dist );
		graph.redistribute( pulpDist, noDistPtr);
		ITI::aux<IndexType,ValueType>::print2DGrid(graph, pulpPartitionDV  );
		comm->synchronize();
	}
	
    //this is needed for supermuc
    std::exit(0);   
	
    return 0;
}
