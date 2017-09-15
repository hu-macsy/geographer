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

#include <scai/dmemo/BlockDistribution.hpp>

#include "FileIO.h"
#include "GraphUtils.h"
#include "Settings.h"

#include <parmetis.h>


typedef double ValueType;
typedef int IndexType;

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
	out << token;
	return out;
}
}

int main(int argc, char** argv) {

	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;
        bool parMetisGeom = true;
        
	desc.add_options()
		("help", "display options")
		("version", "show version")
		("graphFile", value<std::string>(), "read graph from file")
                ("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
		("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
		("coordFormat", value<ITI::Format>(), "format of coordinate file")
		("dimensions", value<int>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
		("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
                ("numBlocks", value<IndexType>(&settings.numBlocks), "Number of blocks to partition to")
                ("geom", value<bool>(&parMetisGeom)->default_value(parMetisGeom), "0: use of parmetisKway (no coordinates), 1: use of ParMetisGeomKway. Default is 1.")
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

	if (!vm.count("graphFile")) {
		std::cout << "Input file needed, specify with --graphFile" << std::endl; //TODO: change into positional argument
	}

	std::string graphFile = vm["graphFile"].as<std::string>();
	std::string coordFile;
	if (vm.count("coordFile")) {
		coordFile = vm["coordFile"].as<std::string>();
	} else {
		coordFile = graphFile + ".xyz";
	}
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    //
    // read the input graph
    //
    
    CSRSparseMatrix<ValueType> graph;
    
    if (vm.count("fileFormat")) {
        graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, settings.fileFormat );
    }else{
        graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile );
    }
    
    const IndexType N = graph.getNumRows();
    
    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    
    SCAI_ASSERT_EQUAL( graph.getNumColumns(),  graph.getNumRows() , "matrix not square");
    SCAI_ASSERT( graph.isConsistent(), "Graph npt consistent");
    
    std::vector<DenseVector<ValueType>> coords;
	if (vm.count("fileFormat")) {
		//ITI::Format format = vm["coordFormat"].as<ITI::Format>();
		coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.fileFormat);
	} else {
		coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
	}

    SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size() , coords[1].getLocalValues().size(), "coordinates not of same size" );

    //
    // convert to parMetis data types
    //
    
    //get the vtx array
    
    IndexType size = comm->getSize()+1;
    scai::hmemo::HArray<IndexType> sendVtx(size, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvVtx(size);
    
    IndexType lb, ub;
    scai::dmemo::BlockDistribution blockDist(N, comm);
    blockDist.getLocalRange(lb, ub, N, comm->getRank(), comm->getSize() );
    //PRINT(*comm<< ": "<< lb << " _ "<< ub);
    

    for(IndexType round=0; round<comm->getSize(); round++){
        SCAI_REGION("ParcoRepart.getBlockGraph.shiftArray");
        {   // write your part 
            scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendVtx );
            sendPartWrite[0]=0;
            sendPartWrite[comm->getRank()+1]=ub;
        }
        comm->shiftArray(recvVtx , sendVtx, 1);
        sendVtx.swap(recvVtx);
    } 

    scai::hmemo::ReadAccess<IndexType> recvPartRead( recvVtx );
    // vtxDist is an array of size numPEs and is replicated in every processor
    idx_t vtxDist[ comm->getSize()+1 ]; 
    vtxDist[0]= 0;
    for(int i=0; i<recvPartRead.size(); i++){
        vtxDist[i+1]= recvPartRead[i+1];
        //vtxDist[i]= recvPartRead[i];
    }
    /*
    for(IndexType i=0; i<recvPartRead.size(); i++){
        PRINT(*comm<< " , " << i <<": " << vtxDist[i]);
    }
    */
    recvPartRead.release();

    // setting xadj=ia and adjncy=ja values, these are the local values of every processor
    scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();
    scai::utilskernel::LArray<IndexType>& ia = localMatrix.getIA();
    scai::utilskernel::LArray<IndexType>& ja = localMatrix.getJA();

    idx_t xadj[ia.size()], adjncy[ja.size()];
    for(int i=0; i<ia.size(); i++){
        SCAI_ASSERT( i < sizeof(xadj)/sizeof(idx_t), "index " << i << " out of bounds");
        xadj[i]= ia[i];
        SCAI_ASSERT( xadj[i] >=0, "negative value for i= "<< i << " , val= "<< xadj[i]);
    }

    for(int i=0; i<ja.size(); i++){
        adjncy[i]= ja[i];
        SCAI_ASSERT( adjncy[i] >=0, "negative value for i= "<< i << " , val= "<< adjncy[i]);
        SCAI_ASSERT( adjncy[i] <N , "too large value for i= "<< i << " , val= "<< adjncy[i]);
    }

    //vwgt , adjwgt store the weigths of edges and vertices. Here we have
    // no weight so are both NULL.
    idx_t* vwgt= NULL;
    idx_t* adjwgt= NULL;
    
    // wgtflag is for the weight and can take 4 values. Here =0.
    idx_t wgtflag= 0;
    
    // numflag: 0 for C-style (start from 0), 1 for Fortrant-style (start from 1)
    idx_t numflag= 0;
    
    // ndims: the number of dimensions
    idx_t ndims = settings.dimensions;

    // the xyz array for coordinates of size dim*localN contains the local coords
    // convert the vector<DenseVector> to idx_t*
    IndexType localN= dist->getLocalSize();
    real_t xyzLocal[ndims*localN];

    if (ndims != 2) {
    	throw std::logic_error("Not yet implemented for dimensions != 2");
    }

    // TODO: only for 2D now
    scai::utilskernel::LArray<ValueType>& localPartOfCoords0 = coords[0].getLocalValues();
    scai::utilskernel::LArray<ValueType>& localPartOfCoords1 = coords[1].getLocalValues();
    
    for(unsigned int i=0; i<localN; i++){
        SCAI_ASSERT( 2*i+1< sizeof(xyzLocal)/sizeof(xyzLocal[0]), "Too large index: " << 2*i+1);
        xyzLocal[2*i]= real_t(localPartOfCoords0[i]);
        xyzLocal[2*i+1]= real_t(localPartOfCoords1[i]);
        //PRINT(*comm <<": "<< xyzLocal[2*i] << ", "<< xyzLocal[2*i+1]);
    }

    // ncon: the numbers of weigths each vertex has. Here 1;
    idx_t ncon = 1;
    
    // nparts: the number of parts to partition (=k)
    //IndexType k = comm->getSize();
    if( !vm.count("numBlocks") ){
        settings.numBlocks = comm->getSize();
    }
    IndexType k = settings.numBlocks;
    idx_t nparts= k;
  
    // tpwgts: array of size ncons*nparts, that is used to specify the fraction of 
    // vertex weight that should be distributed to each sub-domain for each balance
    // constraint. Here we want equal sizes, so every value is 1/nparts.
    real_t tpwgts[ k];
    real_t total = 0;
    for(int i=0; i<sizeof(tpwgts)/sizeof(real_t) ; i++){
	tpwgts[i] = real_t(1)/k;
        //PRINT(*comm << ": " << i <<": "<< tpwgts[i]);
	total += tpwgts[i];
    }

    // ubvec: array of size ncon to specify imbalance for every vertex weigth.
    // 1 is perfect balance and nparts perfect imbalance. Here 1 for now
    real_t ubvec= settings.epsilon + 1;
    
    // options: array of integers for passing arguments.
    // Here, options[0]=0 for the default values.
    idx_t options[1]= {0};
    
    //
    // OUTPUT parameters
    //
    // edgecut: the size of cut
    idx_t edgecut;
    
    // partition array of size localN, contains the block every vertex belongs
    //idx_t* partKway = (idx_t*) malloc(sizeof(idx_t)*localN);
    idx_t partKway[localN];
    
    // comm: the MPI comunicator
    MPI_Comm metisComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &metisComm);
     
    //PRINT(*comm<< ": xadj.size()= "<< sizeof(xadj) << "  adjncy.size=" <<sizeof(adjncy) ); 
    //PRINT(*comm << ": "<< sizeof(xyzLocal)/sizeof(real_t) << " ## "<< sizeof(partKway)/sizeof(idx_t) << " , localN= "<< localN);
    
    SCAI_ASSERT( sizeof(partKway)/sizeof(partKway[0])==localN , sizeof(partKway)/sizeof(partKway[0]) << " , " << localN);
    
    if(comm->getRank()==0){
	    PRINT("dims=" << ndims << ", nparts= " << nparts<<", ubvec= "<< ubvec << ", options="<< *options << ", ncon= "<< ncon );
    }
PRINT0( "" );        
    //
    // get the partitions with parMetis
    //

    std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
    int metisRet;

    //
    // parmetis partition
    //
    if( parMetisGeom ){
        metisRet = ParMETIS_V3_PartGeomKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ndims, xyzLocal, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );
    }else{
        metisRet = ParMETIS_V3_PartKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );
    }
    
    std::chrono::duration<double> partitionKwayTime =  std::chrono::system_clock::now() - beforePartTime;

    double partKwayTime= comm->max(partitionKwayTime.count() );

    
    // convert partition to a DenseVector
    //
    DenseVector<IndexType> partitionKway(dist);
    for(unsigned int i=0; i<localN; i++){
        partitionKway.getLocalValues()[i] = partKway[i];
    }
    ValueType cutKway = ITI::GraphUtils::computeCut(graph, partitionKway, true);
    ValueType imbalanceKway = ITI::GraphUtils::computeImbalance<IndexType, ValueType>( partitionKway, nparts );
    IndexType maxComm = ITI::GraphUtils::computeMaxComm<IndexType, ValueType>( graph, partitionKway, nparts);
    assert(sizeof(xyzLocal)/sizeof(real_t) == 2*sizeof(partKway)/sizeof(idx_t) );
  
    // check correct transformation to DenseVector
    for(int i=0; i<localN; i++){
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        assert( partKway[i]== partitionKway.getLocalValues()[i]);
    }

    char machineChar[255];
    std::string machine;
    gethostname(machineChar, 255);
    if (machineChar) {
        machine = std::string(machineChar);
    }
    
    if(comm->getRank()==0){
    	std::cout << std::endl << "machine:" << machine << " input:" << graphFile << " nodes:" << N << " epsilon:" << settings.epsilon;
        std::cout << "\033[1;36m";
        if( parMetisGeom ){
            std::cout << std::endl << "ParMETIS_V3_PartGeomKway cut= ";
        }else{
            std::cout << std::endl << "ParMETIS_V3_PartKway cut= ";
        }
        
        std::cout<< cutKway <<" imbalance:" << imbalanceKway<<", time for partition: "<< partKwayTime << " , maxComm=" << maxComm << " \033[0m" << std::endl;
        
        //std::cout<< std::endl << "ParMetisGeom cut= "<< cutGeom <<" and imbalance= " << imbalanceGeom<<", time for partition: "<< partGeomTime << std::endl;

    }
    return 0;
}

