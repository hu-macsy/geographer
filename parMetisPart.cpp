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

#include <parmetis.h>


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
    bool parMetisGeom = false;
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
        ("geom", "use ParMetisGeomKway, with coordinates. Default is parmetisKway (no coordinates)")
        
        ("writePartition", "Writes the partition in the outFile.partition file")
        ("outFile", value<std::string>(&settings.outFile), "write result partition into file")
        ("writeDebugCoordinates", value<bool>(&settings.writeDebugCoordinates)->default_value(settings.writeDebugCoordinates), "Write Coordinates of nodes in each block")
		;
        
	variables_map vm;
	store(command_line_parser(argc, argv).
			  options(desc).run(), vm);
	notify(vm);

    parMetisGeom = vm.count("geom");
    writePartition = vm.count("writePartition");
	bool writeDebugCoordinates = settings.writeDebugCoordinates;
	
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
        		
        if(parMetisGeom or writeDebugCoordinates ){
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

    //-----------------------------------------------------
    //
    // convert to parMetis data types
    //
    
    //get the vtx array
    
    IndexType size = comm->getSize();
    scai::hmemo::HArray<IndexType> sendVtx(size+1, static_cast<ValueType>( 0 ));
    scai::hmemo::HArray<IndexType> recvVtx(size+1);
    
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
    idx_t vtxDist[ size+1 ];
    vtxDist[0]= 0;
 
    for(int i=0; i<recvPartRead.size()-1; i++){
        vtxDist[i+1]= recvPartRead[i+1];
    }
    /*
    for(IndexType i=0; i<recvPartRead.size(); i++){
        PRINT(*comm<< " , " << i <<": " << vtxDist[i]);
    }
    */
    recvPartRead.release();

    //
    // setting xadj=ia and adjncy=ja values, these are the local values of every processor
    //
    
    scai::lama::CSRStorage<ValueType>& localMatrix= graph.getLocalStorage();
    
    scai::hmemo::ReadAccess<IndexType> ia( localMatrix.getIA() );
    scai::hmemo::ReadAccess<IndexType> ja( localMatrix.getJA() );
    IndexType iaSize= ia.size();
    
    idx_t* xadj = new idx_t[ iaSize ];
    idx_t* adjncy = new idx_t[ ja.size() ];
    
    for(int i=0; i<iaSize ; i++){
        xadj[i]= ia[i];
        SCAI_ASSERT( xadj[i] >=0, "negative value for i= "<< i << " , val= "<< xadj[i]);
    }

    for(int i=0; i<ja.size(); i++){
        adjncy[i]= ja[i];
        SCAI_ASSERT( adjncy[i] >=0, "negative value for i= "<< i << " , val= "<< adjncy[i]);
        SCAI_ASSERT( adjncy[i] <N , "too large value for i= "<< i << " , val= "<< adjncy[i]);
    }
    ia.release();
    ja.release();


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
    real_t *xyzLocal;

    if( parMetisGeom or writeDebugCoordinates ){
        xyzLocal = new real_t[ ndims*localN ];
        
        std::vector<scai::utilskernel::LArray<ValueType>> localPartOfCoords( ndims );
        for(int d=0; d<ndims; d++){
            localPartOfCoords[d] = coords[d].getLocalValues();
        }
        for(unsigned int i=0; i<localN; i++){
            SCAI_ASSERT_LE_ERROR( ndims*(i+1), ndims*localN, "Too large index, localN= " << localN );
            for(int d=0; d<ndims; d++){
                xyzLocal[ndims*i+d] = real_t(localPartOfCoords[d][i]);
            }
        }
        
    }
    // ncon: the numbers of weigths each vertex has. Here 1;
    idx_t ncon = 1;
    
    // nparts: the number of parts to partition (=k)
    if( !vm.count("numBlocks") ){
        settings.numBlocks = comm->getSize();
    }
    idx_t nparts= settings.numBlocks;
  
    // tpwgts: array of size ncons*nparts, that is used to specify the fraction of 
    // vertex weight that should be distributed to each sub-domain for each balance
    // constraint. Here we want equal sizes, so every value is 1/nparts.
    real_t tpwgts[ nparts ];
    real_t total = 0;
    for(int i=0; i<sizeof(tpwgts)/sizeof(real_t) ; i++){
	tpwgts[i] = real_t(1)/nparts;
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
    idx_t *partKway = new idx_t[ localN ];
    
    // comm: the MPI comunicator
    MPI_Comm metisComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &metisComm);
     
    //PRINT(*comm<< ": xadj.size()= "<< sizeof(xadj) << "  adjncy.size=" <<sizeof(adjncy) ); 
    //PRINT(*comm << ": "<< sizeof(xyzLocal)/sizeof(real_t) << " ## "<< sizeof(partKway)/sizeof(idx_t) << " , localN= "<< localN);
    
    if(comm->getRank()==0){
	    PRINT("dims=" << ndims << ", nparts= " << nparts<<", ubvec= "<< ubvec << ", options="<< *options << ", ncon= "<< ncon );
    }
     
    //
    // get the partitions with parMetis
    //

    double sumKwayTime = 0.0;
    int repeatTimes = 5;
    
    int metisRet;
    
    //
    // parmetis partition
    //
    int r;
    for( r=0; r<repeatTimes; r++){
        
        std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
        if( parMetisGeom ){
            metisRet = ParMETIS_V3_PartGeomKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ndims, xyzLocal, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );  
        }else{
            metisRet = ParMETIS_V3_PartKway( vtxDist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, partKway, &metisComm );
        }
        std::chrono::duration<double> partitionKwayTime =  std::chrono::system_clock::now() - beforePartTime;
        double partKwayTime= comm->max(partitionKwayTime.count() );
        sumKwayTime += partKwayTime;
        
        if( comm->getRank()==0 ){
            std::cout<< "Running time for run number " << r << " is " << partKwayTime << std::endl;
        }
        
        if( sumKwayTime>500){
			std::cout<< "Stopping runs because of excessive running total running time: " << sumKwayTime << std::endl;
            break;
        }
    }

	if( r!=repeatTimes){		// in case it has to break before all the runs are completed
		repeatTimes = r+1;
	}
	if(comm->getRank()==0 ){
        std::cout<<"Number of runs: " << repeatTimes << std::endl;	
    }
    
    double avgKwayTime = sumKwayTime/repeatTimes;

    //
    // free arrays
    //
    delete[] xadj;
    delete[] adjncy;
    if( parMetisGeom or writeDebugCoordinates ){
        delete[] xyzLocal;
    }
    
    //
    // convert partition to a DenseVector
    //
    DenseVector<IndexType> partitionKway(dist);
    for(unsigned int i=0; i<localN; i++){
        partitionKway.getLocalValues()[i] = partKway[i];
    }
    
    // check correct transformation to DenseVector
    for(int i=0; i<localN; i++){
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        assert( partKway[i]== partitionKway.getLocalValues()[i]);
    }
    
    delete[] partKway;
    
    //---------------------------------------------
    //
    // Get metrics
    //
    
    // the constuctor with metrics(comm->getSize()) is needed for ParcoRepart timing details
    struct Metrics metrics(1);
    
    metrics.timeFinalPartition = avgKwayTime;
    
    // uniform node weights
    scai::lama::DenseVector<ValueType> nodeWeights = scai::lama::DenseVector<ValueType>( graph.getRowDistributionPtr(), 1);
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
                if( vm.count("generate") ){
                    outF << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                }else{
                    outF << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon<< std::endl;
                }
                outF << "numBlocks= " << settings.numBlocks << std::endl;
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
            ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partitionKway, settings.outFile+"_parMetisGeom_k_"+std::to_string(nparts)+".partition");    
        }else{
            std::cout<<" write partition" << std::endl;
            ITI::FileIO<IndexType, ValueType>::writePartitionCentral( partitionKway, settings.outFile+"_parMetisGraph_k_"+std::to_string(nparts)+".partition");    
        }
    }

    //settings.writeDebugCoordinates = 0;
    if (writeDebugCoordinates and parMetisGeom) {
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

