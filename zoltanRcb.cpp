// @HEADER
//
// ***********************************************************************
//
//   Zoltan2: A package of combinatorial algorithms for scientific computing
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Karen Devine      (kddevin@sandia.gov)
//                    Erik Boman        (egboman@sandia.gov)
//                    Siva Rajamanickam (srajama@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

/*! \file rcb_C.cpp
    \brief An example of partitioning coordinates with RCB.
*/

#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_InputTraits.hpp>
//#include <Tpetra_Map.hpp>

#include <vector>
#include <cstdlib>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include "FileIO.h"
#include "AuxiliaryFunctions.h"
#include "Settings.h"
#include "Metrics.h"
#include "MeshGenerator.h"

//using namespace std;
//using std::vector;
//using Teuchos::RCP;

/*! \example rcb_C.cpp
    An example of the use of the RCB algorithm to partition coordinate data.
*/

int main(int argc, char *argv[])
{
#ifdef HAVE_ZOLTAN2_MPI                   
  //MPI_Init(&argc, &argv);
  //int rank, nprocs;
  //MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  //int rank=0, nprocs=1;
#endif

  // For convenience, we'll use the Tpetra defaults for local/global ID types
  // Users can substitute their preferred local/global ID types
  //typedef Tpetra::Map<> Map_t;
  //typedef Map_t::local_ordinal_type localId_t;
  //typedef Map_t::global_ordinal_type globalId_t;

  //typedef double scalar_t;
  typedef Zoltan2::BasicUserTypes<ValueType, IndexType, IndexType> myTypes;

  // TODO explain
  typedef Zoltan2::BasicVectorAdapter<myTypes> inputAdapter_t;
  typedef Zoltan2::EvaluatePartition<inputAdapter_t> quality_t;
  typedef inputAdapter_t::part_t part_t;
  //
  
  	using namespace boost::program_options;
	options_description desc("Supported options");
      
	struct Settings settings;
	std::string algo = "rcb";			// the algorithm to be used by zoltan

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
        //("geom", value<bool>(&geomFlag), "use ParMetisGeomKway, with coordinates. Default is parmetisKway (no coordinates)")
		("algo", value<std::string>(&algo), "The algorithm that zoltan will run, choose from: rcb, rib, multijagged, hsfc, patoh, phg, metis, parmetis, pulp, parma, scotch,zoltan")
        
        ("writePartition", "Writes the partition in the outFile.partition file")
        ("outFile", value<std::string>(&settings.outFile), "write result partition into file")
        ("writeDebugCoordinates", value<bool>(&settings.writeDebugCoordinates)->default_value(settings.writeDebugCoordinates), "Write Coordinates of nodes in each block")
		("verbose", "print more info")
		("debug", "more output")
		;
        
	variables_map vm;
	store(command_line_parser(argc, argv).
	options(desc).run(), vm);
	notify(vm);
	
    bool geomFlag = true;
    bool writePartition = false;
	writePartition = vm.count("writePartition");
	bool writeDebugCoordinates = settings.writeDebugCoordinates;
	bool verbose = false;
	if( vm.count("verbose") ){
		verbose = true;
	}
	
	bool debug = false;
	if( vm.count("debug") ){
		debug = true;
	}
	
	const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const IndexType thisPE = comm->getRank();
	const IndexType numPEs = comm->getSize();
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
        		
        if( geomFlag or writeDebugCoordinates ){
            if (vm.count("fileFormat")) {
                coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, settings.fileFormat);
            } else {
                coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions);
            }
            SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size() , coords[1].getLocalValues().size(), "coordinates not of same size" );
        }
        
    } else if(vm.count("generate")){
        if (settings.dimensions != 3) {
            if(thisPE== 0) {
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

        if( thisPE== 0){
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
        if(thisPE==0){
            std::cout<< "Generated structured 3D graph with "<< nodes<< " and "<< edges << " edges."<< std::endl;
        }
        
        //nodeWeights = scai::lama::DenseVector<IndexType>(graph.getRowDistributionPtr(), 1);
    }else{
    	std::cout << "Either an input file or generation parameters are needed. Call again with --graphFile, --quadTreeFile, or --generate" << std::endl;
    	return 126;
    }
    
        
    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
	SCAI_ASSERT_EQ_ERROR( coords[0].getLocalValues().size(), dist->getLocalSize(), "Possible local size mismatch")
	
	const IndexType numBlocks = settings.numBlocks;
	
	//-----------------------------------------------------
    //
    // convert to zoltan data types
    //
    
	IndexType dimensions = settings.dimensions;
	IndexType localN= dist->getLocalSize();
	
	//TODO: point directly to the localCoords data and save time and space for zoltanCoords
	// the local part of coordinates for zoltan
	ValueType *zoltanCoords = new ValueType [dimensions * localN];

	std::vector<scai::utilskernel::LArray<ValueType>> localPartOfCoords( dimensions );
	for(unsigned int d=0; d<dimensions; d++){
		localPartOfCoords[d] = coords[d].getLocalValues();
	}
	IndexType coordInd = 0;
	for(int d=0; d<dimensions; d++){
		//SCAI_ASSERT_LE_ERROR( dimensions*(i+1), dimensions*localN, "Too large index, localN= " << localN );
		for(IndexType i=0; i<localN; i++){
			//zoltanCoords[dimensions*i+d] = localPartOfCoords[d][i];
			//SCAI_ASSERT_LE_ERROR( dimensions*(i+1), dimensions*localN, "Too large index, localN= " << localN );
			SCAI_ASSERT_LT_ERROR( coordInd, localN*dimensions, "Too large coordinate index");
			zoltanCoords[coordInd++] = localPartOfCoords[d][i];
		}
	}
  	
  	ValueType *x = zoltanCoords; 	// localPartOfCoords[0]
	ValueType *y = x + localN;  	// localPartOfCoords[1]
	ValueType *z = 0; 				// localPartOfCoords[2]
	
	if( dimensions==3 ){
		z = y + localN;
	}
	
	std::vector<const ValueType *>coordVec( dimensions );
	std::vector<int> coordStrides(dimensions);
	
	coordVec[0] = zoltanCoords; 	// coordVec[d] = localCoords[d].data(); or something
	coordStrides[0] = 1;
	/*
	coordVec[1] = y; 
	coordStrides[1] = 1;
	if( dimensions==3 ){
		coordVec[2] = z; 
		coordStrides[2] = 1;
	}
	*/
	for( int d=1; d<dimensions; d++){
		coordVec[d] = coordVec[d-1] + localN;
		coordStrides[d] = 1;
	}	
	
	
	if( debug ){
		for(int i=0; i<localN; i++){
			//PRINT( thisPE << ": " << dist->local2global(i) << " coords: ");
			std::cout<< thisPE << ": " << dist->local2global(i) << " coords: ";
			for(int d=0; d<dimensions; d++){
				std::cout<< coordVec[d][i] << ", ";
			}
			std::cout << std::endl;
		}		
	}
	///////////////////////////////////////////////////////////////////////
	// Create parameters for an RCB problem
	
	ValueType tolerance = 1.1;
	
	if (thisPE == 0)
		std::cout << "Imbalance tolerance is " << tolerance << std::endl;
	
	Teuchos::ParameterList params("test params");
	params.set("debug_level", "basic_status");
	params.set("debug_procs", "0");
	params.set("error_check_level", "debug_mode_assertions");
	
	params.set("algorithm", algo);
	params.set("imbalance_tolerance", tolerance );
	params.set("num_global_parts", numBlocks );		   
	


  
  // Create global ids for the coordinates.

  IndexType *globalIds = new IndexType [localN];
  IndexType offset = thisPE * localN;

  //TODO: can also be taken from the distribution?
  for (size_t i=0; i < localN; i++)
    globalIds[i] = offset++;

  
	std::vector<ValueType> localUnitWeight( localN, 1.0);
	std::vector<const ValueType *>weightVec(1);
	weightVec[0] = localUnitWeight.data();
	std::vector<int> weightStrides(1);
	weightStrides[0] = 1;
	
	inputAdapter_t *ia2=new inputAdapter_t(localN, globalIds, coordVec, 
                                         coordStrides, weightVec, weightStrides);
  
	//
	//
	//lala;
	//
	//
	
	
/*

	
  // Create a Zoltan2 input adapter for this geometry. TODO explain

  inputAdapter_t *ia1 = new inputAdapter_t(localN,globalIds,x,y,z,1,1,1);

  // Create a Zoltan2 partitioning problem
	Zoltan2::PartitioningProblem<inputAdapter_t> *problem1 =
           new Zoltan2::PartitioningProblem<inputAdapter_t>(ia1, &params);
      
	
		   
  // Solve the problem
	std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
	problem1->solve();

	std::chrono::duration<double> partitionTmpTime = std::chrono::system_clock::now() - beforePartTime;
	double partitionTime= comm->max(partitionTmpTime.count() );
	
	
  // create metric object where communicator is Teuchos default
  quality_t *metricObject1 = new quality_t(ia1, &params, //problem1->getComm(),
					   &problem1->getSolution());
  // Check the solution.

  if (rank == 0) {
    metricObject1->printMetrics(std::cout);
  }

  if (rank == 0){
    ValueType imb = metricObject1->getObjectCountImbalance();
    if (imb <= tolerance)
      std::cout << "pass: " << imb << std::endl;
    else
      std::cout << "fail: " << imb << std::endl;
    std::cout << std::endl;
  }
  delete metricObject1;

  
	//delete[] globalIds;
	
	//
    // convert partition to a DenseVector
    //

    DenseVector<IndexType> partitionZoltan(dist);
	
	const Zoltan2::PartitioningSolution<inputAdapter_t> &solution1 = problem1->getSolution();
	
	std::vector<IndexType> localBlockSize( numBlocks, 0 );
	
	const int *partAssignments = solution1.getPartListView();
	for(unsigned int i=0; i<localN; i++){
		IndexType thisBlock = partAssignments[i];
		SCAI_ASSERT_LT_ERROR( thisBlock, numBlocks, "found wrong vertex id");
		SCAI_ASSERT_GE_ERROR( thisBlock, 0, "found negetive vertex id");
		partitionZoltan.getLocalValues()[i] = thisBlock;
		localBlockSize[thisBlock]++;
		if( debug )
			PRINT(thisPE << ": " << dist->local2global(i) << " -- " << thisBlock );
	}
	
	// check correct transformation to DenseVector
	for(int i=0; i<localN; i++){
		//PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
		SCAI_ASSERT_EQ_ERROR( partitionZoltan.getLocalValues()[i], partAssignments[i], "Wrong conversion to DenseVector");
	}
	
	if( verbose ){
		PRINT(*comm);
		ITI::aux<IndexType,ValueType>::printVector( localBlockSize );
		PRINT(*comm << ": " << std::accumulate( localBlockSize.begin(), localBlockSize.end(), 0) );
	}
	comm->synchronize();
	if( verbose){
		std::vector<IndexType> globalBlockSize( numBlocks );
		comm->sumImpl( globalBlockSize.data(), localBlockSize.data(), numBlocks, scai::common::TypeTraits<ValueType>::stype);
		if(thisPE==0){
			ITI::aux<IndexType,ValueType>::printVector( globalBlockSize );
			PRINT(*comm << ": " << std::accumulate( globalBlockSize.begin(), globalBlockSize.end(), 0) );
		}
	}
	
	
	struct Metrics metrics(1);
	
	metrics.timeFinalPartition = partitionTime;
	PRINT0("time for partition: " <<  metrics.timeFinalPartition );
	
	metrics.getMetrics( graph, partitionZoltan, nodeWeights, settings );
	
	if( thisPE==0 ){
		printMetricsShort( metrics, std::cout );
	}
	*/
	
	
	//const std::vector<int> zoltanLocalPart (problem1->getSolution().getPartDistribution() );
	//const int *zoltanLocalPart;

	
	/*
	if( solution.oneToOnePartDistribution() ){
		zoltanLocalPart = solution.getProcDistribution();
	}else{
		zoltanLocalPart = solution.getPartDistribution();
	}
	
	for( int i=0; i<localN; i++){
		PRINT(thisPE << ": "<< i << " _ " << globalIds[i] << " >> " << zoltanLocalPart[i] );
	}
    */
    

    
    /*
    // check correct transformation to DenseVector
    for(int i=0; i<localN; i++){
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        assert( partKway[i]== partitionKway.getLocalValues()[i]);
    }
    */
   
  	Zoltan2::PartitioningProblem<inputAdapter_t> *problem2 =
           new Zoltan2::PartitioningProblem<inputAdapter_t>(ia2, &params);	
		   
	std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
	problem2->solve();
  	
	std::chrono::duration<double> partitionTmpTime = std::chrono::system_clock::now() - beforePartTime;
	double partitionTime= comm->max(partitionTmpTime.count() );
		
	quality_t *metricObject2 = new quality_t(ia2, &params, //problem1->getComm(),
					   &problem2->getSolution());
	
	if (thisPE == 0) {
		metricObject2->printMetrics(std::cout);
	}
	// uniform node weights
    scai::lama::DenseVector<ValueType> nodeWeights = scai::lama::DenseVector<ValueType>( graph.getRowDistributionPtr(), 1);
	

	///////////////////////////////////////////////////
	///////////////////////////////////////////////////
	//
    // convert partition to a DenseVector
    //
	DenseVector<IndexType> partitionZoltan2(dist);

	std::vector<IndexType> localBlockSize( numBlocks, 0 );
	
	const Zoltan2::PartitioningSolution<inputAdapter_t> &solution2 = problem2->getSolution();
	const int *partAssignments2 = solution2.getPartListView();
	for(unsigned int i=0; i<localN; i++){
		IndexType thisBlock = partAssignments2[i];
		SCAI_ASSERT_LT_ERROR( thisBlock, numBlocks, "found wrong vertex id");
		SCAI_ASSERT_GE_ERROR( thisBlock, 0, "found negetive vertex id");
		partitionZoltan2.getLocalValues()[i] = thisBlock;
		localBlockSize[thisBlock]++;
		if( debug )
			PRINT( thisPE << ": " << dist->local2global(i) << " -- " << thisBlock );
	}
	for(int i=0; i<localN; i++){
		//PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
		SCAI_ASSERT_EQ_ERROR( partitionZoltan2.getLocalValues()[i], partAssignments2[i], "Wrong conversion to DenseVector");
	}
	
	if( verbose ){
		PRINT(*comm);
		ITI::aux<IndexType,ValueType>::printVector( localBlockSize );
		PRINT(*comm << ": " << std::accumulate( localBlockSize.begin(), localBlockSize.end(), 0) );
	}
	comm->synchronize();
	if( verbose){
		std::vector<IndexType> globalBlockSize( numBlocks );
		comm->sumImpl( globalBlockSize.data(), localBlockSize.data(), numBlocks, scai::common::TypeTraits<ValueType>::stype);
		if(thisPE==0){
			ITI::aux<IndexType,ValueType>::printVector( globalBlockSize );
			PRINT(*comm << ": " << std::accumulate( globalBlockSize.begin(), globalBlockSize.end(), 0) );
		}
	}
		
	if( solution2.getPartDistribution()==NULL ){
		PRINT0("null PART dist");
	}
	if( solution2.getProcDistribution()==NULL ){
		PRINT0("null PROC dist");
	}
	
	struct Metrics metrics2(1);
	
	metrics2.timeFinalPartition = partitionTime;
	PRINT0("time for partition: " <<  metrics2.timeFinalPartition );
	
	metrics2.getMetrics( graph, partitionZoltan2, nodeWeights, settings );
	
	if( thisPE==0 ){
		printMetricsShort( metrics2, std::cout );
	}
	
	//PRINT(*comm << ": " << problem2->getSolution().oneToOnePartDistribution() );
	
/*	
    //---------------------------------------------
    //
    // Get metrics
    //
    
    // the constuctor with metrics(comm->getSize()) is needed for ParcoRepart timing details
    struct Metrics metrics(1);
    
    metrics.timeFinalPartition = partitionTime;
    
    // uniform node weights
    scai::lama::DenseVector<ValueType> nodeWeights = scai::lama::DenseVector<ValueType>( graph.getRowDistributionPtr(), 1);
    metrics.getMetrics( graph, partitionZoltan, nodeWeights, settings );
    
	
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
    
    if(thisPE==0){
		std::cout << "Running " << __FILE__ << std::endl;
        if( vm.count("generate") ){
            std::cout << std::endl << "machine:" << machine << " input: generated mesh,  nodes:" << N << " epsilon:" << settings.epsilon;
        }else{
            std::cout << std::endl << "machine:" << machine << " input: " << vm["graphFile"].as<std::string>() << " nodes:" << N << " epsilon:" << settings.epsilon;
        }
        
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
        if( geomFlag ){    
            std::cout<<" write partition" << std::endl;
            ITI::FileIO<IndexType, ValueType>::writePartitionParallel( partitionZoltan, settings.outFile+"_parMetisGeom_k_"+std::to_string(nparts)+".partition");    
        }else{
            std::cout<<" write partition" << std::endl;
            ITI::FileIO<IndexType, ValueType>::writePartitionCentral( partitionZoltan, settings.outFile+"_parMetisGraph_k_"+std::to_string(nparts)+".partition");    
        }
    }
    

    //settings.writeDebugCoordinates = 0;
    if (writeDebugCoordinates and geomFlag) {
        scai::dmemo::DistributionPtr metisDistributionPtr = scai::dmemo::DistributionPtr(new scai::dmemo::GeneralDistribution( partitionZoltan.getDistribution(), partitionZoltan.getLocalValues() ) );
        scai::dmemo::Redistributor prepareRedist(metisDistributionPtr, coords[0].getDistributionPtr());
        
		for (IndexType dim = 0; dim < settings.dimensions; dim++) {
			SCAI_ASSERT_EQ_ERROR( coords[dim].size(), N, "Wrong coordinates size for coord "<< dim);
			coords[dim].redistribute( prepareRedist );
		}
        
        std::string destPath;
        if( geomFlag ){
            destPath = "partResults/parMetisGeom/blocks_" + std::to_string(settings.numBlocks) ;
        }else{
            destPath = "partResults/parMetis/blocks_" + std::to_string(settings.numBlocks) ;
        }
        boost::filesystem::create_directories( destPath );   		
		ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed( coords, N, settings.dimensions, destPath + "/metisResult");
    }	
    */



	
/*  
  ///////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////
  // Try a problem with weights 
  ///////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////

  scalar_t *weights = new scalar_t [localCount];
  for (size_t i=0; i < localCount; i++){
    weights[i] = 1.0 + scalar_t(rank) / scalar_t(nprocs);
  }

  // Create a Zoltan2 input adapter that includes weights.

  vector<const scalar_t *>coordVec(2);
  vector<int> coordStrides(2);

  coordVec[0] = x; coordStrides[0] = 1;
  coordVec[1] = y; coordStrides[1] = 1;

  vector<const scalar_t *>weightVec(1);
  vector<int> weightStrides(1);

  weightVec[0] = weights; weightStrides[0] = 1;

  inputAdapter_t *ia2=new inputAdapter_t(localCount, globalIds, coordVec, 
                                         coordStrides,weightVec,weightStrides);

  // Create a Zoltan2 partitioning problem

  Zoltan2::PartitioningProblem<inputAdapter_t> *problem2 =
           new Zoltan2::PartitioningProblem<inputAdapter_t>(ia2, &params);

  // Solve the problem

  problem2->solve();

  // create metric object for MPI builds

#ifdef HAVE_ZOLTAN2_MPI
  quality_t *metricObject2 = new quality_t(ia2, &params, //problem2->getComm()
					   MPI_COMM_WORLD,
					   &problem2->getSolution());
#else
  quality_t *metricObject2 = new quality_t(ia2, &params, problem2->getComm(),
					   &problem2->getSolution());
#endif
  // Check the solution.

  if (rank == 0) {
    metricObject2->printMetrics(cout);
  }

  if (rank == 0){
    scalar_t imb = metricObject2->getWeightImbalance(0);
    if (imb <= tolerance)
      std::cout << "pass: " << imb << std::endl;
    else
      std::cout << "fail: " << imb << std::endl;
    std::cout << std::endl;
  }
  delete metricObject2;

  if (localCount > 0){
    delete [] weights;
    weights = NULL;
  }

  ///////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////
  // Try a problem with multiple weights.
  ///////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////

  // Add to the parameters the multicriteria objective.

  params.set("partitioning_objective", "multicriteria_minimize_total_weight");

  // Create the new weights.

  weights = new scalar_t [localCount*3];
  srand(rank);

  for (size_t i=0; i < localCount*3; i+=3){
    weights[i] = 1.0 + rank / nprocs;      // weight idx 1
    weights[i+1] = rank<nprocs/2 ? 1 : 2;  // weight idx 2
    weights[i+2] = rand()/RAND_MAX +.5;    // weight idx 3
  }

  // Create a Zoltan2 input adapter with these weights.

  weightVec.resize(3);
  weightStrides.resize(3);

  weightVec[0] = weights;   weightStrides[0] = 3;
  weightVec[1] = weights+1; weightStrides[1] = 3;
  weightVec[2] = weights+2; weightStrides[2] = 3;

  inputAdapter_t *ia3=new inputAdapter_t(localCount, globalIds, coordVec,
                                         coordStrides,weightVec,weightStrides);

  // Create a Zoltan2 partitioning problem.

  Zoltan2::PartitioningProblem<inputAdapter_t> *problem3 =
           new Zoltan2::PartitioningProblem<inputAdapter_t>(ia3, &params);

  // Solve the problem

  problem3->solve();

  // create metric object where Teuchos communicator is specified

  quality_t *metricObject3 = new quality_t(ia3, &params, problem3->getComm(),
					   &problem3->getSolution());
  // Check the solution.

  if (rank == 0) {
    metricObject3->printMetrics(cout);
  }

  if (rank == 0){
    scalar_t imb = metricObject3->getWeightImbalance(0);
    if (imb <= tolerance)
      std::cout << "pass: " << imb << std::endl;
    else
      std::cout << "fail: " << imb << std::endl;
    std::cout << std::endl;
  }
  delete metricObject3;

  ///////////////////////////////////////////////////////////////////////
  // Try the other multicriteria objectives.

  bool dataHasChanged = false;    // default is true

  params.set("partitioning_objective", "multicriteria_minimize_maximum_weight");
  problem3->resetParameters(&params);
  problem3->solve(dataHasChanged);    

  // Solution changed!

  metricObject3 = new quality_t(ia3, &params, problem3->getComm(),
                                &problem3->getSolution());
  if (rank == 0){
    metricObject3->printMetrics(cout);
    scalar_t imb = metricObject3->getWeightImbalance(0);
    if (imb <= tolerance)
      std::cout << "pass: " << imb << std::endl;
    else
      std::cout << "fail: " << imb << std::endl;
    std::cout << std::endl;
  }
  delete metricObject3;

  params.set("partitioning_objective", "multicriteria_balance_total_maximum");
  problem3->resetParameters(&params);
  problem3->solve(dataHasChanged);    

  // Solution changed!

  metricObject3 = new quality_t(ia3, &params, problem3->getComm(),
                                &problem3->getSolution());
  if (rank == 0){
    metricObject3->printMetrics(cout);
    scalar_t imb = metricObject3->getWeightImbalance(0);
    if (imb <= tolerance)
      std::cout << "pass: " << imb << std::endl;
    else
      std::cout << "fail: " << imb << std::endl;
    std::cout << std::endl;
  }
  delete metricObject3;

  if (localCount > 0){
    delete [] weights;
    weights = NULL;
  }

  ///////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////
  // Using part sizes, ask for some parts to be empty.
  ///////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////

  // Change the number of parts to twice the number of processes to
  // ensure that we have more than one global part.

  params.set("num_global_parts", nprocs*2);

  // Using the initial problem that did not have any weights, reset
  // parameter list, and give it some part sizes.

  problem1->resetParameters(&params);

  part_t partIds[2];
  scalar_t partSizes[2];

  partIds[0] = rank*2;    partSizes[0] = 0;
  partIds[1] = rank*2+1;  partSizes[1] = 1;

  problem1->setPartSizes(2, partIds, partSizes);

  // Solve the problem.  The argument "dataHasChanged" indicates 
  // that we have not changed the input data, which allows the problem
  // so skip some work when re-solving. 

  dataHasChanged = false;

  problem1->solve(dataHasChanged);

  // Obtain the solution

  const Zoltan2::PartitioningSolution<inputAdapter_t> &solution4 =
    problem1->getSolution();

  // Check it.  Part sizes should all be odd.

  const part_t *partAssignments = solution4.getPartListView();

  int numInEmptyParts = 0;
  for (size_t i=0; i < localCount; i++){
    if (partAssignments[i] % 2 == 0)
      numInEmptyParts++;
  }

  if (rank == 0)
    std::cout << "Request that " << nprocs << " parts be empty." <<std::endl;

  // Solution changed!

  metricObject1 = new quality_t(ia1, &params, //problem1->getComm(),
                                &problem1->getSolution());
  // Check the solution.

  if (rank == 0) {
    metricObject1->printMetrics(cout);
  }

  if (rank == 0){
    scalar_t imb = metricObject1->getObjectCountImbalance();
    if (imb <= tolerance)
      std::cout << "pass: " << imb << std::endl;
    else
      std::cout << "fail: " << imb << std::endl;
    std::cout << std::endl;
  }
  delete metricObject1;

  if (coords)
    delete [] coords;

  if (globalIds)
    delete [] globalIds;
*/

	delete[] zoltanCoords;
	
	//delete problem1;
	//delete ia1;
	delete problem2;
	delete ia2;
  //delete problem3;
  //delete ia3;

#ifdef HAVE_ZOLTAN2_MPI
  MPI_Finalize();
#endif

	if (thisPE == 0)
		std::cout << "PASS LALA !!" << std::endl;
	
	exit(0);
	return 0;
}

