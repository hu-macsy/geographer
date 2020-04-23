#pragma once

#include "sys/types.h"
#include "sys/sysinfo.h"
#include "sys/times.h"
#include "sys/vtimes.h"

#include <cxxopts.hpp>

#include "AuxiliaryFunctions.h"
#include "FileIO.h"
#include "Settings.h"
#include "Metrics.h"
#include "MeshGenerator.h"
#include "parseArgs.h"

namespace ITI{

/** Read the needed parameters from the virtual machine and return the input data.
    @param[in] vm The virtual machine with the input parameters
    @param[in/out] settings Some input settings. Some of them might change
    @param[in] comm The communicator
    @param[out] graph The returned input matrix
    @param[out] coords The returned input coordinates
    @param[out] nodeWeights The returned input node weights
*/

template <typename ValueType>
IndexType readInput( 
    const cxxopts::ParseResult& vm,
    Settings& settings,
    const scai::dmemo::CommunicatorPtr& comm,
    scai::lama::CSRSparseMatrix<ValueType>& graph,
    std::vector<scai::lama::DenseVector<ValueType>>& coords,
    std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights ){

    IndexType N;


    if (vm.count("graphFile")) {
        std::string graphFile =  vm["graphFile"].as<std::string>();
        std::string coordFile;

        if (vm.count("coordFile")) {
            coordFile = vm["coordFile"].as<std::string>();
        } else {
            coordFile = graphFile + ".xyz";
        }

        // read the graph
        if (vm.count("fileFormat")) {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, comm, settings.fileFormat );
        } else {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, comm );
        }

        N = graph.getNumRows();

        SCAI_ASSERT_EQUAL( graph.getNumColumns(),  graph.getNumRows(), "matrix not square");
        SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");

        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();

        // set the node weights
        IndexType numReadNodeWeights = nodeWeights.size();

        //Case where we ask to automatically find cpu and memory. This creates an tree with 2 nodeweights.
        // If input has less node weights add unit weights
        if( settings.autoSetCpuMem and numReadNodeWeights!=2){
            if (comm->getRank() == 0) {
                std::cout << "WARNING:\n\toption autoSetCpuMem is activated and it will create a tree with two node weights"<< std::endl;
                std::cout<< "\tbut input has " << numReadNodeWeights << " number of weights. Will adapt (pad or remove) input weights and"<< std::endl;
                std::cout <<"\twill consider only two weights." << std::endl;
            }
            settings.numNodeWeights = 2;
        }

        if (numReadNodeWeights == 0 and settings.numNodeWeights==0) {
            nodeWeights.resize(1);
            nodeWeights[0] = fill<DenseVector<ValueType>>(rowDistPtr, 1.0);
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
                    nodeWeights[i] = fill<DenseVector<ValueType>>(rowDistPtr, 1.0);
                }
                if (comm->getRank() == 0) {
                    std::cout << "Read " << numReadNodeWeights << " weights per node but " << settings.numNodeWeights << " weights were specified, padding with "
                              << settings.numNodeWeights - numReadNodeWeights << " uniform weights. " << std::endl;
                }
            }
        }

        //read the coordinates file
        if (vm.count("coordFormat")) {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, comm, settings.coordFormat);
        } else if (vm.count("fileFormat")) {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, comm, settings.fileFormat);
        } else {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, comm);
        }
        SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size(), coords[1].getLocalValues().size(), "coordinates not of same size" );      

    }else if(vm.count("generate")) {

        N = settings.numX * settings.numY * settings.numZ;

        std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D
        maxCoord[0] = settings.numX;
        maxCoord[1] = settings.numY;
        if(settings.dimensions==3) {
            maxCoord[2] = settings.numZ;
        }

        std::vector<IndexType> numPoints(3); // number of points in each dimension, used only for 3D

        for (IndexType i = 0; i < settings.dimensions; i++) {
            numPoints[i] = maxCoord[i];
        }

        if( comm->getRank()== 0) {
            std::cout<< "Generating for dim= "<< settings.dimensions << " and numPoints= "<< settings.numX << ", " << settings.numY << ", "<< settings.numZ << ", in total "<< N << " number of points" << std::endl;
            std::cout<< "\t\t and maxCoord= "; //<< maxCoord[0] << ", "<< maxCoord[1] << ", " << maxCoord[2]<< std::endl;
            for (IndexType i = 0; i < settings.dimensions; i++) {
                std::cout << maxCoord[i] << ", ";
            }
            std::cout << std::endl;
        }

        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>( rowDistPtr, noDistPtr );

        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<settings.dimensions; i++) {
            coords[i].allocate(coordDist);
            coords[i] = static_cast<ValueType>( 0 );
        }

        // create the adjacency matrix and the coordinates
        ITI::MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist( graph, coords, maxCoord, numPoints, settings.dimensions );

        IndexType nodes= graph.getNumRows();
        IndexType edges= graph.getNumValues()/2;
        if(comm->getRank()==0) {
            std::cout<< "Generated random 3D graph with "<< nodes<< " and "<< edges << " edges."<< std::endl;
        }

        nodeWeights.resize(1);
        nodeWeights[0] = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);

    } else if (vm.count("quadTreeFile")) {
        //if (comm->getRank() == 0) {
        graph = ITI::FileIO<IndexType, ValueType>::readQuadTree(vm["quadTreeFile"].as<std::string>(), coords);
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
            coords[i].redistribute(rowDistPtr);
        }

        nodeWeights.resize(1);
        nodeWeights[0] = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);

    }else{
        std::cout << "No input file was given. Call again with --graphFile, --quadTreeFile" << std::endl;
        return 126;        
    }

    if( not aux<IndexType,ValueType>::checkConsistency( graph, coords, nodeWeights, settings) ){
        throw std::runtime_error("Input not consistent.\nAborting...");
        return -1;
    }

    return N;
}


void printInfo(std::ostream& out, const scai::dmemo::CommunicatorPtr comm, const Settings settings){
    if (comm->getRank() == 0) {
        std::chrono::time_point<std::chrono::system_clock> now =  std::chrono::system_clock::now();
        std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
        out << "date and time: " << std::ctime(&timeNow);

        out<< "commit: "<< version << " , machine: " << settings.machine << " , p: "<< comm->getSize();

        auto oldprecision = std::cout.precision(std::numeric_limits<double>::max_digits10);
        out <<" seed:" << settings.seed << std::endl;
        out.precision(oldprecision);

        out << "Calling command:" << std::endl;
        out << settings.callingCommand << std::endl << std::endl;
    }
}


Settings initialize( const int argc, char** argv, const cxxopts::ParseResult& vm, const scai::dmemo::CommunicatorPtr& comm){

    if (comm->getType() != scai::dmemo::CommunicatorType::MPI) {
        std::cout << "The linked lama version was compiled without MPI. Only sequential partitioning is supported." << std::endl;
    }
      
    std::string callingCommand = ITI::getCallingCommand(argc, argv);
    
    Settings settings = ITI::interpretSettings(vm);
    //add the calling command to the setting so it can be extracted later
    settings.callingCommand = callingCommand;
    if( !settings.isValid )
        throw std::runtime_error("Invalid settings");

    return settings;
}


std::string getOutFileName( const Settings& settings, const std::string& toolName, const scai::dmemo::CommunicatorPtr& comm){
    const IndexType rank = comm->getRank();
    std::string outFile = settings.outFile;

    //want to store partition but outFile was not provided
    if( settings.storePartition and outFile=="-"){
        outFile = toolName+"_k"+ std::to_string(settings.numBlocks);
        if(rank==0){
            std::cout << "Option to store partition was given but no filename (--outFile). Created prefix: " << outFile << std::endl;
        }
    }

    //we are given a directory 
    if( settings.outDir!="-" ) {
        std::string dash = "";
        if( (settings.outDir.compare(settings.outDir.length()-1, 1, "/")!=0) and (settings.outDir.compare(settings.outDir.length()-1, 1, "\\")!=0) and toolName!="" ){
            dash= "/";
        }        
        //but no outFile
        if( settings.outFile=="-" ){
            //set the graphName in order to create the outFile name
            std::string copyName;
            if( settings.fileName!="-" ){
                copyName = settings.fileName;
            }else{     
                copyName = "generate_"+ std::to_string(settings.numX)+ "_"+ std::to_string(settings.numY);
            }
            std::vector<std::string> strs = aux<IndexType,double>::split( copyName, '/' );
            std::string graphName = aux<IndexType,double>::split(strs.back(), '.')[0];
            //add specific folder for each tool
            outFile = settings.outDir+ dash+ toolName+ "/"+ graphName+ "_k"+ std::to_string(settings.numBlocks)+ ".info";
        }
        //we are given both a file name and a directory append toolName
        else{
            //outFile = settings.outDir+ "/"+ settings.outFile;
            std::vector<std::string> strs = aux<IndexType,double>::split( settings.fileName, '/' );
            std::string graphName = aux<IndexType,double>::split(strs.back(), '.')[0];            
            outFile = settings.outDir+ "/"+ graphName+ "_k"+ std::to_string(settings.numBlocks)+ settings.outFile+ ".info";
            if(toolName!=""){
                outFile += ("_"+toolName);
            }
        }
    }

    //we are given just one file name, not a directory
    if( settings.outFile!="-" and settings.outDir=="-"){
        //do nothing
    }

    return outFile;
}

//taken from 
//https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
unsigned long getFreeRam(const scai::dmemo::CommunicatorPtr& comm){

    struct sysinfo memInfo;
    const IndexType rank = comm->getRank();
    const double kb = 1024.0;
    const double mb = kb*1024;
    [[maybe_unused]] const double gb = mb*1024;

    sysinfo (&memInfo);
    long long totalVirtualMem = memInfo.totalram;
    //Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;

    long long totalPhysMem = memInfo.totalram;
    //Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;

    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    //Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;

    unsigned long long freeRam = memInfo.freeram;
    freeRam *= memInfo.mem_unit;

    unsigned long long sharedRam = memInfo.sharedram;
    sharedRam *= memInfo.mem_unit;    
    
    unsigned long long buffRam = memInfo.bufferram;
    buffRam *= memInfo.mem_unit; 

    auto parseLine = [](char* line){
        // This assumes that a digit will be found and the line ends in " Kb".
        int i = strlen(line);
        const char* p = line;
        while (*p <'0' || *p > '9') p++;
        line[i-3] = '\0';
        i = atoi(p);
        return i;
    };

    auto getValue = [&](){ //Note: this value is in KB!
        FILE* file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != NULL){
            if (strncmp(line, "VmRSS:", 6) == 0){
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
        return result;
    };

    PRINT( rank <<  ": totalPhysMem: " << (totalPhysMem/mb) << 
                " MB, physMemUsed: " << physMemUsed/mb << 
                " MB, free ram: " << freeRam/mb <<
                " MB, shared ram: " << sharedRam/mb <<
                " MB, buffered   ram: " << buffRam/mb <<
                " MB, I am using: " << getValue()/kb << " MB" );

    return freeRam;
}


IndexType getCpuFreqLinux(const scai::dmemo::CommunicatorPtr& comm, const int nodeSize=24){
//TODO: how to get CPU speed?

    const IndexType numPEs = comm->getSize();
    const IndexType rank = comm->getRank();

	SCAI_ASSERT_LE_ERROR( nodeSize, numPEs, "The number of processes per node should be less that the number of calling PEs. Set parameter --processPerNode to an appropriate value.");
    SCAI_ASSERT_EQ_ERROR( nodeSize*(int (numPEs/nodeSize)), numPEs, "The size of each node is " << nodeSize <<" but it should be a multiple of the number of calling PEs. Set parameter --processPerNode to an appropriate value.");

    //rank inside this compute node
    const int myInternalRank = rank%nodeSize;

    std::string cpuFreqFile = "/sys/devices/system/cpu/cpu"+ std::to_string(myInternalRank)+ "/cpufreq/scaling_cur_freq";
    std::ifstream file(cpuFreqFile);
    if (file.fail()) {
        throw std::runtime_error("Reading CPU frequency from " + cpuFreqFile + " failed for PE " + std::to_string(rank));
    }

    std::string line;
    std::getline(file, line);
    IndexType freq = std::stoll(line);

    SCAI_ASSERT_GT( freq, 0, "Illegal CPU frequency value");
    
    return freq;
}


template <typename vType>
std::vector<std::vector<vType>> calculateLoadRequests(const scai::dmemo::CommunicatorPtr& comm, const int nodeSize=24){

    const IndexType numPEs = comm->getSize();
    const IndexType rank = comm->getRank();

    //in the version, we have two node weights: cpu frequency and memory size
    std::vector<std::vector<vType>> retWeights (2, std::vector<vType> (numPEs, 0.0) );

    //start with the cpu frequency

    const double myCpuFreq = getCpuFreqLinux(comm, nodeSize);
    std::vector<IndexType> allCpuFreq(numPEs, 0);
    allCpuFreq[rank] = myCpuFreq;

    //replicate all frequencies in all PEs
    comm->sumImpl( allCpuFreq.data(), allCpuFreq.data(), numPEs, scai::common::TypeTraits<IndexType>::stype );

    const IndexType sumCpuFreq = std::accumulate( allCpuFreq.begin(), allCpuFreq.end(), 0 );

    //set first weight relevant to the CPU frequency

    for( int i=0; i<numPEs; i++){
        retWeights[0][i] = ((vType) allCpuFreq[i]) /sumCpuFreq;
    }

    //memory
	//TODO: remove hardcoded long int size
    const unsigned long myFreeRam = getFreeRam(comm)/sizeof(long int); //value returned in bytes; convert to vertex size 

    std::vector<vType> allFreeRam(numPEs, 0.0);
    allFreeRam[rank] = (vType) myFreeRam;

    //replicate all memory capacities in all PEs
    comm->sumImpl( retWeights[1].data(), allFreeRam.data(), numPEs, scai::common::TypeTraits<vType>::stype );
	
	
	for( int i=0; i<numPEs; i++){
	PRINT0( i << ": cpu " << retWeights[0][i] << ", free ram " << retWeights[1][i] );	
	}
	
    return retWeights;
}

}//namespace ITI
