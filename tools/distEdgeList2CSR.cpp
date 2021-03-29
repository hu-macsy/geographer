
#include <memory>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <unistd.h>

#include "FileIO.h"


int main(int argc, char** argv) {
	
	using namespace ITI;
	typedef double ValueType;   //use double
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    IndexType thisPE = comm->getRank();

    if( thisPE==0 ) {
        std::cout << "program reads an edge list stored in multiple files, converts them to a graph and stores the graph "  << std::endl;
        std::cout << "in one file.\n###\tAttention: it should be called with as many processes as the number of files." << std::endl;
        std::cout << "Each process X will try to read the file \"inputFileX\" " << std::endl;
        std::cout << "usage: mpirun -np X  inputFile outputFile (optionally, add \"bin\" to store to bianry format)" << std::endl;
    }

    if( argc<2 ) {
        if( thisPE==0 ) {
            std::cout<< "Wrong number of parameter given: " << argc << std::endl;
        }
        return 0;
    }

    const std::string filename = argv[1];
    if( thisPE==0 ) {
        std::cout<< "Will read from file" << filename << std::endl;
    }

	std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();
	
    bool duplicateEdges = true;
    bool removeSelfLoops = true;
    const scai::lama::CSRSparseMatrix<ValueType> graph = 
        ITI::FileIO<IndexType,ValueType>::readEdgeListDistributed( filename, comm, duplicateEdges, removeSelfLoops);

	std::chrono::duration<double> readListTime = std::chrono::steady_clock::now() - startTime;
    if( thisPE==0 ) {
        std::cout<< "Read distributed edge list in time " << readListTime.count() << std::endl;
    }
	
    if( not graph.isConsistent() ){
        throw std::runtime_error("Graph is not consistent; maybe corrupted edge list files?"); 
    }

    const std::string outFile = argc>2 ? argv[2] : filename+".graph";

    IndexType numV = graph.getNumValues() ;
    if( thisPE==0 ) {
        std::cout<< "Graph has " << graph.getNumRows() << " vertices and " << numV << " edges." <<std::endl;
        std::cout<< "Will store graph in file: " << outFile << std::endl;
    }

    bool binaryFormat = argc==4 ? true:false;

    ITI::FileIO<IndexType,ValueType>::writeGraph( graph, outFile, binaryFormat);

	std::chrono::duration<double> totalTime = std::chrono::steady_clock::now() - startTime;
    if( thisPE==0 ) {
        std::cout<< "Wrote csr file in time " << totalTime.count() - readListTime.count() << std::endl;
    }
	
    return 0;
}

