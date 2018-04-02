
#include <memory>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <unistd.h>

#include "FileIO.h"
//#include "Settings.h"


int main(int argc, char** argv) {
    std::cout << "\nA program converts an alya .dom.geo file to two files: one with the graph in metis format and one for the coordinates."  << std::endl;
    std::cout << "The endings are added automatically. The filenames are: outputFilename.graph and outputFilename.graph.xyz" <<std::endl;
    std::cout << "usage: ./a.out inputFilename outputFilename dimensions numberOfNodes, eg: ./a.out plane.dom.geo planeMetis 3 1000" << std::endl;
	std::cout << "usually, the number of points can be found in the *.dom.dat under \"NODAL_POINTS\""<< std::endl;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if( comm->getSize()>1 ){
        std::cout<< "The converter works only sequantially. Call again without mpirun." << std::endl;
        return -1;
    }
    
    
    if( argc!=5 ){
        std::cout<< "Wrong number of parameter given: " << argc << std::endl; 
        return 0;
    }
    
    IndexType dimensions =  std::stoi( argv[3] );
    if( dimensions<=0 ){
        PRINT0("wrong number of dimensions: " << dimensions);
		return -1;
    }
    
    IndexType N =  std::stoi( argv[4] );
    if( N<=0 ){
        PRINT0("wrong number of nodes: " << N);
		return -1;
    }
    
    std::string inFilename = argv[1];
    std::string outFilename = argv[2];
    
    
    scai::lama::CSRSparseMatrix<ValueType> graph;
    std::vector<DenseVector<ValueType>> coords;
    
	PRINT0( N );	
	
    ITI::FileIO<IndexType, ValueType>::readAlyaCentral( graph, coords, N, dimensions, inFilename);
    
    //IndexType N = graph.getNumRows();
    
    SCAI_ASSERT_EQ_ERROR( N, graph.getNumColumns(), "Matrix is not square.");
    SCAI_ASSERT_EQ_ERROR( N, coords[0].size(), "Number of coordinates do not agree with matrix size.");
    SCAI_ASSERT_EQ_ERROR( coords.size() , dimensions, "Dimensions of coordinates do not agree with dimensions given");
    
    ITI::FileIO<IndexType, ValueType>::writeGraph( graph, outFilename+".graph" );
    
    ITI::FileIO<IndexType, ValueType>::writeCoords( coords, outFilename+".graph.xyz" );
    
    std::cout << "Output written in files: "<< outFilename+".graph" << " and "<< outFilename+".graph.xyz" << std::endl;
}
    
