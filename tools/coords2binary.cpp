
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

    if(thisPE==0){
        std::cout << "program converts a coordinates file into a binary file. "  << std::endl;
        std::cout << "usage: ./a.out dimensions numberOfPoints inputFile outputFile , eg: ./a.out 2 100 coords.xyz coords.bcf" << std::endl;
    }

    if( argc!=5 ) {
        if( thisPE==0 ) {
            std::cout<< "Wrong number of parameter given: " << argc << std::endl;
        }
        return 0;
    }

    IndexType dimensions = std::stoi( argv[1] );
    if( dimensions<=0 ) {
        PRINT0("wrong number of dimensions: " << dimensions);
    }

    //IndexType globalN = std::strtol( argv[2] );
    IndexType globalN = std::stoi( argv[2] );
    SCAI_ASSERT_GT_ERROR( globalN, 0, "Number of points must be positive." );

    std::string inFilename = argv[3];
    std::string outFilename = argv[4];

	std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();
	
    std::vector<scai::lama::DenseVector<ValueType>> coordsOrig = ITI::FileIO<IndexType, ValueType>::readCoords( inFilename, globalN, dimensions);

	std::chrono::duration<double> readTime = std::chrono::steady_clock::now() - startTime;
    if( thisPE==0 ) {
        std::cout<< "Read coords in time " << readTime.count() << std::endl;
    }
	
    ITI::FileIO<IndexType, ValueType>::writeCoordsParallel( coordsOrig, outFilename);

	std::chrono::duration<double> totalTime = std::chrono::steady_clock::now() - startTime;
    if( thisPE==0 ) {
        std::cout<< "Wrote binary coords in time " << totalTime.count() - readTime.count() << std::endl;
		std::cout<< "Coords stored in file " << outFilename << std::endl;
    }
}

