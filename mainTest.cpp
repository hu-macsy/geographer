#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>

#include "MeshIO.h"
#include "ParcoRepart.h"
#include "Settings.h"

typedef double ValueType;
typedef int IndexType;


/* For 2 dimensions reads graph and coordinates from a file, for creates a structured mesh.
*
* Use of parameters: first is dimensions. If dimensions is 2 then a filename must follow from where
* the adjacency matrix of the graph will be read. Also from filename.xyz we read the coordiantes.
* Last is the parameter epsilon, example:
* ./a.out 2 myGraph 0.2
* If dimensions are 3, then must follow three numbers, the number of points in each dimension
* and then parameter epsilon, example:
* ./a.out 3 50 60 70 0.2
*/


//----------------------------------------------------------------------------

int main(int argc, char** argv) {

    // just print the input parameters
    std::cout << "argc =" << argc << std::endl;  
    for (int i = 0; i < argc; i++) {
            std::cout << i << ":" << argv[i] << std::endl;
    }

    IndexType dimensions = atoi(argv[1]);	// first parameter is the number of dimensions
    
    IndexType N = 1; 		// total number of points
    ValueType epsilon;
        
    /* The struct for the settings passed to the partitioner.
    * !!!
    * In the 2D case w do not know the number of points in every direction, only the total number of points.
    * So in 2D numX= totalNumOfPoints and numY=numZ=0.
    * !!!
    */
    struct Settings Settings;

    scai::lama::CSRSparseMatrix<ValueType> graph; 	// the adjacency matrix of the graph
    std::vector<DenseVector<ValueType>> coordinates(dimensions); // the coordinates of the graph

    std::vector<IndexType> numPoints; // number of poitns in each dimension, used only for 3D
    std::vector<ValueType> maxCoord(dimensions); // the max coordinate in every dimensions, used only for 3D

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    // treat differently for 2D and 3D
    
    if(dimensions == 2){
        if( argc<4 ){
            throw std::runtime_error("Wrong number of input parameters for " +  std::to_string(dimensions)  + " dimensions, entered: " + std::to_string(dimensions) );
        }
        std::string graphFile = argv[2];  	// should be the filename for the adjacency matrix
        std::string coordFile = graphFile + ".xyz";
        std::fstream f(graphFile);

        if(f.fail()){
            throw std::runtime_error("File "+ graphFile+ " failed.");
        }
        
        f >> N;				// first line must have total number of nodes and edges
        
        // for 2D we do not know the size of every dimension
        Settings.numX= N;
        Settings.numY= 0;
        Settings.numZ= 0;
        
        std::cout<< "Reading from file \""<< graphFile << "\" and .xyz for coordinates"<< std::endl;
        std::cout<< "Starting for dim= "<< dimensions << " number of points= " << N << std::endl;

        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
        graph = scai::lama::CSRSparseMatrix<ValueType>( rowDistPtr , noDistPtr );

        // read the adjacency matrix and the coordinates from a file
        ITI::MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( graph , graphFile );
        graph.redistribute(rowDistPtr , noDistPtr);
        // take care, when reading from file graph needs redistribution
        
        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<dimensions; i++){ 
            coordinates[i].allocate(coordDist);
            coordinates[i] = static_cast<ValueType>( 0 );
        }
        ITI::MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, coordinates, N );
        coordinates[0].redistribute(coordDist);
        coordinates[1].redistribute(coordDist);
        
        epsilon = atof(argv[3]); 		// next is the parameter epsilon

    }else if(dimensions == 3){
        if( argc<6 ){
            throw std::runtime_error("Wrong number of input parameters for " +  std::to_string(dimensions)  + " dimensions, entered: " + std::to_string(dimensions) );
        }
        // numPoints actually not needed
        numPoints.push_back( atoi(argv[2]) );
        numPoints.push_back( atoi(argv[3]) );
        numPoints.push_back( atoi(argv[4]) );

        Settings.numX= numPoints[0];
        Settings.numY= numPoints[1];
        Settings.numZ= numPoints[2];
        
        N = numPoints[0]* numPoints[1]* numPoints[2];
            
        // set maxCoords same as numPoints
        for(IndexType i=0; i<dimensions; i++){
            maxCoord[i] = (ValueType) numPoints[i];   
        }
        if( comm->getRank()== 0){
            std::cout<< "Starting for dim= "<< dimensions << " and numPoints= "<< numPoints[0] << ", " << numPoints[1] << ", "<< numPoints[2] << ", in total "<< N << " number of points" << std::endl;
            std::cout<< "\t\t and maxCoord= "<< maxCoord[0] << ", "<< maxCoord[1] << ", " << maxCoord[2]<< std::endl;
        }
        
        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph = scai::lama::CSRSparseMatrix<ValueType>( rowDistPtr , noDistPtr );
        
        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<dimensions; i++){ 
            coordinates[i].allocate(coordDist);
            coordinates[i] = static_cast<ValueType>( 0 );
        }

        // create the adjacency matrix and the coordinates
        ITI::MeshIO<IndexType, ValueType>::createStructured3DMesh_dist( graph, coordinates, maxCoord, numPoints);

        epsilon = atof(argv[5]); 		// next is the parameter epsilon

    }else{
        throw std::runtime_error("Dimensions must be either 2 or 3, entered: " + std::to_string(dimensions) );
    }

    // set the rest of the settings
    // should be passed as parameters when calling main
    Settings.dimensions = dimensions;
    Settings.borderDepth = 4;
    Settings.stopAfterNoGainRounds = 10;
    Settings.minGainForNextRound = 1;
    Settings.sfcResolution = 9;  
    Settings.epsilon = epsilon;
    Settings.numBlocks = comm->getSize();
    
    if( comm->getRank() ==0){
        if(dimensions==2){
            Settings.print2D(std::cout);
        }else{
            Settings.print3D(std::cout);
        }
    }
    
    scai::lama::DenseVector<IndexType> partition = ITI::ParcoRepart<IndexType, ValueType>::partitionGraph( graph, coordinates, Settings );
    
    ValueType cut = ITI::ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true); 
    ValueType imbalance = ITI::ParcoRepart<IndexType, ValueType>::computeImbalance( partition, Settings.numBlocks );
    
    std::cout<< "Cut is: "<< cut<< " and imbalance: "<< imbalance << std::endl;
    
}
