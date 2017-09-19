/*
 * IOTest.cpp
 *
 *  Created on: 15.02.2017
 *      Author: moritzl
 */

#include <scai/lama.hpp>
#include <scai/logging.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>
#include <scai/lama/Vector.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>
#include <scai/hmemo/WriteAccess.hpp>
#include <scai/hmemo/ReadAccess.hpp>

#include <scai/utilskernel/LArray.hpp>

#include <memory>

#include "gtest/gtest.h"

#include "ParcoRepart.h"
#include "HilbertCurve.h"
#include "FileIO.h"
#include "MeshGenerator.h"
#include "Settings.h"
#include "quadtree/SpatialTree.h"

typedef double ValueType;
typedef int IndexType;

using scai::lama::CSRStorage;

namespace ITI {

class FileIOTest : public ::testing::Test {

};

//-----------------------------------------------------------------
TEST_F(FileIOTest, testWriteMetis_Dist_3D){

    std::vector<IndexType> numPoints= { 10, 10, 10};
    std::vector<ValueType> maxCoord= { 10, 20, 30};
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];
    std::cout<<"Building mesh of size "<< numPoints[0]<< "x"<< numPoints[1]<< "x"<< numPoints[2] << " , N=" << N <<std::endl;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));

    std::vector<DenseVector<ValueType>> coords(3);
    for(IndexType i=0; i<3; i++){
	  coords[i].allocate(dist);
	  coords[i] = static_cast<ValueType>( 0 );
    }

    scai::lama::CSRSparseMatrix<ValueType> adjM( dist, noDistPointer);

    // create the adjacency matrix and the coordinates
    MeshGenerator<IndexType, ValueType>::createStructured3DMesh_dist(adjM, coords, maxCoord, numPoints);

    // write the mesh in p(=number of PEs) files
    FileIO<IndexType, ValueType>::writeGraphDistributed( adjM, "meshes/dist3D_");

}

//-----------------------------------------------------------------
/* Reads a graph from a file "filename" in METIS format, writes it back into "my_filename" and reads the graph
 * again from "my_filename".
 *
 * Occasionally throws error, probably because own process tries to read the file while some other is still writing in it.
 */

TEST_F(FileIOTest, testReadAndWriteGraphFromFile){
    std::string path = "meshes/slowrot/";
    //std::string file = "Grid8x8";
    std::string file = "slowrot-00000.graph";
    std::string filename= path + file;
    CSRSparseMatrix<ValueType> Graph;
    IndexType N;    //number of points

    std::ifstream f(filename);
    IndexType nodes, edges;
    //In the METIS format the two first number in the file are the number of nodes and edges
    f >>nodes >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist( new scai::dmemo::NoDistribution( nodes ));

    // read graph from file
    {
        SCAI_REGION("testReadAndWriteGraphFromFile.readGraphFromFile");
        Graph = FileIO<IndexType, ValueType>::readGraph(filename);
    }
    N = Graph.getNumColumns();
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues())/2 );

    std::string fileTo= path + std::string("MY_") + file;

    // write the graph you read in a new file
    FileIO<IndexType, ValueType>::writeGraph(Graph, fileTo );

    comm->synchronize();

    // read new graph from the new file we just written
    CSRSparseMatrix<ValueType> Graph2 = FileIO<IndexType, ValueType>::readGraph( fileTo );

    // check that the two graphs are identical
    if(comm->getRank()==0 ){
        std::cout<< "Output written in file: "<< fileTo<< std::endl;
    }
    EXPECT_EQ(Graph.getNumValues(), Graph2.getNumValues() );
    EXPECT_EQ(Graph.l2Norm(), Graph2.l2Norm() );
    EXPECT_EQ(Graph2.getNumValues(), Graph2.l1Norm() );
    EXPECT_EQ( Graph.getNumRows() , Graph2.getNumColumns() );

    // check every element of the  graphs
    {
        SCAI_REGION("testReadAndWriteGraphFromFile.checkArray");
        const CSRStorage<ValueType>& localStorage = Graph.getLocalStorage();
        scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

        const CSRStorage<ValueType>& localStorage2 = Graph2.getLocalStorage();
        scai::hmemo::ReadAccess<ValueType> values2(localStorage2.getValues());

        assert( values.size() == values2.size() );

        for(IndexType i=0; i< values.size(); i++){
            assert( values[i] == values2[i] );
        }
    }
}

//-----------------------------------------------------------------
// read a graph from a file in METIS format and its coordinates in 2D and partition that graph
// usually, graph file: "file.graph", coordinates file: "file.graph.xy" or .xyz
TEST_F(FileIOTest, testPartitionFromFile_dist_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix
    IndexType dim= 2, k= 8, i;
    ValueType epsilon= 0.1;

    std::string path = "./meshes/slowrot/";
    //std::string path = "";
    std::string file= "slowrot-00000.graph";
    std::string grFile= path +file, coordFile= path +file +".xyz";  //graph file and coordinates file
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    //
    k = comm->getSize();
    //

    //read the adjacency matrix from a file
    //std::cout<<"reading adjacency matrix from file: "<< grFile<<" for k="<< k<< std::endl;
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution( nodes ));

    SCAI_REGION_START("testPartitionFromFile_local_2D.readGraphFromFile");
        graph = FileIO<IndexType, ValueType>::readGraph( grFile );
        graph.redistribute( distPtr , noDistPtr);
        //std::cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";
    SCAI_REGION_END("testPartitionFromFile_local_2D.readGraphFromFile");

    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);

    //read the coordiantes from a file
    //std::cout<<"reading coordinates from file: "<< coordFile<< std::endl;

    SCAI_REGION_START("testPartitionFromFile_local_2D.readFromFile2Coords_2D");
    std::vector<DenseVector<ValueType>> coords2D = FileIO<IndexType, ValueType>::readCoords( coordFile, N, dim);
    EXPECT_TRUE(coords2D[0].getDistributionPtr()->isEqual(*distPtr));
    SCAI_REGION_END("testPartitionFromFile_local_2D.readFromFile2Coords_2D");

    EXPECT_EQ(coords2D.size(), dim);
    EXPECT_EQ(coords2D[0].size(), N);

    // print
    /*
    for(IndexType i=0; i<N; i++){
        std::cout<< i<< ": "<< *comm<< " - " <<coords2D[0].getLocalValues()[i] << " , " << coords2D[1].getLocalValues()[i] << std::endl;
    }
    */

    SCAI_REGION_START("testPartitionFromFile_local_2D.partition");

        struct Settings settings;
        settings.numBlocks= k;
        settings.epsilon = epsilon;
        settings.dimensions = dim;
        settings.minGainForNextRound = 10;
        
        //partition the graph
        scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, settings );
        EXPECT_EQ(partition.size(), N);
    SCAI_REGION_END("testPartitionFromFile_local_2D.partition");

}
//-------------------------------------------------------------------------------------------------

TEST_F(FileIOTest, testWriteCoordsDistributed){

    //std::string path = "./meshes/slowrot/";
    std::string path = "";
    std::string file= "Grid8x8";
    std::string grFile= path +file , coordFile= path +file +".xyz";  //graph file and coordinates file
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    IndexType dim=2;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    //scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution( nodes ));

    // every PE reads its own part of the coordinates based on a block distribution
    std::vector<DenseVector<ValueType>> coords2D = FileIO<IndexType, ValueType>::readCoords( coordFile, nodes, dim);
    EXPECT_TRUE(coords2D[0].getDistributionPtr()->isEqual(*distPtr));
    
    FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coords2D, nodes, "writeCoordsDist");
    //TODO: delete files after they have been written!
}

TEST_F(FileIOTest, testReadCoordsOcean) {
	std::string graphFile = "fesom_core2.graph";
	std::string coordFile = "nod2d_core2.out";

	std::vector<DenseVector<ValueType> > coords = FileIO<IndexType, ValueType>::readCoordsOcean(coordFile, 2);
	EXPECT_EQ(126858, coords[0].size());
}

//-------------------------------------------------------------------------------------------------

TEST_F(FileIOTest, testReadQuadTree){
	std::string filename = "cells.dat";

	scai::lama::CSRSparseMatrix<ValueType> matrix = FileIO<IndexType, ValueType>::readQuadTree(filename);

	std::cout << "Matrix has " << matrix.getNumRows() << " rows and " << matrix.getNumValues() << " values " << std::endl;
	EXPECT_TRUE(matrix.isConsistent());
	//IndexType m = std::accumulate(edgeList.begin(), edgeList.end(), 0, [](int previous, std::set<std::shared_ptr<SpatialCell> > & edgeSet){return previous + edgeSet.size();});
	//std::cout << "Read Quadtree with " << edgeList.size() << " nodes and " << m << " edges." << std::endl;
}
//-------------------------------------------------------------------------------------------------

TEST_F(FileIOTest, testReadGraphBinary){
    std::string path = "./meshes/";
    std::string file = "trace10.bfg";   // trace10: n=8699, m=25874
    //std::string file = "Grid16x16.bfg";   // Grid16x16: n= 256, m=480
    //std::string file = "Grid8x8.bfg";   // Grid16x16: n= 64, m=224
    std::string filename= path + file;
    scai::lama::CSRSparseMatrix<ValueType> graph;
    
    //std::vector<DenseVector<ValueType>> dummyWeightContainer;
    graph =  FileIO<IndexType, ValueType>::readGraphBinary(filename);
    
    //assertions
    
    //std::string txtFile= file.substr(0, file.length()-4);
    std::string txtFile= "./meshes/trace/trace-00010.graph";
    std::fstream f(txtFile);
    if (f.fail()) {
        throw std::runtime_error("Reading graph from " + filename + " failed.");
    }
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    IndexType N = graph.getNumRows();
    
    if( N<1000){
        SCAI_ASSERT( graph.checkSymmetry(), "Matrix not symmetric" );
    }
    SCAI_ASSERT( graph.isConsistent(), "Matrix not consistent" );
    
    SCAI_ASSERT_EQ_ERROR( N, nodes, "Mismatch in number of nodes read." );
    SCAI_ASSERT_EQ_ERROR( N, graph.getNumColumns(), "Wrong number of rows and columns" );
    
    IndexType M = graph.getNumValues();
    SCAI_ASSERT_EQ_ERROR( M, edges*2, "Mismatch in number of edges read." );
    
}
//-------------------------------------------------------------------------------------------------

TEST_F(FileIOTest, testReadMatrixMarketFormat){
    std::string path = "./meshes/whitaker3/";
    std::string graphFile = path + "whitaker3.mtx";
    std::string coordFile = path + "whitaker3_coord.mtx";
        
    std::ifstream coordF( coordFile );
    
    // we do not need them for the MatrixMarket format
    IndexType N, dimensions;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    //scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    
    ITI::Format ff = ITI::Format::MATRIXMARKET;
    
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    std::tie( N, dimensions) = FileIO<IndexType, ValueType>::getMatrixMarketCoordsInfos( coordFile );
    PRINT0(" number of points= " << N << ", dimensions= " << dimensions);
    
    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( coordFile, N, dimensions, ff);
    
    std::chrono::duration<double> readTime =  std::chrono::system_clock::now() - startTime;
    
    PRINT0("Read " << coords.size() << " coordinates in time " << readTime.count() );
    
    startTime = std::chrono::system_clock::now();
    
    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( graphFile, ff);
    
    readTime =  std::chrono::system_clock::now() - startTime;
    
    PRINT0("Read  graph in time " << readTime.count() );
    
    
    //assertion - prints
    
    std::cout<< "Coords size= "<< coords[0].size() << " , dimensions= " << coords.size() << std::endl;
    
    SCAI_ASSERT( dimensions=coords.size() , "Dimensions " << dimensions << " do not agree with coordiantes size= " << coords.size() );
    SCAI_ASSERT( N=coords[0].size() , "N= "<< N << " does not agree with coords[0].size()= " << coords[0].size() );
    
    /*
    for( int i=0; i<coords[0].getLocalValues().size(); i++){
        std::cout << *comm << " ";
        for(int d=0; d<dimensions; d++){
            std::cout<< coords[d].getLocalValues()[i] << ", ";
        }
        std::cout << std::endl;
    }
    */
    
    PRINT(*comm << ": localCoords.size()= "<< coords[0].getLocalValues().size() );
    SCAI_ASSERT( coords[0].getLocalValues().size()>0 , "Coordinate vector is PE " << *comm << " is empty");
    for(int d=1; d<dimensions; d++){
        SCAI_ASSERT( coords[d].getLocalValues().size()==coords[d-1].getLocalValues().size() , "Coordinates for different dimension have different sizes, should be the same");
        SCAI_ASSERT( coords[d].getLocalValues().size()>0 , "Coordinate vector is PE " << *comm << " is empty");
    }
    
    {
        const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();    	scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

        for(int i=0; i<10; i++){
            //PRINT0(ja[i]);
        }
    }
}
//-------------------------------------------------------------------------------------------------

TEST_F(FileIOTest, testReadBlockSizes){
    
    std::string path = "./";
    std::string blocksFile = path + "blockSizes.txt";

    std::vector<IndexType> blockSizes = FileIO<IndexType,ValueType>::readBlockSizes(blocksFile, 16);
    
    //aux::printVector( blockSizes );
    SCAI_ASSERT( blockSizes.size()==16 , "Wrong number of blocks, should be 16 but is " << blockSizes.size() );
    
}

} /* namespace ITI */
