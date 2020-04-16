/*
 * IOTest.cpp
 *
 *  Created on: 15.02.2017
 *      Authors: Charilaos Tzovas, Moritz von Looz
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

#include <memory>

#include "gtest/gtest.h"

#include "ParcoRepart.h"
#include "HilbertCurve.h"
#include "FileIO.h"
#include "MeshGenerator.h"
#include "Settings.h"
#include "quadtree/SpatialTree.h"

using scai::lama::CSRStorage;


namespace ITI {

template<typename T>
class FileIOTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(FileIOTest, testTypes);

//-----------------------------------------------------------------
TYPED_TEST(FileIOTest, testWriteMetis_Dist_3D) {
    using ValueType = TypeParam;

    std::vector<IndexType> numPoints= { 10, 10, 10};
    std::vector<ValueType> maxCoord= { 10, 20, 30};
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];
    std::cout<<"Building mesh of size "<< numPoints[0]<< "x"<< numPoints[1]<< "x"<< numPoints[2] << " , N=" << N <<std::endl;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution( N ));

    std::vector<DenseVector<ValueType>> coords(3);
    for(IndexType i=0; i<3; i++) {
        coords[i].allocate(dist);
        coords[i] = static_cast<ValueType>( 0 );
    }

    auto adjM = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>( dist, noDistPointer);

    // create the adjacency matrix and the coordinates
    MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist(adjM, coords, maxCoord, numPoints, 3);

    // write the mesh in p(=number of PEs) files
    FileIO<IndexType, ValueType>::writeGraphDistributed( adjM, FileIOTest<ValueType>::graphPath+"dist3D_");

}

//-----------------------------------------------------------------
/* Reads a graph from a file "filename" in METIS format, writes it back into "my_filename" and reads the graph
 * again from "my_filename".
 *
 * Occasionally throws error, probably because own process tries to read the file while some other is still writing in it.
 */

TYPED_TEST(FileIOTest, testReadAndWriteGraphFromFile) {
    using ValueType = TypeParam;

    std::string file = "Grid8x8";
    //std::string file = "slowrot-00000.graph";
    std::string filename= FileIOTest<ValueType>::graphPath + file;
    CSRSparseMatrix<ValueType> Graph;

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
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues())/2 );

    std::string fileTo= FileIOTest<ValueType>::graphPath + std::string("MY_") + file;

    // write the graph you read in a new file
    FileIO<IndexType, ValueType>::writeGraph(Graph, fileTo );

    comm->synchronize();

    // read new graph from the new file we just wrote
    CSRSparseMatrix<ValueType> Graph2 = FileIO<IndexType, ValueType>::readGraph( fileTo );

    // check that the two graphs are identical
    if(comm->getRank()==0 ) {
        std::cout<< "Output written in file: "<< fileTo<< std::endl;
    }
    EXPECT_EQ(Graph.getNumValues(), Graph2.getNumValues() );
    EXPECT_EQ(Graph.l2Norm(), Graph2.l2Norm() );
    EXPECT_EQ(Graph2.getNumValues(), Graph2.l1Norm() );
    EXPECT_EQ( Graph.getNumRows(), Graph2.getNumColumns() );

    // check every element of the  graphs
    {
        SCAI_REGION("testReadAndWriteGraphFromFile.checkArray");
        const CSRStorage<ValueType>& localStorage = Graph.getLocalStorage();
        scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

        const CSRStorage<ValueType>& localStorage2 = Graph2.getLocalStorage();
        scai::hmemo::ReadAccess<ValueType> values2(localStorage2.getValues());

        EXPECT_EQ( values.size(), values2.size() );

        for(IndexType i=0; i< values.size(); i++) {
            EXPECT_EQ( values[i], values2[i] );
        }
    }
}
//-----------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadAndWriteBinaryGraphFromFile) {
    using ValueType = TypeParam;

    std::string file = "Grid8x8";
    //std::string file = "slowrot-00000.graph";
    std::string filename= FileIOTest<ValueType>::graphPath + file;
    CSRSparseMatrix<ValueType> Graph;

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
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues())/2 );

    std::string fileTo= FileIOTest<ValueType>::graphPath + std::string("MY_") + file+".bgf";

    // write the graph you read in a new file
    FileIO<IndexType, ValueType>::writeGraph(Graph, fileTo, true, false );

    if(comm->getRank()==0 ) {
        std::cout<< "Output written in file: "<< fileTo<< std::endl;
    }

    comm->synchronize();

    // read new graph from the new file we just wrote
    ITI::Format format = ITI::Format::BINARY;
    CSRSparseMatrix<ValueType> Graph2 = FileIO<IndexType, ValueType>::readGraph( fileTo, comm, format );

    // check that the two graphs are identical

    EXPECT_EQ(Graph.getNumValues(), Graph2.getNumValues() );
    EXPECT_EQ(Graph.l2Norm(), Graph2.l2Norm() );
    EXPECT_EQ(Graph2.getNumValues(), Graph2.l1Norm() );
    EXPECT_EQ( Graph.getNumRows(), Graph2.getNumColumns() );

    // check every element of the  graphs
    {
        SCAI_REGION("testReadAndWriteGraphFromFile.checkArray");
        const CSRStorage<ValueType>& localStorage = Graph.getLocalStorage();
        scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());

        const CSRStorage<ValueType>& localStorage2 = Graph2.getLocalStorage();
        scai::hmemo::ReadAccess<ValueType> values2(localStorage2.getValues());

        EXPECT_EQ( values.size(), values2.size() );

        for(IndexType i=0; i< values.size(); i++) {
            EXPECT_EQ( values[i], values2[i] );
        }
    }
}
//-----------------------------------------------------------------

TYPED_TEST(FileIOTest, testWriteGraphWithEdgeWeights) {
    using ValueType = TypeParam;
    const IndexType N = 10;

    //define distributions
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));

    //generate random complete matrix
    auto graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>(dist, noDistPointer);

    scai::lama::MatrixCreator::fillRandom(graph, 1/9.0);

    std::string filename = "./meshes/noEdgeWeights.graph";
    FileIO<IndexType, ValueType>::writeGraph( graph, filename );

    filename = "./meshes/edgeWeights.graph";
    FileIO<IndexType, ValueType>::writeGraph( graph, filename, true );

}


//-----------------------------------------------------------------
// read a graph from a file in METIS format and its coordinates in 2D and partition that graph
// usually, graph file: "file.graph", coordinates file: "file.graph.xy" or .xyz
TYPED_TEST(FileIOTest, testPartitionFromFile_dist_2D) {
    using ValueType = TypeParam;

    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix
    IndexType dim= 2, k= 8;
    ValueType epsilon= 0.1;

    std::string file= "slowrot-00000.graph";
    std::string grFile= FileIOTest<ValueType>::graphPath +file;
    std::string coordFile= grFile+".xyz";  //graph file and coordinates file
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
    graph.redistribute( distPtr, noDistPtr);
    std::cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";
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
    Metrics<ValueType> metrics(settings);

    //partition the graph
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, settings, metrics );
    EXPECT_EQ(partition.size(), N);
    SCAI_REGION_END("testPartitionFromFile_local_2D.partition");

}
//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testWriteCoordsDistributed) {
    using ValueType = TypeParam;

    std::string file= "Grid8x8";
    std::string grFile= FileIOTest<ValueType>::graphPath +file;
    std::string coordFile= grFile +".xyz";   //graph file and coordinates file
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();

    IndexType dim=2;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );

    // every PE reads its own part of the coordinates based on a block distribution
    std::vector<DenseVector<ValueType>> coords2D = FileIO<IndexType, ValueType>::readCoords( coordFile, nodes, dim);
    EXPECT_TRUE(coords2D[0].getDistributionPtr()->isEqual(*distPtr));

    FileIO<IndexType, ValueType>::writeCoordsDistributed( coords2D, dim, "partResults/writeCoordsDist");
    //TODO: delete files after they have been written!
}
//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadCoordsOcean) {
    using ValueType = TypeParam;

    std::string coordFile = FileIOTest<ValueType>::graphPath + "node2d_core2.out";
    const IndexType n = 126858;
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    std::vector<DenseVector<ValueType> > coords = FileIO<IndexType, ValueType>::readCoordsOcean(coordFile, 2, comm);
    EXPECT_EQ(n, coords[0].size());

    for (IndexType d = 0; d < 2; d++) {
        scai::hmemo::ReadAccess<ValueType> rCoords(coords[d].getLocalValues());
        for (IndexType i = 0; i < rCoords.size(); i++ ) {
            EXPECT_TRUE(std::isfinite(rCoords[i]));
        }
    }
}

//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadQuadTree) {
    using ValueType = TypeParam;

    std::string filename = FileIOTest<ValueType>::graphPath + "cells_small.dat";

    scai::lama::CSRSparseMatrix<ValueType> matrix = FileIO<IndexType, ValueType>::readQuadTree(filename);

    std::cout << "Matrix has " << matrix.getNumRows() << " rows and " << matrix.getNumValues() << " values " << std::endl;
    EXPECT_TRUE(matrix.isConsistent());
    //IndexType m = std::accumulate(edgeList.begin(), edgeList.end(), 0, [](int previous, std::set<std::shared_ptr<SpatialCell> > & edgeSet){return previous + edgeSet.size();});
    //std::cout << "Read Quadtree with " << edgeList.size() << " nodes and " << m << " edges." << std::endl;
}

//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadBinaryEdgeList) {
    using ValueType = TypeParam;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    std::string filename = FileIOTest<ValueType>::graphPath + "delaunay-3D-12.edgelist";

    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(filename, comm, ITI::Format::BINARYEDGELIST);

    scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();

    const IndexType n = graph.getNumRows();
    //const IndexType localN = dist->getLocalSize();
    EXPECT_EQ(4096, n);
    EXPECT_TRUE(graph.isConsistent());
    EXPECT_EQ(n, graph.getNumColumns());

    //check degree symmetry
    std::vector<IndexType> inDegree(n);
    std::vector<IndexType> outDegree(n);
    scai::hmemo::ReadAccess<IndexType> ia(graph.getLocalStorage().getIA());
    scai::hmemo::ReadAccess<IndexType> ja(graph.getLocalStorage().getJA());

    for (IndexType i = 0; i < graph.getLocalNumRows(); i++) {
        const IndexType globalI = dist->local2Global(i);//this can be optimized
        outDegree[globalI] = ia[i+1] - ia[i];
    }

    for (IndexType i = 0; i < ja.size(); i++) {
        inDegree[ja[i]]++;
    }

    comm->sumImpl(outDegree.data(), outDegree.data(), n, scai::common::TypeTraits<IndexType>::stype);
    comm->sumImpl(inDegree.data(), inDegree.data(), n, scai::common::TypeTraits<IndexType>::stype);

    for (IndexType i = 0; i < n; i++) {
        EXPECT_EQ(inDegree[i], outDegree[i]);
    }

    //check actual symmetry
    //EXPECT_TRUE(graph.checkSymmetry());

}

//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadGraphBinary) {
    using ValueType = TypeParam;

    std::string file = "trace-00008.bgf";   // trace-08: n=8993, m=13370
    //std::string file = "Grid16x16.bgf";   // Grid16x16: n= 256, m=480
    //std::string file = "Grid8x8.bgf";   // Grid8x8: n= 64, m=224
    std::string filename= FileIOTest<ValueType>::graphPath + file;

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::lama::CSRSparseMatrix<ValueType> graph =  FileIO<IndexType, ValueType>::readGraphBinary(filename, comm);

    //assertions

    //TODO: read same graph with the original reader. Matrices must be identical

    std::string txtFile= FileIOTest<ValueType>::graphPath+"/trace-00008.graph";
    std::fstream f(txtFile);
    if (f.fail()) {
        throw std::runtime_error("Reading graph from " + txtFile + " failed.");
    }
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();

    IndexType N = graph.getNumRows();

    if( N<1000) {
        SCAI_ASSERT( graph.checkSymmetry(), "Matrix not symmetric" );
    }
    SCAI_ASSERT( graph.isConsistent(), "Matrix not consistent" );

    SCAI_ASSERT_EQ_ERROR( N, nodes, "Mismatch in number of nodes read." );
    SCAI_ASSERT_EQ_ERROR( N, graph.getNumColumns(), "Wrong number of rows and columns" );

    IndexType M = graph.getNumValues();
    SCAI_ASSERT_EQ_ERROR( M, edges*2, "Mismatch in number of edges read." );

}
//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadMatrixMarketFormat) {
    using ValueType = TypeParam;

    std::string graphFile = FileIOTest<ValueType>::graphPath + "whitaker3.mtx";
    std::string coordFile = FileIOTest<ValueType>::graphPath + "whitaker3_coord.mtx";

    std::ifstream coordF( coordFile );

    // we do not need them for the MatrixMarket format
    IndexType N, dimensions;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    ITI::Format ff = ITI::Format::MATRIXMARKET;

    std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();

    std::tie( N, dimensions) = FileIO<IndexType, ValueType>::getMatrixMarketCoordsInfos( coordFile );
    PRINT0(" number of points= " << N << ", dimensions= " << dimensions);

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( coordFile, N, dimensions, comm, ff);

    std::chrono::duration<double> readTime =  std::chrono::steady_clock::now() - startTime;
    PRINT0("Read " << coords.size() << " coordinates in time " << readTime.count() );

    startTime = std::chrono::steady_clock::now();
    scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph( graphFile, comm, ff);
    readTime =  std::chrono::steady_clock::now() - startTime;

    PRINT0("Read  graph in time " << readTime.count() );


    //assertion - prints

    std::cout<< "Coords size= "<< coords[0].size() << " , dimensions= " << coords.size() << std::endl;

    SCAI_ASSERT( dimensions=coords.size(), "Dimensions " << dimensions << " do not agree with coordiantes size= " << coords.size() );
    SCAI_ASSERT( N=coords[0].size(), "N= "<< N << " does not agree with coords[0].size()= " << coords[0].size() );

    PRINT(*comm << ": localCoords.size()= "<< coords[0].getLocalValues().size() );
    SCAI_ASSERT( coords[0].getLocalValues().size()>0, "Coordinate vector is PE " << *comm << " is empty");
    for(int d=1; d<dimensions; d++) {
        SCAI_ASSERT( coords[d].getLocalValues().size()==coords[d-1].getLocalValues().size(), "Coordinates for different dimension have different sizes, should be the same");
        SCAI_ASSERT( coords[d].getLocalValues().size()>0, "Coordinate vector is PE " << *comm << " is empty");
    }

    {
        const CSRStorage<ValueType>& localStorage = graph.getLocalStorage();
        scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

        for(int i=0; i<10; i++) {
            //PRINT0(ja[i]);
        }
    }
}
//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadBlockSizes) {

    using ValueType = TypeParam;

    std::string path = projectRoot+"/testing/";
    std::string blocksFile = path + "blockSizes.txt";

    std::vector<std::vector<ValueType>> blockSizes = FileIO<IndexType,ValueType>::readBlockSizes(blocksFile, 16);

    SCAI_ASSERT( blockSizes[0].size()==16, "Wrong number of blocks, should be 16 but is " << blockSizes.size() );

}
//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testWriteCoordsParallel) {
    using ValueType = TypeParam;

    std::string file = FileIOTest<ValueType>::graphPath + "delaunayTest.graph";
    std::ifstream f(file);

    //WARNING: for this example we need dimension 3 because the Schamberger graphs have always 3 coordinates
    IndexType dimensions= 3;
    IndexType N, edges;
    f >> N >> edges;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr blockDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );

    std::vector<DenseVector<ValueType>> coordsOrig = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions, comm);
    EXPECT_TRUE(coordsOrig[0].getDistributionPtr()->isEqual(*blockDist));

    //
    // write coords in parallel
    std::string outFilename = std::string( file+"_parallel.xyz");

    FileIO<IndexType, ValueType>::writeCoordsParallel( coordsOrig, outFilename);

    //now read the coords
    std::vector<DenseVector<ValueType>> coordsBinary =  FileIO<IndexType, ValueType>::readCoordsBinary( outFilename, N, dimensions, comm);


    for( int d=0; d<dimensions; d++) {
        scai::hmemo::ReadAccess<ValueType> localCoordsBinary( coordsBinary[d].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> localCoordsOrig( coordsOrig[d].getLocalValues() );

        SCAI_ASSERT_EQ_ERROR( localCoordsBinary.size(), localCoordsOrig.size(), "Size mismatch");

        //PRINT(*comm << ": size= " << localCoordsOrig.size()<< ", dimension: "<< d);

        for( IndexType i=0; i<localCoordsBinary.size(); i++) {
            SCAI_ASSERT_EQ_ERROR( localCoordsBinary[i], localCoordsOrig[i], *comm << ": Not equal coordinates at index " << i);
        }
    }

}
//-------------------------------------------------------------------------------------------------

TYPED_TEST(FileIOTest, testReadGraphAndCoordsBinary) {
    using ValueType = TypeParam;    

    std::string fileBin = FileIOTest<ValueType>::graphPath + "delaunayTest.bgf";
    std::string coordFileBin = fileBin+".xyz";
    std::string fileMetis = FileIOTest<ValueType>::graphPath + "delaunayTest.graph";
    std::string coordFileMetis = fileMetis+".xyz";

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    //
    // check that graphs are identical
    //
    scai::lama::CSRSparseMatrix<ValueType> graphBin = FileIO<IndexType, ValueType>::readGraphBinary(fileBin, comm);
    scai::lama::CSRSparseMatrix<ValueType> graphMetis = FileIO<IndexType, ValueType>::readGraph(fileMetis);

    if( graphBin.getNumRows()< 500) {
        PRINT0("Checking matrix symmetry...");
        SCAI_ASSERT_ERROR( graphBin.checkSymmetry(), "Graph not symmetric");
        SCAI_ASSERT_ERROR( graphMetis.checkSymmetry(), "Graph not symmetric");
    }
    SCAI_ASSERT_ERROR( graphBin.isConsistent(), "Graph not consistent");
    SCAI_ASSERT_ERROR( graphMetis.isConsistent(), "Graph not consistent");

    {
        scai::hmemo::ReadAccess<ValueType> readBinGraphVal( graphBin.getLocalStorage().getValues() );
        scai::hmemo::ReadAccess<ValueType> readMetisGraphVal( graphMetis.getLocalStorage().getValues() );
        SCAI_ASSERT_EQ_ERROR( readBinGraphVal.size(), readMetisGraphVal.size(), "Matrix mismatch");

        for(int i=0; i<readBinGraphVal.size(); i++) {
            SCAI_ASSERT_EQ_ERROR(readBinGraphVal[i], readBinGraphVal[i], "Matrix value mismatch in position " << i);
        }

        scai::hmemo::ReadAccess<IndexType> readBinGraphIA( graphBin.getLocalStorage().getIA() );
        scai::hmemo::ReadAccess<IndexType> readMetisGraphIA( graphMetis.getLocalStorage().getIA() );
        SCAI_ASSERT_EQ_ERROR( readBinGraphIA.size(), readMetisGraphIA.size(), "Matrix mismatch");
        for(int i=0; i<readBinGraphIA.size(); i++) {
            SCAI_ASSERT_EQ_ERROR(readBinGraphIA[i], readMetisGraphIA[i], "Matrix value mismatch in position " << i);
        }

        scai::hmemo::ReadAccess<IndexType> readBinGraphJA( graphBin.getLocalStorage().getJA() );
        scai::hmemo::ReadAccess<IndexType> readMetisGraphJA( graphMetis.getLocalStorage().getJA() );
        SCAI_ASSERT_EQ_ERROR( readBinGraphJA.size(), readMetisGraphJA.size(), "Matrix mismatch");
        for(int i=0; i<readBinGraphJA.size(); i++) {
            SCAI_ASSERT_EQ_ERROR(readBinGraphJA[i], readMetisGraphJA[i], "Matrix value mismatch in position " << i);
        }
    }

    //
    // check coordinates are identical
    //

    IndexType dimensions = 2;
    IndexType N = graphBin.getNumRows();

    std::vector<DenseVector<ValueType>> coordsBinary =  FileIO<IndexType, ValueType>::readCoordsBinary( coordFileBin, N, dimensions, comm);
    std::vector<DenseVector<ValueType>> coordsMetis =  FileIO<IndexType, ValueType>::readCoords( coordFileMetis, N, dimensions, comm);

    SCAI_ASSERT_EQ_ERROR( coordsBinary.size(), coordsMetis.size(), "Wrong dimension");

    for(int d=0; d<coordsBinary.size(); d++) {
        scai::hmemo::ReadAccess<ValueType> localCoordsBin( coordsBinary[d].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> localCoordsMetis( coordsMetis[d].getLocalValues() );
        SCAI_ASSERT_EQ_ERROR( localCoordsBin.size(), localCoordsMetis.size(), "Local sizes do not agree.");

        for(int i=0; i<localCoordsBin.size(); i++) {
            if( std::abs(localCoordsBin[i] - localCoordsMetis[i]) > 1e-6) {
                std::cout<< localCoordsBin[i] << "-" << localCoordsMetis[i] << "  _  " << localCoordsBin[i]-localCoordsMetis[i];
                std::cout << std::endl;
            }

            SCAI_ASSERT_LT_ERROR( localCoordsBin[i]-localCoordsMetis[i], 1e-6, "Coordinates in position " << coordsBinary[0].getDistributionPtr()->local2Global(i) << " in processor " << comm->getRank() << " do not agree for dimension " << d);
        }
    }

}
//-------------------------------------------------------------------------------------------------

TYPED_TEST (FileIOTest, testWriteDenseVectorCentral) {
    using ValueType = TypeParam;

    const IndexType N = 9001;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    const IndexType k = comm->getSize();
    const IndexType localN = dist->getLocalSize();

    scai::lama::DenseVector<IndexType> partition(dist, 0);
    {
        scai::hmemo::WriteAccess<IndexType> wPart(partition.getLocalValues());
        for (IndexType i = 0; i < localN; i++) {
            IndexType blockId = rand() % k;
            wPart[i] = blockId;
        }
    }

    FileIO<IndexType, ValueType>::writeDenseVectorCentral( partition, "partResults/testWriteDenseVectorCentral.part");
    //TODO: maybe clean up?
}

//-------------------------------------------------------------------------------------------------

TYPED_TEST (FileIOTest, testreadOFFCentral) {
    using ValueType = TypeParam;

    std::string file = FileIOTest<ValueType>::graphPath+ "2.off";

    // open file and read number of nodes and edges
    //

    IndexType numVertices, numEdges;
    {
        std::ifstream f(file);
        if(f.fail())
            throw std::runtime_error("File "+ file + " failed.");

        std::string line;
        std::getline(f, line);   // first line should have the string OFF
        std::getline(f, line);
        std::stringstream ss;
        ss.str( line );

        IndexType numFaces;
        ss >> numVertices >> numFaces >> numEdges;
    }

    scai::lama::CSRSparseMatrix<ValueType> graph;
    std::vector<DenseVector<ValueType>> coords;

    FileIO<IndexType, ValueType>::readOFFTriangularCentral( graph, coords, file );

    IndexType N = coords[0].size();

    SCAI_ASSERT_EQ_ERROR( N, numVertices, "Number of vertices do not agree.");
    SCAI_ASSERT_EQ_ERROR( graph.getNumValues()/2, numEdges, "Number of edges do not agree.");

    /*
    for(IndexType i=0; i<10; i++){
        for(int dim=0; dim<3; dim++){
            std::cout<< coords[dim].getValue(N-i-1) << " ";
        }
        std::cout << std::endl;
    }
    */

    if( N<5000 ) {
        SCAI_ASSERT_EQ_ERROR( true, graph.checkSymmetry(), "Matrix not symmetric");
    }
    SCAI_ASSERT_EQ_ERROR( true, graph.isConsistent(), "Matrix not consistent");

    //PRINT( graph.getNumValues() << " _ " << graph.getNumRows() << " @ " << graph.getNumColumns() );
}
//-------------------------------------------------------------------------------------------------

TYPED_TEST (FileIOTest, testReadEdgeListDistributed) {
    using ValueType = TypeParam;

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if( comm->getSize() != 4 ) {
        if( comm->getRank() == 0) {
            std::cout << "\n\t\t### WARNING: this test reads a distributed file and only works for p=4. You should call again with mpirun -n 4 (maybe also with --gtest_filter=*ReadEdgeListDistributed)." << std::endl<< std::endl;
        }
    } else {

        std::string file = FileIOTest<ValueType>::graphPath + "tmp4/out";

        scai::lama::CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readEdgeListDistributed( file, comm );

        ASSERT_TRUE( graph.isConsistent());
        EXPECT_TRUE( graph.checkSymmetry() );
        EXPECT_EQ( graph.getNumRows(), graph.getNumColumns()) << "Matrix is not square";

        // only for graph: graphPath + "tmp4/out"
        EXPECT_EQ( graph.getNumRows(), 16) << "for files tmp4/out N must be 16";
    }
}
//-------------------------------------------------------------------------------------------------

TYPED_TEST (FileIOTest, testReadPETree) {
    using ValueType = TypeParam;

    std::string file = FileIOTest<ValueType>::graphPath+ "processorTrees/testPEgraph28.txt";

    ITI::CommTree<IndexType, ValueType> tree =  FileIO<IndexType, ValueType>::readPETree( file );
    PRINT("read file " << file );
    tree.checkTree();

    tree.getRoot().print();

}

} /* namespace ITI */
