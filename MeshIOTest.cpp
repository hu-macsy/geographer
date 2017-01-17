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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>

#include "ParcoRepart.h"
#include "gtest/gtest.h"
#include "HilbertCurve.h"
#include "MeshIO.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;
using namespace std;

namespace ITI {

class MeshIOTest : public ::testing::Test {

};

//-------------------------------------------------------------------------------------------------
/* Creates a random 3D mesh and writes in a .graph file the graph in a METIS format and in the
 *  .graph.xyz the coordiantes.
 */
TEST_F(MeshIOTest, testMesh3DCreateRandomMeshWriteInFile_Local_3D) {
    std::vector<DenseVector<ValueType>> coords;
    int numberOfPoints= 100;
    ValueType maxCoord= 1;
    std::string grFile = "meshes/randomTest6.graph";
    std::string coordFile= grFile + ".xyz";

    scai::lama::CSRSparseMatrix<ValueType> adjM(numberOfPoints, numberOfPoints);
    {
        SCAI_REGION("testMesh3DCreateRandomMeshWriteInFile_Local_3D.createRandom3DMesh")
        MeshIO<IndexType, ValueType>::createRandom3DMesh(adjM, coords, numberOfPoints, maxCoord);
    }
    
    {
        SCAI_REGION("testMesh3DCreateRandomMeshWriteInFile_Local_3D.(writeInFileMetisFormat and writeInFileCoords)")
        MeshIO<IndexType, ValueType>::writeInFileMetisFormat( adjM, grFile);
        MeshIO<IndexType, ValueType>::writeInFileCoords( coords, numberOfPoints, coordFile);
    }
}

//----------------------------------------------------------------------------------------
/* Creates a semi-structured 3D mesh given the number of points for each dimension and the maximum
 * corrdinate in each axis. Writes the graph in METIS format in a .graph file and the coordiantes
 * in a .graph.xyz file.
 * */

TEST_F(MeshIOTest, testMesh3DCreateStructuredMesh_Local_3D) {
    std::vector<IndexType> numPoints= {12, 13, 14};
    std::vector<ValueType> maxCoord= {100,180,130};
    IndexType numberOfPoints= numPoints[0]*numPoints[1]*numPoints[2];
    std::vector<DenseVector<ValueType>> coords(3, DenseVector<ValueType>(numberOfPoints, 0));
    std::string grFile = "meshes/structuredTest6.graph";
    std::string coordFile= grFile + ".xyz";

    scai::lama::CSRSparseMatrix<ValueType> adjM( numberOfPoints, numberOfPoints);
    std::cout<<__FILE__<< "  "<< __LINE__<< " , numberOfPoints=" << numberOfPoints<< std::endl;

    {
        SCAI_REGION("testMesh3DCreateStructuredMesh_Local_3D.createStructured3DMesh" )
        MeshIO<IndexType, ValueType>::createStructured3DMesh(adjM, coords, maxCoord, numPoints);
    }
    
    {
        SCAI_REGION("testMesh3DCreateStructuredMesh_Local_3D.(writeInFileMetisFormat and writeInFileCoords)")
        MeshIO<IndexType, ValueType>::writeInFileMetisFormat( adjM, grFile);
        MeshIO<IndexType, ValueType>::writeInFileCoords( coords, numberOfPoints, coordFile);
    }
    
    CSRSparseMatrix<ValueType> graph = scai::lama::CSRSparseMatrix<ValueType>( numberOfPoints, numberOfPoints );
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( graph, grFile );
    
    // check the two matrixes to be equal
    {
        SCAI_REGION("testMesh3DCreateStructuredMesh_Local_3D.checkMatricesEqual");
        for(IndexType i=0; i<adjM.getNumRows(); i++){
            for(IndexType j=0; j<adjM.getNumColumns(); j++){
                EXPECT_EQ( adjM(i,j).Scalar::getValue<ValueType>() , graph(i,j).Scalar::getValue<ValueType>()  );
            }
        }
    }
}
//-----------------------------------------------------------------

TEST_F(MeshIOTest, testPartitionWithRandom3DMesh_Local_3D) {
    IndexType N= 100;
    ValueType maxCoord= 1;
    IndexType dim= 3, k= 8;
    ValueType epsilon= 0.2;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    //the coordinates of the points: coords.size()= dim , coords[dim].size()= N
    std::vector<DenseVector<ValueType>> coords(3);    
    
    //the adjacency matrix
    scai::lama::CSRSparseMatrix<ValueType> adjM(N, N);
    {
        SCAI_REGION("testPartitionWithRandom3DMesh_Local_3D.createRandom3DMesh");
        //random coordinates in 3D stored in coords and the adjacency matrix in adjM
        MeshIO<IndexType, ValueType>::createRandom3DMesh(adjM, coords, N, maxCoord);
    }
    //std::cout<< "Number of nodes= "<< N<< " , Number of edges="<< (adjM.getNumValues()-N)/2 << std::endl;
    
    SCAI_REGION_START("testPartitionWithRandom3DMesh_Local_3D.partitionGraph");
    //get the partition
    DenseVector<IndexType> partition= ParcoRepart<IndexType, ValueType>::partitionGraph( adjM, coords, k, epsilon);
    SCAI_REGION_END("testPartitionWithRandom3DMesh_Local_3D.partitionGraph");
    
    // check partition
    for(IndexType i=0; i<partition.size(); i++){
        EXPECT_LE( partition(i).Scalar::getValue<ValueType>() , k);
        EXPECT_GE( partition(i).Scalar::getValue<ValueType>() , 0);
    }
    
    //calculate and print cut and imbalance
    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(adjM, partition, true);
    std::cout<< "# cut = "<< cut<< " , "<< std::endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    std::cout<< "# imbalance = " << imbalance<< " , "<< std::endl;
    
}

//-----------------------------------------------------------------
/* Reads a graph from a file "filename" in METIS format, writes it back into "my_filename" and reads the graph
 * again from "my_filename".
 */
TEST_F(MeshIOTest, testReadAndWriteGraphFromFile){
    //string path = "./meshes/my_meshes/";
    std::string path = "";
    std::string file = "Grid32x32";
    std::string filename= path + file;
    CSRSparseMatrix<ValueType> Graph;
    IndexType N;    //number of points     
    
    std::ifstream f(filename);
    IndexType nodes, edges;
    //In the METIS format the two first number in the file are the number of nodes and edges
    f >>nodes >> edges; 
    
    dmemo::DistributionPtr dist( new dmemo::NoDistribution( nodes ));
    Graph = scai::lama::CSRSparseMatrix<ValueType>(dist, dist);
    {
        SCAI_REGION("testReadAndWriteGraphFromFile.readFromFile2AdjMatrix");
        MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(Graph, filename );
    }
    N = Graph.getNumColumns();
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues())/2 );
    
    std::string fileTo= path + std::string("MY_") + file;
    
    CSRSparseMatrix<ValueType> Graph2(dist, dist);
    {   
        SCAI_REGION("testReadAndWriteGraphFromFile.(writeInFileMetisFormat and writeInFileCoords)");
        MeshIO<IndexType, ValueType>::writeInFileMetisFormat(Graph, fileTo );
        MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(Graph2, fileTo );
    }
    std::cout<< "Output written in file: "<< fileTo<< std::endl;
    EXPECT_EQ(Graph.getNumValues(), Graph2.getNumValues() );
    EXPECT_EQ(Graph.l2Norm(), Graph2.l2Norm() );
    EXPECT_EQ(Graph2.getNumValues(), Graph2.l1Norm() );
}

//-----------------------------------------------------------------
// read a graph from a file in METIS format and its coordiantes in 2D and partiotion that graph
// usually, graph file: "file.graph", coodinates file: "file.graph.xy" or .xyz
TEST_F(MeshIOTest, testPartitionFromFile_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    std::vector<DenseVector<ValueType>> coords2D(2);        //the coordiantes of each node 
    IndexType dim= 2, k= 10, i;
    ValueType epsilon= 0.1;
    
    //std::string path = "./meshes/my_meshes";s
    std::string path = "";
    std::string file= "Grid32x32";
    std::string grFile= path +file, coordFile= path +file +".xyz";  //graph file and coordinates file
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    //read the adjacency matrix from a file
    std::cout<<"reading adjacency matrix from file: "<< grFile<<" for k="<< k<< std::endl;
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    
    graph = scai::lama::CSRSparseMatrix<ValueType>(distPtr, distPtr);
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( graph , grFile );
    std::cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";

    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    std::cout<<"reading coordinates from file: "<< coordFile<< std::endl;
    
    coords2D[0].allocate(N);
    coords2D[1].allocate(N);
    coords2D[0]= static_cast<ValueType>( 0 );
    coords2D[1]= static_cast<ValueType>( 0 );
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, coords2D, N );
    EXPECT_EQ(coords2D.size(), dim);
    EXPECT_EQ(coords2D[0].size(), N);
    
    //partition the graph
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D,  k, epsilon);
    EXPECT_EQ(partition.size(), N);
    /*
    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true);
    std::cout<< "# cut = "<< cut<< std::endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    std::cout<< "# imbalance = " << imbalance<< std::endl; 
    */ 
}

//-----------------------------------------------------------------
    
}//namespace ITI
