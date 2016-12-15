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

TEST_F(MeshIOTest, testMesh3DCreateRandomMeshWriteInFile) {
vector<DenseVector<ValueType>> coords;
int numberOfPoints= 2000;
ValueType maxCoord= 1;
std::string grFile = "meshes/randomTest5.graph";
std::string coordFile= grFile + ".xyz";

scai::lama::CSRSparseMatrix<ValueType> adjM(numberOfPoints, numberOfPoints);
std::cout<<__FILE__<< "  "<< __LINE__<< endl;

chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
MeshIO<IndexType, ValueType>::createRandom3DMesh(adjM, coords, numberOfPoints, maxCoord);
chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
std::cout<<__FILE__<< "  "<< __LINE__<< " , time for createRandom3DMesh: "<< duration <<endl;

MeshIO<IndexType, ValueType>::writeInFileMetisFormat( adjM, grFile);
MeshIO<IndexType, ValueType>::writeInFileCoords( coords, 3, numberOfPoints, coordFile);

duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() -t2).count();
//t2 = chrono::high_resolution_clock::now();
std::cout<<__FILE__<< "  "<< __LINE__<< " , time to write in files: "<< duration <<endl;
std::cout<< "graph written in files: " << grFile<< " and "<< coordFile<< endl;
}
//-----------------------------------------------------------------
TEST_F(MeshIOTest, testMesh3DCreateStructuredMesh) {
vector<IndexType> numPoints= {15, 17, 25};
//vector<DenseVector<ValueType>> coords(3);//= { DenseVector<ValueType>(numPoints[0],0), DenseVector<ValueType>(numPoints[1],0), DenseVector<ValueType>(numPoints[2],0)};
vector<ValueType> maxCoord= {100,180,130};
IndexType numberOfPoints= numPoints[0]*numPoints[1]*numPoints[2];
vector<DenseVector<ValueType>> coords(3, DenseVector<ValueType>(numberOfPoints, 0));
std::string grFile = "meshes/structuredTest3.graph";
std::string coordFile= grFile + ".xyz";
/*
for(IndexType i=0; i<3; i++){
    coords[i] = DenseVector<ValueType>( numPoints[i], 0);
//std::cout<<__FILE__<< "  "<< __LINE__<< " __ "<< numPoints[i]<< std::endl;
    for(IndexType j=0; j< numPoints[i]; j++)
        coords[i].setValue(j,0);
}
*/

scai::lama::CSRSparseMatrix<ValueType> adjM(numberOfPoints, numberOfPoints);
std::cout<<__FILE__<< "  "<< __LINE__<< " , numberOfPoints=" << numberOfPoints<< std::endl;

chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

MeshIO<IndexType, ValueType>::createStructured3DMesh(adjM, coords, maxCoord, numPoints);

chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
std::cout<<__FILE__<< "  "<< __LINE__<< " , time for creating structured3DMesh: "<< duration <<std::endl;
/*
for(int i=0; i<numberOfPoints; i++)
    cout<< i<<": coords= ["<< coords[0].getValue(i).getValue<ValueType>()<< ", " << coords[1].getValue(i).getValue<ValueType>()<< ", "<< coords[2].getValue(i).getValue<ValueType>()<<"] , " << std::endl;
    
cout<< "\t";
for(int i=0; i<numberOfPoints; i++)
    cout<< i+1<< " ";
cout<<endl;

for(int i=0; i<numberOfPoints; i++){
    cout<< i+1<< ":\t";
    for(int j=0; j<numberOfPoints; j++)
        cout<< adjM.getValue(i,j).getValue<ValueType>()<< " ";
    cout<< endl;
}
 */

MeshIO<IndexType, ValueType>::writeInFileMetisFormat( adjM, grFile);
MeshIO<IndexType, ValueType>::writeInFileCoords( coords, 3, numberOfPoints, coordFile);

duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() -t2).count();
//t2 = chrono::high_resolution_clock::now();
std::cout<<__FILE__<< "  "<< __LINE__<< " , time to write in files: "<< duration <<endl;
std::cout<< "graph written in files: " << grFile<< " and "<< coordFile<< endl;
}
//-----------------------------------------------------------------

TEST_F(MeshIOTest, testPartitionWithRandom3DMeshLocal) {
    IndexType N= 800;
    ValueType maxCoord= 1;
    IndexType dim= 3, k= 8;
    ValueType epsilon= 0.2;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    
    //the coordinates of the points: coords.size()= dim , coords[dim].size()= N
    vector<DenseVector<ValueType>> coords(3);    
    //the adjacency matrix
    scai::lama::CSRSparseMatrix<ValueType> adjM(N, N);
    //random coordinates in 3D stored in coords and the adjacency matrix in adjM
    MeshIO<IndexType, ValueType>::createRandom3DMesh(adjM, coords, N, maxCoord);
    
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
    cout << "time elapsed after creating 3DMesh: "<< duration<< " ms"<< endl;
//std::cout<<__FILE__<< "  "<< __LINE__<< std::endl;
    // adjM.getNumValues() returns N values more !!!
    cout<< "Number of nodes= "<< N<< " , Number of edges="<< (adjM.getNumValues()-N)/2 <<endl;
    
    //get the partition
    DenseVector<IndexType>partition= ParcoRepart<IndexType, ValueType>::partitionGraph( adjM, coords, dim, k, epsilon);
    
    
    //begin: my partition
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
    DenseVector<IndexType> perm( noDistPtr );
    DenseVector<ValueType> indices2(N, 0);
    const std::vector<ValueType> minCoords({0,0,0});  
    const std::vector<ValueType> maxCoords( {coords[0].max().getValue<ValueType>(),       coords[1].max().getValue<ValueType>(),coords[2].max().getValue<ValueType>()} );  
    for (IndexType i = 0; i < N; i++){
        indices2.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords, dim, i, 10, minCoords, maxCoords) );
    }
    
    indices2.sort(perm, true);
   
    scai::lama::DenseVector<IndexType> partition2(N,0);
    IndexType partSize= (int) N/k;
    int j=1, i;
    
    partition2.setValue(  perm.getValue(0).getValue<IndexType>()  ,0);
    for(i=0; i<k; i++){
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        if(j!=N){
            partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i);
            j++;
        }
    }
    
    // because of (int)N/k some point are let out. Just put them in the last part.
    for(;j<N;j++)
        partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i-1);
    //end: my partition
    

    //calculate and print cut and imbalance
    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(adjM, partition, true);
    ValueType cut2= ParcoRepart<IndexType, ValueType>::computeCut(adjM, partition2, true);
    cout<< "# cut 1= "<< cut<< " , ";
    cout<< "cut2= "<<cut2<< endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    ValueType imbalance2= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    cout<< "# imbalance 1= " << imbalance<< " , ";
    cout <<"imbalance 2= "<< imbalance2 << endl;
    
    /*
    for(IndexType i=0; i<N; i++){
        cout<< i<< ": "<<partition.getValue(i).getValue<ValueType>()<< " --> ";
        for(IndexType j=0; j<N; j++){
            if( adjM(i,j)==1)
                cout<< partition.getValue(j).getValue<ValueType>()<< " , ";
        }
        cout<< endl;
    }
    */
    
    
   /* vector<IndexType> numPart(k,0);
    for(IndexType i=0; i<N; i++){
        numPart[ ((int) partition.getValue(i).getValue<ValueType>()) ]+=1;
    }
    for(IndexType i=0; i<k; i++)
        cout<< numPart[i] << endl;
    */
}

//-----------------------------------------------------------------
TEST_F(MeshIOTest, testReadAndWriteGraphFromFile){
    string path = "./meshes/MY_meshes/";
    string file = "Grid32x32";
    string filename= path + file;
    CSRSparseMatrix<ValueType> Graph;
    IndexType N;    //number of points     
    
    ifstream f(filename);
    IndexType nodes, edges;
    //In the METIS format the two first number in the file are the number of nodes and edges
    f >>nodes >> edges; 
    
    dmemo::DistributionPtr dist( new dmemo::NoDistribution( nodes ));
    Graph = scai::lama::CSRSparseMatrix<ValueType>(dist, dist);
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix_Boost(Graph, dist, filename );
    N=Graph.getNumColumns();
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues())/2 );
    
    string fileTo= path + string("MY_") + file;
    
    //CSRSparseMatrix<ValueType> Graph2 =  MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(fileTo );
    CSRSparseMatrix<ValueType> Graph3(dist, dist);
    MeshIO<IndexType, ValueType>::writeInFileMetisFormat(Graph, fileTo );
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix_Boost(Graph3, dist, fileTo );
    
    cout<< "Outpur written in file: "<< fileTo<< endl;
    //EXPECT_EQ(Graph.getNumValues(), Graph2.getNumValues() );
    EXPECT_EQ(Graph.getNumValues(), Graph3.getNumValues() );
    EXPECT_EQ(Graph.l2Norm(), Graph3.l2Norm() );
    EXPECT_EQ(Graph3.getNumValues(), Graph3.l1Norm() );
    
    cout<<"reading from file: "<< filename + string(".xyz")<< endl;
    vector<DenseVector<ValueType>> coords2D({ DenseVector<ValueType>(N,0), DenseVector<ValueType>(N,0) });
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D(filename + string(".xyz"), coords2D, N );
    EXPECT_EQ(coords2D.size()/2, N);
    
}
//-----------------------------------------------------------------
TEST_F(MeshIOTest, testReadGraphFromFile_Boost){
    string path = "./meshes/hugetrace/";
    string file = "hugetrace-00005.graph";
    string filename= path + file;
    CSRSparseMatrix<ValueType>  graph, graph2;
    IndexType N, E;    //number of points and edges
    
    ifstream f(filename);
    //In the METIS format the two first number in the file are the number of nodes and edges
    f >>N >> E;
    
    dmemo::DistributionPtr dist( new dmemo::NoDistribution( N ));
    //Graph = scai::lama::CSRSparseMatrix<ValueType>(dist, dist);
    
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    
    cout<<"reading from file: "<< filename<< " with Boost."<< endl;
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix_Boost(graph, dist, filename );
    EXPECT_EQ(graph.getNumColumns(), graph.getNumRows());    
    EXPECT_EQ(N, graph.getNumColumns());
    EXPECT_EQ(E, (graph.getNumValues())/2 );
    
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
    cout << "time elapsed: "<< duration<< " ms"<< endl;
    
    cout<< endl<<"reading file without Boost"<< endl;
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph2, dist, filename );
    EXPECT_EQ(graph2.getNumColumns(), graph2.getNumRows());    
    EXPECT_EQ(N, graph2.getNumColumns());
    EXPECT_EQ(E, (graph2.getNumValues())/2 );
    
    EXPECT_EQ(graph.l2Norm(), graph2.l2Norm());
    EXPECT_EQ(graph.getNumValues(), graph2.getNumValues());
    
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() -t2 ).count();
    //t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed: "<< duration<< " ms"<< endl;
    
}

//-----------------------------------------------------------------
// read a graph from a file in METIS format and its coordiantes in 2D and partiotion that graph
// usually, graph file: "file.graph", coodinates file: "file.graph.xy" or .xyz
TEST_F(MeshIOTest, testPartitionFromFile_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    vector<DenseVector<ValueType>> coords2D;        //the coordiantes of each node 
    IndexType dim= 2, k= 10, i;
    ValueType epsilon= 0.1;
    
    std::string path = "./meshes/";
    std::string file= "bubbles/bubbles-00010.graph";
    std::string grFile= path +file, coordFile= path +file +".xyz";
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    //read the adjacency matrix from a file
    cout<<"reading adjacency matrix from file: "<< grFile<<" for k="<< k<< endl;
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    graph = scai::lama::CSRSparseMatrix<ValueType>(distPtr, distPtr);
    graph =  MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( grFile );
    cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";

    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    cout<<"reading coordinates from file: "<< coordFile<< endl;

    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N*dim) );
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, coords2D, N );
    EXPECT_EQ(coords2D.size(), dim);
    EXPECT_EQ(coords2D[0].size(), N);
    
    //partiotion the graph
    //scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    //EXPECT_EQ(partition.size(), N);
 
    //my partition
    dmemo::DistributionPtr dist( new dmemo::NoDistribution( N ));
    DenseVector<IndexType> perm(dist);
    DenseVector<ValueType> indices2(N, 0);
    const std::vector<ValueType> minCoords({0,0});  
    //ValueType max= coords2D.max().getValue<ValueType>();
    //const std::vector<ValueType> maxCoords({max, max});
    const std::vector<ValueType> maxCoords({ coords2D[0].max().getValue<ValueType>() , coords2D[1].max().getValue<ValueType>() });
   
    for (IndexType i = 0; i < N; i++){
        indices2.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords2D, dim, i, 10, minCoords, maxCoords) );
    }
    
    indices2.sort(perm, true);
   
    scai::lama::DenseVector<IndexType> partition2(N,0);
    IndexType partSize= (int) N/k;
    
    int j=1;
    partition2.setValue(  perm.getValue(0).getValue<IndexType>()  ,0);
    for(i=0; i<k; i++){
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        if(j!=N){
            partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i);
            j++;
        }
    }
    
    ValueType cut=-1;//= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true);
    ValueType cut2= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    cout<< "# cut 1= "<< cut<<"    and cut2= "<<cut2<< endl;
    ValueType imbalance= -1;// ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    ValueType imbalance2= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    cout<< "# imbalance 1= " << imbalance<<" , imbalance 2= "<< imbalance2 << endl;
    
}
//-----------------------------------------------------------------
//-----------------------------------------------------------------
// read a graph from a file in METIS format and its coordiantes in 2D and partiotion that graph
// usually, graph file: "file.graph", coodinates file: "file,graph.xy" or .xyz
/*
TEST_F(MeshIOTest, testPartitionInGrid_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    DenseVector<ValueType> coords2D;        //the coordiantes of each node 
    IndexType dim= 2, k= 10, i;
    ValueType epsilon= 0.1;
    
    std::string path = "./meshes/";
    std::string file= "Grid32x32";
    std::string grFile= path +file, coordFile= path +file +".xyz";
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    //read the adjacency matrix from a file
    cout<<"reading adjacency matrix from file: "<< grFile<<" for k="<< k<< endl;
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    graph = scai::lama::CSRSparseMatrix<ValueType>(distPtr, distPtr);
    graph =  MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( grFile );
    cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";

    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    cout<<"reading coordinates from file: "<< coordFile<< endl;

    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N*dim) );
    coords2D = MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, N );
    EXPECT_EQ(coords2D.size(), N*dim);
    
    //partiotion the graph
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    EXPECT_EQ(partition.size(), N);
  
    //begin test print
    int gS=32;
    IndexType partViz[gS][gS];
    for(i=0; i<gS; i++)
        for(int j=0; j<gS; j++)
            partViz[i][j]=partition.getValue(i*gS+j).getValue<IndexType>();
    
    for(i=0; i<gS; i++){
        for(int j=0; j<gS; j++)
            cout<< partViz[i][j]<<"-";
        cout<<endl;
    }

    //end test print

    //my partition
    dmemo::DistributionPtr dist( new dmemo::NoDistribution( N ));
    DenseVector<IndexType> perm(dist);
    DenseVector<ValueType> indices2(N, 0);
    const std::vector<ValueType> minCoords({0,0});  
    ValueType max= coords2D.max().getValue<ValueType>();
    const std::vector<ValueType> maxCoords({max, max});
   
    for (IndexType i = 0; i < N; i++){
        indices2.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords2D, dim, i, 10, minCoords, maxCoords) );
    }
    
    indices2.sort(perm, true);
   
    scai::lama::DenseVector<IndexType> partition2(N,0);
    IndexType partSize= (int) N/k;
    
    int j=1;
    partition2.setValue(  perm.getValue(0).getValue<IndexType>()  ,0);
    for(i=0; i<k; i++){
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        if(j!=N){
            partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i);
            j++;
        }
    }
    
    // because of (int)N/k some point are let out. Just put them in the last part.
    for(;j<N;j++)
        partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i-1);
    
    
    //begin test print
    std::cout<<"----------------------------"<< " Partition 2"<< std::endl;
    IndexType partViz2[gS][gS];
    for(i=0; i<gS; i++)
        for(int j=0; j<gS; j++)
            partViz2[i][j]=partition2.getValue(i*gS+j).getValue<IndexType>();
    
    for(i=0; i<gS; i++){
        for(int j=0; j<gS; j++)
            cout<< partViz2[i][j]<<"-";
        cout<<endl;
    }
    
    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true);
    ValueType cut2= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    cout<< "# cut 1= "<< cut<<"    and cut2= "<<cut2<< endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    ValueType imbalance2= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    cout<< "# imbalance 1= " << imbalance<<" , imbalance 2= "<< imbalance2 << endl;
    
}
*/
//-----------------------------------------------------------------
/*
TEST_F(MeshIOTest, testPartitionByHand_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    DenseVector<ValueType> coords2D;        //the coordiantes of each node 
    IndexType dim= 2, k= 10, i;
    
    std::string path = "./meshes/";
    std::string file= "trace/trace-00004.graph";
    std::string grFile= path +file, coordFile= path +file +".xyz";
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    // read the adjacency matrix from a file
    cout<<"reading adjacency matrix from file: "<< grFile<<" for k="<< k<< endl;
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    graph = scai::lama::CSRSparseMatrix<ValueType>(dist, dist);
    graph = MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( grFile );
    cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";

    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    // read the coordiantes from a file
    cout<<"reading coordinates from file: "<< coordFile<< endl;
    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N*dim) );
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, coords2D, N );
    EXPECT_EQ(coords2D.size(), N*dim);

    DenseVector<IndexType> perm(dist);
    DenseVector<ValueType> indices2(N, 0);
    const std::vector<ValueType> minCoords({0,0});  
    ValueType max= coords2D.max().getValue<ValueType>();
    const std::vector<ValueType> maxCoords({max, max});
    
    for (IndexType i = 0; i < N; i++){
        indices2.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords2D, dim, i, 10, minCoords, maxCoords) );
    }
    
    indices2.sort(perm, true);
   
    scai::lama::DenseVector<IndexType> partition2(N,0);
    IndexType partSize= (int) N/k;
    
    int j=1;
    partition2.setValue(  perm.getValue(0).getValue<IndexType>()  ,0);
    for(i=0; i<k; i++){
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        if(j!=N){
            partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i);
            j++;
        }
    }
    
    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    cout<< "# cut = "<< cut << endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    cout<< "# imbalance = " << imbalance << endl;

}*/
//-----------------------------------------------------------------
TEST_F(MeshIOTest, testCSRSparseMatrixSetRow){
    IndexType N=100, i;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDist ( new scai::dmemo::NoDistribution(N) );
    //CSRSparseMatrix<ValueType> ret( dist, dist);
    CSRSparseMatrix<ValueType> ret( N, N);
    
    cout<<"Should be 0: "<< ret.getNumValues()<<endl;
    
    DenseVector<ValueType> row(N, 1);
    cout<<"row.l1Norm should be 100 : "<< row.l1Norm()<< endl;
    ret.setRow( row, 1, utilskernel::binary::ADD);
    DenseVector<ValueType> row2;
    ret.getRow(row2,1);
    cout<<"row2.l1Norm should be 100 : "<< row2.l1Norm()<<", size= "<<row2.size()<< endl;
    //for(i=0; i<N; i++) cout<< row2.getValue(i) <<" _ ";
    cout<<"ret.NumValues and l1Norm should be 100: \t"<< ret.getLocalNumValues()<<" or "<< ret.l1Norm()<<endl;
    
    //try using HArray and setLocalRow
    hmemo::ContextPtr contextPtr = hmemo::Context::getHostPtr();
    ValueType rowValues[N];
    for(i=0; i<N; i++){
        rowValues[i]= 1;
    } 
    hmemo::HArray<ValueType> rowLocal(N, rowValues);
    hmemo::WriteAccess<ValueType> writeAccess( rowLocal, contextPtr );    
    CSRSparseMatrix<ValueType> retLocal(N, N);

    writeAccess.resize(N);
    ValueType* data= writeAccess.get();
    for(i=0; i<N; i++){
        data[i]=1;
    }
    writeAccess.release(); 

    hmemo::ReadAccess<ValueType> readAccess(rowLocal, contextPtr);
    //const ValueType* data2 = readAccess.get();

    retLocal.setLocalRow( rowLocal , 2, utilskernel::binary::ADD);
    readAccess.release();    

    cout<<"retLocal.NumValues and l1Norm should be 100: \t"<< retLocal.getNumValues()<<" or "<< retLocal.l1Norm()<<endl;
    SCAI_LOG_INFO( std::cout, "LALA\n")
    
    //CSRSparseMatrix<ValueType> ret2( N, N);
    //ret2.Matrix::readFromFile("Grid8x8.txt");
    //cout<< ret2.l1Norm() << endl;

}
//-----------------------------------------------------------------   

TEST_F(MeshIOTest, testCSRSparseMatrixFromFile_2D){
    
    vector<DenseVector<ValueType>> coords2D(2);        //the coordinates of each node 
    IndexType dim= 2, k= 40000, i;
    ValueType epsilon= 0.7;
    
    std::string path = "./meshes/";
    //std::string file= "hugebubbles/hugebubbles-00012.graph";
    std::string file= "rotation/rotation-00009.graph";
    std::string grFile= path +file, coordFile= path +file +".xyz";
std::string grFile2 = my_argv[1];
std::cout<< "grFile: "<< grFile2<< std::endl;
grFile = grFile2;
coordFile = grFile +".xyz";
    std::fstream f(grFile);
    unsigned long nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    std::cout<< "Starting for k= "<< k << std::endl;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();  
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    //the graph as an adjacency matrix  
    CSRSparseMatrix<ValueType> graph = scai::lama::CSRSparseMatrix<ValueType>(dist, dist);
    
    // read the adjacency matrix from a file
    cout<<"reading adjacency matrix from file: "<< grFile<< endl;   
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, dist, grFile );
    cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";
    
    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    cout<<"reading coordinates from file: "<< coordFile<< endl;
    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N*dim) );
    coords2D[0]= DenseVector<ValueType>(coordDist);
    coords2D[1]= DenseVector<ValueType>(coordDist);
    
    MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, coords2D, N );
    EXPECT_EQ(coords2D.size(), dim);
    EXPECT_EQ(coords2D[0].size(), N);

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
    cout << "time elapsed for reading files: "<< duration<< " ms"<< endl;
    
    //partition the graph
    //
    // TODO: a new version of partitionGraph
    // for now use local partition.
    //
    //scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    //EXPECT_EQ(partition.size(), N);
    
    chrono::high_resolution_clock::time_point  t3 = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>( t3 - t2 ).count();
    cout << "time elapsed for partitioning: "<< duration<< " ms"<< endl;
    
    t1 = chrono::high_resolution_clock::now();
    //begin: my partition
    DenseVector<IndexType> perm(dist);
    DenseVector<ValueType> indices2(N, 0);
    const std::vector<ValueType> minCoords({0,0});  
    //ValueType max= coords2D.max().getValue<ValueType>()
    const std::vector<ValueType> maxCoords({ coords2D[0].max().getValue<ValueType>() , coords2D[1].max().getValue<ValueType>() });
   
    for (IndexType i = 0; i < N; i++){
        indices2.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords2D, dim, i, 15, minCoords, maxCoords) );
    }
    //t2 = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t3 ).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for GETTING the hilbert indices: "<< duration<< " ms"<<  endl;
    
    indices2.sort(perm, true);
    
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t2).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for SORTING the hilbert indices: "<< duration<< " ms"<<  endl;
    
    scai::lama::DenseVector<IndexType> partition2(N,0);
    IndexType partSize= (int) N/k;
    int j=1;
    
    partition2.setValue(  perm.getValue(0).getValue<IndexType>()  ,0);
    for(i=0; i<k; i++){
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        if(j!=N){
            partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i);
            j++;
        }
    }
    
    // because of (int)N/k some point are let out. Just put them in the last part.
    for(;j<N;j++)
        partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i-1);
    
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t2).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for getting the partition "<< duration<< " ms"<<  endl;   
    ValueType cut3= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    cout<< "cut before fiducciaMattheysesRound= "<< cut3 << endl<< flush;
    
    ParcoRepart<IndexType,ValueType>::fiducciaMattheysesRound(graph, partition2, k, epsilon);
    
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t2).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for one fiducciaMattheysesRound "<< duration<< " ms"<<  endl;    
    

    //end: my partition

    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t3 ).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for \"by hand\"-sfc partitioning: "<< duration<< " ms"<<  endl;
    
    //calculate and print cut and imbalance
    //ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true);
    ValueType cut2= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    //cout<< "# cut 1= "<< cut<< " , ";
    cout<< "cut2= "<<cut2<< endl;
    //ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    ValueType imbalance2= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    //cout<< "# imbalance 1= " << imbalance<< " , ";
    cout <<"imbalance 2= "<< imbalance2 << endl;
    
    t2 = chrono::high_resolution_clock::now();
    cout<<"Test completed in time, chrono:: "<< chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count()<< " ms"<< endl;
}
//-----------------------------------------------------------------
//-----------------------------------------------------------------   

TEST_F(MeshIOTest, testCSRSparseMatrixFromFile_3D){
    
    vector<DenseVector<ValueType>> coords3D;        //the coordinates of each node 
    IndexType dim= 3, k= 4, i;
    ValueType epsilon= 0.7;
    /*
    std::string path = "./meshes/";
    std::string file= "rotation/rotation-00009.graph";
    std::string grFile= path +file, coordFile= path +file +".xyz";
    */
    
    //first argument given is the filename, the second is number of partitions
    std::string grFile = my_argv[1];
    istringstream ss(my_argv[2]);
    if(!(ss>>k)) 
        throw std::runtime_error("argument 2"+ std::string(my_argv[2]) + " invalid.");
    //k= my_argv[2];
    std::cout<< "grFile: "<< grFile<< std::endl;
    //grFile = grFile2;
    std::string coordFile = grFile +".xyz";
    std::fstream f(grFile);
    unsigned long nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    std::cout<< "Starting for k= "<< k << std::endl;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();  
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    //the graph as an adjacency matrix  
    CSRSparseMatrix<ValueType> graph = scai::lama::CSRSparseMatrix<ValueType>(dist, dist);
    
    // read the adjacency matrix from a file
    cout<<"reading adjacency matrix from file: "<< grFile<< endl;   
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, dist, grFile );
    cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";
    
    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    cout<<"reading coordinates from file: "<< coordFile<< endl;
    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N*dim) );
    MeshIO<IndexType, ValueType>::fromFile2Coords_3D(coordFile, coords3D, N );
    EXPECT_EQ(coords3D.size(), dim);
    EXPECT_EQ(coords3D[0].size(), N);

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
    cout << "time elapsed for reading files: "<< duration<< " ms"<< endl;
    
    //partition the graph
    //
    // TODO: a new version of partitionGraph
    // for now use local partition.
    //
    //scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    //EXPECT_EQ(partition.size(), N);
    
    chrono::high_resolution_clock::time_point  t3 = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>( t3 - t2 ).count();
    cout << "time elapsed for partitioning: "<< duration<< " ms"<< endl;
    
    t1 = chrono::high_resolution_clock::now();
    //begin: my partition
    DenseVector<IndexType> perm(dist);
    DenseVector<ValueType> indices(N, 0);
    const std::vector<ValueType> minCoords({0,0,0});  
    const std::vector<ValueType> maxCoords( {coords3D[0].max().getValue<ValueType>(),       coords3D[1].max().getValue<ValueType>(),coords3D[2].max().getValue<ValueType>()}  );
   
    for (IndexType i = 0; i < N; i++){
        indices.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords3D, dim, i, 15, minCoords, maxCoords) );
    }
    //t2 = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t3 ).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for GETTING the hilbert indices: "<< duration<< " ms"<<  endl;
    
    indices.sort(perm, true);
    
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t2).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for SORTING the hilbert indices: "<< duration<< " ms"<<  endl;
    
    scai::lama::DenseVector<IndexType> partition2(N,0);
    IndexType partSize= (int) N/k;
    int j=1;
    
    partition2.setValue(  perm.getValue(0).getValue<IndexType>()  ,0);
    for(i=0; i<k; i++){
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        if(j!=N){
            partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i);
            j++;
        }
    }
    
    // because of (int)N/k some point are let out. Just put them in the last part.
    for(;j<N;j++)
        partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i-1);
    
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t2).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for getting the partition "<< duration<< " ms"<<  endl;   
    
    ValueType cut3= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    cout<< "cut before fiducciaMattheysesRound= "<< cut3 << endl<< flush;
    
    ParcoRepart<IndexType,ValueType>::fiducciaMattheysesRound(graph, partition2, k, epsilon);
    
    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t2).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for one fiducciaMattheysesRound "<< duration<< " ms"<<  endl;    
    
    //end: my partition

    duration = chrono::duration_cast<chrono::milliseconds>( chrono::high_resolution_clock::now() - t3 ).count();
    t2 = chrono::high_resolution_clock::now();
    cout << "time elapsed for \"by hand\"-sfc partitioning: "<< duration<< " ms"<<  endl;
    
    //calculate and print cut and imbalance
    //ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true);
    ValueType cut2= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    //cout<< "# cut 1= "<< cut<< " , ";
    cout<< "cut2= "<<cut2<< endl;
    //ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    ValueType imbalance2= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    //cout<< "# imbalance 1= " << imbalance<< " , ";
    cout <<"imbalance 2= "<< imbalance2 << endl;
    
    t2 = chrono::high_resolution_clock::now();
    cout<<"Test completed in time, chrono:: "<< chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count()<< " ms"<< endl;
}

//-----------------------------------------------------------------
    
}//namespace ITI