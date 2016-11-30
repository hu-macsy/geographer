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

TEST_F(MeshIOTest, testMesh3DCreateMeshWriteInFile) {
vector<DenseVector<ValueType>> coords;
int numberOfPoints= 500;
ValueType maxCoord= 1;

scai::lama::CSRSparseMatrix<ValueType> adjM(numberOfPoints, numberOfPoints);

MeshIO<IndexType, ValueType>::create3DMesh(adjM, coords, numberOfPoints, maxCoord);

MeshIO<IndexType, ValueType>::writeInFileMetisFormat( adjM, "M5_test.graph");

}
//-----------------------------------------------------------------
TEST_F(MeshIOTest, testPartitionWithRandom3DMeshLocal) {
    IndexType N= 500;
    ValueType maxCoord= 1;
    IndexType dim= 3, k= 4;
    ValueType epsilon= 0.6;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    
    //the coordinates of the points
    //coords.size()= N , coords[i].size()= dim
    //TODO: should be reversed!!
    vector<DenseVector<ValueType>> coords;    
    //the adjacency matrix
    scai::lama::CSRSparseMatrix<ValueType> adjM(N, N);
    //random coordinates in 3D stored in coords and the adjacency matrix in adjM
    MeshIO<IndexType, ValueType>::create3DMesh(adjM, coords, N, maxCoord);
    
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
    cout << "time elapsed after creating 3DMesh: "<< duration<< " ms"<< endl;
    
    //must convert vector<DenseVector<ValueType>> to DenseVector<ValueType>...
    DenseVector<ValueType> coords2(N*dim, 0);
    ValueType tmp;
    for(IndexType i=0; i<N; i++){
        for(IndexType d=0; d<dim; d++){
            tmp= coords[i].getValue(d).getValue<ValueType>();
            coords2.setValue(i*dim+d, coords[i].getValue(d));
        }
    }
    
    // adjM.getNumValues() returns N values more !!!
    cout<< "Number of nodes= "<< N<< " , Number of edges="<< (adjM.getNumValues()-N)/2 <<endl;
    //get the partition
    DenseVector<IndexType>partition= ParcoRepart<IndexType, ValueType>::partitionGraph( adjM, coords2, dim, k, epsilon);

    //begin: my partition
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
    DenseVector<IndexType> perm( noDistPtr );
    DenseVector<ValueType> indices2(N, 0);
    const std::vector<ValueType> minCoords({0,0});  
    ValueType max= coords2.max().getValue<ValueType>();
    const std::vector<ValueType> maxCoords({max, max, max});
   
    for (IndexType i = 0; i < N; i++){
        indices2.setValue(i, HilbertCurve<IndexType, ValueType>::getHilbertIndex(coords2, dim, i, 10, minCoords, maxCoords) );
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
    string path = "./meshes/trace/";
    string file = "trace-00000.graph";
    string filename= path + file;
    CSRSparseMatrix<ValueType> Graph;
    IndexType N;    //number of points      
    
    Graph =  MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(filename );
    N=Graph.getNumColumns();
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    
    ifstream f(filename);
    IndexType nodes, edges;
    //In the METIS format the two first number in the file are the number of nodes and edges
    f >>nodes >> edges; 
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues()-N)/2 );
    
    string fileTo= path + string("MY_") + file;
    MeshIO<IndexType, ValueType>::writeInFileMetisFormat(Graph, fileTo );
    CSRSparseMatrix<ValueType> Graph2 =  MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(fileTo );
    
    cout<< "Outpur written in file: "<< fileTo<< endl;
    EXPECT_EQ(Graph.getNumValues(), Graph2.getNumValues() );
    
    cout<<"reading from file: "<< filename + string(".xyz")<< endl;
    DenseVector<ValueType> coords2D = MeshIO<IndexType, ValueType>::fromFile2Coords_2D(filename + string(".xyz"), N );
    EXPECT_EQ(coords2D.size()/2, N);
    
}

//-----------------------------------------------------------------
// read a graph from a file in METIS format and its coordiantes in 2D and partiotion that graph
// usually, graph file: "file.graph", coodinates file: "file,graph.xy" or .xyz
TEST_F(MeshIOTest, testPartitionFromFile_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    DenseVector<ValueType> coords2D;        //the coordiantes of each node 
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
    coords2D = MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, N );
    EXPECT_EQ(coords2D.size(), N*dim);
    
    //partiotion the graph
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    EXPECT_EQ(partition.size(), N);
 
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
    
    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true);
    ValueType cut2= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    cout<< "# cut 1= "<< cut<<"    and cut2= "<<cut2<< endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    ValueType imbalance2= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    cout<< "# imbalance 1= " << imbalance<<" , imbalance 2= "<< imbalance2 << endl;
    
}
//-----------------------------------------------------------------
//-----------------------------------------------------------------
// read a graph from a file in METIS format and its coordiantes in 2D and partiotion that graph
// usually, graph file: "file.graph", coodinates file: "file,graph.xy" or .xyz
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
    
    int j=1, jOld=j;
    partition2.setValue(  perm.getValue(0).getValue<IndexType>()  ,0);
    for(i=0; i<k; i++){
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        //cout<< "part "<< i<< " has size: "<< j-jOld<< endl;
        //jOld=j;
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
//-----------------------------------------------------------------

TEST_F(MeshIOTest, testPartitionByHand_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    DenseVector<ValueType> coords2D;        //the coordiantes of each node 
    IndexType dim= 2, k= 10, i;
    ValueType epsilon= 0.1;
    
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
    coords2D = MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, N );
    EXPECT_EQ(coords2D.size(), N*dim);

    /*  not sure if it works corectly. Get an initial partition with sfc "by hand"
    * 
    //partiotion the graph
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    EXPECT_EQ(partition.size(), N);
    */

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

}
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
    const ValueType* data2 = readAccess.get();

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
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    DenseVector<ValueType> coords2D;        //the coordiantes of each node 
    IndexType dim= 2, k= 10, i;
    ValueType epsilon= 0.1;
    
    std::string path = "./meshes/";
    std::string file= "hugetric/hugetric-00025.graph";
    std::string grFile= path +file, coordFile= path +file +".xyz";
    std::fstream f(grFile);
    IndexType nodes, edges;
    f>> nodes>> edges;
    f.close();
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    clock_t t = clock();
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    
    // read the adjacency matrix from a file
    cout<<"reading adjacency matrix from file: "<< grFile<< endl;
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    graph = scai::lama::CSRSparseMatrix<ValueType>(dist, dist);
    
    MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(graph, dist, grFile );
    cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";
    
    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    cout<<"reading coordinates from file: "<< coordFile<< endl;
    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N*dim) );
    coords2D = MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, N );
    EXPECT_EQ(coords2D.size(), N*dim);

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
    cout << "time elapsed after reading files: "<< duration<< " ms"<< endl;
    
    //partition the graph
    //
    // TODO: a new version of partitionGraph
    // for now use local partition.
    //
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    //EXPECT_EQ(partition.size(), N);
    
    chrono::high_resolution_clock::time_point  t3 = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>( t3 - t2 ).count();
    cout << "time elapsed after partitioning: "<< duration<< " ms"<< endl;
    
    
    //begin: my partition
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
    //end: my partition
    t2 = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>( t2 - t3 ).count();
    cout << "time elapsed after \"by hand\"-sfc partitioning: "<< duration<< " ms"<<  endl;
    
    //calculate and print cut and imbalance
    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition, true);
    ValueType cut2= ParcoRepart<IndexType, ValueType>::computeCut(graph, partition2, true);
    cout<< "# cut 1= "<< cut<< " , ";
    cout<< "cut2= "<<cut2<< endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    ValueType imbalance2= ParcoRepart<IndexType, ValueType>::computeImbalance(partition2, k);
    cout<< "# imbalance 1= " << imbalance<< " , ";
    cout <<"imbalance 2= "<< imbalance2 << endl;
    
    t2 = chrono::high_resolution_clock::now();
    cout<<"Test completed in time, chrono:: "<< chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count()<< " ms"<< endl;
}
//-----------------------------------------------------------------
//
// Really meshy. Maybe needed in the future for testing....
// I will keep for the moment.
/*
TEST_F(MeshIOTest, testCSRSparseMatrixBasedOnBuildPoisson_2D){
    IndexType N=100, i;
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    scai::dmemo::DistributionPtr noDist ( new scai::dmemo::NoDistribution(N) );
    //CSRSparseMatrix<ValueType> ret( dist, dist);
    CSRSparseMatrix<ValueType> matrix;// dist, dist);
    
    // Calculate subdomains, subranges
    PartitionId gridSize[3] = { 1, 1, 1 };
    PartitionId gridRank[3] = { 0, 0, 0 };
    IndexType lb=0;
    
    //comm->getGrid3Rank( gridRank, gridSize );
    dmemo::BlockDistribution::getLocalRange( lb, N, N, gridRank[0], gridSize[0] );
    
    cout <<   *comm << ": rank = (" << gridRank[0] << "," << gridRank[1] << "," << gridRank[2] 
                    << ") of (" << gridSize[0] << "," << gridSize[1] << "," << gridSize[2] << std::endl;
                    
    //read from file
    const string filename= "meshes/trace/trace-00004.graph";
    ifstream file(filename);
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
    else
        cout<< "Reading from file "<< filename<< endl;

    IndexType numEdges;
    file >>N >> numEdges;
    cout<< "#nodes= "<< N<< " , #edges: "<< numEdges<< endl;
    IndexType globalSize = N; // number of rows, columns of full matrix
    std::vector<IndexType> myGlobalIndexes; // row indexes of this processor
    IndexType myNNA = 0; // accumulated sum for number of my entries
   
    // compute local size on this processor
    IndexType localSize = N;
    //for ( IndexType i = 0; i < dimension; i++ )
    //    localSize *= dimUB[i] - dimLB[i];
    
    cout<< *comm << ": has local size = " << localSize<< endl;
    myGlobalIndexes.reserve( localSize );
    
    //this part is used for initializing a distribution where every processor knows
    //its local indices. 
    //Skip for now, not distributed
    //TODO: use this to make it distributed
    for(i=0; i<N; i++){
        //read from file. file should have N lines, one for every vertex
        //row[i] = read line i
        //now, do it for some rows
        if(1){
            //myGlobalIndexes.push_back( i );
            //myNNA++;    
        }
    }
    cout<< *comm << ": has local " << localSize << " rows, nna = " << myNNA<< " and globalSize= "<< globalSize<< endl;    
        
    //for the distributed version
    // allocate and fill local part of the distributed matrix
    //hmemo::HArrayRef<IndexType> indexes(  myGlobalIndexes );
    //dmemo::DistributionPtr distribution( new dmemo::GeneralDistribution( globalSize, indexes, comm ) );
    dmemo::DistributionPtr distribution( dmemo::Distribution::getDistributionPtr("BLOCK", comm, globalSize ) );
    cout<< "distribution = " << *distribution<< endl;
    // create new local CSR data ( # local rows x # columns )
    scai::lama::CSRStorage<double> localMatrix;
    localMatrix.allocate( localSize, globalSize );
    
    // Allocate local matrix with correct sizes and correct first touch in case of OpenMP
    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<double> csrValues;           
    
    
    {        

        //count is the number of numbers in the file. In the Metis format (count-2)/2 are the edges of the graph
        //also, this is the number of non zero elements of the CSR matrix
        //std::size_t count = std::distance(std::istream_iterator<std::string>(file),std::istream_iterator<std::string>());
        
cout<<__LINE__<<": "<< N<< " , # of numbers in file: "<< numEdges<< endl;
cout<<"N= "<< N<< " , numEdges= "<< numEdges<< endl;
        //EXPECT_EQ((count-2)/2, numEdges);
        
        //TODO: for a distributed version this must change as myNNA should be the number of
        //      the local nodes in the processor.
        // *2 because every edge is read twice.
        myNNA = numEdges*2;
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, localSize +1 );
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA, myNNA );
        hmemo::WriteOnlyAccess<double> values( csrValues, myNNA );

        ia[0] = 0;

        std::vector<IndexType> colIndexes;
        std::vector<int> colValues;     

        // compute global indexes this processor is responsibile for and number of non-zero values
        // Important: same loop order as above
        IndexType rowCounter = 0; // count local rows
        IndexType nnzCounter = 0; // count local non-zero elements
        IndexType checkNnz= 0;
cout<<__LINE__<<": #nodes= "<< N<< " and myNNA= "<< myNNA << endl;  
        std::string line;
        std::getline(file, line);
        //for every line, aka for all nodes
        for ( IndexType i=0; i<N; i++ ){
            std::getline(file, line);
            vector< vector<int> > line_integers;
            istringstream iss( line );
            line_integers.push_back( vector<int>( istream_iterator<int>(iss), istream_iterator<int>() ) );
//cout<<__LINE__<<": "<< i<< endl;            
            //ia += the numbers of neighbours of i = line_integers.size()
            ia[rowCounter + 1] = ia[rowCounter] + static_cast<IndexType>( line_integers[0].size() );
//cout<<__LINE__<<": ia["<<rowCounter+1<< "] = "<< ia[rowCounter+1]<<endl;
            checkNnz += line_integers[0].size();               
            //for all numbers in the line, aka all the neighbours of node i
            for(IndexType j=0; j<line_integers[0].size(); j++){
                // -1 because of the METIS format
                ja[nnzCounter]= line_integers[0][j] -1 ;
                // all values are 1 for undirected, no-weigths graph
                values[nnzCounter]= 1;
                ++nnzCounter;
//cout<<__LINE__<<": "<< line_integers[0][j]<< endl; 
            }
              
            ++rowCounter;
//cout<<__LINE__<<": "<< nnzCounter<< endl;  
        }
        
        cout<< "rowCounter=" << rowCounter<< " , nnzCounter="<< nnzCounter<< " , checkNnz= "<< checkNnz<< endl;
        EXPECT_EQ(nnzCounter, checkNnz);
    }

cout<<__LINE__<<": IA.size= "<< csrIA.size()<< " , JA.size= "<< csrJA.size()<< " , Values.size= "<< csrValues.size()<<  endl;                
//localMatrix._MatrixStorage::print();
cout<< "local matrix: ["<< localMatrix.getNumColumns()<<","<<localMatrix.getNumRows()<<"]\n";

    localMatrix.swap( csrIA, csrJA, csrValues );
    cout<< "replace owned data with " << localMatrix<< endl;
    matrix.assign( localMatrix, distribution, distribution ); // builds also halo
    

    
    // but now the local part of matrixA should have the diagonal property as global column // indexes have been localized
    // is not for each storage format the case 
    // SCAI_ASSERT_DEBUG( matrix.getLocalStorage().hasDiagonalProperty(), "local storage data has not diagonal property: " << matrix )

    cout<< "built matrix A = " << matrix<< endl;
    
    // out an output file name to print the matrix in a file and test 
    // if the graph read from the file
    //std::string outFile = "";
    //MeshIO<IndexType, ValueType>::writeInFileMetisFormat(matrix, outFile);

}
*/
//-----------------------------------------------------------------
    
}//namespace ITI