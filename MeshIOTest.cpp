#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <memory>
#include <cstdlib>
#include <fstream>
#include <iostream>

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

TEST_F(MeshIOTest, testMesh3D_1) {
vector<DenseVector<ValueType>> coords;
int numberOfPoints= 500;
ValueType maxCoord= 1;

scai::lama::CSRSparseMatrix<ValueType> adjM(numberOfPoints, numberOfPoints);

MeshIO<IndexType, ValueType>::create3DMesh(adjM, coords, numberOfPoints, maxCoord);

/*
for(int i=0; i<numberOfPoints; i++){
    cout<< i<< ": ";
    for(int j=0; j<3; j++)
        cout<< coords[i].getValue(j)<< ", \t";
    cout<< endl;
}
 
int testDim =10;
vector<DenseVector<ValueType>> coords2;
scai::lama::CSRSparseMatrix<ValueType> adjM2(testDim, testDim);
MeshIO<IndexType, ValueType>::create3DMesh(adjM2, coords2, testDim, maxCoord);
for(int i=0; i<testDim; i++){
    cout<< i<< ": ";
    for(int j=0; j<testDim; j++)
        cout<< adjM2(i,j).Scalar::getValue<ValueType>()<< ", \t";
    cout<< endl;
}
cout<< "l1Norm= "<< adjM2.l1Norm()<< endl;
*/

MeshIO<IndexType, ValueType>::writeInFileMetisFormat( adjM, "M5_test.graph");


}
//-----------------------------------------------------------------
TEST_F(MeshIOTest, testPartitionWithRandom3DMeshLocal) {
    IndexType N= 100;
    ValueType maxCoord= 1;
    IndexType dim= 3, k= 4;
    ValueType epsilon= 0.6;

    //the coordinates of the points
    //coords.size()= N , coords[i].size()= dim
    //TODO: should be reversed!!
    vector<DenseVector<ValueType>> coords;    
    //scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(N));
    //the adjacency matrix
    scai::lama::CSRSparseMatrix<ValueType> adjM(N, N);
    //random coordinates in 3D stored in coords and the adjacency matrix in adjM
    MeshIO<IndexType, ValueType>::create3DMesh(adjM, coords, N, maxCoord);
    
 /* Print data for debug reasons.
  * 
    for( IndexType i=0; i<N; i++){
        for(IndexType j=0; j<N; j++)
            cout<< j<< ": "<< adjM(i,j).getValue<ValueType>()<< " - ";
        cout<<endl;
    }
cout<<endl;
    for(IndexType i=0; i<N; i++){
        for(IndexType d=0; d<dim; d++)
            cout<<coords[i].getValue(d).getValue<ValueType>()<<" _ ";
        cout<< endl;
    }
*/    
    ValueType tmp;
    //must convert vector<DenseVector<ValueType>> to DenseVector<ValueType>...
    DenseVector<ValueType> coords2(N*dim, 0);
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

    for(IndexType i=0; i<N; i++){
        cout<< i<< ": "<<partition.getValue(i).getValue<ValueType>()<< " --> ";
        for(IndexType j=0; j<N; j++){
            if( adjM(i,j)==1)
                cout<< partition.getValue(j).getValue<ValueType>()<< " , ";
        }
        cout<< endl;
    }

    ValueType cut= ParcoRepart<IndexType, ValueType>::computeCut(adjM, partition, true);
    cout<< "# cut is "<< cut<< endl;
    ValueType imbalance= ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    cout<< "# imbalance is " << imbalance<< endl;
    
    vector<IndexType> numPart(k,0);
    for(IndexType i=0; i<N; i++){
        numPart[ ((int) partition.getValue(i).getValue<ValueType>()) ]+=1;
    }
    for(IndexType i=0; i<k; i++)
        cout<< numPart[i] << endl;

}

//-----------------------------------------------------------------
TEST_F(MeshIOTest, testReadAndWriteGraphFromFile){
    string path = "./meshes/";
    string file = "trace-00000.graph";
    string filename= path + file;
    CSRSparseMatrix<ValueType> Graph;
    IndexType N;    //number of points      
    
    Graph =  MeshIO<IndexType, ValueType>::fromFile2AdjMatrix(filename );
    N=Graph.getNumColumns();
    //cout<< Graph.getNumColumns() << " @@ "<< Graph.getNumRows()<<" ## "<<\
        (Graph.getNumValues()-N)/2 << endl;
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    
    ifstream f(filename);
    IndexType nodes, edges;
    //In the METIS format the two first number in the file are the number of nodes and edges
    f >>nodes >> edges; 
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues()-N)/2 );
    
    string fileTo= path + string("MY_") + file;
    MeshIO<IndexType, ValueType>::writeInFileMetisFormat(Graph, fileTo );
    CSRSparseMatrix<ValueType> Graph2 =  MeshIO<IndexType, ValueType>::fromFile2AdjMatrix(fileTo );
    
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
    graph =  MeshIO<IndexType, ValueType>::fromFile2AdjMatrix( grFile );
    cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";

    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    cout<<"reading coordinates from file: "<< coordFile<< endl;

    scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N*dim) );
    coords2D = MeshIO<IndexType, ValueType>::fromFile2Coords_2D(coordFile, N );
    EXPECT_EQ(coords2D.size(), N*dim);
    
    //scale the coordinates in [0,1]
    //DenseVector<ValueType> scaledCoords = coords2D;
    //ValueType maxCoord= coords2D.max().getValue<ValueType>();
    

    //partiotion the graph
    scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, dim,  k, epsilon);
    EXPECT_EQ(partition.size(), N);
  
    int gS=100;
//begin test print
/*
IndexType partViz[gS][gS];
for(i=0; i<gS; i++)
    for(int j=0; j<gS; j++)
        partViz[i][j]=partition.getValue(i*gS+j).getValue<IndexType>();
    
for(i=0; i<gS; i++){
    for(int j=0; j<gS; j++)
        cout<< partViz[i][j]<<"-";
    cout<<endl;
}

for(i=0; i<40; i++){
    cout<< i<<": ["<< coords2D.getValue(i*dim).getValue<ValueType>()<< ", "<< coords2D.getValue(i*dim+1).getValue<ValueType>()<<\
                "]"<< " in partition "<< partition.getValue(i).getValue<ValueType>()<< endl;
    cout<< "\t";
    for(IndexType j=0; j<N; j++){
        if(graph.getValue(i,j)==1)
            cout<< "["<< coords2D.getValue(j*dim).getValue<ValueType>()<< ", "<< coords2D.getValue(j*dim+1).getValue<ValueType>()<<"] "\
                        <<"part<" << partition.getValue(j).getValue<ValueType>()<< "> -- ";
    }
    cout<<endl;
}
*/
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
//std::cout<<"j="<< j<< " , "<< partSize<<" "<< partition2.getValue(0)<< std::endl;
        while(j%partSize!=0 && j<N){
            partition2.setValue(  perm.getValue(j).getValue<IndexType>()  ,i);
            j++;
        }
        if(j!=N){
            partition2.setValue(perm.getValue(j).getValue<IndexType>() ,i);
            j++;
        }
    }

//begin test print
/*
std::cout<<"----------------------------"<< partition2.getValue(0) <<std::endl;
IndexType partViz2[gS][gS];
for(i=0; i<gS; i++)
    for(int j=0; j<gS; j++)
        partViz2[i][j]=partition2.getValue(i*gS+j).getValue<IndexType>();
    
for(i=0; i<gS; i++){
    for(int j=0; j<gS; j++)
        cout<< partViz2[i][j]<<"-";
    cout<<endl;
}
*/
    
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
    std::string file= "bigtric/bigtric-00004.graph";
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
    graph = MeshIO<IndexType, ValueType>::fromFile2AdjMatrix( grFile );
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

}//namespace ITI