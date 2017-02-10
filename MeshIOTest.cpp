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
#include "Settings.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class MeshIOTest : public ::testing::Test {

};

//----------------------------------------------------------------------------------------
/* Creates a semi-structured 3D mesh given the number of points for each dimension and the maximum
 * corrdinate in each axis. Writes the graph in METIS format in a .graph file and the coordiantes
 * in a .graph.xyz file.
 * */

TEST_F(MeshIOTest, testMesh3DCreateStructuredMesh_Local_3D) {
    std::vector<IndexType> numPoints= {8, 7, 10};
    std::vector<ValueType> maxCoord= {100,180,130};
    // set number of points in random
    /*
    srand(time(NULL));
    for(int i=0; i<3; i++){
        numPoints[i] = (IndexType) (rand()%4 +7);
    }
    */
    IndexType numberOfPoints= numPoints[0]*numPoints[1]*numPoints[2];
    
    std::vector<DenseVector<ValueType>> coords(3, DenseVector<ValueType>(numberOfPoints, 0));
    std::string grFile = "meshes/structuredTest7.graph";
    std::string coordFile= grFile + ".xyz";

    
    scai::lama::CSRSparseMatrix<ValueType> adjM( numberOfPoints, numberOfPoints);
    std::cout<<__FILE__<< "  "<< __LINE__<< " , numberOfPoints=" << numberOfPoints << " in every axis: "<< numPoints[0] << ", "<< numPoints[1] << ", "<< numPoints[2] << std::endl;

    {
        SCAI_REGION("testMesh3DCreateStructuredMesh_Local_3D.createStructured3DMesh" )
        MeshIO<IndexType, ValueType>::createStructured3DMesh(adjM, coords, maxCoord, numPoints);
    }
    
    {
        SCAI_REGION("testMesh3DCreateStructuredMesh_Local_3D.(writeInFileMetisFormat and writeInFileCoords)")
        MeshIO<IndexType, ValueType>::writeInFileMetisFormat( adjM, grFile);
        MeshIO<IndexType, ValueType>::writeInFileCoords( coords, numberOfPoints, coordFile);
    }
    
    CSRSparseMatrix<ValueType> graph = MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( grFile );
    
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

TEST_F(MeshIOTest, testCreateStructured3DMeshLocalDegreeSymmetry) {
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	IndexType k = comm->getSize();

	IndexType nroot = 300;
	IndexType n = nroot * nroot * nroot;
	IndexType dimensions = 3;

	if (k > 16) {
		scai::dmemo::DistributionPtr dist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, n) );
		scai::dmemo::DistributionPtr noDistPointer(new scai::dmemo::NoDistribution(n));

		scai::lama::CSRSparseMatrix<ValueType>a(dist, noDistPointer);
		std::vector<ValueType> maxCoord(dimensions, nroot);
		std::vector<IndexType> numPoints(dimensions, nroot);

		std::vector<DenseVector<ValueType>> coordinates(dimensions);
		for(IndexType i=0; i<dimensions; i++){
		  coordinates[i].allocate(dist);
		  coordinates[i] = static_cast<ValueType>( 0 );
		}

		MeshIO<IndexType, ValueType>::createStructured3DMesh_dist(a, coordinates, maxCoord, numPoints);
		ParcoRepart<IndexType, ValueType>::checkLocalDegreeSymmetry(a);
	} else {
		std::cout << "Not tested, since called with <= 16 processes, this implies you don't have enough memory for " << n << " nodes."<< std::endl;
	}
}

//-----------------------------------------------------------------
// Creates the part of a structured mesh in each processor ditributed and checks the matrix and the coordinates.
// For the coordinates checks if there are between min and max and for the matrix if every row has more than 3 and
// less than 6 ones ( every node has 3,4,5, or 6 neighbours).
TEST_F(MeshIOTest, testCreateStructuredMesh_Distributed_3D) {
    std::vector<IndexType> numPoints= { 40, 40, 40};
    std::vector<ValueType> maxCoord= {441, 711, 1160};
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
    MeshIO<IndexType, ValueType>::createStructured3DMesh_dist(adjM, coords, maxCoord, numPoints);
    
    // print local values 
    /*
    for(IndexType i=0; i<dist->getLocalSize(); i++){
        std::cout<< i<< ": "<< *comm<< " - " <<coords[0].getLocalValues()[i] << " , " << coords[1].getLocalValues()[i] << " , " << coords[2].getLocalValues()[i] << std::endl;   
    }
    */
    
    EXPECT_EQ( adjM.getLocalNumColumns() , N);
    EXPECT_EQ( adjM.getLocalNumRows() , coords[0].getLocalValues().size() );
    EXPECT_EQ( true , adjM.getRowDistribution().isEqual(coords[0].getDistribution()) );
    
    // for a 3D structured grid with dimensions AxBxC the number of edges is 3ABC-AB-AC-BC
    IndexType numEdges= 3*numPoints[0]*numPoints[1]*numPoints[2] - numPoints[0]*numPoints[1]\
                                -numPoints[0]*numPoints[2] - numPoints[1]*numPoints[2];
    EXPECT_EQ( adjM.getNumValues() , numEdges*2 );     
    
    IndexType cntCorners= 0 , cntSides= 0, cntEdges= 0;
    
    {
        SCAI_REGION("testCreateStructuredMesh_Distributed_3D.check_adjM_2")
        const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
	const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
        
        for(IndexType i=0; i<ia.size()-1; i++){
            // this checks that the number of non-zero elements in each row is less than 6 
            // (6 is the maximum number of neighbours a node can have in a structured grid)
            EXPECT_LE( ia[i+1]-ia[i], 6 );
            // and also more than 3 which is the minimum
            EXPECT_GE( ia[i+1]-ia[i], 3 );
            
            // count the nodes with 3 edges
            if(ia[i+1]-ia[i] == 3){
                ++cntCorners;
            }
            // count the nodes with 4 edges
            if(ia[i+1]-ia[i] == 4){
                ++cntEdges;
            }
            // count the nodes with 4 edges
            if(ia[i+1]-ia[i] == 5){
                ++cntSides;
            }
        }
    }
    IndexType numX= numPoints[0];
    IndexType numY= numPoints[1];
    IndexType numZ= numPoints[2];
    
    //PRINT( comm->sum(cntCorners) );
    
    // check the global values
    EXPECT_EQ( comm->sum(cntCorners), 8);
    EXPECT_EQ( comm->sum(cntEdges) , 4*(numX+numY+numZ)-24);
    EXPECT_EQ( comm->sum(cntSides) , 2*( (numX-2)*(numY-2)+ (numX-2)*(numZ-2)+ (numY-2)*(numZ-2) )  );
    
    PRINT(", corner nodes= "<< cntCorners << " , edge nodes= "<< cntEdges<< " , side nodes= "<< cntSides);
    
    {
    SCAI_REGION("testCreateStructuredMesh_Distributed_3D.check_coords_2")
    std::vector<scai::utilskernel::LArray<ValueType>> localCoords(3);
    for(IndexType i=0; i<3; i++){
        localCoords[i] = coords[i].getLocalValues();
    }
    for(IndexType i=0; i<localCoords[0].size(); i++){
        EXPECT_LE( localCoords[0][i] , maxCoord[0]);
        EXPECT_GE( localCoords[0][i] , 0);
        EXPECT_LE( localCoords[1][i] , maxCoord[1]);
        EXPECT_GE( localCoords[1][i] , 0);
        EXPECT_LE( localCoords[2][i] , maxCoord[2]);
        EXPECT_GE( localCoords[2][i] , 0);
    }
    }
    
}

//-----------------------------------------------------------------

TEST_F(MeshIOTest, testPartitionWithRandom3DMesh_Local_3D) {
    IndexType N= 40;
    ValueType maxCoord= 1;
    IndexType dim= 3, k= 8;
    ValueType epsilon= 0.2;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    //the coordinates of the points: coords.size()= dim , coords[dim].size()= N
    std::vector<DenseVector<ValueType>> coords(3);    
    
    //the adjacency matrix
    scai::lama::CSRSparseMatrix<ValueType> adjM(N, N);
    
    //random coordinates in 3D stored in coords and the adjacency matrix in adjM
    MeshIO<IndexType, ValueType>::createRandom3DMesh(adjM, coords, N, maxCoord);
    
    //std::cout<< "Number of nodes= "<< N<< " , Number of edges="<< (adjM.getNumValues()-N)/2 << std::endl;
    
    SCAI_REGION_START("testPartitionWithRandom3DMesh_Local_3D.partitionGraph");
    //get the partition
    
    struct Settings Settings;
    DenseVector<IndexType> partition= ParcoRepart<IndexType, ValueType>::partitionGraph( adjM, coords, Settings);
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
TEST_F(MeshIOTest, testWriteMetis_Dist_3D){
    
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
    MeshIO<IndexType, ValueType>::createStructured3DMesh_dist(adjM, coords, maxCoord, numPoints);
    
    // write the mesh in p(=number of PEs) files
    MeshIO<IndexType, ValueType>::writeInFileMetisFormat_dist( adjM, "meshes/dist3D_");
    
}

//-----------------------------------------------------------------
/* Reads a graph from a file "filename" in METIS format, writes it back into "my_filename" and reads the graph
 * again from "my_filename".
 * 
 * Occasionally throws error, probably because onw process tries to read the file while some other is still eriting in it.
 */
TEST_F(MeshIOTest, testReadAndWriteGraphFromFile){
    std::string path = "meshes/bigbubbles/";
    std::string file = "bigbubbles-00010.graph";
    std::string filename= path + file;
    CSRSparseMatrix<ValueType> Graph;
    IndexType N;    //number of points     
    
    std::ifstream f(filename);
    IndexType nodes, edges;
    //In the METIS format the two first number in the file are the number of nodes and edges
    f >>nodes >> edges; 
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    dmemo::DistributionPtr dist( new dmemo::NoDistribution( nodes ));
    
    // read graph from file
    {
        SCAI_REGION("testReadAndWriteGraphFromFile.readFromFile2AdjMatrix");
        Graph = MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(filename );
    }
    N = Graph.getNumColumns();
    EXPECT_EQ(Graph.getNumColumns(), Graph.getNumRows());
    EXPECT_EQ(nodes, Graph.getNumColumns());
    EXPECT_EQ(edges, (Graph.getNumValues())/2 );
    
    std::string fileTo= path + std::string("MY_") + file;
    
    // write the graph you read in a new file
    MeshIO<IndexType, ValueType>::writeInFileMetisFormat(Graph, fileTo );
    
    // read new graph from the new file we just written
    CSRSparseMatrix<ValueType> Graph2 = MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( fileTo );
    
    // check that the two graphs are identical
    std::cout<< "Output written in file: "<< fileTo<< std::endl;
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
// read a graph from a file in METIS format and its coordiantes in 2D and partiotion that graph
// usually, graph file: "file.graph", coodinates file: "file.graph.xy" or .xyz
TEST_F(MeshIOTest, testPartitionFromFile_dist_2D){
    CSRSparseMatrix<ValueType> graph;       //the graph as an adjacency matrix  
    IndexType dim= 2, k= 8, i;
    ValueType epsilon= 0.1;
    
    //std::string path = "./meshes/my_meshes";s
    std::string path = "";
    std::string file= "Grid8x8";
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
    std::cout<<"reading adjacency matrix from file: "<< grFile<<" for k="<< k<< std::endl;
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, nodes) );
    scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution( nodes ));
    
    SCAI_REGION_START("testPartitionFromFile_local_2D.readFromFile2AdjMatrix");
        graph = MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( grFile );
        graph.redistribute( distPtr , noDistPtr);
        std::cout<< "graph has <"<< nodes<<"> nodes and -"<< edges<<"- edges\n";
    SCAI_REGION_END("testPartitionFromFile_local_2D.readFromFile2AdjMatrix");
    
    // N is the number of nodes
    IndexType N= graph.getNumColumns();
    EXPECT_EQ(nodes,N);
    
    //read the coordiantes from a file
    std::cout<<"reading coordinates from file: "<< coordFile<< std::endl;
    
    SCAI_REGION_START("testPartitionFromFile_local_2D.readFromFile2Coords_2D");
    std::vector<DenseVector<ValueType>> coords2D = MeshIO<IndexType, ValueType>::fromFile2Coords( coordFile, N, dim);
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
        
        struct Settings Settings;
        Settings.numBlocks= k;
        Settings.epsilon = epsilon;
        //partition the graph
        scai::lama::DenseVector<IndexType> partition = ParcoRepart<IndexType, ValueType>::partitionGraph(graph, coords2D, Settings );
        EXPECT_EQ(partition.size(), N);
    SCAI_REGION_END("testPartitionFromFile_local_2D.partition");
        
}
//-----------------------------------------------------------------
TEST_F(MeshIOTest, testIndex2_3DPoint){
    std::vector<IndexType> numPoints= {9, 11, 7};
    
    srand(time(NULL));
    for(int i=0; i<3; i++){
        numPoints[i] = (IndexType) (rand()%5 + 10);
    }
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];
    
    for(IndexType i=0; i<N; i++){
        std::tuple<IndexType, IndexType, IndexType> ind = MeshIO<IndexType, ValueType>::index2_3DPoint(i, numPoints);
        EXPECT_LE(std::get<0>(ind) , numPoints[0]-1);
        EXPECT_LE(std::get<1>(ind) , numPoints[1]-1);
        EXPECT_LE(std::get<2>(ind) , numPoints[2]-1);
        EXPECT_GE(std::get<0>(ind) , 0);
        EXPECT_GE(std::get<1>(ind) , 0);
        EXPECT_GE(std::get<2>(ind) , 0);
    }
}

//-----------------------------------------------------------------
    
}//namespace ITI
