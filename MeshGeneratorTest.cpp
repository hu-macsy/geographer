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
#include "MeshGenerator.h"
#include "Settings.h"

typedef double ValueType;
typedef int IndexType;

using namespace scai;

namespace ITI {

class MeshGeneratorTest : public ::testing::Test {

};

//----------------------------------------------------------------------------------------
/* Creates a semi-structured 3D mesh given the number of points for each dimension and the maximum
 * corrdinate in each axis. Writes the graph in METIS format in a .graph file and the coordiantes
 * in a .graph.xyz file.
 * */

TEST_F(MeshGeneratorTest, testMesh3DCreateStructuredMesh_Local_3D) {
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
        MeshGenerator<IndexType, ValueType>::createStructured3DMesh(adjM, coords, maxCoord, numPoints);
    }
    
    {
        SCAI_REGION("testMesh3DCreateStructuredMesh_Local_3D.(writeInFileMetisFormat and writeInFileCoords)")
        IO<IndexType, ValueType>::writeGraphToFile( adjM, grFile);
        IO<IndexType, ValueType>::writeCoordsToFile( coords, numberOfPoints, coordFile);
    }
    
    CSRSparseMatrix<ValueType> graph = IO<IndexType, ValueType>::readGraphFromFile( grFile );
    
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

TEST_F(MeshGeneratorTest, testCreateStructured3DMeshLocalDegreeSymmetry) {
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

		MeshGenerator<IndexType, ValueType>::createStructured3DMesh_dist(a, coordinates, maxCoord, numPoints);
		ParcoRepart<IndexType, ValueType>::checkLocalDegreeSymmetry(a);
	} else {
		std::cout << "Not tested, since called with <= 16 processes, this implies you don't have enough memory for " << n << " nodes."<< std::endl;
	}
}

//-----------------------------------------------------------------
// Creates the part of a structured mesh in each processor ditributed and checks the matrix and the coordinates.
// For the coordinates checks if there are between min and max and for the matrix if every row has more than 3 and
// less than 6 ones ( every node has 3,4,5, or 6 neighbours).
TEST_F(MeshGeneratorTest, testCreateStructuredMesh_Distributed_3D) {
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
    MeshGenerator<IndexType, ValueType>::createStructured3DMesh_dist(adjM, coords, maxCoord, numPoints);
    
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
// Creates the part of a structured mesh in each processor ditributed and checks the matrix and the coordinates.
// For the coordinates checks if there are between min and max and for the matrix if every row has more than 3 and
// less than 6 ones ( every node has 3,4,5, or 6 neighbours).
TEST_F(MeshGeneratorTest, testCreateRandomStructuredMesh_Distributed_3D) {
    std::vector<IndexType> numPoints= { 10, 2, 5};
    std::vector<ValueType> maxCoord= {11, 14, 10};
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
    MeshGenerator<IndexType, ValueType>::createRandomStructured3DMesh_dist(adjM, coords, maxCoord, numPoints);
    
    // print local coordinates
    /*
    for(IndexType i=0; i<dist->getLocalSize(); i++){
        std::cout<< i<< ": "<< *comm<< " - " <<coords[0].getLocalValues()[i] << " , " << coords[1].getLocalValues()[i] << " , " << coords[2].getLocalValues()[i] << std::endl;   
    }
    */
    
    EXPECT_EQ( adjM.getLocalNumColumns() , N);
    EXPECT_EQ( adjM.getLocalNumRows() , coords[0].getLocalValues().size() );
    EXPECT_EQ( true , adjM.getRowDistribution().isEqual(coords[0].getDistribution()) );
    
    // check symmetry in every PE
    ParcoRepart<IndexType, ValueType>::checkLocalDegreeSymmetry( adjM );
    //PRINT(*comm<< ": "<< adjM.getLocalNumValues() );
    //PRINT(*comm<< ": "<< comm->sum(adjM.getLocalNumValues()) );
    
    {
        SCAI_REGION("testCreateRandomStructuredMesh_Distributed_3D.noDist")
        // gather/replicate locally and test whole matrix
        adjM.redistribute(noDistPointer, noDistPointer);
        
        ParcoRepart<IndexType, ValueType>::checkLocalDegreeSymmetry( adjM );
        //PRINT(*comm<<": "<< adjM.getNumValues() );
    }
    
    {
        SCAI_REGION("testCreateRandomStructuredMesh_Distributed_3D.cyclicDist")
        // test with a cyclic distribution
        scai::dmemo::DistributionPtr distCyc ( scai::dmemo::Distribution::getDistributionPtr( "CYCLIC", comm, N) );
        adjM.redistribute( distCyc, noDistPointer);
        
        ParcoRepart<IndexType, ValueType>::checkLocalDegreeSymmetry( adjM );
    }
    
}


//-----------------------------------------------------------------
TEST_F(MeshGeneratorTest, testIndex2_3DPoint){
    std::vector<IndexType> numPoints= {9, 11, 7};
    
    srand(time(NULL));
    for(int i=0; i<3; i++){
        numPoints[i] = (IndexType) (rand()%5 + 10);
    }
    IndexType N= numPoints[0]*numPoints[1]*numPoints[2];
    
    for(IndexType i=0; i<N; i++){
        std::tuple<IndexType, IndexType, IndexType> ind = MeshGenerator<IndexType, ValueType>::index2_3DPoint(i, numPoints);
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
