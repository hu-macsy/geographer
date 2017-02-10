/*
 * MeshIO.cpp
 *
 *  Created on: 22.11.2016
 *      Author: tzovas
 */

#include "MeshIO.h"
#include <chrono>

#include <scai/common/macros/assert.hpp>

using std::string;
using std::list;
using std::ifstream;
using std::istream_iterator;
using std::ofstream;
using std::endl;
using std::istringstream;

/*
struct rectancle{
    IndexType id;
    std::tuple<IndexType, IndexType, IndexType> bottomCorner;
    std::tuple<IndexType, IndexType, IndexType> topCorner;
    
    // 6 kinds of neighbours
    std::vector<IndexType> xMin;    // left  - side of rect that x is minimum
    std::vector<IndexType> xMax;    // right - x is maximum
    std::vector<IndexType> yMin;    // front - y is minimum
    std::vector<IndexType> yMax;    // back  - y is maximum
    std::vector<IndexType> zMin;    // bottom- z is minimum
    std::vector<IndexType> zMax;    // top   - z is maximum
    
    // upon division in 8 subrectancles the inner connections are ...
    
};
*/

namespace ITI{

template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createRandom3DMesh( CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, const int numberOfPoints, const ValueType maxCoord) {
    SCAI_REGION( "MeshIO.createRandom3DMesh" )

    int n = numberOfPoints;
    int i, j;
    
    coords = MeshIO::randomPoints(n, 3, maxCoord);
 
    srand(time(NULL));    
    int bottom= 4, top= 8;
    Scalar dist;
    common::scoped_array<ValueType> adjArray( new ValueType[ n*n ]);
    //initialize matrix with zeros
    for(i=0; i<n; i++)
        for(j=0; j<n; j++)
            adjArray[i*n+j]=0;
        
    for(i=0; i<n; i++){
        int k= ((int) rand()%(top-bottom) + bottom);
        std::list<ValueType> kNNdist(k,maxCoord*1.7);       //max distance* sqrt(3)
        std::list<IndexType> kNNindex(k,0);
        typename std::list<ValueType>::iterator liVal;
        typename std::list<IndexType>::iterator liIndex = kNNindex.begin();
  
        for(j=0; j<n; j++){
            if(i==j) continue;
            DenseVector<ValueType> p1(3,0), p2(3,0);
            p1.setValue(0, coords[0].getValue(i));
            p1.setValue(1, coords[1].getValue(i));
            p1.setValue(2, coords[2].getValue(i));
            
            p2.setValue(0, coords[0].getValue(j));
            p2.setValue(1, coords[1].getValue(j));
            p2.setValue(2, coords[2].getValue(j));
            
            dist = MeshIO<IndexType, ValueType>::dist3D(p1 ,p2);

            liIndex= kNNindex.begin();
            for(liVal=kNNdist.begin(); liVal!=kNNdist.end(); ++liVal){
                if(dist.getValue<ValueType>()< (*liVal)*(*liVal) ){
                    kNNindex.insert(liIndex, j);
                    kNNdist.insert(liVal , dist.getValue<ValueType>());
                    kNNindex.pop_back();
                    kNNdist.pop_back();
                    break;
                }
                if(liIndex!=kNNindex.end()) ++liIndex;
                else break;
            }
        }    
        kNNindex.sort();
        liIndex= kNNindex.begin();

        for(IndexType col=0; col<n; col++){
            if(col== *liIndex){
                //undirected graph, symmetric adjacency matrix
                adjArray[i*n +col] = 1;
                adjArray[col*n +i] = 1;
                if(liIndex!=kNNindex.end()) ++liIndex;
                else  break;
            }
        }   
    }
    
    //brute force zero in the diagonal
    //TODO: should not be needed but sometimes ones appear in the diagonal
    for(i=0; i<n; i++) 
        adjArray[i*n +i]=0;
    
    //TODO: NoDistribution should be "BLOCK"?
    //dmemo::DistributionPtr rep( new dmemo::NoDistribution( n ));
    adjM.setRawDenseData( n, n, adjArray.get() );
    assert(adjM.checkSymmetry() );
 
}
//-------------------------------------------------------------------------------------------------
// coords.size()= 3 , coords[i].size()= N
// here, N= numPoints[0]*numPoints[1]*numPoints[2]
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createStructured3DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, std::vector<ValueType> maxCoord, std::vector<IndexType> numPoints) {
	SCAI_REGION( "MeshIO.createStructured3DMesh" )

	if (coords.size() != 3) {
		throw std::runtime_error("Needs three coordinate vectors, one for each dimension");
	}

	if (numPoints.size() != 3) {
		throw std::runtime_error("Needs three point counts, one for each dimension");
	}

    std::vector<ValueType> offset={maxCoord[0]/numPoints[0], maxCoord[1]/numPoints[1], maxCoord[2]/numPoints[2]};
    IndexType N= numPoints[0]* numPoints[1]* numPoints[2];
    // create the coordinates
    IndexType index=0;
    {
    SCAI_REGION("createStructured3DMesh.setCoordinates");   
    for( IndexType indX=0; indX<numPoints[0]; indX++){
        for( IndexType indY=0; indY<numPoints[1]; indY++){
            for( IndexType indZ=0; indZ<numPoints[2]; indZ++){
                coords[0].setValue(index, indX*offset[0] );
                coords[1].setValue(index, indY*offset[1] );
                coords[2].setValue(index, indZ*offset[2] );
                ++index;
            }
        }
    }
    }

    scai::lama::CSRStorage<ValueType> localMatrix;
    localMatrix.allocate( N, N );
    
    //create the adjacency matrix
    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<ValueType> csrValues;
    // ja and values have size= edges of the graph
    // for a 3D structured grid with dimensions AxBxC the number of edges is 3ABC-AB-AC-BC
    IndexType numEdges= 3*numPoints[0]*numPoints[1]*numPoints[2] - numPoints[0]*numPoints[1]\
                                -numPoints[0]*numPoints[2] - numPoints[1]*numPoints[2];
    {    
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, N +1 );
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA, numEdges*2);
        hmemo::WriteOnlyAccess<ValueType> values( csrValues, numEdges*2);
        ia[0] = 0;
     
        IndexType nnzCounter = 0; // count non-zero elements
        // for every node= for every line of adjM
        for(IndexType i=0; i<N; i++){
            // connect the point with its 6 (in 3D) neighbours
            // neighbour_node: the index of a neighbour of i, can take negative values
            // but in that case we do not add it
            float ngb_node = 0;      
            // the number of neighbours for each node. Can be less that 6.
            int numRowElems= 0;
            ValueType max_offset =  *max_element(offset.begin(),offset.end());
            
            std::tuple<IndexType, IndexType, IndexType> thisPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( i, numPoints);
            
            {
            SCAI_REGION("createStructured3DMesh.setAdjacencyMatrix");
            // for all 6 possible neighbours
            for(IndexType m=0; m<6; m++){
                switch(m){
                    case 0: ngb_node= i+1; break;
                    case 1: ngb_node= i-1; break;
                    case 2: ngb_node = i + numPoints[2]; break;
                    case 3: ngb_node = i - numPoints[2]; break;
                    case 4: ngb_node = i + numPoints[2]*numPoints[1]; break;
                    case 5: ngb_node = i - numPoints[2]*numPoints[1]; break;
                }
                
                if(ngb_node>=0 && ngb_node<N){
                	std::tuple<IndexType, IndexType, IndexType> ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( ngb_node, numPoints);
                    
                    // we need to check distance for the nodes at the outer borders of the grid: eg:
                    // in a 4x4 grid with 16 nodes {0, 1, 2, ...} , for p1= node 3 there is an edge with
                    // p2= node 4 that we should not add (node 3 has coords (0,3) and node 4 has coords (1,0)).
                    // A way to avoid that is check if thery are close enough.
                    // TODO: maybe find another, faster way to avoid adding that kind of edges
                    
                    if(dist3DSquared( thisPoint, ngbPoint) <= 1)
                    {
                        ja[nnzCounter]= ngb_node;       // -1 for the METIS format
                        values[nnzCounter] = 1;         // unweighted edges
                        ++nnzCounter;
                        ++numRowElems;
                    }
                }
            }
            }
            ia[i+1] = ia[i] +static_cast<IndexType>(numRowElems);
        }//for
        SCAI_ASSERT_EQUAL_ERROR(numEdges*2 , ia[ia.size()-1] )
    }
    SCAI_ASSERT_EQUAL_ERROR(numEdges*2 , csrValues.size() )
    SCAI_ASSERT_EQUAL_ERROR(numEdges*2 , csrJA.size() )
    
    localMatrix.swap( csrIA, csrJA, csrValues );
    adjM.assign(localMatrix);
}

//-------------------------------------------------------------------------------------------------
// coords.size()= 3 , coords[i].size()= N
// here, N= numPoints[0]*numPoints[1]*numPoints[2]

template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createStructured3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, std::vector<ValueType> maxCoord, std::vector<IndexType> numPoints) {
    SCAI_REGION( "MeshIO.createStructured3DMesh_dist" )
    
    if (coords.size() != 3) {
        throw std::runtime_error("Needs three coordinate vectors, one for each dimension");
    }
    
    if (numPoints.size() != 3) {
        throw std::runtime_error("Needs three point counts, one for each dimension");
    }
        
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    
    if( !dist->isEqual( coords[0].getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and coordinates dist: "<< coords[0].getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }
    
    std::vector<ValueType> offset={maxCoord[0]/numPoints[0], maxCoord[1]/numPoints[1], maxCoord[2]/numPoints[2]};

    // create the coordinates
       
    // get the local part of the coordinates vectors
    std::vector<scai::utilskernel::LArray<ValueType>* > localCoords(3);
    
    for(IndexType i=0; i<3; i++){
        localCoords[i] = &coords[i].getLocalValues();
    }
    
    IndexType localSize = dist->getLocalSize(); // the size of the local part
    
    IndexType planeSize= numPoints[1]*numPoints[2]; // a YxZ plane
    
    // find which should be the first local coordinate in this processor
    IndexType startingIndex = dist->local2global(0);
    
    IndexType indX = (IndexType) (startingIndex/planeSize) ;
    IndexType indY = (IndexType) ((startingIndex%planeSize)/numPoints[2]);
    IndexType indZ = (IndexType) ((startingIndex%planeSize) % numPoints[2]);
    //PRINT( *comm<< ": " << indX << " ,"<< indY << ", "<< indZ << ", startingIndex= "<< startingIndex);
    
    for(IndexType i=0; i<localSize; i++){
        SCAI_REGION("MeshIO.createStructured3DMesh_dist.setCoordinates");
        
        (*localCoords[0])[i] = indX*offset[0];
        (*localCoords[1])[i] = indY*offset[1];
        (*localCoords[2])[i] = indZ*offset[2];
        //PRINT( *comm << ": "<< (*localCoords[0])[i] << "_ "<< (*localCoords[1])[i] << "_ "<< (*localCoords[2])[i]);
        
        ++indZ;

        if(indZ >= numPoints[2]){   // if z coord reaches maximum, set it to 0 and increase y
            indZ = 0;
            ++indY;
        }
        if(indY >= numPoints[1]){   // if y coord reaches maximum, set it to 0 and increase x
            indY = 0;
            ++indX;
        }

        if(indX >= numPoints[0]){   // reached end of grid
            assert(i == localSize - 1);
        }
    }
    // finish setting the coordinates
    
    
    // start making the local part of the adjacency matrix
    scai::lama::CSRStorage<ValueType> localMatrix;
    localMatrix.allocate( adjM.getLocalNumRows() , adjM.getLocalNumColumns() );
    
    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<ValueType> csrValues;
    // ja and values have size= edges of the graph
    // for a 3D structured grid with dimensions AxBxC the number of edges is 3ABC-AB-AC-BC
    IndexType numEdges= 3*numPoints[0]*numPoints[1]*numPoints[2] - numPoints[0]*numPoints[1]\
                                -numPoints[0]*numPoints[2] - numPoints[1]*numPoints[2];
                                
    {
        SCAI_REGION("MeshIO.createStructured3DMesh_dist.setCSRSparseMatrix");
        IndexType N= numPoints[0]* numPoints[1]* numPoints[2];
        
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, adjM.getLocalNumRows() +1 );
        // we do not know the sizes of ja and values. 6*numOfLocalNodes is safe upper bound for a structured 3D mesh
        // after all the values are written the arrays get resized
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA , 6*adjM.getLocalNumRows() );
        hmemo::WriteOnlyAccess<ValueType> values( csrValues, 6*adjM.getLocalNumRows() );
        ia[0] = 0;
        IndexType nnzCounter = 0; // count non-zero elements
         
        for(IndexType i=0; i<localSize; i++){   // for all local nodes
            IndexType globalInd = dist->local2global(i);    // get the corresponding global index
            // the global id of the neighbouring nodes
            IndexType ngb_node = globalInd;
            IndexType numRowElems= 0;     // the number of neighbours for each node. Can be less than 6.
            // the position of this node in 3D
            std::tuple<IndexType, IndexType, IndexType> thisPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( globalInd, numPoints);
            
            // for all 6 possible neighbours
            for(IndexType m=0; m<6; m++){
                switch(m){
                    case 0: ngb_node= globalInd+1; break;
                    case 1: ngb_node= globalInd-1; break;
                    case 2: ngb_node = globalInd +numPoints[2]; break;
                    case 3: ngb_node = globalInd -numPoints[2]; break;
                    case 4: ngb_node = globalInd +numPoints[2]*numPoints[1]; break;
                    case 5: ngb_node = globalInd -numPoints[2]*numPoints[1]; break;
                }
       
                if(ngb_node>=0 && ngb_node<N){
                    // get the position in the 3D of the neighbouring node
                	std::tuple<IndexType, IndexType, IndexType> ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( ngb_node, numPoints);
                	ValueType distanceSquared = dist3DSquared( thisPoint, ngbPoint);
                	assert(distanceSquared <= numPoints[0]*numPoints[0]+numPoints[1]*numPoints[1]+numPoints[2]*numPoints[2]);
                    
                    if(distanceSquared <= 1)
                    {
                        ja[nnzCounter]= ngb_node;       
                        values[nnzCounter] = 1;         // unweighted edges
                        ++nnzCounter;
                        ++numRowElems;
                    }   
                }
            }
            assert(numRowElems >= 3);
            
            ia[i+1] = ia[i] + numRowElems;
        }
        SCAI_ASSERT_EQUAL_ERROR(numEdges*2 , comm->sum(nnzCounter));
        ja.resize(nnzCounter);
        values.resize(nnzCounter);
    } //read/write block
    
    {
        SCAI_REGION( "MeshIO.createStructured3DMesh_dist.swap_assign" )
        localMatrix.swap( csrIA, csrJA, csrValues );
        adjM.assign(localMatrix , adjM.getRowDistributionPtr() , adjM.getColDistributionPtr() );
    }
}


//-------------------------------------------------------------------------------------------------
// coords.size()= 3 , coords[i].size()= N
// here, N= numPoints[0]*numPoints[1]*numPoints[2]

template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createRandomStructured3DMesh_dist(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, std::vector<ValueType> maxCoord, std::vector<IndexType> numPoints) {
    SCAI_REGION( "MeshIO.createRandomStructured3DMesh_dist" )
    
    if (coords.size() != 3) {
        throw std::runtime_error("Needs three coordinate vectors, one for each dimension");
    }
    
    if (numPoints.size() != 3) {
        throw std::runtime_error("Needs three point counts, one for each dimension");
    }
        
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = adjM.getRowDistributionPtr();
    
    if( !dist->isEqual( coords[0].getDistribution() ) ){
        std::cout<< __FILE__<< "  "<< __LINE__<< ", matrix dist: " << *dist<< " and coordinates dist: "<< coords[0].getDistribution() << std::endl;
        throw std::runtime_error( "Distributions: should (?) be equal.");
    }
    
    //for simplicity
    IndexType numX = numPoints[0] , numY= numPoints[1], numZ= numPoints[2];
    IndexType N = numX* numY* numZ;
    
    std::vector<ValueType> offset={maxCoord[0]/numX, maxCoord[1]/numY, maxCoord[2]/numZ};

    // create the coordinates
       
    // get the local part of the coordinates vectors
    std::vector<scai::utilskernel::LArray<ValueType>* > localCoords(3);
    
    for(IndexType i=0; i<3; i++){
        localCoords[i] = &coords[i].getLocalValues();
    }
    
    IndexType localSize = dist->getLocalSize(); // the size of the local part
    
    IndexType planeSize= numY*numZ;             // a YxZ plane
    
    // find which should be the first local coordinate in this processor
    IndexType startingIndex = dist->local2global(0);
    
    IndexType indX = (IndexType) (startingIndex/planeSize) ;
    IndexType indY = (IndexType) ((startingIndex%planeSize)/numZ);
    IndexType indZ = (IndexType) ((startingIndex%planeSize) % numZ);
    //PRINT( *comm<< ": " << numX << ", "<< numY << ", "<< numZ << ", startingIndex= "<< startingIndex);
    
    for(IndexType i=0; i<localSize; i++){
        SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.setCoordinates");
        
        (*localCoords[0])[i] = indX*offset[0];
        (*localCoords[1])[i] = indY*offset[1];
        (*localCoords[2])[i] = indZ*offset[2];
        //PRINT( *comm << ": "<< (*localCoords[0])[i] << "_ "<< (*localCoords[1])[i] << "_ "<< (*localCoords[2])[i]);
        
        ++indZ;

        if(indZ >= numZ){   // if z coord reaches maximum, set it to 0 and increase y
            indZ = 0;
            ++indY;
        }
        if(indY >= numY){   // if y coord reaches maximum, set it to 0 and increase x
            indY = 0;
            ++indX;
        }
        if(indX >= numX){   // reached and of grid
            assert(i == localSize - 1);
        }
    }
    // finish setting the coordinates
    // this part is the same as the structured mesh
    
    /*
     * start making the local part of the adjacency matrix
    */
    
    // first find the indices of possible neighbours
    
    // the diameter of the box in which we gonna pick indices.
    // we set it as: 1 + 2% of the minimum side length. this is >2 for sides >50
    IndexType boxRadius = 1 + (IndexType) (0.02 * (ValueType) *std::min_element(std::begin(numPoints), std::end(numPoints)));
    
    // radius^3 is the number of possible neighbours, here keep their indices as an array
    std::vector<IndexType> neighbourGlobalIndices;

    PRINT(*comm << " , boxRadius= "<< boxRadius<< " , and num of possible neighbours ="<< pow(2*boxRadius+1, 3) );
    
    for(IndexType x=-boxRadius; x<=boxRadius; x++){
        for(IndexType y=-boxRadius; y<=boxRadius; y++){
                for(IndexType z=-boxRadius; z<=boxRadius; z++){
// changed the code above so we find neighbours only with greater indices    
//    for(IndexType x=0; x<=boxRadius; x++){
//        for(IndexType y=0; y<=boxRadius; y++){
//            for(IndexType z=0; z<=boxRadius; z++){
                // calculate the global index of a possible neighbour and insert it to the vector
                IndexType globalNeighbourIndex= x*planeSize + y*numZ + z;
                neighbourGlobalIndices.push_back( globalNeighbourIndex );
            }
        }
    }
    PRINT(*comm<<", num of neighbours inserted= "<< neighbourGlobalIndices.size() );

    // an upper bound to how many neighbours a vertex can have, 
    // at most as many neighbours as we have
    IndexType ngbUpperBound = std::min(12, (IndexType) neighbourGlobalIndices.size() ); // I do not know, just trying 12
    // TODO:  maybe treat nodes in the faces differently
    IndexType ngbLowerBound = 3;
    srand(time(NULL));
                                
    /*  We must the adjacency matrix symmetric and also we do not know how many edges the graph will
     *  have eventually and we cannot use ia, ja and values arrays to build the CSRSparseMatrix.
     *  We will store all edges in a vector and build the matrix afterwards. 
     *  Also we separate those edges with non-local neighbours so we can set the non-local edge
     *  later.
     */
    
                
    // a set for every local node. localNgbs[i] keeps the neighbours of node i that are also local. We use set in order to prevent the insertion of an index multiple times
    //std::vector< std::list<IndexType> > localNgbs(localSize);
    std::vector< std::set<IndexType> > localNgbs(localSize);
        
    // two vector that keep the edges that nees to be communicated with their global indices
    std::vector<IndexType> localNodeInd;
    std::vector<IndexType> nonLocalNodeInd;
    
        
    for(IndexType i=0; i<localSize; i++){                   // for all local nodes
        SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.findNgbrs");
        IndexType thisGlobalInd = dist->local2global(i);    // get the corresponding global index
        IndexType ngbGlobalInd;                             // the global id of the neighbouring nodes
//PRINT(*comm << ": i= " << i<< ", "<< thisGlobalInd);
        // the position of this node in 3D
        std::tuple<IndexType, IndexType, IndexType>  thisPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( thisGlobalInd, numPoints);
            
        // if point is on the faces it will have only 3 edges
        // TODO: not correct, ridge nodes must have >4 edges and face nodes have >5
        if(std::get<0>(thisPoint)== 0 or std::get<1>(thisPoint)== 0 or std::get<2>(thisPoint)== 0){
            ngbUpperBound =3;
        }
        if(std::get<0>(thisPoint)== numX-1 or std::get<1>(thisPoint)== numY-1 or std::get<2>(thisPoint)== numZ-1){
            ngbUpperBound =3;
        }
        
        // get a random number of neighbours between 3 and ngbUpperBound
        IndexType numOfNeighbours;
        if(ngbUpperBound == ngbLowerBound){         //for nodes on the faces
            numOfNeighbours = ngbUpperBound;
        }else{
            numOfNeighbours = ((int) rand()%(ngbUpperBound- ngbLowerBound) + ngbLowerBound);
        }
        assert( numOfNeighbours < neighbourGlobalIndices.size() );
        
        // find all the global indices of the neighbours
        std::set<IndexType> ngbGloblaIndSet;
        
        for(IndexType j=0; j<numOfNeighbours; j++){
            SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.findNgbrs.findRelativeIndices");
            
            IndexType relativeIndex=0;
            std::tuple<IndexType, IndexType, IndexType>  ngbPoint;
            std::pair<std::set<int>::iterator,bool> setInsertion;
            setInsertion.second= false;
            
            do{
                //int randInd = (int) (rand()%(neighbourGlobalIndices.size()-1) +1 ) ;
                // pick a random index (of those allowed) to greate edge
                unsigned long randInd= (unsigned long) rand()%neighbourGlobalIndices.size() ;

                // not 0 to avoid thisGlobalInd == ngbGlobalInd
                while( relativeIndex==0){
                    randInd= (unsigned long) rand()%neighbourGlobalIndices.size();
                    assert(randInd < neighbourGlobalIndices.size());
                    relativeIndex = neighbourGlobalIndices[ randInd ];
                }
                // neighbour's relative index, relative to thisIndex (this + or - a value)
                relativeIndex = neighbourGlobalIndices[ randInd ];
                
                // the global neighbour's index. At this point it can be <0 or >N, ensure this not happens
                ngbGlobalInd = thisGlobalInd + relativeIndex;
               
                // find a suitable ngbGlobalInd: not same as this, not negative, not >N and close enough
                while( /*(ngbGlobalInd==thisGlobalInd) or*/  (ngbGlobalInd<0) or (ngbGlobalInd>= N) ){
                    // pick new index at random
                    randInd= (unsigned long) rand()%neighbourGlobalIndices.size();
                    relativeIndex = neighbourGlobalIndices[ randInd ];
                    while( relativeIndex==0){
                        randInd= (unsigned long) rand()%neighbourGlobalIndices.size();
                        assert(randInd < neighbourGlobalIndices.size());
                        relativeIndex = neighbourGlobalIndices[ randInd ];
                    }
                    
                    ngbGlobalInd = thisGlobalInd + relativeIndex;
                    
                }
                assert( ngbGlobalInd < N);
                assert( ngbGlobalInd >= 0);
                
                // so here, the neighbour's global index should be: >0 && <N
                ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( ngbGlobalInd, numPoints);
                // if the two points are too far, recalculate neighbour index
                if( dist3DSquared(thisPoint, ngbPoint)> 3*boxRadius*boxRadius ){
                    continue;
                }
                
                // at this point ngbGlobalInd should be valid: >0 && <N && close enough
                // but we may have already inserted
                
                // insert the index to the set. if it already exists then setInsertion.second = false
                setInsertion = ngbGloblaIndSet.insert(ngbGlobalInd);
          
            }while(setInsertion.second==false);
            //finally, we inserted a valid (>0 && <N && close enough) neighbour

        } // for(IndexType j=0; j<numOfNeighbours; j++)
        

        //
        // from here, ngbGloblaIndSet has all the valid global neighbours
        //
        SCAI_ASSERT_EQUAL_ERROR(ngbGloblaIndSet.size(), numOfNeighbours);

        // for all neighbours inserted
        for(typename std::set<IndexType>::iterator it= ngbGloblaIndSet.begin(); it!= ngbGloblaIndSet.end(); ++it ){
            SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.findNgbrs.separateNonLocal");
            ngbGlobalInd = *it;
            localNgbs[i].insert( ngbGlobalInd );       // store the local edge (i, ngbGlobalInd)

            // for the other/symmetric edge we must check if ngbGlobalInd is local or not
            if( dist->isLocal(ngbGlobalInd) ){          // if the neighbour is local
                assert( dist->global2local(ngbGlobalInd) < localSize );
                localNgbs[ dist->global2local(ngbGlobalInd) ].insert( dist->local2global(i) );
            } else{                                     // the neighbour is not local
                localNodeInd.push_back( thisGlobalInd );
                nonLocalNodeInd.push_back( ngbGlobalInd );
            }
        }
    } // for(IndexType i=0; i<localSize; i++)
    PRINT(*comm << ",  num of non-local ngbs= " << nonLocalNodeInd.size() );   
    

    /* 
     * communicate the non-local edges that we must set.
     * idea:
     * we have an edge (u,v) where u is local but v is non-local. In the local part of
     * the adjacency matrix we set adjM(u,v)=1 but must also set adjM(v,u)=1 but v
     * is not local. Store in two vectors the indices: in localNodeInd insert u and on
     * nonLocalNodeInd insert v. Now the edge (v,u) is (localNodeInd[i], nonLocalNodeInd[i]).
     * Make two HArrays of the vectors and pass them around. When you receive the corresponding
     * arrays on will not have local indices (is the localNodeInd from another PE) but there
     * might be local indices in the other array, the nonLocalNodeInd from the other PE.
     * Check if there is an index that is local for you. On array arrRcvPossiblyLocalInd
     * there might be a local index. Suppose that arrRcvPossiblyLocalInd[i] is local, that
     * means that you must add the edge (arrRcvPossiblyLocalInd[i], arrRcvNonLocalInd[i]).
     * After you checked the whole array pass on the array you just received, not your own
     * data again, se very pair of arrays localNodeInd and nonLocalNodeInd will be seen
     */
    // build 2 HArrays from vectors to send data to other PEs
    // first round send your own data, then send what you received
    /*
    SCAI_ASSERT_EQUAL_ERROR(localNodeInd.size() , nonLocalNodeInd.size() );
    scai::hmemo::HArrayRef<IndexType> arrSendLocalNodeInd( localNodeInd );
    scai::hmemo::HArrayRef<IndexType> arrSendNonLocalInd( nonLocalNodeInd );
    */
    
    /* The 2 arrays idea does not seem to work. We will use 1 array and edges wil be
     * in indices [0]-[1] , [2]-[3] .... [i]-[i+1]
     */
    
    SCAI_ASSERT( localNodeInd.size() == nonLocalNodeInd.size(), __FILE__<<", "<< __LINE__<< ": sizes should match" );
    
    // [2*i] is the local endpoint, [2*i+1] is the non-local endpoint
    scai::hmemo::HArray<IndexType> sendPart(2* localNodeInd.size());
    scai::hmemo::WriteAccess<IndexType> sendPartWrite( sendPart );
    for(IndexType i=0; i< localNodeInd.size(); i++){
        sendPartWrite[2*i]= localNodeInd[i];
        sendPartWrite[2*i+1]= nonLocalNodeInd[i];
    }
    sendPartWrite.release();
    //PRINT(*comm << ">> " << sendPart.size() );

    // so on the data this PE receives edge [2*i] is local on the sender and [2*i+1] non-local on the sender
    // so now [2*i+1] may be local to this PE
    scai::hmemo::HArray<IndexType> recvPart;
    
    // this array contains local indices of other PEs.
    //scai::hmemo::HArray<IndexType> arrRcvNonLocalInd;
    // this array might contains local indices of this PE.
    // they must be the same size

    IndexType numPEs = comm->getSize();
    
    // the size of what you gonna send nad what you will receive
    scai::hmemo::HArray<IndexType> sendSize( numPEs,  0 );
    scai::hmemo::HArray<IndexType> recvSize( numPEs,  0  );    
    
    SCAI_ASSERT_EQUAL_ERROR( recvSize.size(), comm->getSize() );
    
    for(IndexType round=0; round<numPEs; round++){
        SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.commSize");
        {
            scai::hmemo::WriteAccess<IndexType> sendSizeWrite( sendSize );
            sendSizeWrite[ comm->getRank()] = sendPart.size();
        }
        //PRINT(*comm <<" _"<< recvSize.size() << " ## "<< sendSize.size() );        
        // communicate first the size of waht you will send and what you will receive
        comm->shiftArray( recvSize, sendSize, 1);
        sendSize.swap( recvSize );
    }
    
    scai::hmemo::ReadAccess<IndexType> readRcvSize( recvSize);
    
    for(IndexType ii=0; ii<recvSize.size(); ii++){
        PRINT(*comm<<"| "<< readRcvSize[ii]);
    }

    
    
    for(IndexType round=1; round<comm->getSize(); round++){
        SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.commNonLocalEdges");
        // to find from what PE you receive
        IndexType indexToRcv = (comm->getRank()-round+numPEs)%numPEs;
        IndexType sizeToRcv = readRcvSize[ indexToRcv ];
        //PRINT("round: "<< round <<" , " << *comm << ": receive from PE: "<< indexToRcv << ", size: "<< sizeToRcv);        
        recvPart.resize( sizeToRcv );

        scai::hmemo::ReadAccess<IndexType> sendSizeRead (sendSize);
        //PRINT( *comm << " , sendPart.size= " << sendPart.size() );
        // send your local part, receive other PE's part
        comm->shiftArray( recvPart, sendPart, 1);
        

        // check if recvPart[2*i+1] is local, if it is must add edge [2*i+1]-[2*i]
        // the symmetric edge [2*i]-[2*i+1] is already been set to the PE that send the part
        // Note that send and recv parts should contain global indices
        for(IndexType i=0; i<recvPart.size(); i=i+2){        
            scai::hmemo::ReadAccess<IndexType> recvPartWrite( recvPart );
            if( dist->isLocal(recvPartWrite[i+1]) ){                       // must add edge
                IndexType localIndex=  recvPartWrite[i+1];
                IndexType nonLocalIndex= recvPartWrite[i];     
                // 0< localIndex< globalN but localNgbs.size()= localN, so must convert it to local
                SCAI_ASSERT( dist->global2local(localIndex) < localNgbs.size(),"too large index: "<< localIndex <<" while size= "<< localNgbs.size() )
                localNgbs[dist->global2local(localIndex) ].insert( nonLocalIndex );      // indices are global
            }
            // if not local do not do anything
        }
        
        // the data you received must be passed on, not your own again
        sendPart.resize(recvPart.size());
        sendPart.swap(recvPart);
    }
    
    
    /* 
     * after gathering all non local indices, set the CSR sparse matrix
     * all the edges to be set should be in localNgbs:
     * localNgbs is a vector with sets, localNgbs.size()= localN and each set localNgbs[i]
     * has stored the neighbours of node i. Care to change between global and local indices 
     * when neccessary.
     */
    
    
    scai::lama::CSRStorage<ValueType> localMatrix;
    localMatrix.allocate( adjM.getLocalNumRows() , adjM.getLocalNumColumns() );
    
    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<ValueType> csrValues;
    // ja.size() = values.size() = number of edges of the graph
    
    IndexType nnzCounter = 0; // count non-zero elements
    
    {
        SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.setCSRSparseMatrix");
        IndexType globalN= numX* numY* numZ;
        
        // Summing the size of all sets. This is the number of all edges.
        IndexType nnzValues=0;
        for(IndexType i=0; i<localNgbs.size(); i++){
            nnzValues += localNgbs[i].size();
        }
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, adjM.getLocalNumRows() +1 );
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA , nnzValues);
        hmemo::WriteOnlyAccess<ValueType> values( csrValues, nnzValues );
        ia[0] = 0;
         
        /*TODO:
         * if the 2  assertions never fail then we can save some time
         * since resizing is never done and there is no need to count the non-zero values
         * and the numRowElems
         */
        
        
        for(IndexType i=0; i<localSize; i++){               // for all local nodes
            // for all the neighbours of node i
            IndexType numRowElems= 0;   // should be == localNgbs[i].size()
            for(typename std::set<IndexType>::iterator it= localNgbs[i].begin(); it!=localNgbs[i].end(); ++it){
                IndexType ngbGlobalInd = *it;       // the global index og the neighbour
                ja[nnzCounter] = ngbGlobalInd;
                values[nnzCounter] = 1;
                SCAI_ASSERT( nnzCounter < nnzValues, __FILE__<<" ,"<<__LINE__<< ": nnzValues not calculated properly")
                ++nnzCounter;
                ++numRowElems;
            }
            ia[i+1] = ia[i] + numRowElems;
            //PRINT(numRowElems << " should be == "<< localNgbs[i].size() );
            SCAI_ASSERT(numRowElems == localNgbs[i].size(),  __FILE__<<" ,"<<__LINE__<<"something is wrong");
            ia[i+1] = ia[i] +static_cast<IndexType>(numRowElems);
        } //for(IndexType i=0; i<localSize; i++)
        ja.resize(nnzCounter);
        values.resize(nnzCounter);
        //PRINT("nnz afterwards= " << nnzCounter << " should be == "<< nnzValues);
        SCAI_ASSERT_EQUAL_ERROR( nnzCounter, nnzValues);
    } //read/write block
    
    
    {
        SCAI_REGION( "MeshIO.createRandomStructured3DMesh_dist.swap_assign" )
        localMatrix.swap( csrIA, csrJA, csrValues );
        adjM.assign(localMatrix , adjM.getRowDistributionPtr() , adjM.getColDistributionPtr() );
    }
    
    //SCAI_REGION_END("MeshIO.createRandomStructured3DMesh_dist.setAdjacencyMatrix");
}

//-------------------------------------------------------------------------------------------------
/*Given the adjacency matrix it writes it in the file "filename" using the METIS format. In the
 * METIS format the first line has two numbers, first is the number on vertices and the second
 * is the number of edges. Then, row i has numbers e1, e2, e3, ... notating the edges:
 * (i, e1), (i, e2), (i, e3), ....
 *  
 */
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::writeInFileMetisFormat (const CSRSparseMatrix<ValueType> &adjM, const std::string filename){
    SCAI_REGION( "MeshIO.writeInFileMetisFormat" )
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    //PRINT(*comm << " In writeInFileMetisFormat");
    
    std::ofstream f;
    std::string oldFile = filename + "OLD";
    f.open(oldFile);
    IndexType cols= adjM.getNumColumns() , rows= adjM.getNumRows();
    IndexType i, j;

    SCAI_REGION_START( "MeshIO.writeInFileMetisFormat.newVersion" )
    // new version
    std::ofstream fNew;
    std::string newFile = filename;// + "NEW";
    fNew.open(newFile);
    
    //assert( true == adjM.checkSymmetry() ); // this can be expensive
    assert(((int) adjM.getNumValues())%2==0); // even number of edges
    assert(cols==rows);
    
    // first line is number of nodes and edges 
    fNew << cols <<" "<< adjM.getNumValues()/2 << std::endl;
    //std::cout << cols <<" "<< adjM.getNumValues()/2 << std::endl;
    
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
    //const scai::hmemo::ReadAccess<IndexType> partAccess(localPart);
        
    for(IndexType i=0; i< ia.size(); i++){        // for all local nodes
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){             // for all the edges of a node
            SCAI_REGION("MeshIO.writeInFileMetisFormat.newVersion.writeInFile");
            fNew << ja[j]+1 << " ";
    	}
    	fNew << std::endl;
    }
    fNew.close();
    SCAI_REGION_END( "MeshIO.writeInFileMetisFormat.newVersion" )
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::writeInFileMetisFormat_dist (const CSRSparseMatrix<ValueType> &adjM, const std::string filename){
    SCAI_REGION("MeshIO.writeInFileMetisFormat_dist")
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    std::string fileTo = filename + std::to_string(comm->getRank());
    std::ofstream f(fileTo);
    if(f.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
    
    // notice that numValues is twice the number of edges of the graph 
    assert(((int) adjM.getNumValues())%2 == 0); // even number of edges
    
    IndexType localNumNodes= adjM.getLocalNumRows();
    f<< localNumNodes <<" "<< adjM.getLocalNumValues()/2 << std::endl; // first line is number of nodes and edges 
    
    // get local part
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
        
    for(IndexType i=0; i< ia.size()-1; i++){                  // for all local nodes
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){             // for all the edges of a node
            f << ja[j]+1 << " ";                            
    	}
    	f << std::endl;
    }
    f.close();
}
//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and their dimension, writes them in file "filename".
 */
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::writeInFileCoords (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const std::string filename){
    SCAI_REGION( "MeshIO.writeInFileCoords" )
    
    std::ofstream f(filename);
    if(f.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");

    IndexType i, j;
    IndexType dimension= coords.size();

    assert(coords.size() == dimension );
    assert(coords[0].size() == numPoints);
    for(i=0; i<numPoints; i++){
        for(j=0; j<dimension; j++)
            f<< coords[j].getValue(i).Scalar::getValue<ValueType>() << " ";
        f<< std::endl;
    }
    
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and transforms 
 * it to the adjacency matrix as a CSRSparseMatrix.
 */

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix(const std::string filename) {
	SCAI_REGION("MeshIO.readFromFile2AdjMatrix");

	std::ifstream file(filename);

	if (file.fail()) {
		throw std::runtime_error("Reading graph from " + filename + " failed.");
	}

	IndexType globalN, globalM;

	file >> globalN >> globalM;

	const ValueType avgDegree = ValueType(2*globalM) / globalN;

	//get distribution and local range
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( globalN ));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll to begin of local range. Neighbors of node i are in line i+1
    std::string line;
    std::getline(file, line);
    for (IndexType i = 0; i < beginLocalRange; i++) {
    	std::getline(file, line);
    }

    std::vector<IndexType> ia(localN+1, 0);
    std::vector<IndexType> ja;

    //we don't know exactly how many edges we are going to have, but in a regular mesh the average degree times the local nodes is a good estimate.
    ja.reserve(localN*avgDegree*1.1);

    //now read in local edges
    for (IndexType i = 0; i < localN; i++) {
    	bool read = std::getline(file, line);
    	assert(read);//if we have read past the end of the file, the node count was incorrect
        std::stringstream ss( line );
        std::string item;
        std::vector<IndexType> neighbors;

        while (std::getline(ss, item, ' ')) {
        	IndexType neighbor = std::stoi(item)-1;//-1 because of METIS format
        	if (neighbor >= globalN || neighbor < 0) {
        		throw std::runtime_error("Found illegal neighbor " + std::to_string(neighbor) + " in line " + std::to_string(i+beginLocalRange));
        	}
        	//std::cout << "Converted " << item << " to " << neighbor << std::endl;
        	neighbors.push_back(neighbor);
        }

        //set Ia array
        ia[i+1] = ia[i] + neighbors.size();
        //copy neighbors to Ja array
        std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(ja));
    }

    //TODO: maybe check that file is not longer than expected

    file.close();

    scai::utilskernel::LArray<ValueType> values(ja.size(), 1);//unweighted edges
    assert(ja.size() == ia[localN]);
    assert(comm->sum(localN) == globalN);

    if (comm->sum(ja.size()) != 2*globalM) {
    	throw std::runtime_error("Expected " + std::to_string(2*globalM) + " edges, got " + std::to_string(comm->sum(ja.size())));
    }

    //assign matrix
    scai::lama::CSRStorage<ValueType> myStorage(localN, globalN, ja.size(), scai::utilskernel::LArray<IndexType>(ia.size(), ia.data()),
    		scai::utilskernel::LArray<IndexType>(ja.size(), ja.data()), values);

    return scai::lama::CSRSparseMatrix<ValueType>(myStorage, dist, noDist);
}


//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads these coordinates and returns a vector of DenseVectors, one for each dimension
 */
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> MeshIO<IndexType, ValueType>::fromFile2Coords( std::string filename, IndexType numberOfPoints, IndexType dimension){
    SCAI_REGION( "MeshIO.fromFile2Coords" )
    IndexType globalN= numberOfPoints;
    std::ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll forward to begin of local range
    std::string line;
    for (IndexType i = 0; i < beginLocalRange; i++) {
    	std::getline(file, line);
    }

    //create result vector
    std::vector<scai::utilskernel::LArray<ValueType> > coords(dimension);
    for (IndexType dim = 0; dim < dimension; dim++) {
    	coords[dim] = scai::utilskernel::LArray<ValueType>(localN);
    }

    //read local range
    for (IndexType i = 0; i < localN; i++) {
		bool read = std::getline(file, line);
		assert(read);//if we have read past the end of the file, the node count was incorrect
		std::stringstream ss( line );
		std::string item;

		IndexType dim = 0;
		while (std::getline(ss, item, ' ') && dim < dimension) {
			ValueType coord = std::stod(item);
			coords[dim][i] = coord;
			dim++;
		}
		if (dim < dimension) {
			throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimension) + " expected in line '" + line + "'");
		}
    }

    std::vector<DenseVector<ValueType> > result(dimension);

    for (IndexType i = 0; i < dimension; i++) {
    	result[i] = DenseVector<ValueType>(coords[i], dist);
    }

    return result;
}

//-------------------------------------------------------------------------------------------------
/* Creates random points in the cube [0,maxCoord] in the given dimensions.
 */
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> MeshIO<IndexType, ValueType>::randomPoints(int numberOfPoints, int dimensions, ValueType maxCoord){
    SCAI_REGION( "MeshIO.randomPoints" )
    int n = numberOfPoints;
    int i, j;
    std::vector<DenseVector<ValueType>> ret(dimensions);
    for (i=0; i<dimensions; i++)
        ret[i] = DenseVector<ValueType>(numberOfPoints, 0);
    
    srand(time(NULL));
    ValueType r;
    for(i=0; i<n; i++){
        for(j=0; j<dimensions; j++){
            r= ((ValueType) rand()/RAND_MAX) * maxCoord;
            ret[j].setValue(i, r);
        }
    }
    return ret;
}

//-------------------------------------------------------------------------------------------------
/* Calculates the distance in 3D.
*/
template<typename IndexType, typename ValueType>
Scalar MeshIO<IndexType, ValueType>::dist3D(DenseVector<ValueType> p1, DenseVector<ValueType> p2){
  SCAI_REGION( "MeshIO.dist3D" )
  Scalar res0, res1, res2, res;
  res0= p1.getValue(0)-p2.getValue(0);
  res0= res0*res0;
  res1= p1.getValue(1)-p2.getValue(1);
  res1= res1*res1;
  res2= p1.getValue(2)-p2.getValue(2);
  res2= res2*res2;
  res = res0+ res1+ res2;
  return scai::common::Math::sqrt( res.getValue<ScalarRepType>() );
}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
ValueType MeshIO<IndexType, ValueType>::dist3DSquared(std::tuple<IndexType, IndexType, IndexType> p1, std::tuple<IndexType, IndexType, IndexType> p2){
  SCAI_REGION( "MeshIO.dist3DSquared" )
  ValueType distX, distY, distZ;

  distX = std::get<0>(p1)-std::get<0>(p2);
  distY = std::get<1>(p1)-std::get<1>(p2);
  distZ = std::get<2>(p1)-std::get<2>(p2);

  ValueType distanceSquared = distX*distX+distY*distY+distZ*distZ;
  return distanceSquared;
}
//-------------------------------------------------------------------------------------------------
// Given a (global) index and the size for each dimension (numPpoints.size()=3) calculates the position
// of the index in 3D. The return value is not the coordinates of the point!
template<typename IndexType, typename ValueType>
std::tuple<IndexType, IndexType, IndexType> MeshIO<IndexType, ValueType>::index2_3DPoint(IndexType index,  std::vector<IndexType> numPoints){
    // a YxZ plane
    IndexType planeSize= numPoints[1]*numPoints[2];
    IndexType xIndex = index/planeSize;
    IndexType yIndex = (index % planeSize) / numPoints[2];
    IndexType zIndex = (index % planeSize) % numPoints[2];
    SCAI_ASSERT(xIndex >= 0, xIndex);
    SCAI_ASSERT(yIndex >= 0, yIndex);
    SCAI_ASSERT(zIndex >= 0, zIndex);
    assert(xIndex < numPoints[0]);
    assert(yIndex < numPoints[1]);
    assert(zIndex < numPoints[2]);
    return std::make_tuple(xIndex, yIndex, zIndex);
}

//-------------------------------------------------------------------------------------------------
template void MeshIO<int, double>::createRandom3DMesh(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coords, const int numberOfPoints,const double maxCoord);
template void MeshIO<int, double>::createStructured3DMesh(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coords, std::vector<double> maxCoord, std::vector<int> numPoints);
template void MeshIO<int, double>::createStructured3DMesh_dist(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coords, std::vector<double> maxCoord, std::vector<int> numPoints);
template void MeshIO<int, double>::createRandomStructured3DMesh_dist(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coords, std::vector<double> maxCoord, std::vector<int> numPoints);
template std::vector<DenseVector<double>> MeshIO<int, double>::randomPoints(int numberOfPoints, int dimensions, double maxCoord);
template void MeshIO<int, double>::writeInFileMetisFormat (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void MeshIO<int, double>::writeInFileMetisFormat_dist (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void MeshIO<int, double>::writeInFileCoords (const std::vector<DenseVector<double>> &coords, int numPoints, const std::string filename);
template CSRSparseMatrix<double> MeshIO<int, double>::readFromFile2AdjMatrix(const std::string filename);
template std::vector<DenseVector<double>>  MeshIO<int, double>::fromFile2Coords( std::string filename, int numberOfCoords, int dimension);
template Scalar MeshIO<int, double>::dist3D(DenseVector<double> p1, DenseVector<double> p2);
template double MeshIO<int, double>::dist3DSquared(std::tuple<int, int, int> p1, std::tuple<int, int, int> p2);
template std::tuple<IndexType, IndexType, IndexType> MeshIO<int, double>::index2_3DPoint(int index,  std::vector<int> numPoints);
} //namespace ITI
