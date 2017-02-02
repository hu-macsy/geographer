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
                if(dist.getValue<ValueType>()<*liVal){
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
                        ja[nnzCounter]= ngb_node;       // -1 for the METIS format
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
    
    IndexType planeSize= numY*numZ; // a YxZ plane
    
    // find which should be the first local coordinate in this processor
    IndexType startingIndex = dist->local2global(0);
    
    IndexType indX = (IndexType) (startingIndex/planeSize) ;
    IndexType indY = (IndexType) ((startingIndex%planeSize)/numZ);
    IndexType indZ = (IndexType) ((startingIndex%planeSize) % numZ);
    //PRINT( *comm<< ": " << indX << " ,"<< indY << ", "<< indZ << ", startingIndex= "<< startingIndex);
    
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
            break;
        }
    }
    // finish setting the coordinates
    // this part is the same as the structured mesh
    
    
    //SCAI_REGION_START("MeshIO.createRandomStructured3DMesh_dist.setAdjacencyMatrix");
    
    /*
     * start making the local part of the adjacency matrix
    */
    
    // first find the indices of possible neighbours
    
    // the diameter of the box in which we gonna pick indices.
    // we set it as: 1 + 5% of the minimum side length. this is >2 for sides > 20
    IndexType boxRadius = 1 + (IndexType) (0.05 * (ValueType) *std::min_element(std::begin(numPoints), std::end(numPoints)));
    
    // radius^3 is the number of possible neighbours, here keep their indices as an array
    std::vector<IndexType> neighbourGlobalIndices;

    //PRINT(*comm << " , boxRadius= "<< boxRadius<< " , and num of possible neighbours ="<< pow(2*boxRadius+1, 3) );
    PRINT(*comm << " , boxRadius= "<< boxRadius<< " , and num of possible neighbours ="<< pow(boxRadius+1, 3) );
    
    IndexType indexCnt= 0;              

//    for(IndexType x=-boxRadius; x<=boxRadius; x++){
//        for(IndexType y=-boxRadius; y<=boxRadius; y++){
//                for(IndexType z=-boxRadius; z<=boxRadius; z++){
// changed the code above so we find neighbours only with greater indices    
    for(IndexType x=0; x<=boxRadius; x++){
        for(IndexType y=0; y<=boxRadius; y++){
            for(IndexType z=0; z<=boxRadius; z++){
                // calculate the global index of a possible neighbour and insert it to the vector
                IndexType globalNeighbourIndex= x*planeSize + y*numZ + z;
                neighbourGlobalIndices.push_back( globalNeighbourIndex );
            }
        }
    }
    PRINT(*comm<<", num of neighbours inserted= "<< neighbourGlobalIndices.size() );
    for(IndexType i=0; i<neighbourGlobalIndices.size(); i++){
        std::cout<< neighbourGlobalIndices[i]<< "| ";
    }
    std::cout<< std::endl;
    
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
        IndexType thisGlobalInd = dist->local2global(i);    // get the corresponding global index
        // the global id of the neighbouring nodes, float because it might take negative value
        IndexType ngbGlobalInd;// = thisGlobalInd;    

        // the position of this node in 3D
        std::tuple<IndexType, IndexType, IndexType>  thisPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( thisGlobalInd, numPoints);
            
        // if point is on the faces it will have only 3 edges
        // TODO: not correct, rim nodes must have >4 edges and face nodes have >5
        if(std::get<0>(thisPoint)== 0 or std::get<1>(thisPoint)== 0 or std::get<2>(thisPoint)== 0){
            ngbUpperBound =3;
        }
        if(std::get<0>(thisPoint)== numX-1 or std::get<1>(thisPoint)== numY-1 or std::get<2>(thisPoint)== numZ-1){
            ngbUpperBound =3;
        }
        
        // get a random number of neighbours between 3 and ngbUpperBound
        IndexType numOfNeighbours;
        if(ngbUpperBound == ngbLowerBound){
            numOfNeighbours = ngbUpperBound;
        }else{
            numOfNeighbours = ((int) rand()%(ngbUpperBound- ngbLowerBound) + ngbLowerBound);
        }
        assert( numOfNeighbours < neighbourGlobalIndices.size() );
        
        // find all the global indices of the neighbours
        // TODO: check std::set, maybe faster for this application, no need for find
        std::set<IndexType> ngbGloblaIndSet;
        
        for(IndexType j=0; j<numOfNeighbours; j++){
            SCAI_REGION("MeshIO.createRandomStructured3DMesh_dist.findRelativeIndices");
            
            IndexType relativeIndex;
            std::tuple<IndexType, IndexType, IndexType>  ngbPoint;
            std::pair<std::set<int>::iterator,bool> setInsertion;
            
            do{
                int randInd = (int) (rand()%(neighbourGlobalIndices.size()-1) +1 ) ;
                assert(randInd != 0 );
                assert(randInd < neighbourGlobalIndices.size());
                relativeIndex = neighbourGlobalIndices[ randInd ];
                relativeIndex = rand()%2==0 ? relativeIndex : -1* relativeIndex;
                
                ngbGlobalInd = thisGlobalInd + relativeIndex;
                ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( ngbGlobalInd, numPoints);
               
                // find a suitable ngbGlobalInd: not negative, not >N and close enough
                while( /* (ngbGlobalInd==thisGlobalInd) or */ (ngbGlobalInd<0) or (ngbGlobalInd>= N) \
                        or (dist3DSquared(thisPoint, ngbPoint)> 3*boxRadius*boxRadius)){ 
                    randInd = (int) (rand()%(neighbourGlobalIndices.size()-1) +1 ) ;
                    relativeIndex = neighbourGlobalIndices[ randInd ];
                    relativeIndex = rand()%2==0 ? relativeIndex : -1* relativeIndex;
                    ngbGlobalInd = thisGlobalInd + relativeIndex;
                    ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( ngbGlobalInd, numPoints);
//PRINT( *comm << ": "<< ngbGlobalInd<<  " -- " << (dist3DSquared(thisPoint, ngbPoint)>3*boxRadius*boxRadius) );  
//PRINT( (ngbGlobalInd> N) << " <> " << (dist3DSquared(thisPoint, ngbPoint)>3*boxRadius*boxRadius) << "  === "<< ( (ngbGlobalInd> N) or (dist3DSquared(thisPoint, ngbPoint)>3*boxRadius*boxRadius) ) );
                }
                // so here, the neighbour's global index should be valid
                assert( ngbGlobalInd < N);
                assert( ngbGlobalInd >= 0);
                
                // insert the index to the set. if it already exists then setInsertion.second = false
                setInsertion = ngbGloblaIndSet.insert(ngbGlobalInd);
//PRINT( *comm << ", randInd= "<< randInd <<  " __ " << relativeIndex );                
            }while(setInsertion.second==false);
        
//PRINT(*comm <<"@@ inserted: "<< ngbGlobalInd );
            /*
            IndexType* ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( thisGlobalInd + relativeIndex, numPoints);
            std::pair<std::set<int>::iterator,bool> setInsertion;
            setInsertion.second = false; // force to enter loop
            
            //IndexType pointDistance = dist3DSquared( thisPoint, ngbPoint);
            while(setInsertion.second == false){
                relativeIndex = neighbourGlobalIndices[ (int) (rand()%(neighbourGlobalIndices.size()) +1 ) ];
                ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( thisGlobalInd + relativeIndex, numPoints);
                // pick a valid point to insert, that is close enough
                while( dist3DSquared( thisPoint, ngbPoint) > 3* boxRadius*boxRadius){
                    relativeIndex = neighbourGlobalIndices[ (int) (rand()%(neighbourGlobalIndices.size()) +1 ) ];
                    ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( thisGlobalInd + relativeIndex, numPoints);
                }
            // insert point if is not already in. If it exists then setInsertion.second= false
            setInsertion = ngbGloblaRelativeInd.insert(relativeIndex);
            }
            */
            
            /*
            // if this index is already added or if it is not correct (either out of bounds
            // or too far),find a new one.
            while( setInsertion.second==false or (dist3DSquared( thisPoint, ngbPoint) > 3* boxRadius*boxRadius) ){
//PRINT( "thisGlobalInd: "<< thisGlobalInd<< " , relativeIndex= "<< relativeIndex<< " , j = " << j<< " , numOfNeighbours= "<< numOfNeighbours-1 << " __ " <<  setInsertion.second << " ++ distSq= "<< dist3DSquared( thisPoint, ngbPoint) << " <> " << 3* boxRadius*boxRadius );
                relativeIndex = neighbourGlobalIndices[ (int) (rand()%(neighbourGlobalIndices.size()) +1 ) ];
                setInsertion = ngbGloblaRelativeInd.insert(relativeIndex);
                ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( thisGlobalInd + relativeIndex, numPoints);
            }
            //ngbGloblaRelativeInd.push_back( relativeIndex );
            */
        }
        // from here, ngbGloblaIndSet has all the valid global neighbours
        
        SCAI_ASSERT_EQUAL_ERROR(ngbGloblaIndSet.size(), numOfNeighbours);

        // for all neighbours
        for(typename std::set<IndexType>::iterator it= ngbGloblaIndSet.begin(); it!= ngbGloblaIndSet.end(); ++it ){

            ngbGlobalInd = *it;
            localNgbs[i].insert( ngbGlobalInd );       // store the local edge (i, ngbGlobalInd)

            // for the other/symmetric edge we must check if ngbGlobalInd is local or not
            
            if( dist->isLocal(ngbGlobalInd) ){          // if the neighbour is local
                assert( dist->global2local(ngbGlobalInd) < localSize );
                localNgbs[ dist->global2local(ngbGlobalInd) ].insert( dist->local2global(i) );
            } else{                                     // the neighbour is not local
//PRINT(*comm<< ": local2global= " << thisGlobalInd <<": edge with non local index "<< ngbGlobalInd );
                localNodeInd.push_back( thisGlobalInd );
                nonLocalNodeInd.push_back( ngbGlobalInd );
            }
        }
    }
PRINT(*comm << ",  num of non-local ngbs= " << nonLocalNodeInd.size() );   
    
    /* 
     * communicate the non-local edges that we must set 
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
    SCAI_ASSERT_EQUAL_ERROR(localNodeInd.size() , nonLocalNodeInd.size() );
    scai::hmemo::HArrayRef<IndexType> arrSendLocalNodeInd( localNodeInd );
    scai::hmemo::HArrayRef<IndexType> arrSendNonLocalInd( nonLocalNodeInd );
    
    // this array contains local indices of other PEs.
    scai::hmemo::HArray<IndexType> arrRcvNonLocalInd;
    // this array might contains local indices of this PE.
    // they must be the same size

    
    std::vector<IndexType> rcvLocalIndices;
    
    for(IndexType round=0; round<comm->getSize(); round++){
/*        
PRINT( *comm << " , sendLocal.size= " << arrSendLocalNodeInd.size() );
        // send your local indices, receive non-local
        comm->shiftArray( arrRcvNonLocalInd, arrSendLocalNodeInd, 1);
PRINT( *comm << ",  rcvNonLocal.size= "<< arrRcvNonLocalInd.size() );
*/

// two communication steps together do not seem to work!

        // send all non-local indices that there is an edge
        // receive idndices that might have local indices
            scai::hmemo::HArray<IndexType> arrRcvPossiblyLocalInd;

PRINT( *comm << " , sendNonLocal.size= "<< arrSendNonLocalInd.size() );
        comm->shiftArray( arrRcvPossiblyLocalInd, arrSendNonLocalInd, 1);
        // arrSendNonLocalInd.swap( arrRcvPossiblyLocalInd );
PRINT(*comm << ", rvcPossiblyLocal.size= " << arrRcvPossiblyLocalInd.size());
        //SCAI_ASSERT_EQUAL_ERROR( arrRcvNonLocalInd.size(), arrRcvPossiblyLocalInd.size() );

        scai::hmemo::ReadAccess<IndexType> rcvPossiblyLocalAccess( arrRcvPossiblyLocalInd );
        //scai::hmemo::ReadAccess<IndexType> arrRcvNonLocalAccess( arrRcvNonLocalInd );
        
        // check arrRcvPossiblyLocalInd to see if there are any local indices
        // if dist*isLocal( arrRcvPossiblyLocalInd[i] ) then add edge
        // (arrRcvPossiblyLocalInd[i] , arrRcvNonLocalInd[i])
        
        // gather locally all local indices received
        for(IndexType i=0; i<arrRcvPossiblyLocalInd.size(); i++){
            if( dist->isLocal (rcvPossiblyLocalAccess[i]) ){
                IndexType rcvLocalIndex = rcvPossiblyLocalAccess[i];
                //IndexType rcvEdgeEndPoint = arrRcvNonLocalAccess[i];
                //localNgbs[ rcvLocalIndex ].insert( rcvEdgeEndPoint );
                rcvLocalIndices.push_back(rcvLocalIndex);
            }
        }
        // write the data you received to be pass on, not your own
        //SCAI_ASSERT_EQUAL_ERROR(arrRcvNonLocalInd.size() , arrSendLocalNodeInd.size() );
        //SCAI_ASSERT_EQUAL_ERROR( arrRcvPossiblyLocalInd.size() , arrSendNonLocalInd.size() );
        
        //arrSendLocalNodeInd.swap( arrRcvNonLocalInd );
        arrSendNonLocalInd.swap( arrRcvPossiblyLocalInd );
        
    }
    
     // print
    for(IndexType i=0; i<localNgbs.size(); i++){
        std::cout<< *comm << "__ "<< i <<":"<< dist->local2global(i) <<std::endl;
        for(typename std::set<IndexType>::iterator it= localNgbs[i].begin(); it!=localNgbs[i].end(); ++it){
            std::cout<< *it <<" _ ";
        }
        std::cout<< std::endl;
    }
    
    /* 
     * after gathering all non local indices, set the CSR sparse matrix
     */
    
    /*
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
        
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, adjM.getLocalNumRows() +1 );
        // we do not know the sizes of ja and values. ngbUpperBound*numOfLocalNodes is safe upper bound for a structured 3D mesh
        // after all the values are written the arrays get resized
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA , ngbUpperBound* adjM.getLocalNumRows() );
        hmemo::WriteOnlyAccess<ValueType> values( csrValues, ngbUpperBound* adjM.getLocalNumRows() );
        ia[0] = 0;
         
        for(IndexType i=0; i<localSize; i++){               // for all local nodes
            IndexType thisGlobalInd = dist->local2global(i);    // get the corresponding global index
            // the global id of the neighbouring nodes, float because it might take negative value
            float ngb_node = thisGlobalInd;      
            int numRowElems= 0;     // the number of neighbours for each node.
            
            // the position of this node in 3D
            IndexType* thisPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( thisGlobalInd, numPoints);
            
            // get a random number of neighbours between 3 and ngbUpperBound
            IndexType numOfNeighbours = ((int) rand()%(ngbUpperBound- ngbLowerBound) + ngbLowerBound);
            
            // for all possible neighbours
            for(IndexType m=0; m<numOfNeighbours; m++){
                // calculate a random index from the vector of all possible indices
                ngb_node = thisGlobalInd + neighbourGlobalIndices[ ((int) rand()%(neighbourGlobalIndices.size()) ) ];
       
                if(ngb_node>=0 && ngb_node<globalN && ngb_node!= thisGlobalInd){
                    // get the position in the 3D of the neighbouring node
                    IndexType* ngbPoint = MeshIO<IndexType, ValueType>::index2_3DPoint( ngb_node, numPoints);
                    
                    if(dist3DSquared( thisPoint, ngbPoint) <= 1)
                    {
                        ja[nnzCounter]= ngb_node;       // -1 for the METIS format
                        values[nnzCounter] = 1;         // unweighted edges
                        ++nnzCounter;
                        ++numRowElems;
                    }   
                }
            }
            
            ia[i+1] = ia[i] +static_cast<IndexType>(numRowElems);
        } //for(IndexType i=0; i<localSize; i++)
        ja.resize(nnzCounter);
        values.resize(nnzCounter);
    } //read/write block
    PRINT("nnz afterwards= " << nnzCounter);
    
    {
        SCAI_REGION( "MeshIO.createRandomStructured3DMesh_dist.swap_assign" )
        localMatrix.swap( csrIA, csrJA, csrValues );
        adjM.assign(localMatrix , adjM.getRowDistributionPtr() , adjM.getColDistributionPtr() );
    }
    */
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
        
    for(IndexType i=0; i< ia.size(); i++){                  // for all local nodes
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

//
//TODO: throws assertion occasionally , more often when p= 3 or 5
//
template<typename IndexType, typename ValueType>
void   MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( lama::CSRSparseMatrix<ValueType> &matrix,  const std::string filename){
    SCAI_REGION( "MeshIO.readFromFile2AdjMatrix" )
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    //PRINT(*comm << " In readFromFile2AdjMatrix");
    
    IndexType N, numEdges;         //number of nodes and edges
    std::ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
   
    file >>N >> numEdges;   

    scai::lama::CSRStorage<double> localMatrix;
    // in a distributed version should be something like that
    // localMatrix.allocate( localSize, globalSize );
    // here is not distributed, local=global
    localMatrix.allocate( N, N );
    
    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<double> csrValues;  
    {
        //TODO: for a distributed version this must change as numNZ should be the number of
        //      the local nodes in the processor, not the global
        // number of Non Zero values. *2 because every edge is read twice.
        IndexType numNZ = numEdges*2;
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, N +1 );
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA, numNZ );
        hmemo::WriteOnlyAccess<double> values( csrValues, numNZ );

        ia[0] = 0;

        std::vector<IndexType> colIndexes;
        std::vector<int> colValues;
        
        IndexType rowCounter = 0; // count "local" rows
        IndexType nnzCounter = 0; // count "local" non-zero elements
        // read the first line and do nothing, contains the number of nodes and edges.
        std::string line;
        std::getline(file, line);
        
        //for every line, aka for all nodes
        for ( IndexType i=0; i<N; i++ ){
            SCAI_REGION( "MeshIO.readFromFile2AdjMatrix.setCSRSparseMatrix")
            std::getline(file, line);            
            std::vector< std::vector<int> > line_integers;
            std::istringstream iss( line );
            line_integers.push_back( std::vector<int>( std::istream_iterator<int>(iss), std::istream_iterator<int>() ) );
            
            //ia += the numbers of neighbours of i = line_integers.size()
            ia[rowCounter + 1] = ia[rowCounter] + static_cast<IndexType>( line_integers[0].size() );
            for(unsigned int j=0, len=line_integers[0].size(); j<len; j++){
                // -1 because of the METIS format
                ja[nnzCounter]= line_integers[0][j] -1 ;
                // all values are 1 for undirected, no-weigths graph    
                values[nnzCounter]= 1;
                ++nnzCounter;
            }            
            ++rowCounter;            
        }        
    }
    
    localMatrix.swap( csrIA, csrJA, csrValues );
    //matrix.assign( localMatrix, distribution, distribution ); // builds also halo
    matrix.assign(localMatrix);
    //PRINT(*comm << " Leaving readFromFile2AdjMatrix");
}


//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads that coordinates and returns
 * the coordinates in a 2 DenseVector, one for each dmension.
 * Every line of the file contais 3 ValueType numbers but the third number is ignored.
 */
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::fromFile2Coords_2D( const std::string filename, std::vector<DenseVector<ValueType>> &coords, IndexType numberOfPoints){
    SCAI_REGION( "MeshIO.fromFile2Coords_2D" )
    IndexType N= numberOfPoints;
    std::ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
    
    //the files, currently, contain 3 numbers in each line but z is always zero
    for(IndexType i=0; i<N; i++){
        ValueType x, y, z;
        file>> x >> y >> z;
        coords[0].setValue(i, x);
        coords[1].setValue(i, y);
    }
}
    
//-------------------------------------------------------------------------------------------------
/*File "filename" contains the 3D coordinates of a graph. The function reads that coordinates and returns
 * them in a vector<DenseVector> where point i=(x,y,z) is in coords[0][i], coords[1][i], coords[2][i].
 * Every line of the file contais 3 ValueType numbers.
 */
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::fromFile2Coords_3D( const std::string filename, std::vector<DenseVector<ValueType>> &coords, IndexType numberOfPoints){
    SCAI_REGION( "MeshIO.fromFile2Coords_3D" )
    IndexType N= numberOfPoints;
    std::ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
    
    //the files, currently, contain 3 numbers in each line
    for(IndexType i=0; i<N; i++){
        ValueType x, y, z;
        file>> x >> y >> z;
        coords[0].setValue(i, x);
        coords[1].setValue(i, y);
        coords[2].setValue(i, z);
    }
    
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
Scalar MeshIO<IndexType, ValueType>::dist3D(DenseVector<ValueType> p1, ValueType *p2){
  SCAI_REGION( "MeshIO.dist3D" )
  Scalar res0, res1, res2, res;
  res0= p1.getValue(0)-p2[0];
  res0= res0*res0;
  res1= p1.getValue(1)-p2[1];
  res1= res1*res1;
  res2= p1.getValue(2)-p2[2];
  res2= res2*res2;
  res = res0+ res1+ res2;
  return scai::common::Math::sqrt( res.getValue<ScalarRepType>() );
}
//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
ValueType MeshIO<IndexType, ValueType>::dist3DSquared(std::tuple<IndexType, IndexType, IndexType> p1, std::tuple<IndexType, IndexType, IndexType> p2){
  SCAI_REGION( "MeshIO.dist3D" )
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
    assert(xIndex >= 0);
    assert(yIndex >= 0);
    assert(zIndex >= 0);
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
template void MeshIO<int, double>::readFromFile2AdjMatrix( CSRSparseMatrix<double> &matrix, const std::string filename);
template void  MeshIO<int, double>::fromFile2Coords_2D( const std::string filename, std::vector<DenseVector<double>> &coords, int numberOfCoords);
template void MeshIO<int, double>::fromFile2Coords_3D( const std::string filename, std::vector<DenseVector<double>> &coords, int numberOfPoints);
template Scalar MeshIO<int, double>::dist3D(DenseVector<double> p1, DenseVector<double> p2);
template Scalar MeshIO<int, double>::dist3D(DenseVector<double> p1, double* p2);
template double MeshIO<int, double>::dist3DSquared(std::tuple<int, int, int> p1, std::tuple<int, int, int> p2);
template std::tuple<IndexType, IndexType, IndexType> MeshIO<int, double>::index2_3DPoint(int index,  std::vector<int> numPoints);
} //namespace ITI
