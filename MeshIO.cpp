/*
 * MeshIO.cpp
 *
 *  Created on: 22.11.2016
 *      Author: tzovas
 */

#include "MeshIO.h"
#include <chrono>

namespace ITI{

template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createRandom3DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, int numberOfPoints, ValueType maxCoord) {
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
    for(i=0; i<n; i++) adjArray[i*n +i]=0;
    
    //TODO: NoDistribution should be "BLOCK"?
    dmemo::DistributionPtr rep( new dmemo::NoDistribution( n ));
    adjM.setRawDenseData( rep, rep, adjArray.get() );
    //assert(adjM.checkSymmetry() );
 
}
//-------------------------------------------------------------------------------------------------
// coords.size()= 3 , coords[i].size()= N
// here, N= numPoints[0]*numPoints[1]*numPoints[2]
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createStructured3DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, std::vector<ValueType> maxCoord, std::vector<IndexType> numPoints) {
    
    std::vector<ValueType> offset={maxCoord[0]/numPoints[0], maxCoord[1]/numPoints[1], maxCoord[2]/numPoints[2]};
    IndexType N= numPoints[0]* numPoints[1]* numPoints[2];
    // create the coordinates
    IndexType index=0;
    index = 0;
    coords[0].setValue(0,0);
    coords[1].setValue(0,0);
    coords[2].setValue(0,0);
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

    scai::lama::CSRStorage<double> localMatrix;
    localMatrix.allocate( N, N );
    
    //create the adjacency matrix
    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<double> csrValues;  
    {
        // ja and values have size= edges of the graph
        // for a 3D structured grid with dimensions AxBxC the number of edges is 3ABC-AB-AC-BC
        IndexType numEdges= 3*numPoints[0]*numPoints[1]*numPoints[2] - numPoints[0]*numPoints[1]\
                                -numPoints[0]*numPoints[2] - numPoints[1]*numPoints[2];
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, N +1 );
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA);
        hmemo::WriteOnlyAccess<double> values( csrValues);    
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
            DenseVector<ValueType> p1(3,0);
            p1.setValue(0,coords[0].getValue(i));
            p1.setValue(1,coords[1].getValue(i));
            p1.setValue(2,coords[2].getValue(i));
            ngb_node = i +1;                             //edge 1
            if(ngb_node>=0 && ngb_node<N){
                // want to do: adjM[i][ngb_node]= 1;
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset )
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;   
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
            ngb_node = i -1;                             //edge 2
            if(ngb_node>=0 && ngb_node<N){
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset )
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
            
            ngb_node = i +numPoints[2];                  //edge 3
            if(ngb_node>=0 && ngb_node<N){
                                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset)
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
                
            ngb_node = i -numPoints[2];                  //edge 4
            if(ngb_node>=0 && ngb_node<N){
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset)
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
            
            ngb_node = i +numPoints[2]*numPoints[1];     //edge 5
            if(ngb_node>=0 && ngb_node<N){
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset )
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
                
            ngb_node = i -numPoints[2]*numPoints[1];     //edge 6
            if(ngb_node>=0 && ngb_node<N){
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset)
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    //-1 for the METIS format
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
            
            ia[i+1] = ia[i] +static_cast<IndexType>(numRowElems);
        }//for
    }
 
    localMatrix.swap( csrIA, csrJA, csrValues );
    adjM.assign(localMatrix);
}
/*
//-------------------------------------------------------------------------------------------------
// coords.size()= 2 , coords[i].size()= N
// here, N= numPoints[0]*numPoints[1]
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createStructured2DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, std::vector<ValueType> maxCoord, std::vector<IndexType> numPoints) {
    
    std::vector<ValueType> offset={maxCoord[0]/numPoints[0], maxCoord[1]/numPoints[1]};
    IndexType N= numPoints[0]* numPoints[1];
    // create the coordinates
    IndexType index=0;
    index = 0;
    coords[0].setValue(0,0);
    coords[1].setValue(0,0);
    for( IndexType indX=0; indX<numPoints[0]; indX++){
        for( IndexType indY=0; indY<numPoints[1]; indY++){
            coords[0].setValue(index, indX*offset[0] );
            coords[1].setValue(index, indY*offset[1] );
            ++index;
        }
    }

    scai::lama::CSRStorage<double> localMatrix;
    localMatrix.allocate( N, N );
    
    //create the adjacency matrix
    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<double> csrValues;  
    {
        // ja and values have size= edges of the graph
        // for a 3D structured grid with dimensions AxBxC the number of edges is 3ABC-AB-AC-BC
        IndexType numEdges= 3*numPoints[0]*numPoints[1]*numPoints[2] - numPoints[0]*numPoints[1]\
                                -numPoints[0]*numPoints[2] - numPoints[1]*numPoints[2];
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, N +1 );
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA);
        hmemo::WriteOnlyAccess<double> values( csrValues);    
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
            DenseVector<ValueType> p1(3,0);
            p1.setValue(0,coords[0].getValue(i));
            p1.setValue(1,coords[1].getValue(i));
            p1.setValue(2,coords[2].getValue(i));
            ngb_node = i +1;                             //edge 1
            if(ngb_node>=0 && ngb_node<N){
                // want to do: adjM[i][ngb_node]= 1;
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset )
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;   
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
            ngb_node = i -1;                             //edge 2
            if(ngb_node>=0 && ngb_node<N){
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset )
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
            
            ngb_node = i +numPoints[2];                  //edge 3
            if(ngb_node>=0 && ngb_node<N){
                                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset)
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
                
            ngb_node = i -numPoints[2];                  //edge 4
            if(ngb_node>=0 && ngb_node<N){
                DenseVector<ValueType> p2(3,0);
                p2.setValue(0,coords[0].getValue(ngb_node));
                p2.setValue(1,coords[1].getValue(ngb_node));
                p2.setValue(2,coords[2].getValue(ngb_node));
                if(dist3D(p1, p2).Scalar::getValue<ValueType>() <= max_offset)
                {
                ja.resize( ja.size()+1);
                values.resize( values.size()+1);
                ja[nnzCounter]= ngb_node;    
                values[nnzCounter] = 1;         // unweighted edges
                ++nnzCounter;
                ++numRowElems;
                }
            }
            
            ia[i+1] = ia[i] +static_cast<IndexType>(numRowElems);
        }//for
    }
 
    localMatrix.swap( csrIA, csrJA, csrValues );
    adjM.assign(localMatrix);
}
*/

//-------------------------------------------------------------------------------------------------
/*Given the adjacency matrix it writes it in the file "filename" using the METIS format. In the
 * METIS format the first line has two numbers, first is the number on vertices and the second
 * is the number of edges. Then, row i has numbers e1, e2, e3, ... notating the edges:
 * (i, e1), (i, e2), (i, e3), ....
 *  
 */

//TODO: must write coordiantes in the filename.xyz file
//      not sure what data type to use for coordinates: a) DenseVector or b)vector<DenseVector> ?
//DONE: Made a separate function for coordiantes
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::writeInFileMetisFormat (const CSRSparseMatrix<ValueType> &adjM, const std::string filename){
    std::ofstream f;
    f.open(filename);
    IndexType cols= adjM.getNumColumns() , rows= adjM.getNumRows();
    IndexType i, j;
    
    //the l1Norm/2 is the number of edges for an undirected, unweighted graph.
    //since is must be an adjacencey matrix cols==rows
    assert(((int) adjM.l1Norm().Scalar::getValue<ValueType>())%2==0);
    assert(cols==rows);
    f<<cols<<" "<< adjM.l1Norm().Scalar::getValue<ValueType>()/2<< std::endl;
    
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            if(adjM(i,j)==1) f<< j+1<< " ";
        }
        f<< std::endl;
    }
    f.close();
}

//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and their dimension, writes them in file "filename".
 */

//TODO:  not sure what data type to use for coordinates: a) DenseVector or b)vector<DenseVector> ?
//here a) coords=DenseVector
//TODO: should be abandoned ###
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::writeInFileCoords (const DenseVector<ValueType> &coords, IndexType dimension, const std::string filename){
    std::ofstream f;
    f.open(filename);
    IndexType i, j;

    // point i has coordiantes: [i*dim],[i*dim+1],...,[i*dim+dim] 
    // the size of the vector/dim must be an integer 
    assert(coords.size()/dimension == std::floor(coords.size()/dimension) );
    for(i=0; i<coords.size()/dimension; i++){
        for(j=0; j<dimension; j++){
            f<< coords.getValue(i*dimension +j).Scalar::getValue<ValueType>() << " ";
        }
        f<< std::endl;
    }
    f.close();
}

//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and their dimension, writes them in file "filename".
 */

//TODO:  not sure what data type to use for coordinates: a) DenseVector or b)vector<DenseVector> ?
// b) coords = vector<DenseVector>
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::writeInFileCoords (const std::vector<DenseVector<ValueType>> &coords, IndexType dimension, IndexType numPoints, const std::string filename){
    std::ofstream f;
    f.open(filename);
    IndexType i, j;

    assert(coords.size() == dimension );
    assert(coords[0].size() == numPoints);
    for(i=0; i<numPoints; i++){
        for(j=0; j<dimension; j++)
            f<< coords[j].getValue(i).Scalar::getValue<ValueType>() << " ";
        f<< std::endl;
    }
    
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and returns
 * it as an adjacency matrix adjM stored as a CSRSparseMatrix.
 * ###
 * TODO: should abandon. Too much memory needed because of the N*N array.
 */
template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType>   MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( const std::string filename){
    IndexType N, E;         //number of nodes and edges
    std::ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
   
    file >>N >> E;    
    CSRSparseMatrix<ValueType> ret(N, N);
    common::scoped_array<ValueType> values( new ValueType[ N * N ] );

    for(IndexType i=0; i<=N; i++){
        std::string line;
        // tokenize each line in the file
        std::getline(file, line);
        std::vector< std::vector<int> > all_integers;
        std::istringstream iss( line );
        all_integers.push_back( std::vector<int>( std::istream_iterator<int>(iss), std::istream_iterator<int>() ) );

        for(unsigned int j=0; j<all_integers.size(); j++){
            for(unsigned int k=0; k<all_integers[j].size(); k++){
                int index =all_integers[j][k];
                // subtract 1 because in the METIS format numbering starts from 1 not 0.
                values[(i-1)*N+index-1] = 1; 
            }
        }        
    }

    dmemo::DistributionPtr rep( new dmemo::NoDistribution( N ) );
    ret.setRawDenseData( rep, rep, values.get() );

    return ret;   
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and transforms 
 * it to the adjacency matrix as a CSRSparseMatrix.
 */
template<typename IndexType, typename ValueType>
void   MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix( lama::CSRSparseMatrix<ValueType> &matrix, dmemo::DistributionPtr  distribution, const std::string filename){
    IndexType N, numEdges;         //number of nodes and edges
    std::ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
   
    file >>N >> numEdges;   

    scai::lama::CSRStorage<double> localMatrix;
    // in a distributed version should be something like that
    //localMatrix.allocate( localSize, globalSize );
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
        
        IndexType rowCounter = 0; // count "local" rows
        IndexType nnzCounter = 0; // count "local" non-zero elements
        // read the first line and do nothing, contains the number of nodes and edges.
        std::string line;
        std::getline(file, line);
        
        //for every line, aka for all nodes
        for ( IndexType i=0; i<N; i++ ){
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
    matrix.assign(localMatrix);//, distRow, distCol);
    //TODO: the completely distributes version
    //reallocate/redistribute the matrix
    //matrix.allocate(distRow, distCol); 
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and transforms 
 * it to the adjacency matrix as a CSRSparseMatrix.
 */
template<typename IndexType, typename ValueType>
void   MeshIO<IndexType, ValueType>::readFromFile2AdjMatrixDistr( lama::CSRSparseMatrix<ValueType> &matrix, const std::string filename){
    const scai::dmemo::DistributionPtr distRow = matrix.getRowDistributionPtr();
    const scai::dmemo::DistributionPtr distCol = matrix.getColDistributionPtr(); //col=noDistribution
    scai::dmemo::CommunicatorPtr comm = distRow->getCommunicatorPtr();
    
    IndexType N, numEdges;         //number of nodes and edges
    std::ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
   
    file >>N >> numEdges;   
    assert( N == distRow->getGlobalSize() );
    
    scai::lama::CSRStorage<double> localMatrix;
    // in a distributed version should be something like that
    //localMatrix.allocate( localSize, globalSize );
    // here is not distributed, local=global
    localMatrix.allocate( distRow->getLocalSize(), distRow->getGlobalSize() );
std::cout<<  __FILE__<< " ,"<<__LINE__<<" == dist:"<< *distRow << " , local.size=" << distRow->getLocalSize()<< " , global.size=" << distRow->getGlobalSize()<< std::endl;    

    hmemo::HArray<IndexType> csrIA;
    hmemo::HArray<IndexType> csrJA;
    hmemo::HArray<double> csrValues;  
    {
        //TODO: for a distributed version this must change as numNZ should be the number of
        //      the local nodes in the processor, not the global
        // number of Non Zero values. *2 because every edge is read twice.
        
        //IndexType numNZ = numEdges*2;
        //hmemo::WriteOnlyAccess<IndexType> ja( csrJA, numNZ );
        //hmemo::WriteOnlyAccess<double> values( csrValues, numNZ );
        // In the distributed version we do not know the number of non zero values
        hmemo::WriteOnlyAccess<IndexType> ja( csrJA);
        hmemo::WriteOnlyAccess<double> values( csrValues);
        hmemo::WriteOnlyAccess<IndexType> ia( csrIA, distRow->getLocalSize() +1 );
        ia[0] = 0;

        std::vector<IndexType> colIndexes;
        std::vector<int> colValues;
        
        IndexType rowCounter = 0; //comm->getRank() * distRow->getLocalSize(); // count "local" rows
        IndexType nnzCounter = 0; // count "local" non-zero elements
        // read the first line and do nothing, contains the number of nodes and edges.
    
        std::string line;
        std::getline(file, line);
        
        //for every line, aka for all nodes
        for ( IndexType i=0; i<N; i++ ){
            if( distRow->isLocal(i)){       //if the line index is local in the processor
                std::getline(file, line);            
                std::vector< std::vector<int> > line_integers;
                std::istringstream iss( line );
                line_integers.push_back( std::vector<int>( std::istream_iterator<int>(iss), std::istream_iterator<int>() ) );
                //ia += the numbers of neighbours of i = line_integers.size()
                ia[rowCounter + 1] = ia[rowCounter] + static_cast<IndexType>( line_integers[0].size() );
                for(unsigned int j=0, len=line_integers[0].size(); j<len; j++){
                    // -1 because of the METIS format
                    ja.resize(ja.size() + 1); 
                    ja[nnzCounter]= line_integers[0][j] -1 ;
                    // all values are 1 for undirected, no-weigths graph    
                    values.resize( values.size() +1 );
                    values[nnzCounter]= 1;
                    ++nnzCounter;
                }                       
                ++rowCounter; 
            }
        }        
    }
    
    localMatrix.swap( csrIA, csrJA, csrValues );
    //matrix.assign( localMatrix, distribution, distribution ); // builds also halo
    matrix.assign(localMatrix, distRow, distCol);
    //TODO: the completely distributes version
    //reallocate/redistribute the matrix
    //matrix.allocate(distRow, distCol); 
}

/*
//-------------------------------------------------------------------------------------------------
// it appears slower than the method above
template<typename IndexType, typename ValueType>
void   MeshIO<IndexType, ValueType>::readFromFile2AdjMatrix_Boost( lama::CSRSparseMatrix<ValueType> &matrix, dmemo::DistributionPtr  distribution, const std::string filename){
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
            std::getline(file, line);            
            std::vector<ValueType> line_integers;
            boost::spirit::qi::phrase_parse( line.begin(), line.end(), 
                                            *boost::spirit::qi::double_,
                                            boost::spirit::ascii::space , line_integers );
            //for (std::vector<double>::size_type z = 0; z < line_integers.size(); ++z)
            //      std::cout << z << ": " << line_integers[z] << std::endl;

            //ia += the numbers of neighbours of i = line_integers.size()
            ia[rowCounter + 1] = ia[rowCounter] + static_cast<IndexType>( line_integers.size() );
            for(unsigned int j=0, len=line_integers.size(); j<len; j++){
                // -1 because of the METIS format
                ja[nnzCounter]= line_integers[j] -1 ;
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
}
*/
//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads that coordinates and returns
 * the coordinates in a DenseVector where point(x,y) is in [x*dim +y].
 * Every line of the file contais 2 ValueType numbers.
 */
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::fromFile2Coords_2D( const std::string filename, std::vector<DenseVector<ValueType>> &coords, IndexType numberOfPoints){
    IndexType N= numberOfPoints;
    //IndexType dim=2;
    //DenseVector<ValueType> ret(N*dim, 0);
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
    int n = numberOfPoints;
    int i, j;
    std::vector<DenseVector<ValueType>> ret(dimensions);
    for (i=0; i<dimensions; i++)
        ret[i] = DenseVector<ValueType>(numberOfPoints, 0);
    
    srand(time(NULL));
    ValueType r;
    for(i=0; i<n; i++){
        //ret[i] = DenseVector<ValueType>(dimensions, 0);
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

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType> struct cube { 
    ValueType x, y ,z;  // the 3D coordinated
    ValueType edge;     // the edge length ofthe cube
    IndexType id;       // the cube's id
    // 6 kind of neighbours
    std::vector<IndexType> neighbours_x0;   // left -   0
    std::vector<IndexType> neighbours_x1;   // right -  1
    std::vector<IndexType> neighbours_y0;   // front -  2
    std::vector<IndexType> neighbours_y1;   // back -   3
    std::vector<IndexType> neighbours_z0;   // bottom - 4
    std::vector<IndexType> neighbours_z1;   // top -    5
    
    std::vector<std::vector<IndexType>> neighbors;
    
    //constructor
    //cube (IndexType);
    cube (IndexType len) : x(0), y(0), z(0), edge(len), id(0) { 
        for(IndexType i=0; i<6; i++){
            neighbors.push_back(  std::vector<IndexType>() );
        }
    }
    
    cube( ValueType a, ValueType b, ValueType c, IndexType len, IndexType ID){
        x= a;
        y= b;
        z= c;
        edge= len;
        id= ID;
        for(IndexType i=0; i<6; i++){
            neighbors.push_back(  std::vector<IndexType>() );
        }
    }
    
    void print(){
        std::cout<< "id: "<< id << " , coords: ("<< x << ", "<< y << ", "<< z << ") , edge= "<< edge << std::endl;
    }
    
    // returns the corresponding neighbours
    // dim = 0,1 or 2 for dimensions x,y and z respectively 
    std::vector<IndexType> getNeighbours( IndexType dim, IndexType direction){
        if( dim==0){
            if(direction==0){
                return neighbours_x0;
            }else{
                return neighbours_x1;
            }
        }else if( dim==1 ){
            if(direction==0){
                return neighbours_y0;
            }else{
                return neighbours_y1;
            }
        }else{ //dim==2
            if(direction==0){
                return neighbours_z0;
            }else{
                return neighbours_z1;
            }
        }
    }
    
    std::vector<IndexType> getNeighbours( IndexType side){
        assert( side < 6);
        return neighbors[side];
    }
    
    
    
    // given a vector it sets it according tp dim and direction
    // dim = 0,1 or 2 for dimensions x,y and z respectively 
    void setNeighbours( IndexType dim, IndexType direction, std::vector<IndexType> input){
        if( dim==0){
            if(direction==0){
                neighbours_x0= input;
            }else{
                neighbours_x1= input;
            }
        }else if( dim==1 ){
            if(direction==0){
                neighbours_y0= input;
            }else{
                neighbours_y1= input;
            }
        }else{ //dim==2
            if(direction==0){
                neighbours_z0= input;
            }else{
                neighbours_z1= input;
            }
        }   
    }
    
    void setNeighbours( IndexType side, std::vector<IndexType> input){
        neighbors[side] = input;
    }
    
    void printNeighbors(){
        for(IndexType i=0; i<6; i++){
            std::cout<< "Neighbors on side "<< i<< ":\t";
            for(IndexType j=0; j< neighbors[i].size(); j++){
                std::cout<< neighbors[i][j]<< " , ";
            }
            std::cout<< std::endl;
        }
    }
    
};
//typedef struct <IndexType, ValueType>cube cube;

// coords.size()= 3 , coords[i].size()= N
// here, N= numPoints[0]*numPoints[1]*numPoints[2]
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::createUnstructured3DMesh(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coords, std::vector<ValueType> startPoint, ValueType edgeLen) {
 
    if(startPoint.size() != 3){
        throw std::runtime_error("The starting point must have " + std::to_string(3) + " dimensions.");
    }
    
    //std::vector<ValueType> offset={maxCoord[0]/numPoints[0], maxCoord[1]/numPoints[1], maxCoord[2]/numPoints[2]};
    //IndexType N= numPoints[0]* numPoints[1]* numPoints[2];
    
    std::vector< struct cube<IndexType, ValueType> > global_cubes; // where all cubes are stores
    std::vector< std::pair<IndexType, IndexType>> global_edges; // where all edges are stored
    
    // initialize the cubes list
    struct cube<IndexType, ValueType> tmp(startPoint[0], startPoint[1], startPoint[2], edgeLen, 0);
    // initialize neighbor's lists as empty
    tmp.neighbours_x0= {};
    tmp.neighbours_x1= {};
    tmp.neighbours_y0= {};
    tmp.neighbours_y1= {};
    tmp.neighbours_z0= {};
    tmp.neighbours_z1= {}; 
    
    global_cubes.push_back( tmp );
    
    IndexType rounds= 2;
    srand( time(NULL));
    for(IndexType i=0; i<rounds; i++){
        // pick a cube at random
        IndexType rand_ind= rand() % global_cubes.size();
        struct cube<IndexType, ValueType> fatherCube = global_cubes[rand_ind];
        
std::cout<<__FILE__<< "  "<< __LINE__<<", random_ind= "<< rand_ind << "  " ;
fatherCube.print();
        // divide cube in position rand_ind
        // every cube is divided in 8 subcubes (sc)
        unsigned int globalSize = global_cubes.size();
        for(IndexType scI=0; scI<8; scI++){
            struct cube<IndexType, ValueType> sc(0);
            unsigned b0 = ( scI >> 0) &1;
            unsigned b1 = ( scI >> 1) &1;
            unsigned b2 = ( scI >> 2) &1;
            
            // set edge length
            sc.edge= fatherCube.edge/2;
            
            // set coordinates
            sc.x = fatherCube.x + b0*sc.edge;
            sc.y = fatherCube.y + b1*sc.edge;
            sc.z = fatherCube.z + b2*sc.edge;
            
             // set id 
            if(scI==0){
                // the first node/cube replaces fatherCube
                sc.id= fatherCube.id;
                //global_cubes[rand_ind]= sc;
            }else{
                // the are added to the end
                sc.id= global_cubes.size();
                //global_cubes.push_back(sc);            
            }      
            
            /*
            // set neighbours. Every node/cube has 6 kinds of neighbours
            for(IndexType k=0; k<6; k++){
                {
                // for b0 - x
                sc.setNeighbours( 0, b0, fatherCube.getNeighbours( 0 , b0) ); // 0 for dim x
                //sc.setNeighbours( b0, fatherCube.getNeighbours( b0 ) ); // 0 for dim x
                IndexType revb0;
                if(b0==0){
                    revb0=1;
                }else{
                    revb0=0;
                }
                std::vector<IndexType> tmp0 = sc.getNeighbours( 0, revb0);
                std::vector<IndexType> tmp0_2 = sc.getNeighbours(revb0);
                
                IndexType ngb0 = scI^100;
                if(ngb0==0){
                    tmp0.push_back(rand_ind);
                    //sc.neighbors[k].push_back(rand_ind);
                }else{
                    tmp0.push_back(global_cubes.size() +scI);
                    //sc.neighbors[k].push_back(global_cubes.size() +scI);
                }
                sc.setNeighbours( 0 , revb0, tmp0 ); 
                }
                
                {
                // for b1 - y
                sc.setNeighbours( 1, b1, fatherCube.getNeighbours( 1 , b1) ); // 0 for dim x
                IndexType revb1;
                if(b1==0){
                    revb1=1;
                }else{
                    revb1=0;
                }
                std::vector<IndexType> tmp1 = sc.getNeighbours( 1, revb1);
                IndexType ngb1 = scI^010;
                if(ngb1==0){
                    tmp1.push_back(rand_ind);
                }else{
                    tmp1.push_back(global_cubes.size() +scI);
                }
                sc.setNeighbours( 1 , revb1, tmp1 );
                }
                
                {
                // for b2 - z
                sc.setNeighbours( 2, b2, fatherCube.getNeighbours( 2 , b2) ); // 0 for dim x
                IndexType revb2;
                if(b2==0){
                    revb2=1;
                }else{
                    revb2=0;
                }
                std::vector<IndexType> tmp2 = sc.getNeighbours( 2, revb2);
                IndexType ngb2 = scI^001;
                if(ngb2==0){
                    tmp2.push_back(rand_ind);
                }else{
                    tmp2.push_back(global_cubes.size() +scI);
                }
                sc.setNeighbours( 2 , revb2, tmp2 );
                }
            } // for(IndexType k=0; k<6; k++)
            */
            
            
            // the edges inherited form the fatherCube
            sc.neighbors[b0] = fatherCube.neighbors[b0];
            //for(IndexType m=0; m<sc.neighbors[b0].size(); m++){
            if( ! sc.neighbors[b0].empty() ){
                for(typename std::vector<IndexType>::iterator it=sc.neighbors[b0].begin(); it!=sc.neighbors[b0].end(); it++){
  std::cout<<__FILE__<< "  "<< __LINE__<< ", scI:"<< scI << " ,sc.n["<< b0<< "].size()= "<< sc.neighbors[b0].size() << " , *it= "<< *it<< " , global.size():" << global_cubes.size()<<  std::endl;                  
                    if( global_cubes[*it].x > sc.x+sc.edge || global_cubes[*it].x+global_cubes[*it].edge < sc.x){
std::cout<< global_cubes[*it].x << " >  " <<  sc.x+sc.edge  << " || "<<  global_cubes[*it].x+global_cubes[*it].edge <<" < "<< sc.x << std::endl;
                        sc.neighbors[b0].erase(it);
                    }else{
std::cout<<__FILE__<< "  "<< __LINE__<< std::endl;
                        global_cubes[*it].neighbors[1-b0].push_back(sc.id);
                    }
                }
            }
            
            sc.neighbors[2+b1] = fatherCube.neighbors[2+b1];
            if( ! sc.neighbors[2+b1].empty() ){
                for(typename std::vector<IndexType>::iterator it=sc.neighbors[2+ b1].begin(); it!=sc.neighbors[2+b1].end(); it++){
  std::cout<<__FILE__<< "  "<< __LINE__<< ", scI:"<< scI << " ,sc.n["<< 2+b1<< "].size()= "<< sc.neighbors[2+b1].size() << " , *it= "<< *it<< " , global.size():" << global_cubes.size()<<  std::endl;                  
                    if( global_cubes[*it].y > sc.y+sc.edge || global_cubes[*it].y+global_cubes[*it].edge < sc.y){
std::cout<< global_cubes[*it].y << " >  " <<  sc.y+sc.edge  <<  " || "<<  global_cubes[*it].y+global_cubes[*it].edge <<" < "<< sc.y << std::endl;
                        sc.neighbors[b1].erase(it);
                    }else{
                        global_cubes[*it].neighbors[3-b1].push_back(sc.id);
                    }
                }
            }
            
            sc.neighbors[4+b2] = fatherCube.neighbors[4+b2];
            if( ! sc.neighbors[4+b2].empty() ){
                for(typename std::vector<IndexType>::iterator it=sc.neighbors[4+b2].begin(); it!=sc.neighbors[4+b2].end(); it++){
  std::cout<<__FILE__<< "  "<< __LINE__<< ", scI:"<< scI << " ,sc.n["<< 4+b2 << "].size()= "<< sc.neighbors[4+b2].size() << " , *it= "<< *it<< " , global.size():" << global_cubes.size()<<  std::endl;                  
                    if( global_cubes[*it].z > sc.z+sc.edge || global_cubes[*it].z+global_cubes[*it].edge < sc.z){
std::cout<< global_cubes[*it].z << " >  " <<  sc.z+sc.edge  <<  " || "<<  global_cubes[*it].z+global_cubes[*it].edge<< " < "<< sc.z << std::endl;
                        sc.neighbors[b2].erase(it);
                    }else{
                        global_cubes[*it].neighbors[5-b2].push_back(sc.id);
                    }
                }
            }
            
            //set edges with brother nodes/cubes
            IndexType ngb = scI^1;
std::cout<<__FILE__<< "  "<< __LINE__<< " , ngb= "<< ngb<< " , glSize+ngb-1= "<< globalSize + ngb -1 << std::endl;
            if( ngb == 0){
                sc.neighbors[1-b0].push_back(rand_ind);
            }else{
                sc.neighbors[1-b0].push_back(globalSize + ngb -1);
            }
            ngb= scI^2;
std::cout<<__FILE__<< "  "<< __LINE__<< " , "<< ngb<< std::endl;
            if( ngb == 0){
                sc.neighbors[3-b1].push_back(rand_ind);
            }else{
                sc.neighbors[3-b1].push_back(globalSize + ngb -1);
            }
            ngb= scI^4;
std::cout<<__FILE__<< "  "<< __LINE__<< " , "<< ngb<< std::endl;
            if( ngb == 0){
                sc.neighbors[5-b2].push_back(rand_ind);
            }else{
                sc.neighbors[5-b2].push_back(globalSize + ngb -1);
            }
            
            // set id 
            // add it to the global_cubes list
            if(scI==0){
                // the first node/cube replaces fatherCube
                //sc.id= fatherCube.id;
                global_cubes[rand_ind]= sc;
            }else{
                // the are added to the end
                //sc.id= global_cubes.size();
                global_cubes.push_back(sc);            
            }
            
            

//sc.print();

        }
        
        // we replace the father node (fatherCube) with the first subcube so no need to erase
        //
        // fatherCube must be removed from global_cubes
        //typename std::vector<struct cube<IndexType, ValueType>>::iterator iter = std::next( global_cubes.begin(), rand_ind);
        //global_cubes.erase(iter);
        
    for(int k=0; k<global_cubes.size(); k++){
        std::cout<< k << ": "; 
        global_cubes[k].print();
        global_cubes[k].printNeighbors();
    }
        
        // the number ids and should be equal the number of cubes/nodes
        assert(global_cubes.size()-1 == global_cubes[ global_cubes.size()-1].id);
//std::cout<<__FILE__<< "  "<< __LINE__<< " , global_cubes.size()= "<< global_cubes.size()<< " <> "<< global_cubes[ global_cubes.size()-1].id << std::endl; 
        
    }

    // insert the new edges accordingly
    // because in the process of dividing the cubes, the cube divided is deleted, the edges should be
    // inserted after the division process is completed

    // insert the ccordinates
    
}

//-------------------------------------------------------------------------------------------------
template void MeshIO<int, double>::createRandom3DMesh(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coords,  int numberOfPoints, double maxCoord);

template void MeshIO<int, double>::createStructured3DMesh(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coords, std::vector<double> maxCoord, std::vector<int> numPoints);

template std::vector<DenseVector<double>> MeshIO<int, double>::randomPoints(int numberOfPoints, int dimensions, double maxCoord);

template Scalar MeshIO<int, double>::dist3D(DenseVector<double> p1, DenseVector<double> p2);

template void MeshIO<int, double>::writeInFileMetisFormat (const CSRSparseMatrix<double> &adjM, const std::string filename);

template void MeshIO<int, double>::writeInFileCoords (const DenseVector<double> &coords, int dimension, const std::string filename);

template void MeshIO<int, double>::writeInFileCoords (const std::vector<DenseVector<double>> &coords, int dimension, int numPoints, const std::string filename);

template CSRSparseMatrix<double>  MeshIO<int, double>::readFromFile2AdjMatrix(const std::string filename);

template void MeshIO<int, double>::readFromFile2AdjMatrix( CSRSparseMatrix<double> &matrix, dmemo::DistributionPtr distribution, const std::string filename);

template void MeshIO<int, double>::readFromFile2AdjMatrixDistr( lama::CSRSparseMatrix<double> &matrix, const std::string filename);

template void  MeshIO<int, double>::fromFile2Coords_2D( const std::string filename, std::vector<DenseVector<double>> &coords, int numberOfCoords);

template void MeshIO<int, double>::fromFile2Coords_3D( const std::string filename, std::vector<DenseVector<double>> &coords, int numberOfPoints);

template void MeshIO<int, double>::createUnstructured3DMesh(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coords, std::vector<double> startPoint, double edgeLen);
//template double MeshIO<int, double>:: randomValueType( double max);
} //namespace ITI
