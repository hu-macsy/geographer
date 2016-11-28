/*
 * MeshIO.cpp
 *
 *  Created on: 22.11.2016
 *      Author: tzovas
 */


#include "MeshIO.h"
//#include "ParcoRepart.h"
//#include "HilbertCurve.h"

namespace ITI{

template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::create3DMesh(CSRSparseMatrix<ValueType> &adjM, vector<DenseVector<ValueType>> &coords, int numberOfPoints, ValueType maxCoord) {
    int n = numberOfPoints;
    int i, j;
    
    coords = MeshIO::randomPoints(n, 3, maxCoord);
/*    for(i=0; i<n; i++){
        for(int j=0; j<3; j++)
            cout<< i<< ": "<< coords[i].getValue(j)<< ", ";
        cout<< endl;
    }
*/ 
    srand(time(NULL));    
    int bottom= 4, top= 8;
    Scalar dist;
    common::scoped_array<ValueType> adjArray( new ValueType[ n*n ]);
    //initialize matrix with zeros
    for(i=0; i<numberOfPoints; i++)
        for(j=0; j<numberOfPoints; j++)
            adjArray[i*n+j]=0;
        
    
    for(i=0; i<numberOfPoints; i++){
        int k= ((int) rand()%(top-bottom) + bottom);
        list<ValueType> kNNdist(k,maxCoord*1.7);       //max distance* sqrt(3)
        list<IndexType> kNNindex(k,0);
        typename list<ValueType>::iterator liVal;
        typename list<IndexType>::iterator liIndex = kNNindex.begin();
        
        for(j=0; j<numberOfPoints; j++){
            if(i==j) continue;
            dist = MeshIO<IndexType, ValueType>::dist3D(coords[i], coords[j]);

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
    
/*    cout<<endl;
    for(int ee=0; ee<n*n; ee++){
        if(ee%n==0) cout<<endl;
        cout<< adjArray[ee]<<" , ";
    }
    cout<<endl;
*/    
    //brute force zero in the diagonal
    //TODO: should not be needed but sometimes ones appear in the diagonal
    for(i=0; i<n; i++) adjArray[i*n +i]=0;
    
    //TODO: NoDistribution should be "BLOCK"?
    dmemo::DistributionPtr rep( new dmemo::NoDistribution( n ));
    adjM.setRawDenseData( rep, rep, adjArray.get() );
    assert(adjM.checkSymmetry() );
    
/*
    cout<< string(30, '-')<< adjM.checkSymmetry()<<endl;
    for(int p=0; p<n; p++){
        for(int q=0; q<n; q++)
            cout<< adjM(p,q).Scalar::getValue<ValueType>() <<" , ";
        cout<<endl;
    }
*/        

}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
vector<DenseVector<ValueType>> MeshIO<IndexType, ValueType>::randomPoints(int numberOfPoints, int dimensions, ValueType maxCoord){
    int n = numberOfPoints;
    int i, j;
    vector<DenseVector<ValueType>> ret(n);

    srand(time(NULL));
    ValueType r;
    
    for(i=0; i<n; i++){
        ret[i] = DenseVector<ValueType>(dimensions, 0);
        for(j=0; j<dimensions; j++){
            r= ((ValueType) rand()/RAND_MAX) * maxCoord;
            ret[i].setValue(j,r);
        }
    }
    return ret;
}

//-------------------------------------------------------------------------------------------------
/*Given the adjacency matrix it writes it in the file "filename" using the METIS format. In the
 * METIS format the first line has two numbers, first is the number on vertices and the second
 * is the number of edges. Then, row i has numbers e1, e2, e3, ... notating the edges:
 * (i, e1), (i, e2), (i, e3), ....
 *  
 */

//TODO: must write coordiantes in the filename.xyz file
//      not sure what data type to use for coordinates: a) DenseVector or b)vector<DenseVector> ?
template<typename IndexType, typename ValueType>
void MeshIO<IndexType, ValueType>::writeInFileMetisFormat (const CSRSparseMatrix<ValueType> &adjM, const string filename){
    ofstream f, fcoords;
    f.open(filename);
    fcoords.open(filename + string(".xyz"));
    IndexType cols= adjM.getNumColumns() , rows= adjM.getNumRows();
    IndexType i, j;
    
    //cout<<"NumCols= "<< cols<< " NumRows= "<< rows<< " , liNorm="<< adjM.l1Norm().Scalar::getValue<ValueType>() <<\
            " getNumValues(): "<< adjM.getNumValues()<< endl;
            
    //the l1Norm/2 is the number of edges for an undirected, unweighted graph.
    //since is must be an adjacencey matrix cols==rows
    assert(((int) adjM.l1Norm().Scalar::getValue<ValueType>())%2==0);
    assert(cols==rows);
    f<<cols<<" "<< adjM.l1Norm().Scalar::getValue<ValueType>()/2<< endl;
    
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            if(adjM(i,j)==1) f<< j+1<< " ";
        }
        f<< endl;
    }
    
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and returns
 * it as an adjacency matrix adjM stored as a CSRSparseMatrix.
 */
template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType>   MeshIO<IndexType, ValueType>::fromFile2AdjMatrix( const string filename){
    IndexType N, E;         //number of nodes and edges
    ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
   
    file >>N >> E;    
    CSRSparseMatrix<ValueType> ret(N, N);
    //DenseVector<ValueType> row(N, 5);
    //ret.setRow( row, 1, utilskernel::binary::COPY);
std::cout<<"file:"<<__FILE__ <<", "<<__LINE__<<std::endl; 
    common::scoped_array<ValueType> values( new ValueType[ N * N ] );

    //ValueType Val[N*N];
std::cout<<"file:"<<__FILE__ <<", "<<__LINE__<<std::endl;    
    for(IndexType i=0; i<=N; i++){
        std::string line;
        IndexType index;
        // tokenize each line in the file
        std::getline(file, line);
        vector< vector<int> > all_integers;
        istringstream iss( line );
        all_integers.push_back( vector<int>( istream_iterator<int>(iss), istream_iterator<int>() ) );

        for(IndexType j=0; j<all_integers.size(); j++){
            for(int oo=0;oo<all_integers[j].size(); oo++){
                index =all_integers[j][oo];
std::cout<<"file:"<<__FILE__ <<", "<<__LINE__<< "   , j= "<< j<< "  , oo="<< oo<< " ## "<< (i-1)*N+index-1<< std::endl;                   
                // subtract 1 because in the METIS format numbering starts from 1 not 0.
                values[(i-1)*N+index-1] = 1; 
            }
        }        
    }

    
    dmemo::DistributionPtr rep( new dmemo::NoDistribution( N ) );
    ret.setRawDenseData( rep, rep, values.get() );
/*
    for(IndexType i=0; i<50; i++){
        for(IndexType j=0; j<50; j++)
            cout<< i<< ","<< j<< ": <"<< ret.getValue(i,j).Scalar::getValue<ValueType>()<< ">  ";
        cout<<endl;
    }
*/

//cout<< ret.l1Norm()<< " - "<< ret.getNumValues()<< endl;
std::cout<<"file:"<<__FILE__ <<", "<<__LINE__<<std::endl;    

    return ret;
}
//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads that graph and returns
 * the coordinates in a DenseVector where point(x,y) is in [x*dim +y].
 * Every line of the file contais 2 ValueType numbers.
 */
template<typename IndexType, typename ValueType>
DenseVector<ValueType>   MeshIO<IndexType, ValueType>::fromFile2Coords_2D( const string filename, IndexType numberOfPoints){
    IndexType N= numberOfPoints;
    IndexType dim=2;
    DenseVector<ValueType> ret(N*dim, 0);
    ifstream file(filename);
    
    if(file.fail()) 
        throw std::runtime_error("File "+ filename+ " failed.");
    
    //the files, currently, contain 3 numbers in each line but z is always zero
    for(IndexType i=0; i<N; i++){
        ValueType x, y, z;
        file>> x >> y >> z;
        ret.setValue(i*dim, x);
        ret.setValue(i*dim+1, y);
        //ret.setValue(i*dim+2, z);
    }
    
    return ret;
}
    
//
// private functions
//

//-------------------------------------------------------------------------------------------------
// Calculates the distance in 3D.
//
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
template void MeshIO<int, double>::create3DMesh(CSRSparseMatrix<double> &adjM, vector<DenseVector<double>> &coords,  int numberOfPoints, double maxCoord);
template vector<DenseVector<double>> MeshIO<int, double>::randomPoints(int numberOfPoints, int dimensions, double maxCoord);
template Scalar MeshIO<int, double>::dist3D(DenseVector<double> p1, DenseVector<double> p2);
template void MeshIO<int, double>::writeInFileMetisFormat (const CSRSparseMatrix<double> &adjM, const string filename); 
template CSRSparseMatrix<double>  MeshIO<int, double>::fromFile2AdjMatrix(const string filename);
template DenseVector<double>   MeshIO<int, double>::fromFile2Coords_2D( const string filename, int numberOfCoords);

//template double MeshIO<int, double>:: randomValueType( double max);
} //namespace ITI
