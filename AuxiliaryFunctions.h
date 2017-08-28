#pragma once

#include <chrono>
#include <fstream>

#include <scai/lama/DenseVector.hpp>
#include "GraphUtils.h"

namespace ITI{

class aux{
public:
    typedef int IndexType;
    typedef double ValueType;
    
//------------------------------------------------------------------------------   
    
static void writeHeatLike_local_2D(std::vector<IndexType> input,IndexType sideLen, IndexType dim, const std::string filename){
    std::ofstream f(filename);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");
    
    f<< "$map2 << EOD" << std::endl;
    
    for(IndexType i=0; i<sideLen; i++){
        for(IndexType j=0; j<sideLen; j++){
            //for(IndexType d=0; d<dim; d++){
            f<< j << " " << i << " " << input[i*sideLen+j] << std::endl;
            //PRINT( i/dim<< " " << i%dim << " " << input[i*dim +dim] );
        }
        f<< std::endl;
    }
    f<< "EOD"<< std::endl;
    f<< "set title \"Pixeled partition for file " << filename << "\" " << std::endl;
    f << "plot '$map2' using 2:1:3 with image" << std::endl;
}    
//------------------------------------------------------------------------------

static void writeHeatLike_local_2D(scai::hmemo::HArray<IndexType> input,IndexType sideLen, IndexType dim, const std::string filename){
    std::ofstream f(filename);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");
    
    f<< "$map2 << EOD" << std::endl;
    scai::hmemo::ReadAccess<IndexType> rInput( input );
    
    for(IndexType i=0; i<sideLen; i++){
        for(IndexType j=0; j<sideLen; j++){
            f<< j << " " << i << " " << rInput[i*sideLen+j] << std::endl;
        }
        f<< std::endl;
    }
    rInput.release();
    f<< "EOD"<< std::endl;
    f<< "set title \"Pixeled partition for file " << filename << "\" " << std::endl;
    f << "plot '$map2' using 2:1:3 with image" << std::endl;
}    
//------------------------------------------------------------------------------

static void print2DGrid(scai::lama::CSRSparseMatrix<ValueType>& adjM, scai::lama::DenseVector<IndexType>& partition  ){
    
    IndexType N= adjM.getNumRows();
    
    IndexType numX = std::sqrt(N);
    IndexType numY = numX;
    SCAI_ASSERT_EQ_ERROR(N , numX*numY, "Input not a grid" );
    
    if( numX>65 ){
        PRINT("grid too big to print, aborting.");
        return;
    }
        
    //get the border nodes
    scai::lama::DenseVector<IndexType> border(adjM.getColDistributionPtr(), 0);
    border = GraphUtils::getBorderNodes( adjM , partition);
    
    IndexType partViz[numX][numY];   
    IndexType bordViz[numX][numY]; 
    for(int i=0; i<numX; i++)
        for(int j=0; j<numY; j++){
            partViz[i][j]=partition.getValue(i*numX+j).getValue<IndexType>();
            bordViz[i][j]=border.getValue(i*numX+j).getValue<IndexType>();
        }

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    comm->synchronize();

    if(comm->getRank()==0 ){            
        std::cout<<"----------------------------"<< " Partition  "<< *comm << std::endl;    
        for(int i=0; i<numX; i++){
            for(int j=0; j<numY; j++){
                if(bordViz[i][j]==1) 
                    std::cout<< "\033[1;31m"<< partViz[i][j] << "\033[0m" <<"-";
                else
                    std::cout<< partViz[i][j]<<"-";
            }
            std::cout<< std::endl;
        }
    }

}
//------------------------------------------------------------------------------

template<typename T>
static void printVector( std::vector<T> v){
    for(int i=0; i<v.size(); i++){
        std::cout<< v[i] << ", ";
    }
    std::cout<< "\b\b\b" << std::endl;
}

//------------------------------------------------------------------------------

/*  From pixel (int) coords, either in 2S or 3D, to a 1D index. 
 * Only for cubes, where every side has the same length, maxLen;
 */
//static IndexType pixel2Index(IndexType pixel1, IndexType maxLen, IndexType dimension){
//}
 
 /** The l1 distance of two pixels in 2D if their given as a 1D distance.
  * @param[in] pixel1 The index of the first pixel.
  * @param[in] pixel1 The index of the second pixel.
  * @param[in] sideLen The length of the side of the cube.
  * 
  * @return The l1 distance of the pixels.
  */
static IndexType pixelL1Distance2D(IndexType pixel1, IndexType pixel2, IndexType sideLen){
     
     IndexType col1 = pixel1/sideLen;
     IndexType col2 = pixel2/sideLen;
     
     IndexType row1 = pixel1%sideLen;
     IndexType row2 = pixel2%sideLen;
     
     return std::abs(col1-col2) + std::abs(row1-row2);;
}

static ValueType pixelL2Distance2D(IndexType pixel1, IndexType pixel2, IndexType sideLen){
     
     IndexType col1 = pixel1/sideLen;
     IndexType col2 = pixel2/sideLen;
     
     IndexType row1 = pixel1%sideLen;
     IndexType row2 = pixel2%sideLen;
     
     return std::pow( ValueType (std::pow(std::abs(col1-col2),2) + std::pow(std::abs(row1-row2),2)) , 0.5);
}
//------------------------------------------------------------------------------

/* Given a (global) index and the size for each dimension (numPpoints.size()=3) calculates the position
 * of the index in 3D. The return value is not the coordinates of the point!
 * */
static std::tuple<IndexType, IndexType, IndexType> index2_3DPoint(IndexType index,  std::vector<IndexType> numPoints){
    // a YxZ plane
    SCAI_ASSERT( numPoints.size()==3 , "Wrong dimensions, should be 3");
    
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

static std::tuple<IndexType, IndexType> index2_2DPoint(IndexType index,  std::vector<IndexType> numPoints){
    SCAI_ASSERT( numPoints.size()==2 , "Wrong dimensions, should be 2");
    
    IndexType xIndex = index/numPoints[1];
    IndexType yIndex = index%numPoints[1];

    SCAI_ASSERT(xIndex >= 0, xIndex);
    SCAI_ASSERT(yIndex >= 0, yIndex);

    SCAI_ASSERT(xIndex < numPoints[0], xIndex << " for index: "<< index);
    SCAI_ASSERT(yIndex < numPoints[1], yIndex << " for index: "<< index);

    return std::make_tuple(xIndex, yIndex);
} 
 
}; //class aux
}// namespace ITI
