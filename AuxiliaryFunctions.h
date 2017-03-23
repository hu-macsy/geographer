#pragma once

#include <chrono>
#include <fstream>

#include "PrioQueue.h"
#include "ParcoRepart.h"
#include "MultiLevel.h"
#include "HilbertCurve.h"
#include "Settings.h"
#include "FileIO.h"


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
    border = ParcoRepart<IndexType,ValueType>::getBorderNodes( adjM , partition);
    
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
   
}; //class aux
}// namespace ITI
