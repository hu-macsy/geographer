#pragma once

#include <scai/lama.hpp>
#include <scai/lama/Vector.hpp>
//#include <scai/lama/Scalar.hpp>

#include "Settings.h"

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define BUILD_COMMIT_STRING STR_VALUE(BUILD_COMMIT)
#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl


namespace ITI {

    template <typename IndexType, typename ValueType>
    class RecursiveBisection{
    public:

        static void getPartition(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings);

        static scai::dmemo::CommunicatorPtr bisection(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings);
        
        static scai::lama::DenseVector<ValueType> partition1D(scai::lama::DenseVector<ValueType>& nodeWeights, IndexType k1, IndexType dimensionToPartition, IndexType sideLen, Settings settings);   
        
        /** Functions to transform a 1D index to 2D or 3D given the side length of the cubical grid.
         * For example, in a 4x4 grid, indexTo2D(1)=(0,1), indexTo2D(4)=(1,0) and indexTo2D(13)=(3,1)
         * 
         * @param[in] ind The index to transform.
         * @param[in] sideLen The side length of the 2D or 3D cube/grid.
         * @param[in] dimensions The dimension of the cube/grid (either 2 or 3).
         * @return A vector containing the index for every dimension. The size of the vector is equal to dimensions.
         */
        static std::vector<IndexType> indexToCoords(IndexType ind, IndexType sideLen, IndexType dimensions);
        
        static IndexType getLocalExtent( scai::lama::DenseVector<IndexType>& nodeWeights, IndexType dim, IndexType totalDims);
        
    private:
        static std::vector<IndexType> indexTo2D(IndexType ind, IndexType sideLen);
        
        static std::vector<IndexType> indexTo3D(IndexType ind, IndexType sideLen);
    };

}