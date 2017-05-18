#pragma once

#include <scai/lama.hpp>
#include <scai/lama/Vector.hpp>
//#include <scai/lama/Scalar.hpp>

#include <climits>

#include "Settings.h"

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define BUILD_COMMIT_STRING STR_VALUE(BUILD_COMMIT)
#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl


namespace ITI {

    template <typename IndexType, typename ValueType>
    class MultiSection{
    public:

        static void getPartition(scai::lama::DenseVector<ValueType>& nodeWeights, IndexType sideLen, Settings settings);
        
        /** Calculates the projection of all points in the bounding box (bBox) in the given dimension. Every PE
         *  creates an array of appropriate length, calculates the projection for its local coords and then
         *  calls a all2all sum routine. We assume a uniform grid so the exact coordianates can be refered just
         *  by the index of the -nodeWeights- vector.
         * 
         * @param[in] nodeWeights The weigh of each point.
         * @param[in] bBox The bounding box given as two vectors; one for the bottom point and one for the top point. For all dimensions i, should be: bBox.first(i)< bBox.second(i).
         * @param[in] dimensiontoProject The dimension in which we wish to project the weights. Should be more or equal to 0 and less than d (where d are the total dimensions).
         * @param[in] sideLen The length of the side of the whole uniform, square grid.
         * @param[in] setting A settigns struct passing various arguments.
         * @return Return an vector where in each position is the sum of the weights of the corespondig coordianate (not the same).

         * Example: bBox={(5,10),(8,15)} and dimensionToProject=0 (=x). Then the return vector has size |8-5|=3. return[0] is the sum of the coordinates in the bBox which have their 0-coordinate equal to 5, return[1] fot he points with 0-coordinate equal to 3 etc. If dimensionToProject=1 then return vector has size |10-15|=5.
         */
        static std::vector<ValueType> projection1D( scai::lama::DenseVector<ValueType>& nodeWeights, std::pair<std::vector<ValueType>,std::vector<ValueType>>& bBox, IndexType dimensionToProject, IndexType sideLen, Settings settings);
        
        static std::vector<ValueType> partition1D(scai::lama::DenseVector<ValueType>& nodeWeights, std::pair<std::vector<ValueType>,std::vector<ValueType>>& bBox, IndexType k1, IndexType dimensionToPartition, IndexType sideLen, Settings settings);   
        
        /**Checks if the given index is in the given bounding box. Index corresponds to a uniform matrix given
         * as a 1D array/vector. 
         * 
         * @param[in] coords The coordinates of the input point.
         * @param[in] bBox The bounding box given as two vectors; one for the bottom point and one for the top point. For all dimensions i, should be: bBox.first(i)< bBox.second(i).
         * @param[in] sideLen The length of the side of the whole uniform, square grid.
         * 
         */
        static bool inBBox( std::vector<IndexType>& coords, std::pair<std::vector<ValueType>, std::vector<ValueType>>& bBox, IndexType sideLen);
        /** Functions to transform a 1D index to 2D or 3D given the side length of the cubical grid.
         * For example, in a 4x4 grid, indexTo2D(1)=(0,1), indexTo2D(4)=(1,0) and indexTo2D(13)=(3,1)
         * 
         * @param[in] ind The index to transform.
         * @param[in] sideLen The side length of the 2D or 3D cube/grid.
         * @param[in] dimensions The dimension of the cube/grid (either 2 or 3).
         * @return A vector containing the index for every dimension. The size of the vector is equal to dimensions.
         */
        static std::vector<IndexType> indexToCoords(IndexType ind, IndexType sideLen, IndexType dimensions);
        
    private:
        static std::vector<IndexType> indexTo2D(IndexType ind, IndexType sideLen);
        
        static std::vector<IndexType> indexTo3D(IndexType ind, IndexType sideLen);
    };

}