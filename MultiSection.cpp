/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"

namespace ITI {

/*    
     pick dimension in which to first partition
     first bBox= {(0,0,0,...), (max,max,max,...)}
    
     get first 1D partition in the choosen dimension
    
     form bounding boxes and get further partitions for the other dimensions
*/
template<typename IndexType, typename ValueType>
void MultiSection<IndexType, ValueType>::getPartition(scai::lama::DenseVector<ValueType>& nodeWeights, IndexType sideLen, Settings settings) {
    SCAI_REGION("MultiSection.getPartition");
	
    const IndexType k = settings.numBlocks;    
    const IndexType dim = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    
    // for all dimensions i: first[i]<second[i] 
    struct rectangle bBox;
    
    // at first the bounding box is the whole space
    for(int d=0; d<dim; d++){
        bBox.bottom.push_back(0);
        bBox.top.push_back(sideLen);
    }
    
    std::vector<rectangle> allRectangles;
    allRectangles.push_back(bBox);
        
    //TODO: this maybe costly...
    // calculate the projections for every dimension
    // after we have the projections in every dimension we must choose the first dimension to partition
    
    std::vector<std::vector<ValueType>> projectionsVector(dim);
    ValueType minExtent = LONG_MAX;
    IndexType chosenDim=0;
    
    for(int r=0; r<allRectangles.size(); r++){
        struct rectangle thisRectangle;
        for(int d=0; d<dim; d++){
            projectionsVector[d] = MultiSection<IndexType, ValueType>::projection( nodeWeights, thisRectangle, d, sideLen, settings);
            // extent = max - min
            ValueType extent = std::max_element( projectionsVector[d].begin(), projectionsVector[d].end() ) - std::min_element( projectionsVector[d].begin(), projectionsVector[d].end() );
            // chosenDim: the dimension with the smallest extent is choosen
            if(extent< minExtent){
                minExtent = extent;
                chosenDim = d;
            }
        }
        
        //perform 1D partitioning for the chosen dimension
        std::vector<ValueType> projection = MultiSection<IndexType, ValueType>::projection( nodeWeights, bBox, chosenDim, sideLen, settings);
        std::vector<ValueType> part1D, weightPerPart;
        std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1D( projection, k, settings);
        
    }
    
}
//---------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
std::vector<ValueType> MultiSection<IndexType, ValueType>::projection(scai::lama::DenseVector<ValueType>& nodeWeights, struct rectangle& bBox, IndexType dimensionToProject, IndexType sideLen, Settings settings){
    SCAI_REGION("MultiSection.projection");
    
    const IndexType dimension = settings.dimensions;
    
    // for all dimensions i: bottomCorner[i]<topCorner[i] 
    std::vector<ValueType> bottomCorner = bBox.bottom, topCorner = bBox.top;
    SCAI_ASSERT( bottomCorner.size()==topCorner.size(), "Dimensions of bBox corners do not agree");
    SCAI_ASSERT( bottomCorner.size()==dimension, "bounding box dimension, "<< topCorner.size() << " do not agree with settings.dimension= "<< dimension);
    for(int i=0; i<dimension; i++){
        SCAI_ASSERT(bottomCorner[i]< topCorner[i], "bounding box corners are wrong: bottom coord= "<< bottomCorner[i] << " and top coord= "<< topCorner[i] << " for dimension "<< i );
    }
    
    const IndexType dim2proj = dimensionToProject; // shorter name
    const IndexType projLength = topCorner[dim2proj]-bottomCorner[dim2proj];
    if(projLength<1){
        throw std::runtime_error("function: projection, line:" +std::to_string(__LINE__) +", the length of the projection is " +std::to_string(projLength) + " and is not correct");
    }
    
    std::vector<ValueType> projection(projLength, 0);
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = nodeWeights.size();
    
    // calculate projection for local coordinates
    {
        SCAI_REGION("MultiSection.projection.localProjection");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
        
        for(int i=0; i<localN; i++){
            const IndexType globalIndex = inputDist->local2global(i);
            std::vector<IndexType> coords = indexToCoords(globalIndex, sideLen, dimension); // check the global index
            if( inBBox(coords, bBox, sideLen) ){   
                IndexType relativeIndex =  coords[dim2proj]-bottomCorner[dim2proj];
                SCAI_ASSERT( relativeIndex>=0, "Index to calculate projection is negative: dimension to project= "<< dim2proj<< ", bottom dimension= "<<  bottomCorner[dim2proj]<< " and this coord= "<< coords[dim2proj]);
                SCAI_ASSERT( relativeIndex<projLength, "Index to calculate projection is too big= "<< relativeIndex<<": dimension to project= "<< dim2proj<< ", bottom dimension= "<<  bottomCorner[dim2proj]<< " and this coord= "<< coords[dim2proj]);
                projection[relativeIndex] += localWeights[i];
            }
        }
    }
    // here, the projection of the local points has been calculated

    // must sum all local projections from all PEs
    std::vector<ValueType> globalProj(projLength);
    {
        SCAI_REGION("MultiSection.projection.sumImpl");
        comm->sumImpl( globalProj.data(), projection.data(), projLength, scai::common::TypeTraits<ValueType>::stype);
    }
    
    return globalProj;
}
//---------------------------------------------------------------------------------------
// Checks if given index is in the bounding box bBox.
template<typename IndexType, typename ValueType>
bool MultiSection<IndexType, ValueType>::inBBox( std::vector<IndexType>& coords, struct rectangle& bBox, IndexType sideLen){
    SCAI_REGION("MultiSection.inBBox");
    
    IndexType dimension = bBox.top.size();
    
    SCAI_ASSERT( coords.size()==dimension, "Dimensions do not agree.");
    if(dimension>3){
        throw std::runtime_error("function: inBBox, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }
    
    // for all dimensions i: bottom(i)<top(i) 
    std::vector<ValueType> bottom = bBox.bottom, top = bBox.top;
    
    bool ret = true;
    
    for(int i=0; i<dimension; i++){
        SCAI_ASSERT( coords[i]<sideLen, "Coordinate "<< coords[i] << " for dimension "<< i <<" is too big, bigget than the side of the whole grid: "<< sideLen);
        // TODO: ensure if it should be coords[i]>=top[i] or coords[i]>top[i
        if(coords[i]>=top[i] or coords[i]<bottom[i]){
            ret = false;
            break;
        }
    }
        
    return ret;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::pair<std::vector<ValueType>,std::vector<ValueType>> MultiSection<IndexType, ValueType>::partition1D( std::vector<ValueType>& projection,  IndexType k, Settings settings){
    SCAI_REGION("MultiSection.partition1D");
    
    const IndexType dimension = settings.dimensions;
    
    ValueType totalWeight = std::accumulate(projection.begin(), projection.end(), 0);
    ValueType averageWeight = totalWeight/k;
//PRINT("totalWeight = "<< totalWeight << " , averageWeight = "<< averageWeight);

    std::vector<ValueType> partHyperplanes(k-1,-9);
    std::vector<ValueType> weightPerPart(k,-9);
    IndexType part=0;
    ValueType thisPartWeight = 0;
    ValueType epsilon = 0.05;   // imbalance parameter
    
    /*
     * TODO: change to a dynamic programming or iterative (or whatever) algorithm that is optimal (?)
     */
    
    // greedy 1D partition (a 2-approx solution?)
    for(int i=0; i<projection.size(); i++){
        thisPartWeight += projection[i];
        if( thisPartWeight > averageWeight*(1+epsilon)){
//PRINT(i<<": " << part << " _ " << thisPartWeight);
            SCAI_ASSERT(part < partHyperplanes.size(), "index: "<< part << " too big, must be < "<< partHyperplanes.size() )
            // choose between keeping the projection[i] in the sum, having something more than the average
            // or do not add projection[i] and get something below average
            if( averageWeight-(thisPartWeight-projection[i]) < thisPartWeight-averageWeight ){
                ValueType hyperplane = i-1;
                partHyperplanes[part]= hyperplane;
                // calculate new total weight left and new average weight
                totalWeight = totalWeight - thisPartWeight + projection[i];
                weightPerPart[part] = thisPartWeight - projection[i];                
                --i;
            }else{  // choose solution that is more than the average
                ValueType hyperplane = i;
                partHyperplanes[part]= hyperplane;
                // calculate new total weight left and new average weight
                totalWeight = totalWeight - thisPartWeight;
                weightPerPart[part] = thisPartWeight;
            }
            averageWeight = totalWeight/(k-part-1);
            thisPartWeight = 0;
            ++part;
        }
    }
    
    weightPerPart[part] = totalWeight;
  
    return std::make_pair(partHyperplanes, weightPerPart);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> MultiSection<IndexType, ValueType>::indexToCoords(IndexType ind, IndexType sideLen, IndexType dim){
    SCAI_REGION("MultiSection.indexToCoords");
    
    if(dim==2){
        return  MultiSection<IndexType, ValueType>::indexTo2D( ind, sideLen);
    }else if(dim==3){
        return MultiSection<IndexType, ValueType>::indexTo3D( ind, sideLen);
    }else{
        throw std::runtime_error("function: indexToCoords, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }
    
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> MultiSection<IndexType, ValueType>::indexTo2D(IndexType ind, IndexType sideLen){
    
    IndexType gridSize= sideLen*sideLen;
    
    if( ind>gridSize){
        PRINT("Index "<< ind <<" too big, should be < gridSize= "<< gridSize);
        throw std::runtime_error("Wrong index");
    }
    
    IndexType x = ind/sideLen;
    IndexType y = ind%sideLen;
    
    return std::vector<IndexType>{x, y};
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> MultiSection<IndexType, ValueType>::indexTo3D(IndexType ind, IndexType sideLen){
    
    IndexType gridSize= sideLen*sideLen*sideLen;
    
    if( ind>gridSize){
        PRINT("Index "<< ind <<" too big, should be < gridSize= "<< gridSize);
        throw std::runtime_error("Wrong index");
    }
    
    IndexType planeSize= sideLen*sideLen; // a YxZ plane
    
    IndexType x = ind/planeSize;
    IndexType y = (ind%planeSize)/sideLen;
    IndexType z = (ind%planeSize)%sideLen;
    
    return std::vector<IndexType>{ x, y, z };
}
//---------------------------------------------------------------------------------------

template void MultiSection<int, double>::getPartition(scai::lama::DenseVector<double>& nodeWeights, IndexType sideLen, Settings settings);

template std::vector<double> MultiSection<int, double>::projection( scai::lama::DenseVector<double>& nodeWeights, struct rectangle& bBox, int dimensionToProject, int sideLen, Settings settings);

template bool MultiSection<int, double>::inBBox( std::vector<int>& coords, struct rectangle& bBox, int sideLen);
    
template  std::pair<std::vector<double>,std::vector<double>> MultiSection<int, double>::partition1D( std::vector<double>& array, int k, Settings settings);

template std::vector<int> MultiSection<int, double>::indexToCoords(int ind, int sideLen, int dim);


};