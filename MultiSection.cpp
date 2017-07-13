/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"
#include <numeric>

namespace ITI {

template<typename IndexType, typename ValueType>
std::priority_queue< rectangle, std::vector<rectangle>, rectangle> MultiSection<IndexType, ValueType>::getPartition( const scai::lama::DenseVector<ValueType>& nodeWeights, const IndexType sideLen, Settings settings) {
    SCAI_REGION("MultiSection.getPartition");
	
    const IndexType k = settings.numBlocks;    
    const IndexType dim = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    
    // for all dimensions i: bBox.bottom[i]<bBox.top[i]
    struct rectangle bBox;
    
    // at first the bounding box is the whole space
    for(int d=0; d<dim; d++){
        bBox.bottom.push_back(0);
        bBox.top.push_back(sideLen);
    }

    // TODO: try to avoid that, probably not needed
    ValueType totalWeight = nodeWeights.sum().scai::lama::Scalar::getValue<ValueType>();
    ValueType averageWeight = totalWeight/k;

    bBox.weight = totalWeight;

    std::priority_queue< rectangle, std::vector<rectangle>, rectangle> allRectangles ;

    allRectangles.push(bBox);
    
    //TODO: find dim integers for multisection
    // if k=2 then it is a bisection
    //
    IndexType k1 = 2;
    //
    
    while( allRectangles.size() < k){
        SCAI_REGION("MultiSection.getPartition.forAllRectangles");
        
        struct rectangle thisRectangle;          
        if( !allRectangles.empty() ){
            thisRectangle = allRectangles.top();
            allRectangles.pop();
        }else{
            throw std::runtime_error("allRectangles is empty, this should not have happend.");
        }

        ValueType maxExtent = 0;
        ValueType minDifference = LONG_MAX;
        std::vector<ValueType> chosenProjection, thisProjection;
        // chosenDim: the dimension with the smallest extent is choosen
        IndexType chosenDim = -1;
        
        /*Two way to find in with dimension to project:
         * 1) just pick the dimension of the bounding box that has the largest extent: only one projection
         * 2) project in every dimension and pick the one in which the difference between the maximum and minimum value is the smallest: d projections
         * 3) TODO: maybe we can change (2) and calculate the variance of the projection and pick the one with the biggest
         * */
             
        //TODO: since this is done locally, we can also get the 1D partition in every dimension and choose the best one
        //      maybe not the fastest way but probably would give better quality
        for(int d=0; d<dim; d++){
            if(settings.useExtent){
                SCAI_REGION("MultiSection.getPartition.useExtent");
                ValueType extent = thisRectangle.top[d] - thisRectangle.bottom[d];
                if( extent>maxExtent ){
                    maxExtent = extent;
                    chosenDim= d;
                
                    // this way we take only the projection to the dimension with the largest extent.
                    chosenProjection = MultiSection<IndexType, ValueType>::projection( nodeWeights, thisRectangle, d, sideLen, settings);
                }
            }else{          //use difference
                SCAI_REGION("MultiSection.getPartition.useDiff");
                thisProjection = MultiSection<IndexType, ValueType>::projection( nodeWeights, thisRectangle, d, sideLen, settings);
                // variance = max - min
                ValueType difference = std::max_element( thisProjection.begin(), thisProjection.end() ) - std::min_element( thisProjection.begin(), thisProjection.end() );
                if( difference< minDifference){
                    minDifference = difference;
                    chosenDim = d;
                    chosenProjection = thisProjection;
                }
            }
        }

        //perform 1D partitioning for the chosen dimension
        std::vector<ValueType> part1D, weightPerPart;
        std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1D( chosenProjection, k1, settings);

        SCAI_REGION_START("MultiSection.getPartition.createRectanglesAndPush");
        // create the new rectangles and add them to the queue
ValueType dbg_rectW=0;
        struct rectangle newRect;
        newRect.bottom = thisRectangle.bottom;
        newRect.top = thisRectangle.top;
        
        //first rectangle
        newRect.top[chosenDim] = thisRectangle.bottom[chosenDim]+part1D[0]+1;
        newRect.weight = weightPerPart[0];
        allRectangles.push( newRect );        
dbg_rectW += newRect.weight;


        for(int h=0; h<part1D.size()-1; h++ ){
            //change only the chosen dimension
            newRect.bottom[chosenDim] = thisRectangle.bottom[chosenDim]+part1D[h];
            newRect.top[chosenDim] = thisRectangle.bottom[chosenDim]+part1D[h+1]+1;
            newRect.weight = weightPerPart[h];
            allRectangles.push( newRect );
dbg_rectW += newRect.weight;                
        }
        
        //last rectangle
        newRect.bottom[chosenDim] = thisRectangle.bottom[chosenDim]+part1D.back()+1;
        newRect.top = thisRectangle.top;
        newRect.weight = weightPerPart.back();
        allRectangles.push( newRect );
dbg_rectW += newRect.weight;        

//TODO: only for debuging, remove variable dbg_rectW
SCAI_ASSERT( dbg_rectW==thisRectangle.weight, "Rectangle weight not correct"); 
        SCAI_REGION_END("MultiSection.getPartition.createRectanglesAndPush");

    }
    
    return allRectangles;
}
//---------------------------------------------------------------------------------------

//TODO: Now it works only for k1=2 => bisection, handle k not a power of 2.
//TODO: Find numbers k1,k2,...,kd such that k1*k2*...*kd=k to perform multisection

template<typename IndexType, typename ValueType>
std::vector<ValueType> MultiSection<IndexType, ValueType>::projection(const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle& bBox, const IndexType dimensionToProject, const IndexType sideLen, Settings settings){
    SCAI_REGION("MultiSection.projection");
    
    const IndexType dimension = settings.dimensions;
    
    // for all dimensions i: bottomCorner[i]<topCorner[i] 
    std::vector<ValueType> bottomCorner = bBox.bottom, topCorner = bBox.top;
    SCAI_ASSERT( bottomCorner.size()==topCorner.size(), "Dimensions of bBox corners do not agree");
    SCAI_ASSERT( bottomCorner.size()==dimension, "Bounding box dimension, "<< topCorner.size() << " do not agree with settings.dimension= "<< dimension);
    
    for(int i=0; i<dimension; i++){
        SCAI_ASSERT(bottomCorner[i]< topCorner[i], "Bounding box corners are wrong: bottom coord= "<< bottomCorner[i] << " and top coord= "<< topCorner[i] << " for dimension "<< i );
        SCAI_ASSERT( topCorner[i]<=sideLen, "The bounding box is out of the grid bounds. Top corner in dimension   "<< i << " is " << topCorner[i] << " while the grid's side is "<< sideLen);
    }
    
    const IndexType dim2proj = dimensionToProject; // shorter name
    const IndexType projLength = topCorner[dim2proj]-bottomCorner[dim2proj];
    if(projLength<1){
        throw std::runtime_error("function: projection, line:" +std::to_string(__LINE__) +", the length of the projection is " +std::to_string(projLength) + " and is not correct");
    }
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = nodeWeights.size();
    
    std::vector<ValueType> projection(projLength, 0);
    
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
bool MultiSection<IndexType, ValueType>::inBBox( const std::vector<IndexType>& coords, const struct rectangle& bBox, const IndexType sideLen){
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
        // TODO: ensure if it should be coords[i]>=top[i] or coords[i]>top[i]
        if(coords[i]>=top[i] or coords[i]<bottom[i]){
            ret = false;
            break;
        }
    }
        
    return ret;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::pair<std::vector<ValueType>,std::vector<ValueType>> MultiSection<IndexType, ValueType>::partition1D( const std::vector<ValueType>& projection, const IndexType k, Settings settings){
    SCAI_REGION("MultiSection.partition1D");
    
    const IndexType dimension = settings.dimensions;
    
    ValueType totalWeight = std::accumulate(projection.begin(), projection.end(), 0);
    ValueType averageWeight = totalWeight/k;

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
        if( thisPartWeight > averageWeight /* *(1+epsilon)*/){
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
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle& bBox, const IndexType sideLen, Settings settings){
    SCAI_REGION("MultiSection.getRectangleWeight");
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    
    const IndexType dimension = bBox.top.size();
    ValueType localWeight=0;
    
    {
        SCAI_REGION("MultiSection.getRectangleWeight.localWeight");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
        
        for(int i=0; i<localN; i++){
            const IndexType globalIndex = inputDist->local2global(i);
            std::vector<IndexType> coords = indexToCoords(globalIndex, sideLen, dimension); // check the global index
            if( inBBox(coords, bBox, sideLen) ){ 
                localWeight += localWeights[i];
            }
        }
    }
    
    // sum all local weights
    return comm->sum(localWeight);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> MultiSection<IndexType, ValueType>::indexToCoords(const IndexType ind, const IndexType sideLen, const IndexType dim){
    SCAI_REGION("MultiSection.indexToCoords");
    
    IndexType gridSize= std::pow(sideLen, dim);
    
    if( ind>gridSize){
        PRINT("Index "<< ind <<" too big, should be < gridSize= "<< gridSize);
        throw std::runtime_error("Wrong index");
    }
    
    if(ind<0){
        throw std::runtime_error("Wrong index" + std::to_string(ind) + " should be positive or zero.");
    }
    
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

    IndexType x = ind/sideLen;
    IndexType y = ind%sideLen;
    
    return std::vector<IndexType>{x, y};
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> MultiSection<IndexType, ValueType>::indexTo3D(IndexType ind, IndexType sideLen){
    
    IndexType planeSize= sideLen*sideLen; // a YxZ plane
    
    IndexType x = ind/planeSize;
    IndexType y = (ind%planeSize)/sideLen;
    IndexType z = (ind%planeSize)%sideLen;
    
    return std::vector<IndexType>{ x, y, z };
}
//---------------------------------------------------------------------------------------

template std::priority_queue< rectangle, std::vector<rectangle>, rectangle> MultiSection<int, double>::getPartition( const scai::lama::DenseVector<double>& nodeWeights, const IndexType sideLen, Settings settings);

template std::vector<double> MultiSection<int, double>::projection( const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const int dimensionToProject, const int sideLen, Settings settings);

template bool MultiSection<int, double>::inBBox( const std::vector<int>& coords, const struct rectangle& bBox, const int sideLen);
    
template  std::pair<std::vector<double>,std::vector<double>> MultiSection<int, double>::partition1D( const std::vector<double>& array, const int k, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const int sideLen, Settings settings);

template std::vector<int> MultiSection<int, double>::indexToCoords( const int ind, const int sideLen, const int dim);


};
