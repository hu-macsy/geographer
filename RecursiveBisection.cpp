/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"

namespace ITI {

template<typename IndexType, typename ValueType>
void MultiSection<IndexType, ValueType>::getPartition(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings) {


}
//---------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
scai::dmemo::CommunicatorPtr MultiSection<IndexType, ValueType>::bisection(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings) {


}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
IndexType MultiSection<IndexType, ValueType>::getLocalExtent(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType dim, IndexType totalDims){
 
     const scai::dmemo::DistributionPtr dist = nodeWeights.getDistributionPtr();
     const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
     IndexType localN = dist->getLocalSize();
     
     for(IndexType i=0; i<localN; i++){
        IndexType globalIndex = dist->local2global(i);
//TODO: finish it, search in local indices to get the max extent for this dimension        
     }
     
     return 0;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> MultiSection<IndexType, ValueType>::projection1D(scai::lama::DenseVector<ValueType>& nodeWeights, std::pair<std::vector<ValueType>,std::vector<ValueType>> bBox, IndexType dimensionToProject, IndexType sideLen, Settings settings){
    SCAI_REGION("MultiSection.projection1D");
    const IndexType dimension = settings.dimensions;
    
    // for all dimensions i: bottomCorner[i]<topCorner[i] 
    std::vector<ValueType> bottomCorner = bBox.first, topCorner = bBox.second;
    SCAI_ASSERT( bottomCorner.size()==topCorner.size(), "Dimensions of bBox corners do not agree");
    SCAI_ASSERT( bottomCorner.size()==dimension, "bounding box dimension, "<< topCorner.size() << " do not agree with settings.dimension= "<< dimension);
    for(int i=0; i<dimension; i++){
        SCAI_ASSERT(bottomCorner[i]< topCorner[i], "bounding box corners are wrong: bottom coord= "<< bottomCorner[i] << " and top coord= "<< topCorner[i] << " for dimension "<< i );
    }
    
    const IndexType dim2proj = dimensionToProject; // shorter name
    const IndexType projLength = topCorner[dim2proj]-bottomCorner[dim2proj];
    if(projLength<1){
        throw std::runtime_error("function: projection1D, line:" +std::to_string(__LINE__) +", the length of the projection is " +std::to_string(projLength) + " and is not correct");
    }
    
    std::vector<ValueType> projection(projLength, 0);
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = nodeWeights.size();
    
    // calculate projection for local coordinates
    {
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
    // here, the projection of the local points is calculated
for(int i=0; i<projLength; i++){
    std::cout<< *comm<< ": "<< projection[i]<< " ,";
}
std::cout<< std::endl;
    // must sum all local projections from all PEs
    std::vector<ValueType> globalProj(projLength);
    comm->sumImpl( globalProj.data(), projection.data(), projLength, scai::common::TypeTraits<ValueType>::stype);
    
    return globalProj;
}
//---------------------------------------------------------------------------------------
// Checks if given index is in the bounding box bBox.
template<typename IndexType, typename ValueType>
bool MultiSection<IndexType, ValueType>::inBBox( std::vector<IndexType> coords, std::pair<std::vector<ValueType>, std::vector<ValueType>> bBox, IndexType sideLen){

    IndexType dimension = bBox.first.size();
    
    SCAI_ASSERT( coords.size()==dimension, "Dimensions do not agree.");
    if(dimension>3){
        throw std::runtime_error("function: inBBox, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }
    
    // for all dimensions i: bottom(i)<top(i) 
    std::vector<ValueType> bottom = bBox.first, top = bBox.second;
    
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
std::vector<ValueType> MultiSection<IndexType, ValueType>::partition1D(scai::lama::DenseVector<ValueType>& nodeWeights, IndexType k1, IndexType dimensionToPartition, IndexType sideLen, Settings settings){
    
    const IndexType dimension = settings.dimensions;
    SCAI_ASSERT(dimensionToPartition < dimension, "Dimension to partition is wrong, must be less than "<< dimension << " but it is " << dimensionToPartition );
    //SCAI_ASSERT(maxPoint.size() == dimension);
    
    const scai::dmemo::DistributionPtr dist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
     
    const IndexType globalN = nodeWeights.getDistributionPtr()->getGlobalSize();
    const IndexType localN = nodeWeights.getDistributionPtr()->getLocalSize();

    // assuming a square grid 
    scai::lama::DenseVector<ValueType> projectionSum(sideLen,0);
        
    {
        scai::hmemo::ReadAccess<ValueType> localWeights(nodeWeights.getLocalValues());   
        scai::hmemo::WriteAccess<ValueType> localSum(projectionSum.getLocalValues());
        
        for(int i=0; i<localN; i++){
            IndexType globalIndex = dist->local2global(i);
            std::vector<IndexType> coords = indexToCoords( globalIndex, sideLen, dimension);
            localSum[ coords[dimensionToPartition] ] += localWeights[i];
        }
    }
    
    // the global sum of the weights projection in the given dimension
    // the vector -projectionSum- is replicated to every PE
    comm->sumArray( projectionSum.getLocalValues() );
    
    ValueType totalWeight = projectionSum.sum().scai::lama::Scalar::getValue<ValueType>();
    ValueType averageWeight = totalWeight/k1;
    
PRINT0(averageWeight);
if(comm->getRank() ==0){
    for(int i=0; i<sideLen; i++){
        std::cout<< projectionSum.getLocalValues()[i] << " , ";
    }
    std::cout << std::endl;
}
    
    std::vector<ValueType> partHyperplanes(k1-1,-9);
    IndexType part=1;
    scai::utilskernel::LArray<ValueType> projectionSum_local = projectionSum.getLocalValues();
    ValueType thisPartWeight = 0;
    
    // TODO: pick the most balanced of the two choices; add parameter epsilon
    // if the weight of this part get more than the average return the previous index as hyperplane
    for(int i=0; i<sideLen;i++){
        thisPartWeight += projectionSum_local[i];
        if( thisPartWeight > part*averageWeight){
            SCAI_ASSERT(part-1 < partHyperplanes.size(), "index: "<< part-1 << " too big, must be < "<< partHyperplanes.size() )
            partHyperplanes[part-1]= i-1;
            ++part;
            thisPartWeight =0;
        }
    }
     
    return partHyperplanes;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> MultiSection<IndexType, ValueType>::indexToCoords(IndexType ind, IndexType sideLen, IndexType dim){
    
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

template void MultiSection<int, double>::getPartition(scai::lama::DenseVector<int>& nodeWeights, int k,  scai::dmemo::CommunicatorPtr comm, Settings settings);

template scai::dmemo::CommunicatorPtr MultiSection<int, double>::bisection(scai::lama::DenseVector<int>& nodeWeights, int k, scai::dmemo::CommunicatorPtr comm, Settings settings);

template std::vector<double> MultiSection<int, double>::projection1D(scai::lama::DenseVector<double>& nodeWeights, std::pair<std::vector<double>,std::vector<double>> bBox, int dimensionToProject, int sideLen, Settings settings);

template bool MultiSection<int, double>::inBBox( std::vector<int> coords, std::pair<std::vector<double>, std::vector<double>> bBox, int sideLen);
    
template std::vector<double> MultiSection<int, double>::partition1D(scai::lama::DenseVector<double>& nodeWeights, int k1, int dimensionToPartition, int sideLen, Settings settings);

template int MultiSection<int, double>::getLocalExtent(scai::lama::DenseVector<int>& nodeWeights, int dim, int totalDims);

template std::vector<int> MultiSection<int, double>::indexToCoords(int ind, int sideLen, int dim);


};