/*
 * RecursiveBisection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "RecursiveBisection.h"

namespace ITI {

template<typename IndexType, typename ValueType>
void RecursiveBisection<IndexType, ValueType>::getPartition(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings) {


}
//---------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
scai::dmemo::CommunicatorPtr RecursiveBisection<IndexType, ValueType>::bisection(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType k, scai::dmemo::CommunicatorPtr comm, Settings settings) {


}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
IndexType RecursiveBisection<IndexType, ValueType>::getLocalExtent(scai::lama::DenseVector<IndexType>& nodeWeights, IndexType dim, IndexType totalDims){
 
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
std::vector<ValueType> RecursiveBisection<IndexType, ValueType>::partition1D(scai::lama::DenseVector<ValueType>& nodeWeights, IndexType k1, IndexType dimensionToPartition, IndexType sideLen, Settings settings){
    
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
std::vector<IndexType> RecursiveBisection<IndexType, ValueType>::indexToCoords(IndexType ind, IndexType sideLen, IndexType dim){
    
    if(dim==2){
        return  RecursiveBisection<IndexType, ValueType>::indexTo2D( ind, sideLen);
    }else if(dim==3){
        return RecursiveBisection<IndexType, ValueType>::indexTo3D( ind, sideLen);
    }else{
        throw std::runtime_error("function: indexToCoords, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }
    
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> RecursiveBisection<IndexType, ValueType>::indexTo2D(IndexType ind, IndexType sideLen){
    
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
std::vector<IndexType> RecursiveBisection<IndexType, ValueType>::indexTo3D(IndexType ind, IndexType sideLen){
    
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

template void RecursiveBisection<int, double>::getPartition(scai::lama::DenseVector<int>& nodeWeights, int k,  scai::dmemo::CommunicatorPtr comm, Settings settings);

template scai::dmemo::CommunicatorPtr RecursiveBisection<int, double>::bisection(scai::lama::DenseVector<int>& nodeWeights, int k, scai::dmemo::CommunicatorPtr comm, Settings settings);

template std::vector<double> RecursiveBisection<int, double>::partition1D(scai::lama::DenseVector<double>& nodeWeights, int k1, int dimensionToPartition, int sideLen, Settings settings);

template int RecursiveBisection<int, double>::getLocalExtent(scai::lama::DenseVector<int>& nodeWeights, int dim, int totalDims);

template std::vector<int> RecursiveBisection<int, double>::indexToCoords(int ind, int sideLen, int dim);


};