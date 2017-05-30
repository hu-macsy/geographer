/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"

namespace ITI {
    
//TODO: Now it works only for k1=2 => bisection, handle k not a power of 2.
//TODO: Find numbers k1,k2,...,kd such that k1*k2*...*kd=k to perform multisection
//TODO: Rectangles are always non-intersecting => calculate all the projections at once
//TODO(?):  Enforce initial partition and keep track which PEs need to communicate for each projection
//TODO(?): Add an optimal algorithm for 1D partition
//TODO(kind of): Keep in mind semi-strucutured grids
    
template<typename IndexType, typename ValueType>
std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> MultiSection<IndexType, ValueType>::getPartition( const scai::lama::DenseVector<ValueType>& nodeWeights, const IndexType sideLen, Settings settings) {
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
//PRINT0( "totalWeight =" << totalWeight << ", averageWeight= " << averageWeight);

    bBox.weight = totalWeight;

    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
    
    //TODO: find integers for multisection
    // if k=2 then it is a bisection
    //
    IndexType k1 = 2;
    //

    IndexType numLeaves = root->getNumLeaves();

    while( numLeaves < k){
        SCAI_REGION("MultiSection.getPartition.forAllRectangles");
PRINT0(numLeaves);         
        ValueType maxExtent = 0;
        //std::vector<ValueType> minDifference ( numLeaves, LONG_MAX );        

        std::vector<IndexType> chosenDim ( numLeaves, -1);
        
        /* 
         * WARNING: projections[i], chosenDim[i] and numLeaves[i] should all refer to the same leaf/rectangle i
         */

        // a vector with pointers to all the neave nodes of the tree
        std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = root->getAllLeaves();        
        SCAI_ASSERT( allLeaves.size()==numLeaves, "Wrong number of leaves.");

        /*Two way to find in with dimension to project:
         * 1) just pick the dimension of the bounding box that has the largest extent and then project: only one projection
         * 2) project in every dimension and pick the one in which the difference between the maximum and minimum value is the smallest: d projections
         * 3) TODO: maybe we can change (2) and calculate the variance of the projection and pick the one with the biggest
         * */
             
        //TODO: since this is done locally, we can also get the 1D partition in every dimension and choose the best one
        //      maybe not the fastest way but probably would give better quality
        
        // choose the dimension to project for all leaves/rectangles
        if( settings.useExtent ){
            // for all leaves/rectangles
            for( int l=0; l<allLeaves.size(); l++){
                struct rectangle thisRectangle = allLeaves[l]->getRect();
                maxExtent = 0;
                for(int d=0; d<dim; d++){
                    SCAI_REGION("MultiSection.getPartition.useExtent");
                    ValueType extent = thisRectangle.top[d] - thisRectangle.bottom[d];
                    if( extent>maxExtent ){
                        maxExtent = extent;
                        chosenDim[l] = d;
                    }
                }
            }
        }
        // in chosenDim we have stored the desired dimension to project for all the leaf nodes

        // a vector of size numLeaves. projections[i] is the projection of leaf/rectangle i in the chosen dimension
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projection( nodeWeights, root, chosenDim, sideLen, settings);
                
        SCAI_ASSERT( projections.size()==numLeaves, "Wrong number of projections"); 
                /*
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
            */
        
        
        for(int l=0; l<numLeaves; l++){        
            SCAI_REGION("MultiSection.getPartition.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<ValueType> part1D, weightPerPart;
            std::vector<ValueType> thisProjection = projections[l];
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1D( thisProjection, k1, settings);

            //TODO: make sure that projections[l] and allLeaves[l] refer to the same rectangle
            struct rectangle thisRectangle = allLeaves[l]->getRect();
    /*        
    PRINT0("chosenDim= "<<  chosenDim);
    for(int gg=0; gg<weightPerPart.size(); gg++){
        PRINT0(weightPerPart[gg]);
        if( gg<part1D.size()){
            PRINT0(part1D[gg]);
        }
    }
    PRINT0("-----");
    */
            IndexType thisChosenDim = chosenDim[l];
            
            // create the new rectangles and add them to the queue
ValueType dbg_rectW=0;
            struct rectangle newRect;
            newRect.bottom = thisRectangle.bottom;
            newRect.top = thisRectangle.top;
            
            //first rectangle
            newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[0]+1;
            newRect.weight = weightPerPart[0];
            root->insert( newRect );     
dbg_rectW += newRect.weight;
//if(comm->getRank()==0) { newRect.print();}

            for(int h=0; h<part1D.size()-1; h++ ){
                //change only the chosen dimension
                newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h];
                newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h+1]+1;
                newRect.weight = weightPerPart[h];
                root->insert( newRect );
dbg_rectW += newRect.weight;                
            }
            
            //last rectangle
            newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D.back()+1;
            newRect.top = thisRectangle.top;
            newRect.weight = weightPerPart.back();
            root->insert( newRect );
dbg_rectW += newRect.weight;    
        

//TODO: only for debuging, remove variable dbg_rectW
SCAI_ASSERT( dbg_rectW==thisRectangle.weight, "Rectangle weight not correct"); 
        }
        numLeaves = root->getNumLeaves();
    }
    

    return root->getAllLeaves();
}
//---------------------------------------------------------------------------------------

//TODO: Generalize to take the projection of multiple, non-intersecting rectangles. 

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> MultiSection<IndexType, ValueType>::projection(const scai::lama::DenseVector<ValueType>& nodeWeights, const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot, const std::vector<IndexType>& dimensionToProject, const IndexType sideLen, Settings settings){
    SCAI_REGION("MultiSection.projection");
    
    const IndexType dimension = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    
    IndexType numLeaves = treeRoot->getNumLeaves();
    
    std::vector<std::vector<ValueType>> projections(numLeaves);
    
    IndexType leafIndex = treeRoot->indexLeaves(0);
    SCAI_ASSERT( numLeaves==leafIndex, "Wrong leaf indexing");
    SCAI_ASSERT( numLeaves==dimensionToProject.size(), "Wrong dimensionToProject vector size.");
    
    //TODO: pass allLeaves as argument since we already calculate them in getPartition
    
    const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = treeRoot->getAllLeaves();
    SCAI_ASSERT( allLeaves.size()==numLeaves, "Not consistent number of leaf nodes.");
    
    // reserve space for every projection
    for(IndexType l=0; l<numLeaves; l++){
        SCAI_REGION("MultiSection.projection.reserveSpace");
        const IndexType dim2proj = dimensionToProject[l];
        // the length for every projection in the chosen dimension
        IndexType projLength = allLeaves[l]->getRect().top[dim2proj] - allLeaves[l]->getRect().bottom[dim2proj];
        if(projLength<1){
            throw std::runtime_error("function: projection, line:" +std::to_string(__LINE__) +", the length of the projection is " +std::to_string(projLength) + " and is not correct");
        }
//PRINT0(l<<" ++ "<< projLength );   
        //projections[l].reserve(projLength);
        projections[l].assign( projLength, 0 );
    }
    
    // calculate projection for local coordinates
    {
        SCAI_REGION("MultiSection.projection.localProjection");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
        
        for(int i=0; i<localN; i++){
            const IndexType globalIndex = inputDist->local2global(i);
            std::vector<IndexType> coords = indexToCoords(globalIndex, sideLen, dimension); // check the global index
            //TODO: avoid the conversion to vector<double>
            std::vector<ValueType> coordsVal( coords.begin(), coords.end() );
            
            // a pointer to the cell that contains point i
            std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell = treeRoot->contains( coordsVal );
            IndexType thisLeafID = thisRectCell->getLeafID();
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0");
            SCAI_ASSERT( thisLeafID<projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];
            IndexType relativeIndex = coords[dim2proj]-thisRectCell->getRect().bottom[dim2proj];

            SCAI_ASSERT( relativeIndex<projections[ thisLeafID ].capacity(), "Wrong relative index: "<< relativeIndex << " should be < "<< projections[ thisLeafID ].capacity() );
//PRINT0(thisLeafID << " , "<< relativeIndex << " _ " <<localWeights[i] );   
            projections[ thisLeafID ][relativeIndex] += localWeights[i];
        }
    }
    // here, the projection of the local points has been calculated

    
    // must sum all local projections from all PEs
    std::vector<std::vector<ValueType>> globalProj(numLeaves);
    for(int i=0; i<numLeaves; i++){
        SCAI_REGION("MultiSection.projection.sumImpl");
PRINT0(projections[i].size() << " +__+ " << projections[i].capacity() );
        globalProj[i].assign( projections[i].size() ,0 );
        comm->sumImpl( globalProj[i].data(), projections[i].data(), projections[i].size(), scai::common::TypeTraits<ValueType>::stype);
PRINT0( globalProj[i].size()) ;
    }
    
    return globalProj;
}
//---------------------------------------------------------------------------------------

//TODO: Use an optimal algorithm or maybe both and add a user parameter
template<typename IndexType, typename ValueType>
std::pair<std::vector<ValueType>, std::vector<ValueType>> MultiSection<IndexType, ValueType>::partition1D( const std::vector<ValueType>& projection, const IndexType k, Settings settings){
    SCAI_REGION("MultiSection.partition1D");
    
    const IndexType dimension = settings.dimensions;
    
    ValueType totalWeight = std::accumulate(projection.begin(), projection.end(), 0);
    ValueType averageWeight = totalWeight/k;
    
    if(projection.size()==0){
        throw std::runtime_error( "In MultiSection::partition1D, input projection vector is empty");
    }
//for(int i : projection)      PRINT( projection[i] );

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

//TODO: can this be faster? with a tree-like structure?
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
//TODO: generalize for more dimensions and for non-cubic grids

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

template std::vector<std::shared_ptr<rectCell<int,double>>> MultiSection<int, double>::getPartition( const scai::lama::DenseVector<double>& nodeWeights, const IndexType sideLen, Settings settings);

template std::vector<std::vector<double>> MultiSection<int, double>::projection( const scai::lama::DenseVector<double>& nodeWeights, const std::shared_ptr<rectCell<int,double>> treeRoot, const std::vector<int>& dimensionToProject, const int sideLen, Settings settings);

template bool MultiSection<int, double>::inBBox( const std::vector<int>& coords, const struct rectangle& bBox, const int sideLen);
    
template  std::pair<std::vector<double>,std::vector<double>> MultiSection<int, double>::partition1D( const std::vector<double>& array, const int k, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const int sideLen, Settings settings);

template std::vector<int> MultiSection<int, double>::indexToCoords( const int ind, const int sideLen, const int dim);


};