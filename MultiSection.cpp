/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"
#include <numeric>

namespace ITI {
    
//TODO: Now it works only for k=x^(1/dim) for int x. Handle the general case.
//TODO: Find numbers k1,k2,...,kd such that k1*k2*...*kd=k to perform multisection
//TODO(?): Enforce initial partition and keep track which PEs need to communicate for each projection
//TODO(?): Add an optimal algorithm for 1D partition
//TODO(kind of): Keep in mind semi-strucutured grids
    


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::getPartitionNonUniform(const scai::lama::CSRSparseMatrix<ValueType> &input, const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, struct Settings settings ){ 
    SCAI_REGION("MultiSection.getPartition");
    
    const scai::dmemo::DistributionPtr inputDistPtr = input.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDistPtr->getCommunicatorPtr();
    
    const IndexType k = settings.numBlocks;
    const IndexType dim = settings.dimensions;
    const IndexType globalN = inputDistPtr->getGlobalSize();
    const IndexType localN = inputDistPtr->getLocalSize();

    //
    // check input arguments for sanity
    //
    if( coordinates.size()!=dim ){
        throw std::runtime_error("Wrong number of settings.dimensions and coordinates.size(). They must be the same");
    }

    if( globalN != coordinates[0].size() ) {
        throw std::runtime_error("Matrix has " + std::to_string(globalN) + " rows, but " + std::to_string(coordinates[0].size())
        + " coordinates are given.");
    }
    
    if( globalN != input.getNumColumns()) {
        throw std::runtime_error("Matrix must be quadratic.");
    }
    
    if( !input.isConsistent()) {
        throw std::runtime_error("Input matrix inconsistent");
    }
    
    if( k > globalN) {
        throw std::runtime_error("Creating " + std::to_string(k) + " blocks from " + std::to_string(globalN) + " elements is impossible.");
    }
     
    //
    // get minimum and maximum of the coordinates
    //
    std::vector<ValueType> minCoords(dim, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dim, std::numeric_limits<ValueType>::lowest());
    std::vector<ValueType> scaledMin(dim, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> scaledMax(dim, std::numeric_limits<ValueType>::lowest());
    
    //
    // scale the local coordinates so the projections are not too big and relative to the input size
    //
    //TODO: since we scale only the local part, this can be turned into a vector<vector<T>>
    //TODO: assert if scaledMin and scaledMax are always 0 and scale respectivelly
    std::vector<scai::lama::DenseVector<IndexType>> scaledCoords(dim);
    
    for(IndexType d=0; d<dim; d++){
        scaledCoords[d].allocate(inputDistPtr);
        scaledCoords[d] = static_cast<ValueType>( 0 );
    }
    
    // scale= N^(1/d): this way the scaled max is N^(1/d) and this is also the maximum size of the projection arrays
    IndexType scale = std::pow( globalN, 1.0/dim);
    
    {
        SCAI_REGION( "MultiSection.getPartitionNonUniform.minMaxAndScale" )
        
        for (IndexType d = 0; d < dim; d++) {
            //get local parts of coordinates
            const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[d].getLocalValues();
            for (IndexType i = 0; i < localN; i++) {
                ValueType coord = localPartOfCoords[i];
                if (coord < minCoords[d]) minCoords[d] = coord;
                if (coord > maxCoords[d]) maxCoords[d] = coord;
            }
        }
        
        //  communicate to get global min / max
        // WARNING/TODO: because the max coordinate of a bounding box does belong in the box ( box=[min,manx) )
        //          we must +1. or TODO: do it inside the getRectanglesNonUniform() function
        for (IndexType d = 0; d < dim; d++) {
            minCoords[d] = comm->min(minCoords[d]);
            maxCoords[d] = comm->max(maxCoords[d])+1;
            scaledMax[d] = scale;
            scaledMin[d] = 0;
        }
        
        for (IndexType d = 0; d < dim; d++) {
            //get local parts of coordinates
            const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[d].getLocalValues();
            scai::hmemo::WriteOnlyAccess<IndexType> wScaledCoord( scaledCoords[d].getLocalValues() );
            
            SCAI_ASSERT( localN==wScaledCoord.size() , "Wrong size of local part.");
            
            for (IndexType i = 0; i < localN; i++) {
                ValueType normalizedCoord = (localPartOfCoords[i] - minCoords[d])/(maxCoords[d]-minCoords[d]);
                IndexType scaledCoord =  normalizedCoord * scale; 
                wScaledCoord[i] = scaledCoord;
                SCAI_ASSERT( scaledCoord >=0 and scaledCoord<=scale, "Wrong scaled coordinate " << scaledCoord << " is either negative or more than "<< scale);
                
                if (scaledCoord < scaledMin[d]) scaledMin[d] = scaledCoord;
                if (scaledCoord > scaledMax[d]) scaledMax[d] = scaledCoord;
            }
            scaledMin[d] = comm->min( scaledMin[d] );
            scaledMax[d] = comm->max( scaledMax[d] );      
        }
    }
    
    for (IndexType d=0; d<dim; d++) {
        SCAI_ASSERT( scaledMax[d]<= std::pow(globalN, 1.0/dim), "Scaled maximum value "<< scaledMax[d] << " is too large. should be less than " << std::pow(globalN, 1.0/dim) );
        if( scaledMin[d]!=0 ){
            //TODO: it works even if scaledMin is not 0 but the projection arrays will start from 0 and the first 
            //      elements will just always be 0.
            PRINT(":");
            throw std::logic_error("Minimum scaled value should be 0 but it is " + std::to_string(scaledMin[d]) );
        }
    }
    SCAI_ASSERT( localN==scaledCoords[0].getLocalValues().size(), "Wrong size of scaled coordinates vector: localN= "<< localN << " and scaledCoords.size()= " << scaledCoords[0].getLocalValues().size() );
    
    //
    // get a partitioning into rectangles
    //
    std::shared_ptr<rectCell<IndexType,ValueType>> root = MultiSection<IndexType, ValueType>::getRectanglesNonUniform( input, scaledCoords, nodeWeights, scaledMin, scaledMax, settings);
    
    const IndexType numLeaves = root->getNumLeaves();
    
    SCAI_ASSERT( numLeaves==k , "Returned number of rectangles is not equal k, rectangles.size()= " << numLeaves << " and k= "<< k );
    
    //
    // set the partition of every local point/vertex according to which rectangle it belongs to
    //
    scai::lama::DenseVector<IndexType> partition( inputDistPtr, -1 );
    
    {
        SCAI_REGION( "MultiSection.getPartitionNonUniform.setLocalPartition" )
        scai::hmemo::WriteOnlyAccess<IndexType> wLocalPart( partition.getLocalValues() );
        
        for(IndexType i=0; i<localN; i++){
            std::vector<IndexType> point(dim);            
            //TODO: probably not the most efficient way to do it, maybe avoid ReadAccess
            for(IndexType d=0; d<dim; d++){
                scai::hmemo::ReadAccess<IndexType> coordAccess( scaledCoords[d].getLocalValues() );
                coordAccess.getValue( point[d], i);
            }   
            wLocalPart[i] = root->contains( point )->getLeafID();
        }
    }
    
    return partition;
}

//--------------------------------------------------------------------------------------- 
    
template<typename IndexType, typename ValueType>
std::shared_ptr<rectCell<IndexType,ValueType>> MultiSection<IndexType, ValueType>::getRectangles( const scai::lama::DenseVector<ValueType>& nodeWeights, const IndexType sideLen, struct Settings settings) {
    SCAI_REGION("MultiSection.getRectangles");
	
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

    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
    
    //TODO: find integers for multisection
    // if k=2 then it is a bisection
    //
    IndexType k1 = 2;
    //
    
    // from k get d numbers such that their product equals k
    // TODO: now k must be number such that k^(1/d) is an integer, drop this condition, generalize
    const ValueType sqrtK = std::pow( k,  1.0/dim );

    if( !std::floor(sqrtK)==sqrtK ){
        PRINT0("Input k= "<< k << " and sqrt(k)= "<< sqrtK );
        throw std::logic_error("Number of blocks not a square number");
    }
    
    // sqrtK is not correct, it is -1 but not sure if always
    IndexType intSqrtK = sqrtK;
    
    if( std::pow( intSqrtK+1, dim ) == k){
        intSqrtK++;
    }
    SCAI_ASSERT( std::pow( intSqrtK, dim ) == k, "Wrong square root of k. k= "<< k << ", pow(k, 1/d)= " << intSqrtK);
    
    // number of cuts for each dimensions
    std::vector<IndexType> numCuts( dim, intSqrtK );

    IndexType numLeaves = root->getNumLeaves();

    for(typename std::vector<IndexType>::iterator thisDimCuts=numCuts.begin(); thisDimCuts!=numCuts.end(); ++thisDimCuts ){
        SCAI_REGION("MultiSection.getRectangles.forAllRectangles");

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
        if( settings.multisectionUseExtent ){
            SCAI_REGION("MultiSection.getRectangles.forAllRectangles.useExtent");
            // for all leaves/rectangles
            for( int l=0; l<allLeaves.size(); l++){
                struct rectangle thisRectangle = allLeaves[l]->getRect();
                maxExtent = 0;
                for(int d=0; d<dim; d++){
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

        for(int l=0; l<numLeaves; l++){        
            SCAI_REGION("MultiSection.getRectangles.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<ValueType> part1D, weightPerPart;
            std::vector<ValueType> thisProjection = projections[l];
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1D( thisProjection, *thisDimCuts, settings);
            // TODO: possibly expensive assertion
            SCAI_ASSERT( std::accumulate(thisProjection.begin(), thisProjection.end(), 0)==std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0), "Weights are wrong." )
            
            //TODO: make sure that projections[l] and allLeaves[l] refer to the same rectangle
            struct rectangle thisRectangle = allLeaves[l]->getRect();

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

            for(int h=0; h<part1D.size()-1; h++ ){
                //change only the chosen dimension
                newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h]+1;
                newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h+1]+1;
                newRect.weight = weightPerPart[h+1];
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
SCAI_ASSERT( dbg_rectW==thisRectangle.weight, "Rectangle weights not correct. dbg_rectW= " << dbg_rectW << " , this.weight= "<< thisRectangle.weight);

        }
        numLeaves = root->getNumLeaves();
    }
    
    return root;
    //return root->getAllLeaves();
}
//---------------------------------------------------------------------------------------

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
        projections[l].assign( projLength, 0 );
    }
              
    // calculate projection for local coordinates
    {
        SCAI_REGION("MultiSection.projection.localProjection");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
        
        for(int i=0; i<localN; i++){
            SCAI_REGION_START("MultiSection.projection.localProjection.indexAndCopyCoords");
            const IndexType globalIndex = inputDist->local2global(i);
            std::vector<IndexType> coords = indexToCoords(globalIndex, sideLen, dimension); // check the global index
            //TODO: avoid the conversion to vector<double>
            std::vector<ValueType> coordsVal( coords.begin(), coords.end() );
            SCAI_REGION_END("MultiSection.projection.localProjection.indexAndCopyCoords");
            
            // a pointer to the cell that contains point i
            SCAI_REGION_START("MultiSection.projection.localProjection.contains");
            std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell = treeRoot->contains( coordsVal );
            SCAI_REGION_END("MultiSection.projection.localProjection.contains");
            
            //TODO: in the partition this should not happen. But it may happen in a more general case
            // if this point is not contained in any rectangle
            if( thisRectCell==NULL ){
                continue;
            }
            
            IndexType thisLeafID = thisRectCell->getLeafID();
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0");
            SCAI_ASSERT( thisLeafID<projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];
            IndexType relativeIndex = coords[dim2proj]-thisRectCell->getRect().bottom[dim2proj];

            SCAI_ASSERT( relativeIndex<projections[ thisLeafID ].capacity(), "Wrong relative index: "<< relativeIndex << " should be < "<< projections[ thisLeafID ].capacity() << " (and thisRect.bottom= "<< thisRectCell->getRect().bottom[dim2proj]  << " )" );

            projections[ thisLeafID ][relativeIndex] += localWeights[i];
        }
    }
    // here, the projection of the local points has been calculated
    
    // must sum all local projections from all PEs
    //TODO: sum using one call to comm->sum()
    // data of vector of vectors are not stored continuously. Maybe copy to a large vector and then add
    std::vector<std::vector<ValueType>> globalProj(numLeaves);
    for(int i=0; i<numLeaves; i++){
        SCAI_REGION("MultiSection.projection.sumImpl");
        globalProj[i].assign( projections[i].size() ,0 );
        comm->sumImpl( globalProj[i].data(), projections[i].data(), projections[i].size(), scai::common::TypeTraits<ValueType>::stype);
    }
    
    return globalProj;
}
//---------------------------------------------------------------------------------------

// The non-uniform grid case. Now we take as input the adjacency matrix of a graph and also the coordinates.
   
template<typename IndexType, typename ValueType>
std::shared_ptr<rectCell<IndexType,ValueType>> MultiSection<IndexType, ValueType>::getRectanglesNonUniform( 
    const scai::lama::CSRSparseMatrix<ValueType> &input,
    const std::vector<scai::lama::DenseVector<IndexType>> &coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<ValueType>& minCoords,
    const std::vector<ValueType>& maxCoords,
    Settings settings) {
    SCAI_REGION("MultiSection.getRectanglesNonUniform");
	
    const IndexType k = settings.numBlocks;    
    const IndexType dim = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    
    SCAI_ASSERT( coordinates.size()==dim , "Dimensions given and size of coordinates do not agree." );
    SCAI_ASSERT( minCoords.size()==maxCoords.size() and maxCoords.size()==dim , "Wrong size of maxCoords or minCoords." );
    
    for(int d=0; d<dim; d++){
        SCAI_ASSERT( minCoords[d]<maxCoords[d] , "Minimum coordinates should be less than the maximum coordinates.");
    }
    
    //
    //decide the number of multisection for every dimension
    //

    // from k get d numbers such that their product equals k
    // TODO: now k must be number such that k^(1/d) is an integer, drop this condition, generalize
    const ValueType sqrtK = std::pow( k,  1.0/dim );

    if( !std::floor(sqrtK)==sqrtK ){
        PRINT0("Input k= "<< k << " and sqrt(k)= "<< sqrtK );
        throw std::logic_error("Number of blocks not a square number");
    }
    
    // sqrtK is not correct, it is -1 but not sure if always
    IndexType intSqrtK = sqrtK;
    
    if( std::pow( intSqrtK+1, dim ) == k){
        intSqrtK++;
    }
    SCAI_ASSERT( std::pow( intSqrtK, dim ) == k, "Wrong square root of k. k= "<< k << ", pow(k, 1/d)= " << intSqrtK);
    
    //TODO: now for every dimension we have sqrtK cuts. This can be generalized so we have different number of cuts
    //  for each multisection but even more, different cuts for every block.
    //TODO: maybe if the algorithm dynamically decides in how many parts it will mutlisect each rectangle/block?
    
    // number of cuts for each dimensions
    std::vector<IndexType> numCuts;
    
    // if the bisection option is chosen the algorithm performs a bisection
    if( !settings.multisectionBisect ){
        numCuts = std::vector<IndexType>( dim, intSqrtK );
    }else{        
        SCAI_ASSERT( k && !(k & (k-1)) , "k is not a power of 2 and this is required for now");  
        numCuts = std::vector<IndexType>( log2(k) , 2 );
    }
    
    //
    // initialize the tree
    //
    
    // for all dimensions i: bBox.bottom[i]<bBox.top[i]
    struct rectangle bBox;
    
    // at first the bounding box is the whole space
    for(int d=0; d<dim; d++){
        bBox.bottom.push_back( minCoords[d]);
        bBox.top.push_back( maxCoords[d] );
    }

    // TODO: try to avoid that
    ValueType totalWeight = nodeWeights.sum().scai::lama::Scalar::getValue<ValueType>();
    ValueType averageWeight = totalWeight/k;

    bBox.weight = totalWeight;

    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
    
    IndexType numLeaves = root->getNumLeaves();
    
    //
    //multisect in every dimension
    //
    
    for(typename std::vector<IndexType>::iterator thisDimCuts=numCuts.begin(); thisDimCuts!=numCuts.end(); ++thisDimCuts ){
        SCAI_REGION("MultiSection.getRectanglesNonUniform.forAllRectangles");

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
        
        //TODO: useExtent is the only option. Add another or remove settings.useExtent
        // choose the dimension to project for all leaves/rectangles
        if( settings.multisectionUseExtent or true){
            SCAI_REGION("MultiSection.getRectanglesNonUniform.forAllRectangles.useExtent");
            // for all leaves/rectangles
            for( int l=0; l<allLeaves.size(); l++){
                struct rectangle thisRectangle = allLeaves[l]->getRect();
                maxExtent = 0;
                for(int d=0; d<dim; d++){
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
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projectionNonUniform( coordinates, nodeWeights, root, chosenDim, settings);

        SCAI_ASSERT( projections.size()==numLeaves, "Wrong number of projections"); 
 
        for(int l=0; l<numLeaves; l++){        
            SCAI_REGION("MultiSection.getRectanglesNonUniform.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<ValueType> part1D, weightPerPart;
            std::vector<ValueType> thisProjection = projections[l];
            IndexType thisChosenDim = chosenDim[l];

            //partiD.size() = thisDimCuts-1 , weightPerPart.size = thisDimCuts 
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1D( thisProjection, *thisDimCuts, settings);

            // TODO: possibly expensive assertion
            SCAI_ASSERT( std::accumulate(thisProjection.begin(), thisProjection.end(), 0)==std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0), "Weights are wrong." )
            
            //TODO: make sure that projections[l] and allLeaves[l] refer to the same rectangle
            struct rectangle thisRectangle = allLeaves[l]->getRect();
            
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

            for(int h=0; h<part1D.size()-1; h++ ){
                //change only the chosen dimension
                newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h]+1;
                newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h+1]+1;
                newRect.weight = weightPerPart[h+1];
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
SCAI_ASSERT( dbg_rectW==thisRectangle.weight, "Rectangle weights not correct. dbg_rectW= " << dbg_rectW << " , this.weight= "<< thisRectangle.weight);

        }
        numLeaves = root->getNumLeaves();
    }
    
    std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> ret = root->getAllLeaves();
    SCAI_ASSERT( ret.size()==numLeaves , "Number of leaf nodes not correct, ret.size()= "<< ret.size() << " but numLeaves= "<< numLeaves );

    return root;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> MultiSection<IndexType, ValueType>::projectionNonUniform( 
    const std::vector<scai::lama::DenseVector<IndexType>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
    const std::vector<IndexType>& dimensionToProject,
    Settings settings){
    SCAI_REGION("MultiSection.projectionNonUniform");
    
    const IndexType dimension = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    
    IndexType numLeaves = treeRoot->getNumLeaves();
    
    std::vector<std::vector<ValueType>> projections(numLeaves); // 1 projection per rectangle/leaf
    
    IndexType leafIndex = treeRoot->indexLeaves(0);
    SCAI_ASSERT( numLeaves==leafIndex, "Wrong leaf indexing");
    SCAI_ASSERT( numLeaves==dimensionToProject.size(), "Wrong dimensionToProject vector size.");

    //TODO: pass allLeaves as argument since we already calculate them in getPartition
    
    const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = treeRoot->getAllLeaves();
    SCAI_ASSERT( allLeaves.size()==numLeaves, "Not consistent number of leaf nodes.");
    
    //
    // reserve space for every projection
    //
    for(IndexType l=0; l<numLeaves; l++){
        SCAI_REGION("MultiSection.projectionNonUniform.reserveSpace");
        const IndexType dim2proj = dimensionToProject[l];
        // the length for every projection in the chosen dimension
        IndexType projLength = allLeaves[l]->getRect().top[dim2proj] - allLeaves[l]->getRect().bottom[dim2proj];
        if(projLength<1){
            throw std::runtime_error("function: projectionNonUnifo, line:" +std::to_string(__LINE__) +", the length of the projection is " +std::to_string(projLength) + " and is not correct");
        }
        projections[l].assign( projLength, 0 );
    }
    
    //
    // calculate projection for local coordinates
    //
    {
        SCAI_REGION("MultiSection.projectionNonUniform.localProjection");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
        
        for(int i=0; i<localN; i++){
            SCAI_REGION_START("MultiSection.projectionNonUniform.localProjection.CopyCoords");
            std::vector<IndexType> coords;
            for(int c=0; c<dimension; c++){
                 coords.push_back( coordinates[c].getLocalValues()[i] );
            }
            SCAI_REGION_END("MultiSection.projectionNonUniform.localProjection.CopyCoords");
            
            // a pointer to the cell that contains point i
            SCAI_REGION_START("MultiSection.projectionNonUniform.localProjection.contains");
            std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell = treeRoot->contains( coords );
            SCAI_REGION_END("MultiSection.projectionNonUniform.localProjection.contains");
            
            //TODO: in the partition this should not happen. But it may happen in a more general case
            // if this point is not contained in any rectangle
            if( thisRectCell==NULL ){
                continue;
            }
            
            IndexType thisLeafID = thisRectCell->getLeafID();
            if( thisLeafID==-1 and comm->getRank()==0 ){
                PRINT0( "Owner rectangle for point is ");
                thisRectCell->getRect().print();
                PRINT0( thisRectCell->getLeafID() );
            }
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0 , for coords= "<< coords[0] << ", "<< coords[1] );
            SCAI_ASSERT( thisLeafID<projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];
            IndexType relativeIndex = coords[dim2proj]-thisRectCell->getRect().bottom[dim2proj];

            SCAI_ASSERT( relativeIndex<projections[ thisLeafID ].capacity(), "Wrong relative index: "<< relativeIndex << " should be < "<< projections[ thisLeafID ].capacity() << " (and thisRect.bottom= "<< thisRectCell->getRect().bottom[dim2proj]  << " , thisRect.top= "<< thisRectCell->getRect().top[dim2proj] << ")" );

            projections[ thisLeafID ][relativeIndex] += localWeights[i];
        }
    }
    //
    // sum all local projections from all PEs
    //
    //TODO: sum using one call to comm->sum()
    // data of vector of vectors are not stored continuously. Maybe copy to a large vector and then add
    std::vector<std::vector<ValueType>> globalProj(numLeaves);
    for(int i=0; i<numLeaves; i++){
        SCAI_REGION("MultiSection.projectionNonUniform.sumImpl");
        globalProj[i].assign( projections[i].size() ,0 );
        comm->sumImpl( globalProj[i].data(), projections[i].data(), projections[i].size(), scai::common::TypeTraits<ValueType>::stype);
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
scai::lama::DenseVector<ValueType>  MultiSection<IndexType, ValueType>::convert2Uniform(scai::lama::CSRSparseMatrix<ValueType>& input, std::vector<scai::lama::DenseVector<ValueType>>& coordinates, struct Settings settings){
    SCAI_REGION("MultiSection.convert2Uniform");
    
    const scai::dmemo::DistributionPtr inputDist = input.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    
    const IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();
    
    std::vector<ValueType> minCoords(dimensions, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    
    {
        SCAI_REGION( "MultiSection.initialPartition.minMax" )
        for (IndexType dim = 0; dim < dimensions; dim++) {
            //get local parts of coordinates
            scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[dim].getLocalValues();
            for (IndexType i = 0; i < localN; i++) {
                ValueType coord = localPartOfCoords[i];
                if (coord < minCoords[dim]) minCoords[dim] = coord;
                if (coord > maxCoords[dim]) maxCoords[dim] = coord;
            }
        }

        //communicate to get global min / max
        for (IndexType dim = 0; dim < dimensions; dim++) {
            minCoords[dim] = comm->min(minCoords[dim]);
            maxCoords[dim] = comm->max(maxCoords[dim]);
        }
    }
    
    scai::lama::DenseVector<ValueType> ret;
    
    for(IndexType d=0; d<dimensions; d++){
        IndexType resolution = maxCoords[d] / std::pow(globalN, 1.0/dimensions);
        //PRINT0(resolution);
    }
    
    return ret;
}
//---------------------------------------------------------------------------------------

// Checks if given index is in the bounding box bBox.
template<typename IndexType, typename ValueType>
bool MultiSection<IndexType, ValueType>::inBBox( const std::vector<IndexType>& coords, const struct rectangle& bBox){
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
            if( inBBox(coords, bBox) ){ 
                localWeight += localWeights[i];
            }
        }
    }
    
    // sum all local weights
    return comm->sum(localWeight);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<scai::lama::DenseVector<ValueType>>& coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle& bBox, const std::vector<ValueType> maxCoords, Settings settings){
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
            std::vector<IndexType> coords;
            for(int d=0; d<dimension; d++){
                coords.push_back( coordinates[d].getLocalValues()[i] );
                //TODO: remove assertion, probably not needed
                SCAI_ASSERT( coords.back()<maxCoords[d], "Coordinate too big, coords.back()= " << coords.back() << " , maxCoords[d]= "<< maxCoords[d] );
            }
            if( inBBox(coords, bBox) ){ 
                localWeight += localWeights[i];
            }
        }
    }
    
    // sum all local weights
    return comm->sum(localWeight);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<scai::lama::DenseVector<IndexType>>& coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle& bBox, const std::vector<ValueType> maxCoords, Settings settings){
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
            std::vector<IndexType> coords;
            for(int d=0; d<dimension; d++){
                coords.push_back( coordinates[d].getLocalValues()[i] );
                //TODO: remove assertion, probably not needed
                SCAI_ASSERT( coords.back()<maxCoords[d], "Coordinate too big, coords.back()= " << coords.back() << " , maxCoords[d]= "<< maxCoords[d] );
            }
            if( inBBox(coords, bBox) ){ 
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
    SCAI_REGION("MultiSection.indexTo2D");
    IndexType x = ind/sideLen;
    IndexType y = ind%sideLen;
    
    return std::vector<IndexType>{x, y};
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<IndexType> MultiSection<IndexType, ValueType>::indexTo3D(IndexType ind, IndexType sideLen){
    SCAI_REGION("MultiSection.indexTo3D");
    IndexType planeSize= sideLen*sideLen; // a YxZ plane
    
    IndexType x = ind/planeSize;
    IndexType y = (ind%planeSize)/sideLen;
    IndexType z = (ind%planeSize)%sideLen;
    
    return std::vector<IndexType>{ x, y, z };
}
//---------------------------------------------------------------------------------------

template scai::lama::DenseVector<int> MultiSection<int, double>::getPartitionNonUniform(const scai::lama::CSRSparseMatrix<double> &input, const std::vector<scai::lama::DenseVector<double>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, struct Settings Settings );

template std::shared_ptr<rectCell<int,double>> MultiSection<int, double>::getRectangles( const scai::lama::DenseVector<double>& nodeWeights, const IndexType sideLen, Settings settings);

template std::vector<std::vector<double>> MultiSection<int, double>::projection( const scai::lama::DenseVector<double>& nodeWeights, const std::shared_ptr<rectCell<int,double>> treeRoot, const std::vector<int>& dimensionToProject, const int sideLen, Settings settings);

template std::shared_ptr<rectCell<int,double>> MultiSection<int, double>::getRectanglesNonUniform( const scai::lama::CSRSparseMatrix<double> &input, const std::vector<scai::lama::DenseVector<int>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const std::vector<double>& minCoords, const std::vector<double>& maxCoords, Settings settings);

template std::vector<std::vector<double>> MultiSection<int, double>::projectionNonUniform( const std::vector<scai::lama::DenseVector<int>>& coordinates, const scai::lama::DenseVector<double>& nodeWeights, const std::shared_ptr<rectCell<int,double>> treeRoot, const std::vector<int>& dimensionToProject, Settings settings);

template scai::lama::DenseVector<double>  MultiSection<int, double>::convert2Uniform(scai::lama::CSRSparseMatrix<double> &input, std::vector<scai::lama::DenseVector<double>> &coordinates, struct Settings Settings);
    
template bool MultiSection<int, double>::inBBox( const std::vector<int>& coords, const struct rectangle& bBox);
    
template  std::pair<std::vector<double>,std::vector<double>> MultiSection<int, double>::partition1D( const std::vector<double>& array, const int k, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const int sideLen, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const std::vector<scai::lama::DenseVector<double>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const std::vector<double> maxCoords, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const std::vector<scai::lama::DenseVector<int>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const std::vector<double> maxCoords, Settings settings);

template std::vector<int> MultiSection<int, double>::indexToCoords( const int ind, const int sideLen, const int dim);


};
