/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"
#include "ParcoRepart.h"    //TODO: this is here for compute cut and imbalance. not really needed

#include <numeric>

namespace ITI {
    
//TODO: Now it works only for k=x^(1/dim) for int x. Handle the general case.
//TODO: Find numbers k1,k2,...,kd such that k1*k2*...*kd=k to perform multisection
//TODO(?): Enforce initial partition and keep track which PEs need to communicate for each projection
//TODO(?): Add an optimal algorithm for 1D partition
//TODO(kind of): Keep in mind semi-structured grids
    


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::getPartitionNonUniform(const scai::lama::CSRSparseMatrix<ValueType> &input, const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, struct Settings settings ){ 
    SCAI_REGION("MultiSection.getPartition");
    
    std::chrono::time_point<std::chrono::steady_clock> start, afterMultSect;
    start = std::chrono::steady_clock::now();
    
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
    //TODO: assert if scaledMin and scaledMax are always 0 and scale respectively
    std::vector<scai::lama::DenseVector<IndexType>> scaledCoords(dim);
    
    // localPoints[i].size()= dimensions
    std::vector< std::vector<IndexType> > localPoints( localN, std::vector<IndexType>(dim,0) );
    
    for(IndexType d=0; d<dim; d++){
        scaledCoords[d].allocate(inputDistPtr);
        scaledCoords[d] = static_cast<ValueType>( 0 );
    }
    
    // scale= N^(1/d): this way the scaled max is N^(1/d) and this is also the maximum size of the projection arrays
    ValueType scale = std::pow( globalN /*WARNING*/ -1 , 1.0/dim);
PRINT0( scale );
    {
        SCAI_REGION( "MultiSection.getPartitionNonUniform.minMaxAndScale" )
        
        for (IndexType d = 0; d < dim; d++) {
            //get local parts of coordinates
            scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[d].getLocalValues() );
            for (IndexType i = 0; i < localN; i++) {
                ValueType coord = localPartOfCoords[i];
                if (coord < minCoords[d]) minCoords[d] = coord;
                if (coord > maxCoords[d]) maxCoords[d] = coord;
            }
        }
        
        //  communicate to get global min / max
        //
        for (IndexType d = 0; d < dim; d++) {
            SCAI_REGION( "MultiSection.getPartitionNonUniform.minMaxAndScale.minMax" )
            minCoords[d] = comm->min(minCoords[d]);
            maxCoords[d] = comm->max(maxCoords[d]);
            scaledMax[d] = int(scale) ;
            scaledMin[d] = 0;
        }
        
        PRINT0("max coord= " << *std::max_element(maxCoords.begin(), maxCoords.end() ) << "  and max scaled coord= " << *std::max_element(scaledMax.begin(), scaledMax.end() ) );
        
        for (IndexType d = 0; d < dim; d++) {
            
            ValueType thisDimScale = scale/(maxCoords[d]-minCoords[d]);
            ValueType thisDimMin = minCoords[d];
            scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[d].getLocalValues() );
            
            /*
             * TODO: this code scales the coordinates a little bit faster but there is some problem in the rounding afterwards
            
            SCAI_REGION_START("MultiSection.getPartitionNonUniform.minMaxAndScale.lamaScale" )
            scaledCoords[d] = (coordinates[d] - scai::lama::DenseVector<ValueType>( coordinates[d].getDistributionPtr(), minCoords[d]) );
            scaledCoords[d] = scaledCoords[d] * thisDimScale;
            SCAI_REGION_END("MultiSection.getPartitionNonUniform.minMaxAndScale.lamaScale" )
            */
            
            //SCAI_REGION_START("MultiSection.getPartitionNonUniform.minMaxAndScale.byHandScale" )
            //get local parts of coordinates
            scai::hmemo::WriteOnlyAccess<IndexType> wScaledCoord( scaledCoords[d].getLocalValues() );
            
            SCAI_ASSERT( localN==wScaledCoord.size() , "Wrong size of local part.");
            
            for (IndexType i = 0; i < localN; i++) {
                ValueType normalizedCoord = localPartOfCoords[i] - thisDimMin;
                IndexType scaledCoord =  normalizedCoord * thisDimScale; 
                wScaledCoord[i] = scaledCoord;
                //
                localPoints[i][d] = scaledCoord;
                //
                SCAI_ASSERT( scaledCoord >=0 and scaledCoord<=scale, "Wrong scaled coordinate " << scaledCoord << " is either negative or more than "<< scale);
            }
            //SCAI_REGION_END("MultiSection.getPartitionNonUniform.minMaxAndScale.byHandScale" )
        }
    }
    
    for (IndexType d=0; d<dim; d++) {
        SCAI_ASSERT( scaledMax[d]<= scale   , "Scaled maximum value "<< scaledMax[d] << " is too large. should be less than " << scale );
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
    std::shared_ptr<rectCell<IndexType,ValueType>> root = MultiSection<IndexType, ValueType>::getRectanglesNonUniform( input, localPoints, nodeWeights, scaledMin, scaledMax, settings);
    
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
            /*
            std::vector<IndexType> point(dim);            
            //TODO: probably not the most efficient way to do it, maybe avoid ReadAccess
            for(IndexType d=0; d<dim; d++){
                scai::hmemo::ReadAccess<IndexType> coordAccess( scaledCoords[d].getLocalValues() );
                coordAccess.getValue( point[d], i);
            } 
            */
            try{
                wLocalPart[i] = root->getContainingLeaf( localPoints[i] )->getLeafID();
            }catch( const std::logic_error& e ){
                PRINT0( e.what() );
                std::terminate();
            }
        }
    }
    /*
    ValueType cut = ParcoRepart<IndexType, ValueType>::computeCut( input, partition, false);
    ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(partition, k);
    if (comm->getRank() == 0) {
        afterMultSect = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = afterMultSect-start;
        if( settings.bisect ){
            std::cout << "\033[1;36mWith Bisection (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
        }else{
            std::cout << "\033[1;36mWith MultiSection (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
        }
        std::cout<< "and imbalance= "<< imbalance << "\033[0m" << std::endl;
    }
    */
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
        bBox.top.push_back(sideLen -1); //WARNING: changes rectangle to be [bot, top], not [bot, top)
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
        if( settings.useExtent or true){
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
            SCAI_REGION("MultiSection.getRectangles.forAllRectangles.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<IndexType> part1D;
            std::vector<ValueType> weightPerPart, thisProjection = projections[l];
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DOptimal( thisProjection, *thisDimCuts, settings);
 
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
        
            for(int h=0; h<part1D.size()-1; h++ ){
                //change only the chosen dimension
                newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h];
                newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h+1]-1;
                newRect.weight = weightPerPart[h];
                root->insert( newRect );
dbg_rectW += newRect.weight;                
            }
            
            //last rectangle
            newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D.back();
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
        SCAI_ASSERT( dim2proj>=0 and dim2proj<=dimension , "Wrong dimension to project to: " << dim2proj);
        
        // the length for every projection in the chosen dimension
        IndexType projLength = allLeaves[l]->getRect().top[dim2proj] - allLeaves[l]->getRect().bottom[dim2proj] /*WARNING*/ +1; 

        if(projLength<2){
            throw std::runtime_error("function: projection, line:" +std::to_string(__LINE__) +", the length of projection/leaf " + std::to_string( l) +" is " +std::to_string(projLength) + " and is not correct. Number of leaves = " + std::to_string(numLeaves) );
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
            std::vector<ValueType> coords = indexToCoords<ValueType>(globalIndex, sideLen, dimension); // check the global index
            SCAI_REGION_END("MultiSection.projection.localProjection.indexAndCopyCoords");
            
            // a pointer to the cell that contains point i
            //SCAI_REGION_START("MultiSection.projection.localProjection.getContainingLeaf");
            std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell;
            
            //TODO: in the partition this should not happen. But it may happen in a more general case
            // if this point is not contained in any rectangle
            try{
                SCAI_REGION("MultiSection.projection.localProjection.contains");
                thisRectCell = treeRoot->getContainingLeaf( coords );
            }
            catch( const std::logic_error& e){
                PRINT(*comm <<": Function getContainingLeaf returns an " << e.what() << " exception");
                for( int d=0; d<dimension; d++)
                    std::cout<< coords[d] << ", ";
                std::cout<< std::endl;
                //std::cout<< std::endl << " and root:"<< std::endl;
                //treeRoot->getRect().print();
                continue;   
                //std::terminate();   // not allowed in our case
            }
            //SCAI_REGION_END("MultiSection.projection.localProjection.getContainingLeaf");
            
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
    const std::vector< std::vector<IndexType> > &coordinates,
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
    
    SCAI_ASSERT( coordinates.size()==localN , "Size of coordinatred vector is not right" );
    SCAI_ASSERT( coordinates[0].size()==dim ,"Dimensions given and size of coordinates do not agree." );
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
    
    // TODO/check: sqrtK is not correct, it is -1 but not sure if always
    IndexType intSqrtK = sqrtK;
    
    //TODO: now for every dimension we have sqrtK cuts. This can be generalized so we have different number of cuts
    //  for each multisection but even more, different cuts for every block.
    //TODO: maybe if the algorithm dynamically decides in how many parts it will mutlisect each rectangle/block?
    
    // number of cuts for each dimensions
    std::vector<IndexType> numCuts;
    
    // if the bisection option is chosen the algorithm performs a bisection
    if( !settings.bisect ){
        if( std::pow( intSqrtK+1, dim ) == k){
            intSqrtK++;
        }
        SCAI_ASSERT( std::pow( intSqrtK, dim ) == k, "Wrong square root of k. k= "<< k << ", pow( sqrtK, 1/d)= " << std::pow(intSqrtK,dim));
    
        numCuts = std::vector<IndexType>( dim, intSqrtK );
    }else{        
        SCAI_ASSERT( k && !(k & (k-1)) , "k is not a power of 2 and this is required for now for bisection");  
        numCuts = std::vector<IndexType>( log2(k) , 2 );
    }
    
    //
    // initialize the tree
    //
    
    // for all dimensions i: bBox.bottom[i]<bBox.top[i]
    struct rectangle bBox;
    
    // at first the bounding box is the whole space
    // WARNING: because the max coordinate of a bounding box does belong in the box ( box=[min,manx) ) we must +1.
    for(int d=0; d<dim; d++){
        bBox.bottom.push_back( minCoords[d]);
        bBox.top.push_back( maxCoords[d] );
    }

    // TODO: try to avoid that
    ValueType totalWeight = nodeWeights.sum().scai::lama::Scalar::getValue<ValueType>();
    ValueType averageWeight = totalWeight/k;

    bBox.weight = totalWeight;
if( comm->getRank()==0 ){
    PRINT("");    
    bBox.print();    
}
    
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
        if( settings.useExtent or true){
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
            SCAI_REGION("MultiSection.getRectanglesNonUniform.forAllRectangles.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<IndexType> part1D;
            std::vector<ValueType> weightPerPart, thisProjection = projections[l];
            IndexType thisChosenDim = chosenDim[l];            

//for(int rr=0; rr<thisProjection.size(); rr++)  PRINT0( rr << ": " << thisProjection[rr] );

            //part1D.size() = *thisDimCuts , weightPerPart.size = *thisDimCuts 
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DOptimal( thisProjection, *thisDimCuts, settings);
            
//for(int k=0; k<part1D.size(); k++)    PRINT0("dim= " << thisChosenDim << " , thisDimCuts= " << *thisDimCuts << " :: " << part1D[k] );

            SCAI_ASSERT( part1D.size()== *thisDimCuts , "Wrong size of 1D partition")
            SCAI_ASSERT( weightPerPart.size()== *thisDimCuts , "Wrong size of 1D partition")
            
//for(int i=0; i<*thisDimCuts; i++)    PRINT0(i<< ": " << part1D[i] << " ++ " << weightPerPart[i] );

            // TODO: possibly expensive assertion
            SCAI_ASSERT( std::accumulate(thisProjection.begin(), thisProjection.end(), 0)==std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0), "Weights are wrong, totalWeight of thisProjection= "  << std::accumulate(thisProjection.begin(), thisProjection.end(), 0) << " , total weight of weightPerPart= " << std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0) );
            
            //TODO: make sure that projections[l] and allLeaves[l] refer to the same rectangle
            struct rectangle thisRectangle = allLeaves[l]->getRect();
            ValueType thisRectWeight = thisRectangle.weight;
            ValueType optWeight = thisRectWeight/(*thisDimCuts);
            ValueType maxWeight = 0;
            
            // create the new rectangles and add them to the queue
ValueType dbg_rectW=0;
            struct rectangle newRect;
            newRect.bottom = thisRectangle.bottom;
            newRect.top = thisRectangle.top;

            for(int h=0; h<part1D.size()-1; h++ ){
                //change only the chosen dimension
                newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h];
                newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h+1]-1;
                newRect.weight = weightPerPart[h];
                root->insert( newRect );
                if(newRect.weight>maxWeight){
                    maxWeight = newRect.weight;
                }
if( comm->getRank()==0 ) newRect.print();
dbg_rectW += newRect.weight;                
            }
            
            //last rectangle
            newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D.back();
            newRect.top = thisRectangle.top;
            newRect.weight = weightPerPart.back();
            root->insert( newRect );
            if(newRect.weight>maxWeight){
                maxWeight = newRect.weight;
            }
if( comm->getRank()==0 )  newRect.print();            
dbg_rectW += newRect.weight;    
            
PRINT0("this rect imbalance= " << (maxWeight-optWeight)/optWeight << "  (opt= " << optWeight << " , max= "<< maxWeight << ")" );

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
    const std::vector<std::vector<IndexType>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
    const std::vector<IndexType>& dimensionToProject,
    Settings settings){
    SCAI_REGION("MultiSection.projectionNonUniform");
    
    const IndexType dimension = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    
    const IndexType numLeaves = treeRoot->getNumLeaves();
    SCAI_ASSERT( numLeaves>0, "Zero or negative number of leaves.")
    
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
        IndexType projLength = allLeaves[l]->getRect().top[dim2proj] - allLeaves[l]->getRect().bottom[dim2proj]  /*WARNING*/  +1;
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
            /*
            SCAI_REGION_START("MultiSection.projectionNonUniform.localProjection.CopyCoords");
            std::vector<IndexType> coords(dimension);
            for(int c=0; c<dimension; c++){
                 coords[c] = coordinates[c].getLocalValues()[i] ;               
            }
            SCAI_REGION_END("MultiSection.projectionNonUniform.localProjection.CopyCoords");
            */
            // a pointer to the cell that contains point i
            SCAI_REGION_START("MultiSection.projectionNonUniform.localProjection.getContainingLeaf");
            std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell;
            
            //TODO: in the partition this should not happen. But it may happen in a more general case
            // if this point is not contained in any rectangle
            try{
                thisRectCell = treeRoot->getContainingLeaf( coordinates[i] );
            }
            catch( const std::logic_error& e){
                PRINT("Function getContainingLeaf returns an " << e.what() << " exception for point: ");
                for( int d=0; d<dimension; d++)
                    std::cout<< coordinates[i][d] << ", ";
                std::cout<< std::endl << " and root:"<< std::endl;
                treeRoot->getRect().print();
                std::terminate();   // not allowed in our case
            }
            SCAI_REGION_END("MultiSection.projectionNonUniform.localProjection.getContainingLeaf");
            
            IndexType thisLeafID = thisRectCell->getLeafID();
            if( thisLeafID==-1 and comm->getRank()==0 ){
                PRINT0( "Owner rectangle for point is ");
                thisRectCell->getRect().print();
                PRINT0( thisRectCell->getLeafID() );
                // terminate() ??
            }
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0 , for coords= "<< coordinates[i][0] << ", "<< coordinates[i][1] );
            SCAI_ASSERT( thisLeafID<projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];
            IndexType relativeIndex = coordinates[i][dim2proj]-thisRectCell->getRect().bottom[dim2proj];

            SCAI_ASSERT( relativeIndex<=projections[ thisLeafID ].capacity(), "Wrong relative index: "<< relativeIndex << " should be <= "<< projections[ thisLeafID ].capacity() << " (and thisRect.bottom= "<< thisRectCell->getRect().bottom[dim2proj]  << " , thisRect.top= "<< thisRectCell->getRect().top[dim2proj] << ")" );

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
        SCAI_ASSERT( i<globalProj.size() and i<projections.size() , "Index too large");
        
        globalProj[i].assign( projections[i].size(), 0 );
        comm->sumImpl( globalProj[i].data(), projections[i].data(), projections[i].size(), scai::common::TypeTraits<ValueType>::stype);
    }
    
    return globalProj;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>, std::vector<ValueType>> MultiSection<IndexType, ValueType>::partition1DGreedy( const std::vector<ValueType>& projection, const IndexType k, Settings settings){
    SCAI_REGION("MultiSection.partition1DGreedy");
    
    const IndexType dimension = settings.dimensions;
    
    ValueType totalWeight = std::accumulate(projection.begin(), projection.end(), 0);
    ValueType averageWeight = totalWeight/k;
  
    if(projection.size()==0){
        throw std::runtime_error( "In MultiSection::partition1DGreedy, input projection vector is empty");
    }

    std::vector<IndexType> partHyperplanes(k,-9);
    std::vector<ValueType> weightPerPart(k,-9);
    
    partHyperplanes[0] = 0;
    IndexType part = 1;
    ValueType thisPartWeight = 0;
    
    // greedy 1D partition (a 2-approx solution?)
    for(int i=0; i<projection.size(); i++){
        if( part>k) break;
        //SCAI_ASSERT( part<k , "Index " << part << " too big, must be < " << k << " and i= " << i );
        thisPartWeight += projection[i];
        if( thisPartWeight > averageWeight ){
            SCAI_ASSERT(part < partHyperplanes.size(), "index: "<< part << " too big, must be < "<< partHyperplanes.size() )
            // choose between keeping the projection[i] in the sum, having something more than the average
            // or do not add projection[i] and get something below average
            //if( 2*(averageWeight-thisPartWeight)+projection[i] < 0 ){
                partHyperplanes[part]= i;
                // calculate new total weight left and new average weight
                totalWeight = totalWeight - thisPartWeight + projection[i];
                weightPerPart[part-1] = thisPartWeight - projection[i];                
                --i;
            /*
        }else{  // choose solution that is more than the average
                partHyperplanes[part]= i;
                // calculate new total weight left and new average weight
                totalWeight = totalWeight - thisPartWeight;
                weightPerPart[part] = thisPartWeight;
            }
            */
            averageWeight = totalWeight/(k-part);
            thisPartWeight = 0;
            ++part;
        }
    }
   
    weightPerPart[k-1] = totalWeight;

    return std::make_pair(partHyperplanes, weightPerPart);
}
//---------------------------------------------------------------------------------------
// Based on algorithm Nicol found in Pinar, Aykanat, 2004, "Fast optimal load balancing algorithms for 1D partitioning"
//TODO: In the same paper thers is a better, but more complicated, algorithm called Nicol+

template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>, std::vector<ValueType>> MultiSection<IndexType, ValueType>::partition1DOptimal( const std::vector<ValueType>& nodeWeights, const IndexType k, Settings settings){
    
    const IndexType N = nodeWeights.size();

    //
    //create the prefix sum array
    //
    std::vector<ValueType> prefixSum( N+1 , 0);
    
    prefixSum[0] = 0;// nodeWeights[0];
    
    for(IndexType i=1; i<N+1; i++ ){
        prefixSum[i] = prefixSum[i-1] + nodeWeights[i-1];
    }
    
    ValueType totalWeight = prefixSum.back();

    ValueType lowerBound, upperBound;
    lowerBound = totalWeight/k;         // the optimal average weight
    upperBound = totalWeight;
  
    std::vector<IndexType> partIndices(k, -9);
    std::vector<ValueType> weightPerPart(k, -9);
    partIndices[0]=0;
    
    for(IndexType p=1; p<k; p++){
        //IndexType indexLow = p==0 ? 0 : partIndices[p-1];
        IndexType indexLow = partIndices[p-1];
        IndexType indexHigh = N;
        while( indexLow<indexHigh ){
            IndexType indexMid = (indexLow+indexHigh)/2;
            ValueType tmpSum = prefixSum[indexMid] - prefixSum[std::max(partIndices[p-1],0)];
//PRINT("lB= " << lowerBound << " , uB= " << upperBound << " __ indexLow= "<< indexLow << " mid= "<< indexMid << " indexHigh = " << indexHigh );              
//PRINT(p << ": " << tmpSum);            
            if( lowerBound<=tmpSum and tmpSum<upperBound){
                if( probe(prefixSum, k, tmpSum) ){
                    indexHigh = indexMid;
                    upperBound = tmpSum;
                }else{
                    indexLow = indexMid+1;
                    lowerBound = tmpSum;
                }
            }else if(tmpSum>=upperBound){
                indexHigh = indexMid;
            }else{
                indexLow=indexMid+1;
            }
        }
        
        partIndices[p] = indexHigh;      
        weightPerPart[p-1] = prefixSum[indexHigh] - prefixSum[std::max(partIndices[p-1],0)];
//PRINT(p << " :: "<< partIndices[p] << " __ "<< weightPerPart[p-1] );  
    }
//PRINT(prefixSum[ partIndices.back()-1 ]);    
    weightPerPart[k-1] = totalWeight - prefixSum[ partIndices.back() ];
    
    return std::make_pair(partIndices, weightPerPart);
}
//---------------------------------------------------------------------------------------
// Search if there is a partition of the weights array into k parts where the maximum weight of a part is <=target.

//TODO: return also the splitters found
template<typename IndexType, typename ValueType>
bool MultiSection<IndexType, ValueType>::probe(const std::vector<ValueType>& prefixSum, const IndexType k, const ValueType target){

    const IndexType N = prefixSum.size();
    IndexType p = 1;
    const IndexType offset = N/k;
    IndexType step = offset;
    ValueType sumOfPartition = target;
    
    ValueType totalWeight = prefixSum.back();

    bool ret = false;
    
    if(target*k >= totalWeight){
        std::vector<IndexType> splitters( k-1, 0);

        while( p<k and sumOfPartition<totalWeight){
            while( prefixSum[step]<sumOfPartition and step<N){
                step += offset;
                step = std::min( step , N-1);
                SCAI_ASSERT( step<N , "Variable step is too large: " << step);
            }

            splitters[p-1] = std::lower_bound( prefixSum.begin()+(step-offset), prefixSum.begin()+step, sumOfPartition ) - prefixSum.begin();

            sumOfPartition = prefixSum[splitters[p-1]] + target;
            ++p;
        }

        if( sumOfPartition>=totalWeight ){
            ret = true;
        }
    }
    
    return ret;
}
//---------------------------------------------------------------------------------------

// Checks if given index is in the bounding box bBox.
template<typename IndexType, typename ValueType>
template<typename T>
bool MultiSection<IndexType, ValueType>::inBBox( const std::vector<T>& coords, const struct rectangle& bBox){
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
        if(coords[i]>top[i] or coords[i]<bottom[i]){
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
            std::vector<IndexType> coords = indexToCoords<IndexType>(globalIndex, sideLen, dimension); // check the global index
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
template<typename T>
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<scai::lama::DenseVector<T>>& coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle& bBox, const std::vector<ValueType> maxCoords, Settings settings){
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
            std::vector<T> coords;
            for(int d=0; d<dimension; d++){
                coords.push_back( coordinates[d].getLocalValues()[i] );
                //TODO: remove assertion, probably not needed
                SCAI_ASSERT( coords.back()<=maxCoords[d], "Coordinate too big, coords.back()= " << coords.back() << " , maxCoords[d]= "<< maxCoords[d] );
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
template<typename T>
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<std::vector<T>>& coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle& bBox, const std::vector<ValueType> maxCoords, Settings settings){
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
            std::vector<T> coords= coordinates[i];
            /*
            for(int d=0; d<dimension; d++){
                coords.push_back( coordinates[d].getLocalValues()[i] );
                //TODO: remove assertion, probably not needed
                SCAI_ASSERT( coords.back()<=maxCoords[d], "Coordinate too big, coords.back()= " << coords.back() << " , maxCoords[d]= "<< maxCoords[d] );
            }
            */
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
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexToCoords(const IndexType ind, const IndexType sideLen, const IndexType dim){
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
        return  MultiSection<IndexType, ValueType>::indexTo2D<T>( ind, sideLen);
    }else if(dim==3){
        return MultiSection<IndexType, ValueType>::indexTo3D<T>( ind, sideLen);
    }else{
        throw std::runtime_error("function: indexToCoords, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }
    
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexTo2D(IndexType ind, IndexType sideLen){
    SCAI_REGION("MultiSection.indexTo2D");
    T x = ind/sideLen;
    T y = ind%sideLen;
    
    return std::vector<T>{x, y};
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexTo3D(IndexType ind, IndexType sideLen){
    SCAI_REGION("MultiSection.indexTo3D");
    IndexType planeSize= sideLen*sideLen; // a YxZ plane

    T x = ind/planeSize;
    T y = (ind%planeSize)/sideLen;
    T z = (ind%planeSize)%sideLen;
    
    return std::vector<T>{ x, y, z };
}
//---------------------------------------------------------------------------------------

template scai::lama::DenseVector<int> MultiSection<int, double>::getPartitionNonUniform(const scai::lama::CSRSparseMatrix<double> &input, const std::vector<scai::lama::DenseVector<double>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, struct Settings Settings );

template std::shared_ptr<rectCell<int,double>> MultiSection<int, double>::getRectangles( const scai::lama::DenseVector<double>& nodeWeights, const IndexType sideLen, Settings settings);

template std::vector<std::vector<double>> MultiSection<int, double>::projection( const scai::lama::DenseVector<double>& nodeWeights, const std::shared_ptr<rectCell<int,double>> treeRoot, const std::vector<int>& dimensionToProject, const int sideLen, Settings settings);

template std::shared_ptr<rectCell<int,double>> MultiSection<int, double>::getRectanglesNonUniform( const scai::lama::CSRSparseMatrix<double> &input, const std::vector<std::vector<int>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const std::vector<double>& minCoords, const std::vector<double>& maxCoords, Settings settings);

template std::vector<std::vector<double>> MultiSection<int, double>::projectionNonUniform( const std::vector<std::vector<int>>& coordinates, const scai::lama::DenseVector<double>& nodeWeights, const std::shared_ptr<rectCell<int,double>> treeRoot, const std::vector<int>& dimensionToProject, Settings settings);
    
template bool MultiSection<int, double>::inBBox( const std::vector<int>& coords, const struct rectangle& bBox);
    
template  std::pair<std::vector<int>,std::vector<double>> MultiSection<int, double>::partition1DGreedy( const std::vector<double>& array, const int k, Settings settings);

template  std::pair<std::vector<int>,std::vector<double>> MultiSection<int, double>::partition1DOptimal( const std::vector<double>& array, const int k, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const int sideLen, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const std::vector<scai::lama::DenseVector<int>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const std::vector<double> maxCoords, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const std::vector<scai::lama::DenseVector<double>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const std::vector<double> maxCoords, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const std::vector<std::vector<double>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const std::vector<double> maxCoords, Settings settings);

template double MultiSection<int, double>::getRectangleWeight( const std::vector<std::vector<int>> &coordinates, const scai::lama::DenseVector<double>& nodeWeights, const struct rectangle& bBox, const std::vector<double> maxCoords, Settings settings);

template std::vector<int> MultiSection<int, double>::indexToCoords( const int ind, const int sideLen, const int dim);

template bool MultiSection<int, double>::probe(const std::vector<double>& prefixSum, const int k, const double target);

};
