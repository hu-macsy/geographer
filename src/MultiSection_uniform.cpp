/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"
#include "GraphUtils.h"

#include <numeric>

namespace ITI {
    
//TODO: Now it works only for k=x^(1/dim) for int x. Handle the general case.
//TODO: Find numbers k1,k2,...,kd such that k1*k2*...*kd=k to perform multisection
//TODO(?): Enforce initial partition and keep track which PEs need to communicate for each projection
//TODO(?): Add an optimal algorithm for 1D partition
//TODO(kind of): Keep in mind semi-structured grids
    

template<typename IndexType, typename ValueType>
std::shared_ptr<rectCell<IndexType,ValueType>> MultiSection<IndexType, ValueType>::getRectangles( const scai::lama::DenseVector<ValueType>& nodeWeights, const IndexType sideLen, struct Settings settings) {
    SCAI_REGION("MultiSection.getRectangles");
	
    const IndexType k = settings.numBlocks;    
    const IndexType dim = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    
    // for all dimensions i: bBox.bottom[i]<bBox.top[i]
    struct rectangle bBox;
    
    // at first the bounding box is the whole space
    for(int d=0; d<dim; d++){
        bBox.bottom.push_back(0);
        bBox.top.push_back(sideLen -1); //WARNING: changes rectangle to be [bot, top], not [bot, top)
    }

    // TODO: try to avoid that, probably not needed
    ValueType totalWeight = nodeWeights.sum();
    ValueType averageWeight = totalWeight/k;

    bBox.weight = totalWeight;

    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
    
    // from k get d numbers such that their product equals k
    // TODO: now k must be number such that k^(1/d) is an integer, drop this condition, generalize
    const ValueType sqrtK = std::pow( k,  1.0/dim );
	
	//TODO: rounding and checking with std::floor does not work for 3 dimensions
	//TODO: check if std::round is OK
    //if( std::floor(sqrtK)!=sqrtK ){
	if( std::abs( std::round(sqrtK) - sqrtK) > 0.000001 ){
		//PRINT0( sqrtK << " != " << std::round(sqrtK) );
        PRINT0("Input k= "<< k << " and sqrt(k)= "<< sqrtK );
        throw std::logic_error("Number of blocks not a square number");
    }
    
    // number of cuts for each dimensions
    std::vector<IndexType> numCuts;
    
    // if the bisection option is chosen the algorithm performs a bisection
    if( settings.bisect==0 ){
        if( settings.cutsPerDim.empty() ){        // no user-specific number of cuts
            IndexType intSqrtK = sqrtK;
            if( std::pow( intSqrtK+1, dim ) == k){
                intSqrtK++;
            }
            SCAI_ASSERT( std::pow( intSqrtK, dim ) == k, "Wrong square root of k. k= "<< k << ", pow( sqrtK, 1/d)= " << std::pow(intSqrtK,dim));
        
            numCuts = std::vector<IndexType>( dim, intSqrtK );
        }else{                                  // user-specific number of cuts per dimensions
            numCuts = settings.cutsPerDim;
        }
    }else if( settings.bisect==1 ){        
        SCAI_ASSERT( k && !(k & (k-1)) , "k is not a power of 2 and this is required for now for bisection");  
        numCuts = std::vector<IndexType>( log2(k) , 2 );
    }
    /* TODO: actually use cutsPerDim
	else if( settings.msOptions==2 ){        
        numCuts = settings.cutsPerDim;
    }else{
        std::cout << "Wrong value " << settings.msOptions << " for option msOptions" << std::endl;
        std::terminate();
    }
    */
    
    IndexType numLeaves = root->getNumLeaves();

    for(typename std::vector<IndexType>::iterator thisDimCuts=numCuts.begin(); thisDimCuts!=numCuts.end(); ++thisDimCuts ){
        SCAI_REGION("MultiSection.getRectangles.forAllRectangles");

        ValueType maxExtent = 0;      

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
        for( IndexType l=0; l<allLeaves.size(); l++){
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
        // in chosenDim we have stored the desired dimension to project for all the leaf nodes

        // a vector of size numLeaves. projections[i] is the projection of leaf/rectangle i in the chosen dimension
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projection( nodeWeights, root, chosenDim, sideLen, settings);

        SCAI_ASSERT( projections.size()==numLeaves, "Wrong number of projections"); 

        for(IndexType l=0; l<numLeaves; l++){        
            SCAI_REGION("MultiSection.getRectangles.forAllRectangles.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<IndexType> part1D;
            std::vector<ValueType> weightPerPart, thisProjection = projections[l];
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DOptimal( thisProjection, *thisDimCuts, settings);

            // TODO: possibly expensive assertion
            SCAI_ASSERT_EQ_ERROR( std::accumulate(thisProjection.begin(), thisProjection.end(), 0.0), std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0.0), "Weights are wrong." );
            
            //TODO: make sure that projections[l] and allLeaves[l] refer to the same rectangle
            struct rectangle thisRectangle = allLeaves[l]->getRect();

            IndexType thisChosenDim = chosenDim[l];
            
            // create the new rectangles and add them to the queue
            struct rectangle newRect;
            newRect.bottom = thisRectangle.bottom;
            newRect.top = thisRectangle.top;
        
            for(IndexType h=0; h<part1D.size()-1; h++ ){
                //change only the chosen dimension
                newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h];
                newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h+1]-1;
                newRect.weight = weightPerPart[h];
                root->insert( newRect );               
            }
            
            //last rectangle
            newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D.back();
            newRect.top = thisRectangle.top;
            newRect.weight = weightPerPart.back();
            root->insert( newRect );

			//TODO: only for debuging, remove variable dbg_rectW
			//SCAI_ASSERT_LE( dbg_rectW-thisRectangle.weight, 0.0000001, "Rectangle weights not correct, their difference is: " << dbg_rectW-thisRectangle.weight);
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
        // a pointer to the cell that contains point i
        std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell;
        struct rectangle thisRect;
        
        for(IndexType i=0; i<localN; i++){
            SCAI_REGION_START("MultiSection.projection.localProjection.indexAndCopyCoords");
            const IndexType globalIndex = inputDist->local2Global(i);
            std::vector<ValueType> coords = indexToCoords<ValueType>(globalIndex, sideLen, dimension); // check the global index
            SCAI_REGION_END("MultiSection.projection.localProjection.indexAndCopyCoords");
            
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
                continue;   
            }
            
            IndexType thisLeafID = thisRectCell->getLeafID();
            thisRect = thisRectCell->getRect();
            
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0");
            SCAI_ASSERT( thisLeafID<projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];
            IndexType relativeIndex = coords[dim2proj]-thisRect.bottom[dim2proj];

            SCAI_ASSERT( relativeIndex<projections[ thisLeafID ].capacity(), "Wrong relative index: "<< relativeIndex << " should be < "<< projections[ thisLeafID ].capacity() << " (and thisRect.bottom= "<< thisRect.bottom[dim2proj]  << " )" );

            projections[ thisLeafID ][relativeIndex] += localWeights[i];
        }
    }
    // here, the projection of the local points has been calculated
    
    // must sum all local projections from all PEs
    //TODO: sum using one call to comm->sum()
    // data of vector of vectors are not stored continuously. Maybe copy to a large vector and then add
    std::vector<std::vector<ValueType>> globalProj(numLeaves);
    for(IndexType i=0; i<numLeaves; i++){
        SCAI_REGION("MultiSection.projection.sumImpl");
        globalProj[i].assign( projections[i].size() ,0 );
        comm->sumImpl( globalProj[i].data(), projections[i].data(), projections[i].size(), scai::common::TypeTraits<ValueType>::stype);
    }
    
    return globalProj;
}
//---------------------------------------------------------------------------------------

template class MultiSection<IndexType, ValueType>;

};
