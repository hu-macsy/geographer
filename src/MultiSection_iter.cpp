/*
 * MultiSection_iterative.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"
#include "GraphUtils.h"
#include "AuxiliaryFunctions.h"

#include <numeric>

namespace ITI {


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::getPartitionIter( 
	const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
	const scai::lama::DenseVector<ValueType>& nodeWeights,
	struct Settings settings ){ 

    SCAI_REGION("MultiSection.getPartitionIter");
    
    std::chrono::time_point<std::chrono::steady_clock> start, afterMultSect;
    start = std::chrono::steady_clock::now();
    
    const scai::dmemo::DistributionPtr inputDistPtr = coordinates[0].getDistributionPtr();
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

    if( k > globalN) {
        throw std::runtime_error("Creating " + std::to_string(k) + " blocks from " + std::to_string(globalN) + " elements is impossible.");
    }

//
// difference with previous approach: do not scale coordinates
//

    std::vector<ValueType> minCoords(dim, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dim, std::numeric_limits<ValueType>::lowest());

    std::tie(minCoords, maxCoords) = aux<IndexType,ValueType>::getGlobalMinMaxCoords( coordinates );

    //
    //convert coordinates to a vector<vector>
    //

	std::vector<point> localPoints( localN, point(dim,0.0) );

	for (IndexType d=0; d<dim; d++) {
    	scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[d].getLocalValues() );
    	for (IndexType i=0; i<localN; i++) {
    		localPoints[i][d] = localPartOfCoords[i];
    	}
    }

    //
    // get a partitioning into rectangles
    //

    std::shared_ptr<rectCell<IndexType,ValueType>> root = MultiSection<IndexType, ValueType>::getRectanglesIter( localPoints, nodeWeights, minCoords, maxCoords, settings);
    
    const IndexType numLeaves = root->getNumLeaves();
    
    SCAI_ASSERT( numLeaves==k , "Returned number of rectangles is not equal k, rectangles.size()= " << numLeaves << " and k= "<< k );
    
    return MultiSection<IndexType, ValueType>::setPartition( root, inputDistPtr, coordinates);


}//getPartitionIter


template<typename IndexType, typename ValueType>
std::shared_ptr<rectCell<IndexType,ValueType>> MultiSection<IndexType, ValueType>::getRectanglesIter( 
    //const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<point>& localPoints,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<ValueType>& minCoords,
    const std::vector<ValueType>& maxCoords,
    Settings settings) {

	const IndexType k = settings.numBlocks;    
    const IndexType dim = settings.dimensions;

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    //const IndexType globalN = inputDist->getGlobalSize();
    
    SCAI_ASSERT_EQ_ERROR( localPoints.size(), localN , "Size of coordinates vector is not right" );
    SCAI_ASSERT_EQ_ERROR( localPoints[0].size(),dim ,"Dimensions given and size of coordinates do not agree." );
    SCAI_ASSERT( minCoords.size()==maxCoords.size() and maxCoords.size()==dim , "Wrong size of maxCoords or minCoords." );
    
    for(int d=0; d<dim; d++){
        SCAI_ASSERT( minCoords[d]<maxCoords[d] , "Minimum coordinates should be less than the maximum coordinates.");
    }

    //
    //decide the number of multisection for every dimension
    //
    
    //TODO: maybe if the algorithm dynamically decides in how many parts it will multisect each rectangle/block?
    
    // number of cuts for each dimensions
    std::vector<IndexType> numCuts;

    // if the bisection option is chosen the algorithm performs a bisection
    if( settings.bisect==0 ){
        if( settings.cutsPerDim.empty() ){        // no user-specific number of cuts
            // from k get d numbers such that their product equals k
            // TODO: now k must be number such that k^(1/d) is an integer, drop this condition, generalize
            ValueType sqrtK = std::pow( k, 1.0/dim );
            IndexType intSqrtK = sqrtK;
			//PRINT( sqrtK << " _ " << intSqrtK );            
            // TODO/check: sqrtK is not correct, it is -1 but not sure if always
            
            if( std::pow( intSqrtK+1, dim ) == k){
                intSqrtK++;
            }
            SCAI_ASSERT_EQ_ERROR( std::pow( intSqrtK, dim ), k, "Wrong square root of k. k= "<< k << ", sqrtK= " << sqrtK << ", intSqrtK " << intSqrtK );
            numCuts = std::vector<IndexType>( dim, intSqrtK );
        }else{                                  // user-specific number of cuts per dimensions
            numCuts = settings.cutsPerDim;
        }
    }else{        
        SCAI_ASSERT( k && !(k & (k-1)) , "k is not a power of 2 and this is required for now for bisection");  
        numCuts = std::vector<IndexType>( log2(k) , 2 );
    }

	SCAI_ASSERT_EQ_ERROR( numCuts.size(), dim, "Wrong dimensions or vector size.");
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

    ValueType totalWeight = nodeWeights.sum();

    bBox.weight = totalWeight;
    
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );
    
    IndexType numLeaves = root->getNumLeaves();

    //
    //multisect in every dimension
    //
    
    for(typename std::vector<IndexType>::iterator thisDimCuts=numCuts.begin(); thisDimCuts!=numCuts.end(); ++thisDimCuts ){
        SCAI_REGION("MultiSection.getRectanglesNonUniform.forAllRectangles");

PRINT0("about to cut into " << *thisDimCuts);
        /*Two ways to find in which dimension to project:
         * 1) just pick the dimension of the bounding box that has the largest extent and then project: only one projection
         * 2) project in every dimension and pick the one in which the difference between the maximum and minimum value is the smallest: d projections
         * TODO: maybe we can change (2) and calculate the variance of the projection and pick the one with the biggest
         * */

        // a vector with pointers to all the leaf nodes of the tree
        std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = root->getAllLeaves();        
        SCAI_ASSERT( allLeaves.size()==numLeaves, "Wrong number of leaves.");
     
        //TODO: since this is done locally, we can also get the 1D partition in every dimension and choose the best one
        //      maybe not the fastest way but probably would give better quality


        //std::vector<ValueType> maxExtents( numLeaves,0 ); //store the extent for every leaf

        std::vector<IndexType> chosenDim ( numLeaves, -1); //the chosen dim to project for every leaf

        //the hyperplane coordinate for every leaf in the chosen dimension
		std::vector<std::vector<ValueType>> hyperplanes( numLeaves, (std::vector<ValueType> (*thisDimCuts+1,0)) ); 
        
        // choose the dimension to project for each leaf/rectangle
        for( IndexType l=0; l<numLeaves; l++){
            struct rectangle thisRectangle = allLeaves[l]->getRect();
            ValueType maxExtent =0;
            for(int d=0; d<dim; d++){
                ValueType extent = thisRectangle.top[d] - thisRectangle.bottom[d];
                if( extent>maxExtent ){
                    maxExtent = extent;
                    chosenDim[l] = d;
                }
            }
            ValueType meanHyperplaneOffset = maxExtent/ *thisDimCuts;
            for( int c=1; c<*thisDimCuts; c++){
            	hyperplanes[l][c] = hyperplanes[l][c-1] + meanHyperplaneOffset;
            }
        }
        // in chosenDim we have stored the desired dimension to project for all the leaf nodes

        
        //do{
			// a vector of size numLeaves. projections[i] is the projection of leaf/rectangle i in the chosen dimension
    	    std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projectionIter( localPoints, nodeWeights, root, allLeaves, hyperplanes, chosenDim, settings);

        	SCAI_ASSERT( projections.size()==numLeaves, "Wrong number of projections"); 

        	//- adapt hyperplanes according to imbalanced rectangles

        //while( imbalance<settings.epsilon or numIter==maxIter)


	}

}//getRectanglesIter


//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> MultiSection<IndexType, ValueType>::projectionIter( 
	//const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
	const std::vector<point> &coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
    const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>>& allLeaves,
    const std::vector<std::vector<ValueType>>& hyperplanes,
    const std::vector<IndexType>& dimensionToProject,
    Settings settings){

	const IndexType dimension = settings.dimensions;
    
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    
    const IndexType numLeaves = treeRoot->getNumLeaves();
    SCAI_ASSERT( numLeaves>0, "Zero or negative number of leaves.")
    
    IndexType leafIndex = treeRoot->indexLeaves(0);
    SCAI_ASSERT( numLeaves==leafIndex, "Wrong leaf indexing");
    SCAI_ASSERT( numLeaves==dimensionToProject.size(), "Wrong dimensionToProject vector size.");
    SCAI_ASSERT( numLeaves==hyperplanes.size(), "Wrong hyperplanes vector size.");

    //const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = treeRoot->getAllLeaves();
    SCAI_ASSERT( allLeaves.size()==numLeaves, "Not consistent number of leaf nodes.");    

    //
    // reserve space for every projection
	const IndexType numCuts = hyperplanes[0].size();
	std::vector<std::vector<ValueType>> projections( numLeaves, std::vector<ValueType>(numCuts, 0.0) ); // 1 projection per rectangle/leaf

	//
    // calculate projection for local coordinates
    //
    {
  		SCAI_REGION("MultiSection.projectionIter.localProjection");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );
        std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell;
        
        for(IndexType i=0; i<localN; i++){
            SCAI_REGION_START("MultiSection.projectionIter.localProjection.getContainingLeaf");
            
            // if this point is not contained in any rectangle
            //TODO: in the partition this should not happen. But it may happen in a more general case
            try{

//this expects a point; maybe better convert coordinates from vector<DenseVector> to vector<vector>, as before but without scaling them

                thisRectCell = treeRoot->getContainingLeaf( coordinates[i] );
            }
            catch( const std::logic_error& e){
                PRINT("Function getContainingLeaf returns an " << e.what() << " exception for point: ");
                for( int d=0; d<dimension; d++)
                    std::cout<< coordinates[i][d] << ", ";
                std::cout<< std::endl << " and root:"<< std::endl;
                treeRoot->getRect().print(std::cout);
                std::terminate();   // not allowed in our case
            }
            SCAI_REGION_END("MultiSection.projectionIter.localProjection.getContainingLeaf");
            
            IndexType thisLeafID = thisRectCell->getLeafID();
            
            //print some info if somethibg went wrong
            if( thisLeafID==-1 and comm->getRank()==0 ){
                PRINT0( "Owner rectangle for point is ");
                thisRectCell->getRect().print(std::cout);
                PRINT0( thisRectCell->getLeafID() );
            }
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0 , for coords= "<< coordinates[i][0] << ", "<< coordinates[i][1] );
            SCAI_ASSERT( thisLeafID<projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];
            
            //IndexType relativeIndex = coordinates[i][dim2proj]-thisRectCell->getRect().bottom[dim2proj];
            typename std::vector<ValueType>::iterator upBound = std::upper_bound(hyperplanes[thisLeafID].begin(), hyperplanes[thisLeafID].end(), coordinates[i][dim2proj] );
            IndexType relativeIndex = (upBound-hyperplanes[thisLeafID].begin())-1;
SCAI_ASSERT_LE( coordinates[i][dim2proj], hyperplanes[thisLeafID][relativeIndex], "Wrong relative index" );

            SCAI_ASSERT( relativeIndex<=projections[thisLeafID].capacity(), "Wrong relative index: "<< relativeIndex << " should be <= "<< projections[ thisLeafID ].capacity() << " (and thisRect.bottom= "<< thisRectCell->getRect().bottom[dim2proj]  << " , thisRect.top= "<< thisRectCell->getRect().top[dim2proj] << ")" );

            projections[thisLeafID][relativeIndex] += localWeights[i];
        }    
	}	

}//projectionIter

template class MultiSection<IndexType, ValueType>;

}//ITI