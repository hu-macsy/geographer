/*
 * MultiSection.cpp
 *
 *  Created on: 11.04.2017
 */

#include "MultiSection.h"
#include "GraphUtils.h"
#include "AuxiliaryFunctions.h"

#include <numeric>

namespace ITI {

//TODO: Now it works only for k=x^(1/dim) for int x. Handle the general case.
//TODO: Find numbers k1,k2,...,kd such that k1*k2*...*kd=k to perform multisection
//TODO(?): Enforce initial partition and keep track which PEs need to communicate for each projection
//TODO(?): Add an optimal algorithm for 1D partition
//TODO(kind of): Keep in mind semi-structured grids
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::computePartition(
    const scai::lama::CSRSparseMatrix<ValueType> &input,
    const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    struct Settings settings ) {

    const scai::dmemo::DistributionPtr inputDistPtr = input.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDistPtr->getCommunicatorPtr();

    const IndexType k = settings.numBlocks;
    const IndexType dim = settings.dimensions;
    const IndexType globalN = inputDistPtr->getGlobalSize();
    const IndexType localN = inputDistPtr->getLocalSize();

    //
    // check input arguments for sanity
    //
    if( coordinates.size()!=dim ) {
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
    /*
    std::vector<ValueType> minCoords(dim, std::numeric_limits<ValueType>::max());
    std::vector<ValueType> maxCoords(dim, std::numeric_limits<ValueType>::lowest());
    std::tie(minCoords, maxCoords) = aux<IndexType,ValueType>::getGlobalMinMaxCoords( coordinates );
    */
    //get global min and max
    std::vector<ValueType> minCoords(dim);
    std::vector<ValueType> maxCoords(dim);
    for (int d=0; d<dim; d++) {
        minCoords[d] = coordinates[d].min();
        maxCoords[d] = coordinates[d].max();
    }

    if( settings.useIter ) { //in this case, do not scale coords
        std::vector<point> localPoints( localN, point(dim,0.0) );

        for (IndexType d = 0; d < dim; d++) {
            scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[d].getLocalValues() );
            for (IndexType i=0; i<localN; i++) {
                localPoints[i][d] = localPartOfCoords[i];
            }
        }
        return computePartition( input, localPoints, nodeWeights, minCoords, maxCoords, settings );

    } else if( not settings.useIter) {
        std::vector<intPoint> intLocalPoints( localN, intPoint(dim,0) );

        ValueType scale = std::pow( globalN /*WARNING*/ -1, 1.0/dim);
        std::vector<IndexType> scaledMin(dim, 0);
        std::vector<IndexType> scaledMax(dim, scale);

        PRINT0("max coord= " << *std::max_element(maxCoords.begin(), maxCoords.end() ) << "  and max scaled coord= " << *std::max_element(scaledMax.begin(), scaledMax.end() ) );

        for (IndexType d = 0; d < dim; d++) {
            scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[d].getLocalValues() );

            ValueType thisDimScale = scale/(maxCoords[d]-minCoords[d]);
            for (IndexType i = 0; i < localN; i++) {
                ValueType normalizedCoord = localPartOfCoords[i] - minCoords[d];
                IndexType scaledCoord =  normalizedCoord * thisDimScale;
                intLocalPoints[i][d] = scaledCoord;
                SCAI_ASSERT( scaledCoord >=0 and scaledCoord<=scale, "Wrong scaled coordinate " << scaledCoord << " is either negative or more than "<< scale);
            }
        }
        return computePartition( input, intLocalPoints, nodeWeights, scaledMin, scaledMax, settings );

    } else {
        PRINT0("Currently, only supporting IndexType and ValueType for coordinate type.\nAborting");
        throw std::runtime_error("Not supported data type");
    }

}


template<typename IndexType, typename ValueType>
template<typename T>
scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::computePartition(
    const scai::lama::CSRSparseMatrix<ValueType>& input,
    const std::vector<std::vector<T>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<T> &minCoords,
    const std::vector<T> &maxCoords,
    struct Settings settings ) {
    SCAI_REGION("MultiSection.computePartition");

    std::chrono::time_point<std::chrono::steady_clock> start, afterMultSect;
    start = std::chrono::steady_clock::now();

    const scai::dmemo::DistributionPtr inputDistPtr = input.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDistPtr->getCommunicatorPtr();

    const IndexType k = settings.numBlocks;
    const IndexType dim = settings.dimensions;
    const IndexType globalN = inputDistPtr->getGlobalSize();
    const IndexType localN = inputDistPtr->getLocalSize();

    //
    // get a partitioning into rectangles
    //

    std::shared_ptr<rectCell<IndexType,ValueType>> root = MultiSection<IndexType, ValueType>::getRectanglesNonUniform( input, coordinates, nodeWeights, minCoords, maxCoords, settings);

    const IndexType numLeaves = root->getNumLeaves();

    SCAI_ASSERT( numLeaves==k, "Returned number of rectangles is not equal k, rectangles.size()= " << numLeaves << " and k= "<< k );

    return MultiSection<IndexType, ValueType>::setPartition( root, inputDistPtr, coordinates);
}

//---------------------------------------------------------------------------------------

// The non-uniform grid case. Now we take as input the adjacency matrix of a graph and also the coordinates.

template<typename IndexType, typename ValueType>
template<typename T>
std::shared_ptr<rectCell<IndexType,ValueType>> MultiSection<IndexType, ValueType>::getRectanglesNonUniform(
            const scai::lama::CSRSparseMatrix<ValueType>& input,
            const std::vector<std::vector<T>>& coordinates,
            const scai::lama::DenseVector<ValueType>& nodeWeights,
            const std::vector<T>& minCoords,
            const std::vector<T>& maxCoords,
Settings settings) {
    SCAI_REGION("MultiSection.getRectanglesNonUniform");

    const IndexType k = settings.numBlocks;
    const IndexType dim = settings.dimensions;

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();
    //const IndexType globalN = inputDist->getGlobalSize();

    SCAI_ASSERT_EQ_ERROR( coordinates.size(), localN, "Size of coordinates vector is not right" );
    SCAI_ASSERT_EQ_ERROR( coordinates[0].size(),dim,"Dimensions given and size of coordinates do not agree." );
    SCAI_ASSERT( minCoords.size()==maxCoords.size() and maxCoords.size()==dim, "Wrong size of maxCoords or minCoords." );

    for(int d=0; d<dim; d++) {
        SCAI_ASSERT_LT_ERROR( minCoords[d], maxCoords[d], "Minimum coordinates should be less than the maximum coordinates.");
    }

    //
    //decide the number of multisection for every dimension
    //

    //TODO: now for every dimension we have sqrtK cuts. This can be generalized so we have different number of cuts
    //  for each multisection but even more, different cuts for every block.
    //TODO: maybe if the algorithm dynamically decides in how many parts it will multisect each rectangle/block?

    // number of cuts for each dimensions
    std::vector<IndexType> numCuts;

    // if the bisection option is chosen the algorithm performs a bisection
    if( settings.bisect==0 ) {
        if( settings.cutsPerDim.empty() ) {       // no user-specific number of cuts
            // from k get d numbers such that their product equals k
            // TODO: now k must be number such that k^(1/d) is an integer, drop this condition, generalize
            ValueType sqrtK = std::pow( k, 1.0/dim );
            IndexType intSqrtK = sqrtK;
            //PRINT( sqrtK << " _ " << intSqrtK );
            // TODO/check: sqrtK is not correct, it is -1 but not sure if always

            if( std::pow( intSqrtK+1, dim ) == k) {
                intSqrtK++;
            }
            SCAI_ASSERT_EQ_ERROR( std::pow( intSqrtK, dim ), k, "Wrong square root of k. k= "<< k << ", sqrtK= " << sqrtK << ", intSqrtK " << intSqrtK );
            numCuts = std::vector<IndexType>( dim, intSqrtK );
        } else {                                 // user-specific number of cuts per dimensions
            numCuts = settings.cutsPerDim;
        }
        SCAI_ASSERT_EQ_ERROR( numCuts.size(), dim, "Wrong dimensions or vector size.");
    } else {
        SCAI_ASSERT( k && !(k & (k-1)), "k is not a power of 2 and this is required for now for bisection");
        numCuts = std::vector<IndexType>( log2(k), 2 );
    }

    //
    // initialize the tree
    //

    // for all dimensions i: bBox.bottom[i]<bBox.top[i]
    struct rectangle<ValueType> bBox;

    // at first the bounding box is the whole space
    for(int d=0; d<dim; d++) {
        bBox.bottom.push_back( minCoords[d]);
        bBox.top.push_back( maxCoords[d] );
    }

    // TODO: try to avoid that
    ValueType totalWeight = nodeWeights.sum();
    bBox.weight = totalWeight;

    //if(comm->getRank()==0)  bBox.print( std::cout );

    // create the root of the tree that contains the whole grid
    std::shared_ptr<rectCell<IndexType,ValueType>> root( new rectCell<IndexType,ValueType>(bBox) );

    if( not settings.useIter ) {
        SCAI_ASSERT( (std::is_same<T,IndexType>::value), "IndexType is required for the non-iterative approach" );
        MultiSection<IndexType, ValueType>::projectAnd1Dpartition( root, coordinates, nodeWeights, numCuts, maxCoords );
    } else if (settings.useIter) {
        //TODO: this is not necessary, we can have the iterative approach with IndexType coords
        //SCAI_ASSERT( std::is_same<T,ValueType>::value, "ValueType is for the non-iterative approach" );
        MultiSection<IndexType, ValueType>::iterativeProjectionAndPart( root, coordinates, nodeWeights, numCuts, settings );
    } else {
        PRINT0("Currently, only supporting IndexType and ValueType for coordinate type.\nAborting");
        throw std::runtime_error("Not supported data type");
    }

    const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> &ret = root->getAllLeaves();
    IndexType numLeaves = root->getNumLeaves();
    SCAI_ASSERT( ret.size()==numLeaves, "Number of leaf nodes not correct, ret.size()= "<< ret.size() << " but numLeaves= "<< numLeaves );

    return root;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
IndexType MultiSection<IndexType, ValueType>::projectAnd1Dpartition(
    std::shared_ptr<rectCell<IndexType,ValueType>>& root,
    const std::vector<std::vector<T>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<IndexType>& numCuts,
    const std::vector<T>& maxCoords) {

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType dim = coordinates[0].size();
    IndexType numLeaves = root->getNumLeaves();

    //
    //multisect in every dimension
    //

    for(typename std::vector<IndexType>::const_iterator thisDimCuts=numCuts.begin(); thisDimCuts!=numCuts.end(); ++thisDimCuts ) {
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

        std::vector<IndexType> chosenDim ( numLeaves, -1); //the chosen dim to project for every leaf

        //the hyperplane coordinate for every leaf in the chosen dimension
        //this is used only in the iterative approach
        //std::vector<std::vector<ValueType>> hyperplanes( numLeaves, (std::vector<ValueType> (*thisDimCuts+1,0)) );

        // choose the dimension to project for each leaf/rectangle
        for( IndexType l=0; l<allLeaves.size(); l++) {
            struct rectangle<ValueType> thisRectangle = allLeaves[l]->getRect();
            ValueType maxExtent = 0;
            for(int d=0; d<dim; d++) {
                ValueType extent = thisRectangle.top[d] - thisRectangle.bottom[d];
                if( extent>maxExtent ) {
                    maxExtent = extent;
                    chosenDim[l] = d;
                }
            }
        }

        // in chosenDim we have stored the desired dimension to project for all the leaf nodes

        // a vector of size numLeaves. projections[i] is the projection of leaf/rectangle i in the chosen dimension

        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projectionNonUniform( coordinates, nodeWeights, root, chosenDim);

        SCAI_ASSERT_EQ_ERROR( projections.size(), numLeaves, "Wrong number of projections");
        PRINT0("numLeaves= " << numLeaves);

        for(IndexType l=0; l<numLeaves; l++) {
            SCAI_REGION("MultiSection.getRectanglesNonUniform.forAllRectangles.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<IndexType> part1D;
            std::vector<ValueType> weightPerPart, thisProjection = projections[l];
            IndexType thisChosenDim = chosenDim[l];

            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DOptimal( thisProjection, *thisDimCuts);
            SCAI_ASSERT( part1D.size()== *thisDimCuts, "Wrong size of 1D partition")
            SCAI_ASSERT( weightPerPart.size()== *thisDimCuts, "Wrong size of 1D partition")

            // TODO: possibly expensive assertion
            SCAI_ASSERT( std::accumulate(thisProjection.begin(), thisProjection.end(), 0.0)==std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0.0), "Weights are wrong for leaf "<< l << ": totalWeight of thisProjection= "  << std::accumulate(thisProjection.begin(), thisProjection.end(), 0.0) << " , total weight of weightPerPart= " << std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0.0) );

            //TODO: make sure that projections[l] and allLeaves[l] refer to the same rectangle
            struct rectangle<ValueType> thisRectangle = allLeaves[l]->getRect();

            ValueType optWeight = thisRectangle.weight/(*thisDimCuts);
            ValueType maxWeight = 0;

            // create the new rectangles and add them to the queue
            //ValueType dbg_rectW=0;
            struct rectangle<ValueType> newRect;
            newRect.bottom = thisRectangle.bottom;
            newRect.top = thisRectangle.top;

            for(IndexType h=0; h<part1D.size()-1; h++ ) {
                //change only the chosen dimension
                newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h];
                newRect.top[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D[h+1]-1;
                newRect.weight = weightPerPart[h];
                root->insert( newRect );
                SCAI_ASSERT_GT_ERROR( newRect.weight, 0, "Aborting: found rectangle with 0 weight, in leaf " << l << " , creating rectangle number " << h << " for hyperplane " <<part1D[h] << ". Maybe inappropriate input data or needs bigger scaling.");
                if(newRect.weight>maxWeight) {
                    maxWeight = newRect.weight;
                }
                //dbg_rectW += newRect.weight;
                //if(comm->getRank()==0) newRect.print(std::cout);
                //PRINT0("this rect imbalance= " << (newRect.weight-optWeight)/optWeight << "  (opt= " << optWeight << " , myWeight= "<< newRect.weight << ")" );
            }

            //last rectangle
            SCAI_ASSERT_LE_ERROR(part1D.back(), maxCoords[thisChosenDim], "Partition hyperplane bigger than max coordinate. Probaby too dense data to find solution." );
            newRect.bottom[thisChosenDim] = thisRectangle.bottom[thisChosenDim]+part1D.back();
            newRect.top = thisRectangle.top;
            newRect.weight = weightPerPart.back();
            SCAI_ASSERT_GT_ERROR( newRect.weight, 0, "Found rectangle with 0 weight, maybe inappropriate input data or needs bigger scaling of the coordinates (aka refinement) to find suitable hyperplane).");
            root->insert( newRect );
            if(newRect.weight>maxWeight) {
                maxWeight = newRect.weight;
            }
            //dbg_rectW += newRect.weight;
            //if(comm->getRank()==0) newRect.print(std::cout);
            //PRINT0("this rect imbalance= " << (newRect.weight-optWeight)/optWeight << "  (opt= " << optWeight << " , myWeight= "<< newRect.weight << ")" );

            //TODO: only for debuging, remove variable dbg_rectW
            //SCAI_ASSERT_LE_ERROR( dbg_rectW-thisRectangle.weight, 0.0000001, "Rectangle weights not correct: dbg_rectW-this.weight= " << dbg_rectW - thisRectangle.weight);
        }
        numLeaves = root->getNumLeaves();
        PRINT0("numLeaves= " << numLeaves);
    }

    return numLeaves;
}//projectAnd1Dpartition

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
std::vector<std::vector<ValueType>> MultiSection<IndexType, ValueType>::projectionNonUniform(
                                     const std::vector<std::vector<T>>& coordinates,
                                     const scai::lama::DenseVector<ValueType>& nodeWeights,
                                     const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
const std::vector<IndexType>& dimensionToProject) {
    SCAI_REGION("MultiSection.projectionNonUniform");

    const IndexType dimension = coordinates[0].size();

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();

    const IndexType numLeaves = treeRoot->getNumLeaves();
    SCAI_ASSERT( numLeaves>0, "Zero or negative number of leaves.")

    IndexType leafIndex = treeRoot->indexLeaves(0);
    SCAI_ASSERT( numLeaves==leafIndex, "Wrong leaf indexing");
    SCAI_ASSERT( numLeaves==dimensionToProject.size(), "Wrong dimensionToProject vector size.");

    //TODO: pass allLeaves as argument since we already calculate them in computePartition

    const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = treeRoot->getAllLeaves();
    SCAI_ASSERT( allLeaves.size()==numLeaves, "Not consistent number of leaf nodes.");

    //
    // reserve space for every projection
    //
    std::vector<std::vector<ValueType>> projections(numLeaves); // 1 projection per rectangle/leaf

    for(IndexType l=0; l<numLeaves; l++) {
        SCAI_REGION("MultiSection.projectionNonUniform.reserveSpace");
        const IndexType dim2proj = dimensionToProject[l];
        // the length for every projection in the chosen dimension
        IndexType projLength = allLeaves[l]->getRect().top[dim2proj] - allLeaves[l]->getRect().bottom[dim2proj]  /*WARNING*/  +1;
        if(projLength<1) {
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
        std::shared_ptr<rectCell<IndexType,ValueType>> thisRectCell;

        for(IndexType i=0; i<localN; i++) {
            SCAI_REGION_START("MultiSection.projectionNonUniform.localProjection.getContainingLeaf");

            //TODO: in the partition this should not happen. But it may happen in a more general case
            // if this point is not contained in any rectangle
            try {
                thisRectCell = treeRoot->getContainingLeaf( coordinates[i] );
            }
            catch( const std::logic_error& e) {
                PRINT("Function getContainingLeaf returns an " << e.what() << " exception for point: ");
                for( int d=0; d<dimension; d++)
                    std::cout<< coordinates[i][d] << ", ";
                std::cout<< std::endl << " and root:"<< std::endl;
                treeRoot->getRect().print(std::cout);
                std::terminate();   // not allowed in our case
            }
            SCAI_REGION_END("MultiSection.projectionNonUniform.localProjection.getContainingLeaf");

            IndexType thisLeafID = thisRectCell->getLeafID();

            //print some info if somethibg went wrong
            if( thisLeafID==-1 and comm->getRank()==0 ) {
                PRINT0( "Owner rectangle for point is ");
                thisRectCell->getRect().print(std::cout);
                PRINT0( thisRectCell->getLeafID() );
            }
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0 , for coords= "<< coordinates[i][0] << ", "<< coordinates[i][1] );
            SCAI_ASSERT( thisLeafID<projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];
            IndexType relativeIndex = coordinates[i][dim2proj]-thisRectCell->getRect().bottom[dim2proj];

            SCAI_ASSERT( relativeIndex<=projections[ thisLeafID ].capacity(), "Wrong relative index: "<< relativeIndex << " should be <= "<< projections[ thisLeafID ].capacity() << " (and thisRect.bottom= "<< thisRectCell->getRect().bottom[dim2proj]  << " , thisRect.top= "<< thisRectCell->getRect().top[dim2proj] << ")" );

            projections[thisLeafID][relativeIndex] += localWeights[i];
        }
    }
    //
    // sum all local projections from all PEs
    //
    //TODO: sum using one call to comm->sum()
    // data of vector of vectors are not stored continuously. Maybe copy to a large vector and then add
    std::vector<std::vector<ValueType>> globalProj(numLeaves);

    for(IndexType i=0; i<numLeaves; i++) {
        SCAI_REGION("MultiSection.projectionNonUniform.sumImpl");
        SCAI_ASSERT( i<globalProj.size() and i<projections.size(), "Index too large");

        globalProj[i].assign( projections[i].size(), 0 );
        comm->sumImpl( globalProj[i].data(), projections[i].data(), projections[i].size(), scai::common::TypeTraits<ValueType>::stype);
    }

    return globalProj;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::pair<std::vector<IndexType>, std::vector<ValueType>> MultiSection<IndexType, ValueType>::partition1DGreedy( const std::vector<ValueType>& projection, const IndexType k, Settings settings) {
    SCAI_REGION("MultiSection.partition1DGreedy");

    ValueType totalWeight = std::accumulate(projection.begin(), projection.end(), 0.0);
    ValueType averageWeight = totalWeight/k;

    if(projection.size()==0) {
        throw std::runtime_error( "In MultiSection::partition1DGreedy, input projection vector is empty");
    }

    std::vector<IndexType> partHyperplanes(k,-9);
    std::vector<ValueType> weightPerPart(k,-9);

    partHyperplanes[0] = 0;
    IndexType part = 1;
    ValueType thisPartWeight = 0;

    // greedy 1D partition (a 2-approx solution?)
    for(IndexType i=0; i<projection.size(); i++) {
        if( part>=k) break;
        thisPartWeight += projection[i];
        if( thisPartWeight > averageWeight ) {
            SCAI_ASSERT(part <= partHyperplanes.size(), "index: "<< part << " too big, must be < "<< partHyperplanes.size() )

            // choose between keeping the projection[i] in the sum, having something more than the average
            // or do not add projection[i] and get something below average

            if( thisPartWeight-averageWeight> averageWeight-(thisPartWeight-projection[i]) ) {  //solution without projection[i]
                partHyperplanes[part]= i;
                // calculate new total weight left
                totalWeight = totalWeight - (thisPartWeight-projection[i]);
                weightPerPart[part-1] = thisPartWeight-projection[i];
                --i;
            } else { // choose solution that is more than the average
                partHyperplanes[part]= i+1;
                // calculate new total weight left
                totalWeight = totalWeight - thisPartWeight;
                weightPerPart[part-1] = thisPartWeight;
            }
            SCAI_ASSERT_UNEQUAL( k,part, "Division with 0");
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
std::pair<std::vector<IndexType>, std::vector<ValueType>> MultiSection<IndexType, ValueType>::partition1DOptimal(
            const std::vector<ValueType>& nodeWeights,
const IndexType k) {

    const IndexType N = nodeWeights.size();

    //
    //create the prefix sum array
    //
    std::vector<ValueType> prefixSum( N+1, 0.0);

    prefixSum[0] = IndexType(0);// nodeWeights[0];

    for(IndexType i=1; i<N+1; i++ ) {
        prefixSum[i] = prefixSum[i-1] + nodeWeights[i-1];
    }

    ValueType totalWeight = prefixSum.back();

    ValueType lowerBound, upperBound;
    lowerBound = totalWeight/k;         // the optimal average weight
    upperBound = totalWeight;

    std::vector<IndexType> partIndices(k, IndexType(-9));
    std::vector<ValueType> weightPerPart(k, IndexType(-9) );
    partIndices[0]=0;

    for(IndexType p=1; p<k; p++) {
        IndexType indexLow = partIndices[p-1];
        IndexType indexHigh = N;
        while( indexLow<indexHigh ) {

            IndexType indexMid = (indexLow+indexHigh)/2;
            ValueType tmpSum = prefixSum[indexMid] - prefixSum[std::max(partIndices[p-1], IndexType(0) )];

            if( lowerBound<=tmpSum and tmpSum<upperBound) {
                if( probe(prefixSum, k, tmpSum) ) {
                    indexHigh = indexMid;
                    upperBound = tmpSum;
                } else {
                    indexLow = indexMid+1;
                    lowerBound = tmpSum;
                }
            } else if(tmpSum>=upperBound) {
                indexHigh = indexMid;
            } else {
                indexLow=indexMid+1;
            }

        }
        partIndices[p] = indexHigh-1;
        weightPerPart[p-1] = prefixSum[indexHigh-1] - prefixSum[std::max(partIndices[p-1], IndexType(0) )];
    }

    weightPerPart[k-1] = totalWeight - prefixSum[ partIndices.back() ];

    return std::make_pair(partIndices, weightPerPart);
}
//---------------------------------------------------------------------------------------

// Search if there is a partition of the weights array into k parts where the maximum weight of a part is <=target.

//TODO: return also the splitters found
template<typename IndexType, typename ValueType>
bool MultiSection<IndexType, ValueType>::probe(const std::vector<ValueType>& prefixSum, const IndexType k, const ValueType target) {

    const IndexType N = prefixSum.size();
    IndexType p = 1;
    const IndexType offset = N/k;
    IndexType step = offset;
    ValueType sumOfPartition = target;

    ValueType totalWeight = prefixSum.back();

    bool ret = false;

    if(target*k >= totalWeight) {
        std::vector<IndexType> splitters( k-1, 0);

        while( p<k and sumOfPartition<totalWeight) {
            while( prefixSum[step]<sumOfPartition and step<N) {
                step += offset;
                step = std::min( step, N-1);
                SCAI_ASSERT( step<N, "Variable step is too large: " << step);
            }
            IndexType spliter =std::lower_bound( prefixSum.begin()+(step-offset), prefixSum.begin()+step, sumOfPartition ) - prefixSum.begin()-1;
            splitters[p-1] = spliter;

            sumOfPartition = prefixSum[splitters[p-1]] + target;
            ++p;
        }

        if( sumOfPartition>=totalWeight ) {
            ret = true;
        }
    }

    return ret;
}
//---------------------------------------------------------------------------------------
// Search if there is a partition of the weights array into k parts where the maximum weight of a part is <=target.

template<typename IndexType, typename ValueType>
std::pair<bool,std::vector<IndexType>> MultiSection<IndexType, ValueType>::probeAndGetSplitters(const std::vector<ValueType>& prefixSum, const IndexType k, const ValueType target) {

    //const IndexType N = prefixSum.size();
    IndexType p = 1;
    ValueType sumOfPartition = target;

    ValueType totalWeight = prefixSum.back();

    bool ret = false;
    std::vector<IndexType> spliters( k-1, 0);

    if(target*k >= totalWeight) {
        while( p<k and sumOfPartition<totalWeight) {
            IndexType spliter = std::lower_bound( prefixSum.begin(), prefixSum.end(), sumOfPartition ) - prefixSum.begin() -1;
            spliters[p-1] = spliter;

            sumOfPartition = prefixSum[spliter] + target;
            ++p;
        }

        if( sumOfPartition>=totalWeight ) {
            ret = true;
        }
    }
    return std::make_pair(ret,spliters);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::setPartition(
    std::shared_ptr<rectCell<IndexType,ValueType>> root,
    const scai::dmemo::DistributionPtr distPtr,
    const std::vector<std::vector<T>>& localPoints) {
    SCAI_REGION("MultiSection.setPartition");

    const IndexType localN = distPtr->getLocalSize();
    const scai::dmemo::CommunicatorPtr comm = distPtr->getCommunicatorPtr();

    scai::hmemo::HArray<IndexType> localPartition;

    scai::hmemo::WriteOnlyAccess<IndexType> wLocalPart(localPartition, localN);

    for(IndexType i=0; i<localN; i++) {
        try {
            wLocalPart[i] = root->getContainingLeaf( localPoints[i] )->getLeafID();
        } catch( const std::logic_error& e ) {
            PRINT0( e.what() );
            std::terminate();
        }
    }
    wLocalPart.release();

    scai::lama::DenseVector<IndexType> partition(distPtr, std::move(localPartition));

    return partition;
}
//---------------------------------------------------------------------------------------
//for the uniform grid case

template<typename IndexType, typename ValueType>
std::shared_ptr<rectCell<IndexType,ValueType>> MultiSection<IndexType, ValueType>::getRectangles( const scai::lama::DenseVector<ValueType>& nodeWeights, const IndexType sideLen, struct Settings settings) {
    SCAI_REGION("MultiSection.getRectangles");

    const IndexType k = settings.numBlocks;
    const IndexType dim = settings.dimensions;

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();

    // for all dimensions i: bBox.bottom[i]<bBox.top[i]
    struct rectangle<ValueType> bBox;

    // at first the bounding box is the whole space
    for(int d=0; d<dim; d++) {
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
    if( std::abs( std::round(sqrtK) - sqrtK) > 0.000001 ) {
        //PRINT0( sqrtK << " != " << std::round(sqrtK) );
        PRINT0("Input k= "<< k << " and sqrt(k)= "<< sqrtK );
        throw std::logic_error("Number of blocks not a square number");
    }

    // number of cuts for each dimensions
    std::vector<IndexType> numCuts;

    // if the bisection option is chosen the algorithm performs a bisection
    if( settings.bisect==0 ) {
        if( settings.cutsPerDim.empty() ) {       // no user-specific number of cuts
            IndexType intSqrtK = sqrtK;
            if( std::pow( intSqrtK+1, dim ) == k) {
                intSqrtK++;
            }
            SCAI_ASSERT( std::pow( intSqrtK, dim ) == k, "Wrong square root of k. k= "<< k << ", pow( sqrtK, 1/d)= " << std::pow(intSqrtK,dim));

            numCuts = std::vector<IndexType>( dim, intSqrtK );
        } else {                                 // user-specific number of cuts per dimensions
            numCuts = settings.cutsPerDim;
        }
    } else if( settings.bisect==1 ) {
        SCAI_ASSERT( k && !(k & (k-1)), "k is not a power of 2 and this is required for now for bisection");
        numCuts = std::vector<IndexType>( log2(k), 2 );
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

    for(typename std::vector<IndexType>::iterator thisDimCuts=numCuts.begin(); thisDimCuts!=numCuts.end(); ++thisDimCuts ) {
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
        for( IndexType l=0; l<allLeaves.size(); l++) {
            struct rectangle<ValueType> thisRectangle = allLeaves[l]->getRect();
            maxExtent = 0;
            for(int d=0; d<dim; d++) {
                ValueType extent = thisRectangle.top[d] - thisRectangle.bottom[d];
                if( extent>maxExtent ) {
                    maxExtent = extent;
                    chosenDim[l] = d;
                }
            }
        }
        // in chosenDim we have stored the desired dimension to project for all the leaf nodes

        // a vector of size numLeaves. projections[i] is the projection of leaf/rectangle i in the chosen dimension
        std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projection( nodeWeights, root, chosenDim, sideLen, settings);

        SCAI_ASSERT( projections.size()==numLeaves, "Wrong number of projections");

        for(IndexType l=0; l<numLeaves; l++) {
            SCAI_REGION("MultiSection.getRectangles.forAllRectangles.createRectanglesAndPush");
            //perform 1D partitioning for the chosen dimension
            std::vector<IndexType> part1D;
            std::vector<ValueType> weightPerPart, thisProjection = projections[l];
            std::tie( part1D, weightPerPart) = MultiSection<IndexType, ValueType>::partition1DOptimal( thisProjection, *thisDimCuts);

            // TODO: possibly expensive assertion
            SCAI_ASSERT_EQ_ERROR( std::accumulate(thisProjection.begin(), thisProjection.end(), 0.0), std::accumulate( weightPerPart.begin(), weightPerPart.end(), 0.0), "Weights are wrong." );

            //TODO: make sure that projections[l] and allLeaves[l] refer to the same rectangle
            struct rectangle<ValueType> thisRectangle = allLeaves[l]->getRect();

            IndexType thisChosenDim = chosenDim[l];

            // create the new rectangles and add them to the queue
            struct rectangle<ValueType> newRect;
            newRect.bottom = thisRectangle.bottom;
            newRect.top = thisRectangle.top;

            for(IndexType h=0; h<part1D.size()-1; h++ ) {
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
//for the uniform grid case

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> MultiSection<IndexType, ValueType>::projection(const scai::lama::DenseVector<ValueType>& nodeWeights, const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot, const std::vector<IndexType>& dimensionToProject, const IndexType sideLen, Settings settings) {
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

    //TODO: pass allLeaves as argument since we already calculate them in computePartition

    const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = treeRoot->getAllLeaves();
    SCAI_ASSERT( allLeaves.size()==numLeaves, "Not consistent number of leaf nodes.");

    // reserve space for every projection
    for(IndexType l=0; l<numLeaves; l++) {
        SCAI_REGION("MultiSection.projection.reserveSpace");
        const IndexType dim2proj = dimensionToProject[l];
        SCAI_ASSERT( dim2proj>=0 and dim2proj<=dimension, "Wrong dimension to project to: " << dim2proj);

        // the length for every projection in the chosen dimension
        IndexType projLength = allLeaves[l]->getRect().top[dim2proj] - allLeaves[l]->getRect().bottom[dim2proj] /*WARNING*/ +1;

        if(projLength<2) {
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
        struct rectangle<ValueType> thisRect;

        for(IndexType i=0; i<localN; i++) {
            SCAI_REGION_START("MultiSection.projection.localProjection.indexAndCopyCoords");
            const IndexType globalIndex = inputDist->local2Global(i);
            std::vector<ValueType> coords = indexToCoords<ValueType>(globalIndex, sideLen, dimension); // check the global index
            SCAI_REGION_END("MultiSection.projection.localProjection.indexAndCopyCoords");

            //TODO: in the partition this should not happen. But it may happen in a more general case
            // if this point is not contained in any rectangle
            try {
                SCAI_REGION("MultiSection.projection.localProjection.contains");
                thisRectCell = treeRoot->getContainingLeaf( coords );
            }
            catch( const std::logic_error& e) {
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
    for(IndexType i=0; i<numLeaves; i++) {
        SCAI_REGION("MultiSection.projection.sumImpl");
        globalProj[i].assign( projections[i].size(),0 );
        comm->sumImpl( globalProj[i].data(), projections[i].data(), projections[i].size(), scai::common::TypeTraits<ValueType>::stype);
    }

    return globalProj;
}
//---------------------------------------------------------------------------------------

// Checks if given index is in the bounding box bBox.
template<typename IndexType, typename ValueType>
template<typename T>
bool MultiSection<IndexType, ValueType>::inBBox( const std::vector<T>& coords, const struct rectangle<ValueType>& bBox) {
    SCAI_REGION("MultiSection.inBBox");

    IndexType dimension = bBox.top.size();

    SCAI_ASSERT( coords.size()==dimension, "Dimensions do not agree.");
    if(dimension>3) {
        throw std::runtime_error("function: inBBox, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }

    // for all dimensions i: bottom(i)<top(i)
    std::vector<ValueType> bottom = bBox.bottom, top = bBox.top;

    bool ret = true;

    for(int i=0; i<dimension; i++) {
        if(coords[i]>top[i] or coords[i]<bottom[i]) {
            ret = false;
            break;
        }
    }

    return ret;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle<ValueType>& bBox, const IndexType sideLen, Settings settings) {
    SCAI_REGION("MultiSection.getRectangleWeight");

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();

    const IndexType dimension = bBox.top.size();
    ValueType localWeight=0;

    {
        SCAI_REGION("MultiSection.getRectangleWeight.localWeight");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );

        for(IndexType i=0; i<localN; i++) {
            const IndexType globalIndex = inputDist->local2Global(i);
            std::vector<IndexType> coords = indexToCoords<IndexType>(globalIndex, sideLen, dimension); // check the global index
            if( inBBox(coords, bBox) ) {
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
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight(
    const std::vector<scai::lama::DenseVector<T>> &coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const struct rectangle<ValueType>& bBox,
    Settings settings) {
    SCAI_REGION("MultiSection.getRectangleWeight");

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();

    const IndexType dimension = bBox.top.size();
    ValueType localWeight=0;

    {
        SCAI_REGION("MultiSection.getRectangleWeight.localWeight");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );

        for(IndexType i=0; i<localN; i++) {
            std::vector<T> coords;
            for(int d=0; d<dimension; d++) {
                coords.push_back( coordinates[d].getLocalValues()[i] );
            }
            if( inBBox(coords, bBox) ) {
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
ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<std::vector<T>>& coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const  struct rectangle<ValueType>& bBox, Settings settings) {
    SCAI_REGION("MultiSection.getRectangleWeight");

    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType localN = inputDist->getLocalSize();

    //const IndexType dimension = bBox.top.size();
    ValueType localWeight=0;

    {
        SCAI_REGION("MultiSection.getRectangleWeight.localWeight");
        scai::hmemo::ReadAccess<ValueType> localWeights( nodeWeights.getLocalValues() );

        for(IndexType i=0; i<localN; i++) {
            std::vector<T> coords= coordinates[i];
            if( inBBox(coords, bBox) ) {
                localWeight += localWeights[i];
            }
        }
    }
    // sum all local weights
    return comm->sum(localWeight);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> MultiSection<IndexType, ValueType>::getBlockGraphFromTree_local( const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot ) {
    SCAI_REGION("MultiSection.getBlockGraphFromTree_local");

    std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>> allLeaves = treeRoot->getAllLeaves();
    const IndexType numLeaves = allLeaves.size();
    SCAI_ASSERT_EQ_ERROR( numLeaves, treeRoot->getNumLeaves(), "Number of leaves is wrong");


    //TODO: has size k^2, change that to save memory and time
    std::unique_ptr<ValueType[]> rawArray( new ValueType[ numLeaves*numLeaves ] );

    for(IndexType l=0; l<numLeaves; l++) {
        for(IndexType l2=0; l2<numLeaves; l2++) {
            //TODO: merge first and last cases
            if( l==l2) {
                rawArray[ l +l2*numLeaves] = 0;
                rawArray[ l*numLeaves +l2] = 0;
            } else if( allLeaves[l]->getRect().isAdjacent( allLeaves[l2]->getRect() ) ) {
                rawArray[ l +l2*numLeaves] = 1;
                rawArray[ l*numLeaves +l2] = 1;
            } else {
                rawArray[ l +l2*numLeaves] = 0;
                rawArray[ l*numLeaves +l2] = 0;
            }
        }
    }

    scai::lama::CSRSparseMatrix<ValueType> ret;
    ret.setRawDenseData( numLeaves, numLeaves, rawArray.get() );
    return ret;
}
//---------------------------------------------------------------------------------------

//TODO: generalize for more dimensions and for non-cubic grids
template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexToCoords(const IndexType ind, const IndexType sideLen, const IndexType dim) {
    SCAI_REGION("MultiSection.indexToCoords");

    const IndexType gridSize= std::pow(sideLen, dim);

    if( ind>gridSize) {
        PRINT("Index "<< ind <<" too big, should be < gridSize= "<< gridSize);
        throw std::runtime_error("Wrong index");
    }

    if(ind<0) {
        throw std::runtime_error("Wrong index" + std::to_string(ind) + " should be positive or zero.");
    }

    if(dim==2) {
        return  MultiSection<IndexType, ValueType>::indexTo2D<T>( ind, sideLen);
    } else if(dim==3) {
        return MultiSection<IndexType, ValueType>::indexTo3D<T>( ind, sideLen);
    } else {
        throw std::runtime_error("function: indexToCoords, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }

}
//---------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexToCoords(const IndexType ind, const std::vector<IndexType> sideLen) {
    SCAI_REGION("MultiSection.indexToCoords");

    const IndexType gridSize= std::accumulate( sideLen.begin(), sideLen.end(), 1, std::multiplies<IndexType>() );
    const IndexType dim = sideLen.size();

    if( ind>gridSize) {
        PRINT("Index "<< ind <<" too big, should be < gridSize= "<< gridSize);
        throw std::runtime_error("Wrong index");
    }

    if(ind<0) {
        throw std::runtime_error("Wrong index" + std::to_string(ind) + " should be positive or zero.");
    }

    if(dim==2) {
        return  MultiSection<IndexType, ValueType>::indexTo2D<T>( ind, sideLen);
    } else if(dim==3) {
        return MultiSection<IndexType, ValueType>::indexTo3D<T>( ind, sideLen);
    } else {
        throw std::runtime_error("function: indexToCoords, line:" +std::to_string(__LINE__) +", supporting only 2 or 3 dimensions");
    }

}
//---------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexTo2D(IndexType ind, IndexType sideLen) {
    SCAI_REGION("MultiSection.indexTo2D");
    T x = ind/sideLen;
    T y = ind%sideLen;

    return std::vector<T> {x, y};
}
//---------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexTo2D(IndexType ind, std::vector<IndexType> sideLen) {
    SCAI_REGION("MultiSection.indexTo2D");
    assert(sideLen.size()==2);

    T x = ind/sideLen[1];
    T y = ind%sideLen[1];

    return std::vector<T> {x, y};
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexTo3D(IndexType ind, IndexType sideLen) {
    SCAI_REGION("MultiSection.indexTo3D");
    IndexType planeSize= sideLen*sideLen; // a YxZ plane

    T x = ind/planeSize;
    T y = (ind%planeSize)/sideLen;
    T z = (ind%planeSize)%sideLen;

    return std::vector<T> { x, y, z };
}
//---------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
template<typename T>
std::vector<T> MultiSection<IndexType, ValueType>::indexTo3D(IndexType ind, std::vector<IndexType> sideLen) {
    SCAI_REGION("MultiSection.indexTo3D");
    assert( sideLen.size()==3 );

    IndexType planeSize= sideLen[1]*sideLen[2]; // a YxZ plane

    T x = ind/planeSize;
    T y = (ind%planeSize)/sideLen[2];
    T z = (ind%planeSize)%sideLen[2];

    return std::vector<T> { x, y, z };
}
//---------------------------------------------------------------------------------------

//
// instantiations
//

template class MultiSection<IndexType, double>;
template class MultiSection<IndexType, float>;

/*
template IndexType MultiSection<IndexType, ValueType>::projectAnd1Dpartition(
    std::shared_ptr<rectCell<IndexType,ValueType>>& root,
    const std::vector<std::vector<IndexType>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<IndexType>& numCuts,
    const std::vector<IndexType>& maxCoords);


template ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<scai::lama::DenseVector<IndexType>> &coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const struct rectangle& bBox, Settings settings);

template ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<scai::lama::DenseVector<ValueType>> &coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const struct rectangle& bBox, Settings settings);

template ValueType MultiSection<IndexType, ValueType>::getRectangleWeight( const std::vector<std::vector<IndexType>> &coordinates, const scai::lama::DenseVector<ValueType>& nodeWeights, const struct rectangle& bBox, Settings settings);

template std::vector<IndexType> MultiSection<IndexType, ValueType>::indexToCoords(const IndexType ind, const IndexType sideLen, const IndexType dim);

template std::vector<ValueType> MultiSection<IndexType, ValueType>::indexToCoords(const IndexType ind, const IndexType sideLen, const IndexType dim);

//TODO/check: are these instantiations needed? update: probably yes
template scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::setPartition( std::shared_ptr<rectCell<IndexType,ValueType>> root, const scai::dmemo::DistributionPtr  distPtr, const std::vector<std::vector<IndexType>>& localPoints);

template scai::lama::DenseVector<IndexType> MultiSection<IndexType, ValueType>::setPartition( std::shared_ptr<rectCell<IndexType,ValueType>> root, const scai::dmemo::DistributionPtr  distPtr, const std::vector<std::vector<ValueType>>& localPoints);
*/

};
