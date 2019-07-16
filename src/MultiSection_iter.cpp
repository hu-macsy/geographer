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

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
IndexType MultiSection<IndexType, ValueType>::iterativeProjectionAndPart(
    std::shared_ptr<rectCell<IndexType,ValueType>> root,
    const std::vector<std::vector<T>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<IndexType>& numCuts,
    Settings settings) {

    SCAI_REGION("MultiSection.iterativeProjectionAndPart");
    const scai::dmemo::DistributionPtr inputDist = nodeWeights.getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    const IndexType dim = coordinates[0].size();
    IndexType numLeaves = root->getNumLeaves();

    //
    //multisect in every dimension
    //

    //if not using bisection, numCuts.size()=dimensions

    for(typename std::vector<IndexType>::const_iterator thisDimCuts=numCuts.begin(); thisDimCuts!=numCuts.end(); ++thisDimCuts ) {
        SCAI_REGION("MultiSection.iterativeProjectionAndPart.forAllRectangles");
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
        // in chosenDim we have stored the desired dimension to project for all the leaf nodes
        std::vector<IndexType> chosenDim ( numLeaves, -1); //the chosen dim to project for every leaf

        //the hyperplane coordinate for every leaf in the chosen dimension
        //this is used only in the iterative approach
        std::vector<std::vector<ValueType>> hyperplanes( numLeaves, (std::vector<ValueType> (*thisDimCuts+1,0)) );

        // choose the dimension to project for each leaf/rectangle
        for( IndexType l=0; l<allLeaves.size(); l++) {
            struct rectangle thisRectangle = allLeaves[l]->getRect();
            ValueType maxExtent = 0;
            for(int d=0; d<dim; d++) {
                ValueType extent = thisRectangle.top[d] - thisRectangle.bottom[d];
                if( extent>maxExtent ) {
                    maxExtent = extent;
                    chosenDim[l] = d;
                }
            }
            //determine the hyperplanes for every leaf

            ValueType meanHyperplaneOffset = maxExtent/ *thisDimCuts;
            for( int c=1; c<*thisDimCuts; c++) {
                hyperplanes[l][c] = hyperplanes[l][c-1] + meanHyperplaneOffset;
            }

        }

        ValueType maxImbalance = 0.0;
        IndexType numIterations = 0;

        do {
            // projections[i] is the projection of leaf/rectangle i in the chosen dimension; projections.size()=numLeaves

            std::vector<std::vector<ValueType>> projections = MultiSection<IndexType, ValueType>::projectionIter( coordinates, nodeWeights, root, allLeaves, hyperplanes, chosenDim);

            SCAI_ASSERT_EQ_ERROR( projections.size(), numLeaves, "Wrong number of projections");
            //PRINT0("numLeaves= " << numLeaves);

            //balance the hyperplaned for every leaf
            for(IndexType l=0; l<numLeaves; l++) {
                SCAI_REGION("MultiSection.getRectanglesNonUniform.forAllRectangles.forLeaves");

                const IndexType thisChosenDim = chosenDim[l];
                struct rectangle thisRectangle = allLeaves[l]->getRect();
                std::vector<ValueType>& thisHyperplanes = hyperplanes[l];
                const std::vector<ValueType>& thisProjection = projections[l];

                ValueType optWeight = thisRectangle.weight/(*thisDimCuts);

                for( unsigned int h=0; h<thisHyperplanes.size(); h++) {
                    ValueType imbalance = (ValueType (thisHyperplanes[h]-optWeight)/optWeight);
                    PRINT0(thisHyperplanes[h] << " -- opt= " << optWeight);
                }

            }
            numIterations++;
        } while( maxImbalance<settings.epsilon or numIterations<settings.maxIterations );

        numLeaves = root->getNumLeaves();
        //PRINT0("numLeaves= " << numLeaves);
    }

    return numLeaves;

}// iterativeProjectionAndPart
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename T>
std::vector<std::vector<ValueType>> MultiSection<IndexType, ValueType>::projectionIter(
                                     const std::vector<std::vector<T>> &coordinates,
                                     const scai::lama::DenseVector<ValueType>& nodeWeights,
                                     const std::shared_ptr<rectCell<IndexType,ValueType>> treeRoot,
                                     const std::vector<std::shared_ptr<rectCell<IndexType,ValueType>>>& allLeaves,
                                     const std::vector<std::vector<ValueType>>& hyperplanes,
const std::vector<IndexType>& dimensionToProject) {

    const IndexType dimension = coordinates[0].size();

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

        for(IndexType i=0; i<localN; i++) {
            // if this point is not contained in any rectangle
            //TODO: in the partition this should not happen. But it may happen in a more general case
            try {
                SCAI_REGION("MultiSection.projectionIter.localProjection.getContainingLeaf");
                thisRectCell = treeRoot->getContainingLeaf( coordinates[i] );
            }
            catch( const std::logic_error& e) {
                PRINT("Function getContainingLeaf returns an " << e.what() << " exception for point: ");
                for( int d=0; d<dimension; d++) {
                    std::cout<< coordinates[i][d] << ", ";
                }
                std::cout<< std::endl << " and root:"<< std::endl;
                treeRoot->getRect().print(std::cout);
                std::terminate();   // not allowed in our case
            }

            IndexType thisLeafID = thisRectCell->getLeafID();

            //print some info if something went wrong
            if( thisLeafID==-1 and comm->getRank()==0 ) {
                PRINT0( "Owner rectangle for point is ");
                thisRectCell->getRect().print(std::cout);
                PRINT0( thisRectCell->getLeafID() );
            }
            SCAI_ASSERT( thisLeafID!=-1, "leafID for containing rectCell must be >0 , for coords= "<< coordinates[i][0] << ", "<< coordinates[i][1] );
            SCAI_ASSERT_LT_ERROR( thisLeafID, projections.size(), "Index too big.");

            // the chosen dimension to project for this rectangle
            const IndexType dim2proj = dimensionToProject[ thisLeafID ];

            //relativeIndex is the index of the hyperplane such that
            // hyperplane[relativeIndex] < coord <= hyperplane[relativeIndex+1]
            typename std::vector<ValueType>::const_iterator upBound = std::upper_bound(hyperplanes[thisLeafID].begin(), hyperplanes[thisLeafID].end(), coordinates[i][dim2proj] );
            IndexType relativeIndex = (upBound-hyperplanes[thisLeafID].begin())/****** -1 ********/ -1;
            SCAI_ASSERT_GE_ERROR( coordinates[i][dim2proj], hyperplanes[thisLeafID][relativeIndex], "Wrong relative index: " << relativeIndex << " for dimension " << dim2proj << " leafID " << thisLeafID );
            SCAI_ASSERT_LE_ERROR( coordinates[i][dim2proj],
                                  hyperplanes[thisLeafID][std::min(numCuts-1,relativeIndex+1)], "Wrong relative index: " << relativeIndex << " for dimension " << dim2proj << " leafID " << thisLeafID );

            SCAI_ASSERT_LE_ERROR( relativeIndex, projections[thisLeafID].capacity(), "Wrong relative index: "<< relativeIndex << " should be <= "<< projections[ thisLeafID ].capacity() << " (and thisRect.bottom= "<< thisRectCell->getRect().bottom[dim2proj]  << " , thisRect.top= "<< thisRectCell->getRect().top[dim2proj] << ")" );

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

}//projectionIter
//---------------------------------------------------------------------------------------

//
// instantiations
//

template class MultiSection<IndexType, ValueType>;


template IndexType MultiSection<IndexType, ValueType>::iterativeProjectionAndPart(
    std::shared_ptr<rectCell<IndexType,ValueType>> root,
    const std::vector<std::vector<IndexType>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<IndexType>& numCuts,
    Settings settings);

template IndexType MultiSection<IndexType, ValueType>::iterativeProjectionAndPart(
    std::shared_ptr<rectCell<IndexType,ValueType>> root,
    const std::vector<std::vector<ValueType>>& coordinates,
    const scai::lama::DenseVector<ValueType>& nodeWeights,
    const std::vector<IndexType>& numCuts,
    Settings settings);


}//ITI