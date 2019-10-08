/*
 * Wrappers.cpp
 *
 *  Created on: 02.02.2018
 *      Author: tzovas
 */

#include <parmetis.h>

#include <scai/partitioning/Partitioning.hpp>

#include "Wrappers.h"
#include "Mapping.h"
#include "AuxiliaryFunctions.h"
#include "zoltanWrapper.h"
#include "parmetisWrapper.h"


namespace ITI {

//using ValueType= real_t;

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::partition(
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    bool nodeWeightsFlag,
    Tool tool,
    struct Settings &settings,
    Metrics<ValueType> &metrics	) {

    using ITI::parmetisWrapper;
    using ITI::zoltanWrapper;

    scai::lama::DenseVector<IndexType> partition;
    switch( tool) {
    case Tool::parMetisGraph:
        partition = parmetisWrapper<IndexType, ValueType>::metisPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, 0, settings, metrics);
        break;
    case Tool::parMetisGeom:
        partition =  metisPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, 1, settings, metrics);
        break;
    case Tool::parMetisSFC:
        partition = metisPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, 2, settings, metrics);
        break;
    case Tool::zoltanRIB:
        partition = zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rib", settings, metrics);
        break;
    case Tool::zoltanRCB:
        partition = zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rcb", settings, metrics);
        break;
    case Tool::zoltanMJ:
        partition = zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "multijagged", settings, metrics);
        break;
    case Tool::zoltanSFC:
        partition = zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "hsfc", settings, metrics);
        break;
    default:
        throw std::runtime_error("Wrong tool given to partition.\nAborting...");
        partition = scai::lama::DenseVector<IndexType>(graph.getLocalNumRows(), -1 );
    }

    if( settings.mappingRenumbering ) {
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        PRINT0("Applying renumbering of blocks based on the SFC index of their centers.");
        std::chrono::time_point<std::chrono::system_clock> startRnb = std::chrono::system_clock::now();

        Mapping<IndexType,ValueType>::applySfcRenumber( coordinates, nodeWeights, partition, settings );

        std::chrono::duration<double> elapTime = std::chrono::system_clock::now() - startRnb;
        PRINT0("renumbering time " << elapTime.count() );
    }

    return partition;
}
//-----------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::partition(
    const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    bool nodeWeightsFlag,
    Tool tool,
    struct Settings &settings,
    Metrics<ValueType> &metrics	) {

    //create dummy graph as the these tools do not use it.
    const scai::dmemo::DistributionPtr distPtr = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution(distPtr->getGlobalSize()) );
    const scai::lama::CSRSparseMatrix<ValueType> graph = scai::lama::zero<CSRSparseMatrix<ValueType>>( distPtr, noDistPtr );

    scai::lama::DenseVector<IndexType> retPart;
    switch( tool ) {
    case Tool::parMetisGraph:
    case Tool::parMetisGeom:
        PRINT("Tool "<< tool <<" requires the graph to compute a partition but no graph was given.");
        throw std::runtime_error("Missing graph.\nAborting...");
        break;
    case Tool::parMetisSFC:
    case Tool::zoltanRIB:
    case Tool::zoltanRCB:
    case Tool::zoltanMJ:
    case Tool::zoltanSFC:
        //call partition function
        retPart =  partition( graph, coordinates, nodeWeights, nodeWeightsFlag, tool, settings, metrics);
        break;
    default:
        throw std::runtime_error("Wrong tool given to partition.\nAborting...");
        retPart = scai::lama::DenseVector<IndexType>(graph.getLocalNumRows(), -1 );
    }//switch

    return retPart;
}
//-----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::repartition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    bool nodeWeightsFlag,
    Tool tool,
    struct Settings &settings,
    Metrics<ValueType> &metrics) {

    using ITI::parmetisWrapper;
    using ITI::zoltanWrapper;

    switch( tool) {
    // for repartition, metis uses the same function
    case Tool::parMetisGraph:
    case Tool::parMetisGeom:
    case Tool::parMetisSFC:
        throw std::runtime_error("Unfortunatelly, Current version does not support repartitioning with parmetis.\nAborting...");
    //TODO: parmetis needs consective indices for the vertices; must reindex vertices
    //return metisRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, settings, metrics);

    case Tool::zoltanRIB:
        return zoltanPartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rib", settings, metrics);

    case Tool::zoltanRCB:
        return zoltanRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "rcb", settings, metrics);

    case Tool::zoltanMJ:
        return zoltanRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "multijagged", settings, metrics);

    case Tool::zoltanSFC:
        return zoltanRepartition( graph, coordinates, nodeWeights, nodeWeightsFlag, "hsfc", settings, metrics);

    default:
        throw std::runtime_error("Wrong tool given to repartition.\nAborting...");
        return scai::lama::DenseVector<IndexType>(graph.getLocalNumRows(), -1 );
    }
}
//-----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> Wrappers<IndexType, ValueType>::refine(
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const scai::lama::DenseVector<IndexType> partition,
        struct Settings &settings,
        Metrics<ValueType> &metrics
    ){

    //only parmetis refinement is available
    return parmetisWrapper<IndexType,ValueType>::refine( graph, coords, nodeWeights, partition, settings, metrics );
}

//template class Wrappers<IndexType, double>;
//template class Wrappers<IndexType, float>;
template class Wrappers<IndexType, real_t>;

}//namespace
