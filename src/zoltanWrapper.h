#pragma once
#include "Wrappers.h"

//needed in main. Added it here so no new dependencies are needed.
#include <Tpetra_Core.hpp>
#include <Zoltan2_PartitioningProblem.hpp>


namespace ITI {

/** @brief Class for external partitioning tools like zoltan.
*/

template <typename IndexType, typename ValueType>
class zoltanWrapper : public Wrappers<IndexType, ValueType> {

public:

    /** Partitions a graph using some algorithm from zoltan. 
        \sa Wrappers::partition()
    */

    virtual scai::lama::DenseVector<IndexType> partition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const Tool tool,        
        const struct Settings &settings,
        Metrics<ValueType> &metrics);
    
    /** @brief Version for tools that do not need the graph as input.
        Partitions a graph using some algorithm from zoltan. 
        \sa Wrappers::partition()
    */
    scai::lama::DenseVector<IndexType> partition (
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        struct Settings &settings,
        Metrics<ValueType> &metrics);    

    /** Repartitions a graph using some algorithm from zoltan. 
        \sa Wrappers::repartition()
    */    

    virtual scai::lama::DenseVector<IndexType> repartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const Tool tool,
        const struct Settings &settings,
        Metrics<ValueType> &metrics);

private:

    static scai::lama::DenseVector<IndexType> zoltanCoreCoords (
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const std::string algo,
        const bool repart,
        const struct Settings &settings,
        Metrics<ValueType> &metrics);

    static scai::lama::DenseVector<IndexType> zoltanCoreGraph (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const std::string algo,
        const bool repart,
        const struct Settings &settings,
        Metrics<ValueType> &metrics);

    static Teuchos::ParameterList setParams( 
        const std::string algo,
        const Settings settings,
        const bool repart=false,
		const IndexType numWeights=1, 
        const IndexType thisPE=0);

    static std::vector<std::vector<ValueType>> extractLocalNodeWeights(
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag );

    template<typename Adapter>
    static scai::lama::DenseVector<IndexType> runZoltanAlgo(
        Zoltan2::PartitioningProblem<Adapter> *problem,
        const scai::dmemo::DistributionPtr dist,
        const Settings& settings,
        const scai::dmemo::CommunicatorPtr comm,
        Metrics<ValueType> &metrics );

    static std::string tool2String( ITI::Tool tool);

    };//class zoltanWrapper
} /* namespace ITI */
