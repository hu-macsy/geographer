#pragma once
#include "Wrappers.h"

#include "parhip_interface.h"

namespace ITI {

/** @brief Class for external partitioning tools like zoltan.
*/

template <typename IndexType, typename ValueType>
class parhipWrapper : public Wrappers<IndexType,ValueType> {
    
public:

    /** Partitions a graph using some algorithm from zoltan. 
        \sa Wrappers::partition()
    */

    virtual scai::lama::DenseVector<IndexType> partition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const Tool tool,
        const struct Settings &settings,
        Metrics<ValueType> &metrics);

    /** Repartitions a graph using some algorithm from zoltan. 
        \sa Wrappers::repartition()
    */    

    virtual scai::lama::DenseVector<IndexType> repartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const bool nodeWeightsFlag,
        const Tool tool,
        const struct Settings &settings,
        Metrics<ValueType> &metrics);

private:
/*
    static scai::lama::DenseVector<IndexType> parhipCore (
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        bool repart,
        struct Settings &settings,
        Metrics<ValueType> &metrics);
*/
    };//class parhipWrapper
} /* namespace ITI */
