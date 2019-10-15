#pragma once
#include "Wrappers.h"

namespace ITI {

/** @brief Class for external partitioning tools like zoltan.
*/

template <typename IndexType, typename ValueType>
class zoltanWrapper : public Wrappers<IndexType, ValueType> {

//friend class Wrappers<IndexType,ValueType>;
	
public:

    /** Partitions a graph using some algorithm from zoltan. 
        \sa Wrappers::partition()
    */

    static scai::lama::DenseVector<IndexType> partition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        struct Settings &settings,
        Metrics<ValueType> &metrics);

    /** Repartitions a graph using some algorithm from zoltan. 
        \sa Wrappers::repartition()
    */    

    static scai::lama::DenseVector<IndexType> repartition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        struct Settings &settings,
        Metrics<ValueType> &metrics);

private:

    static scai::lama::DenseVector<IndexType> zoltanCore (
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        bool repart,
        struct Settings &settings,
        Metrics<ValueType> &metrics);

    };//class zoltanWrapper
} /* namespace ITI */
