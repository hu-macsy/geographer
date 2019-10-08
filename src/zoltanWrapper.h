#pragma once
#include "Wrappers.h"

namespace ITI {

/** @brief Class for external partitioning tools like zoltan.
*/

template <typename IndexType, typename ValueType>
class zoltanWrapper {

public:

    static scai::lama::DenseVector<IndexType> partition (
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        bool nodeWeightsFlag,
        std::string algo,
        struct Settings &settings,
        Metrics<ValueType> &metrics);

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