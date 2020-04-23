#pragma once

#include <scai/lama.hpp>
#include <scai/lama/DenseVector.hpp>

#include "Settings.h"
#include "Metrics.h"

namespace ITI {

/** @brief Map the blocks of a partitioned graph to a processor graph, i.e., the physical network.
*/

template <typename IndexType, typename ValueType>
class Mapping {

public:

    /* Implementation of the Hoefler, Snir mapping algorithm copied from Roland Glantz
    code as found in TiMEr.

    @param[in] blockGraph The graph to be mapped. Typically, it is created for a
    partitioned input/application graph calling GraphUtils::getBlockGraph
    @param[in] PEGraph The graph of the psysical network, ie. the processor
    graph. The two graph must have the same number of nodes n.
    @return A vector of size n indicating which block should be mapped to
    which processor. Example, if ret[4]=10, then block 4 will be mapped to
    processor 10.
    */
    /*
    	std::vector<IndexType> rolandMapping_local(
    		scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
    		scai::lama::CSRSparseMatrix<ValueType>& PEGraph);
    */
    /**Implementation of the Hoefler, Snir mapping algorithm copied from libTopoMap
    library

    @param[in] blockGraph The graph to be mapped. Typically, it is created for a
    partitioned input/application graph calling GraphUtils::getBlockGraph
    @param[in] PEGraph The graph of the psysical network, ie. the processor
    graph. The two graph must have the same number of nodes n.
    @return A vector of size n indicating which block should be mapped to
    which processor. Example, if ret[4]=10, then block 4 will be mapped to
    processor 10.
    **/

    // copy and convert/reimplement code from libTopoMap,
    // http://htor.inf.ethz.ch/research/mpitopo/libtopomap/,
    // function TPM_Map_greedy found in file libtopomap.cpp around line 580

    static std::vector<IndexType> torstenMapping_local(
        const scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
        const scai::lama::CSRSparseMatrix<ValueType>& PEGraph);

    /**Check if a given mapping is valid. It checks the size of the graphs and the mapping and a checksum.
    The mapping is from the \p blockGraph to the \p PEGraph,
    i.e., we map blocks to PEs.

    @param[in] blockGraph The graph to be mapped. Typically, it is created for a
    partitioned input/application graph calling GraphUtils::getBlockGraph
    @param[in] PEGraph The graph of the physical network, i.e., the processor
    graph. The two graph must have the same number of nodes n.
    @param[in] mapping \p mapping[i]=j means that block i is mapped to PE j.
    mapping.size()= number of nodes of the graphs
    @return true if the mapping is valid, false otherwise.
    */
    static bool isValid(
        const scai::lama::CSRSparseMatrix<ValueType>& blockGraph,
        const scai::lama::CSRSparseMatrix<ValueType>& PEGraph,
        std::vector<IndexType> mapping);

    /** Provide a renumbering of blocks. We find the center for every block in the partition
    *   and we sort the centers according to their SFC (hilbert curve) index. This suggests
    *   a renumbering as follows: if R is the returned vector, renumber block R[i] to i.
    *   So, if R=[4,5,...] we can renumber 4 to 0 since block 4 has the "minimum" center,
    *   i.e., the center closest to (0,0). Similarly, renumber block 5 to 1 since its center
    *   is the second center on the SFC curve etc.
    *
    *	@param[in] coordinates The coordinates of the points, coordinates.size()=dim and
    *	coordinates[i].size()=N
    *	@param[in] nodeWeights The weights of the points
    *	@param[in] partition A given partition of the points.
    *	@param[in] settings Settings struct
    * 	@return A vector of size k (i.e., the number of blocks=max(partition)-1). It suggest
    *	a renumbering of the blocks where block R[i] should be renumbered to i.
    */
    //TODO:	probably node weights are not used or needed

    static std::vector<IndexType> getSfcRenumber(
        const std::vector<scai::lama::DenseVector<ValueType>>& coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights,
        const scai::lama::DenseVector<IndexType>& partition,
        const Settings settings);


    /** Calculates a renumbering of the blocks using getSfcRenumber and also applies it to
    *	the partition.
    *	@param[in] coordinates The coordinates of the points, coordinates.size()=dim and
    *	coordinates[i].size()=N
    *	@param[in] nodeWeights The weights of the points
    *	@param[in,out] partition A given partition of the points. This is changed according
    *	to the renumbering. The number of points per block should remain unchanged.
    *	@param[in] settings Settings struct
    * 	@return The mapping vector: ret.size()=k, ret[i]=j means that block i is renumbered to j.
    */
    static std::vector<IndexType> applySfcRenumber(
        const std::vector<scai::lama::DenseVector<ValueType>>& coordinates,
        const std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights,
        scai::lama::DenseVector<IndexType>& partition,
        const Settings settings);

private:
    class max_compare_func {
    public:
        bool operator()(std::pair<double,int> x, std::pair<double,int> y) {
            if(x.first < y.first) return true;
            return false;
        }
    };
};//class Mapping

}//namespace ITI
