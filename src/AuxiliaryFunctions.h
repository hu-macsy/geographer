/*
 * A collection of several output and mesh functions.
 * TODO: maybe split, move the mesh-related functions to MeshGenerator?
 */

#pragma once

#include <chrono>
#include <fstream>
#include <chrono>
#include <algorithm>

#include <scai/lama.hpp>
#include <scai/lama/DenseVector.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>
#include <scai/dmemo/RedistributePlan.hpp>

#include "GraphUtils.h"
#include "Settings.h"
#include "CommTree.h"

namespace ITI {

using namespace scai::lama;

/** @brief A collection of several for output and mesh helper functions (mainly).
*/

template <typename IndexType, typename ValueType>
class aux {
public:

    /** Print to std::cout some timing information.
    @param[in] The starting point from which is calculated how much time has passed.
    */

    static void timeMeasurement(std::chrono::time_point<std::chrono::high_resolution_clock> timeStart) {

        std::chrono::duration<ValueType,std::ratio<1>> time = std::chrono::high_resolution_clock::now() - timeStart;
        ValueType elapTime = time.count() ;

        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        const IndexType thisPE = comm->getRank();
        const IndexType numPEs = comm->getSize();

        std::vector<ValueType> allTimes(numPEs,0);

        //set local time in your position
        allTimes[thisPE] = elapTime;

        //replicate all times in all PEs, TODO: gather to root PE instead
        comm->sumImpl(allTimes.data(), allTimes.data(), numPEs, scai::common::TypeTraits<ValueType>::stype);

        if( thisPE==0 ) {
            if( numPEs <33 ) {
                for(int i=0; i<numPEs; i++) {
                    std::cout << i << ": " << allTimes[i] << " _ ";
                }
                std::cout << std::endl;
            }
            typename std::vector<ValueType>::iterator maxTimeIt = std::max_element( allTimes.begin(), allTimes.end() );
            IndexType maxTimePE = std::distance( allTimes.begin(), maxTimeIt );
            typename std::vector<ValueType>::iterator minTimeIt = std::min_element( allTimes.begin(), allTimes.end() );
            IndexType minTimePE = std::distance( allTimes.begin(), minTimeIt );

            IndexType slowPEs5=0, slowPEs8=0;
            for(int i=0; i<numPEs; i++ ) {
                if(allTimes[i]>0.5*(*maxTimeIt) + *minTimeIt*(0.5))
                    ++slowPEs5;
                if(allTimes[i]>0.8*(*maxTimeIt) + *minTimeIt*(0.2))
                    ++slowPEs8;
            }

            std::cout<< "max time: " << *maxTimeIt << " from PE " << maxTimePE << std::endl;
            std::cout<< "min time: " << *minTimeIt << " from PE " << minTimePE << std::endl;
            std::cout<< "there are " << slowPEs5 << " that did more than 50% of max time and "<< slowPEs8 << " with more than 80%" << std::endl;
        }

    }

//------------------------------------------------------------------------------

    /** Given the node weights of a square grid as a vector, write them in a file that can be
    viewed as a hear map using gnuplots.

    @warning Only works for square grids, i.e., sideLen*sideLen=input.size()
    @param[in] input The node weights as a 1D vector.
    @param[in] sideLen The side length of the square grid.
    @param[in] filename The name of the file to store the data.
    */
    static void writeHeatLike_local_2D(std::vector<IndexType> input, IndexType sideLen, const std::string filename) {
        std::ofstream f(filename);
        if(f.fail())
            throw std::runtime_error("File "+ filename+ " failed.");

        if( sideLen*sideLen!=input.size() ) {
            throw std::logic_error(" Function writeHeatLike_local_2D only works for square grids/\nAborting...");
        }

        f<< "$map2 << EOD" << std::endl;

        for(IndexType i=0; i<sideLen; i++) {
            for(IndexType j=0; j<sideLen; j++) {
                f<< j << " " << i << " " << input[i*sideLen+j] << std::endl;
            }
            f<< std::endl;
        }
        f<< "EOD"<< std::endl;
        f<< "set title \"Pixeled partition for file " << filename << "\" " << std::endl;
        f << "plot '$map2' using 2:1:3 with image" << std::endl;
    }
//------------------------------------------------------------------------------

    /** Overloaded version where node weights are an HArray.
    \overload
    */

    static void writeHeatLike_local_2D(scai::hmemo::HArray<IndexType> input, IndexType sideLen, const std::string filename) {
        std::ofstream f(filename);
        if(f.fail())
            throw std::runtime_error("File "+ filename+ " failed.");

        f<< "$map2 << EOD" << std::endl;
        scai::hmemo::ReadAccess<IndexType> rInput( input );

        for(IndexType i=0; i<sideLen; i++) {
            for(IndexType j=0; j<sideLen; j++) {
                f<< j << " " << i << " " << rInput[i*sideLen+j] << std::endl;
            }
            f<< std::endl;
        }
        rInput.release();
        f<< "EOD"<< std::endl;
        f<< "set title \"Pixeled partition for file " << filename << "\" " << std::endl;
        f << "plot '$map2' using 2:1:3 with image" << std::endl;
    }
//------------------------------------------------------------------------------

    /** Prints a square grid and also marks the borders between the blocks.
    @warning Only works for square grids, i.e., adjM.getNumRows() is an square number.
    @param[in] adjM The graph,
    @param[in] partition The partition of the graph.
    */
    static void print2DGrid(const scai::lama::CSRSparseMatrix<ValueType>& adjM, const scai::lama::DenseVector<IndexType>& partition  ) {

        IndexType N= adjM.getNumRows();

        IndexType numX = std::sqrt(N);
        IndexType numY = numX;
        SCAI_ASSERT_EQ_ERROR(N, numX*numY, "Input not a grid" );

        if( numX>65 ) {
            PRINT("grid too big to print, aborting.");
            return;
        }

        //get the border nodes
        scai::lama::DenseVector<IndexType> border(adjM.getColDistributionPtr(), 0);
        border = ITI::GraphUtils<IndexType, ValueType>::getBorderNodes( adjM, partition);

        IndexType partViz[numX][numY];
        IndexType bordViz[numX][numY];
        for(int i=0; i<numX; i++)
            for(int j=0; j<numY; j++) {
                partViz[i][j]=partition.getValue(i*numX+j);
                bordViz[i][j]=border.getValue(i*numX+j);
            }

        scai::dmemo::CommunicatorPtr comm = adjM.getRowDistributionPtr()->getCommunicatorPtr();
        comm->synchronize();

        if(comm->getRank()==0 ) {
            std::cout<<"----------------------------"<< " Partition  "<< *comm << std::endl;
            for(int i=0; i<numX; i++) {
                for(int j=0; j<numY; j++) {
                    if(bordViz[i][j]==1)
                        std::cout<< "\033[1;31m"<< partViz[i][j] << "\033[0m" <<"-";
                    else
                        std::cout<< partViz[i][j]<<"-";
                }
                std::cout<< std::endl;
            }
        }

    }
//------------------------------------------------------------------------------

    /** Prints a vector
    */
    template<typename T>
    static void printVector( std::vector<T> v) {
        for(int i=0; i<v.size(); i++) {
            std::cout<< v[i] << ", ";
        }
        std::cout<< "\b\b\n" << std::endl;
    }

//------------------------------------------------------------------------------

    /** Splits a string according to some delimiter.
    @param[in] s The string to be slitted.
    @param[in] delim The delimiters according to which we gonna split the string.
    @return A vector of the string separated by delim.
    */
    static std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }


    /** The l1 distance of two pixels in 2D if their given as a 1D distance.
     * @param[in] pixel1 The index of the first pixel.
     * @param[in] pixel1 The index of the second pixel.
     * @param[in] sideLen The length of the side of the cube.
     *
     * @return The l1 distance of the pixels.
     */

    static IndexType pixelL1Distance2D(IndexType pixel1, IndexType pixel2, IndexType sideLen) {

        IndexType col1 = pixel1/sideLen;
        IndexType col2 = pixel2/sideLen;

        IndexType row1 = pixel1%sideLen;
        IndexType row2 = pixel2%sideLen;

        return absDiff(col1, col2) + absDiff(row1, row2);;
    }



    static ValueType pixelL2Distance2D(IndexType pixel1, IndexType pixel2, IndexType sideLen) {

        IndexType col1 = pixel1/sideLen;
        IndexType col2 = pixel2/sideLen;

        IndexType row1 = pixel1%sideLen;
        IndexType row2 = pixel2%sideLen;

        return std::pow( ValueType (std::pow(absDiff(col1, col2),2) + std::pow(absDiff(row1, row2),2)), 0.5);
    }

    //template<T>
    static ValueType pointDistanceL2( std::vector<ValueType> p1, std::vector<ValueType> p2) {
        SCAI_REGION("aux.pointDistanceL2");

        const IndexType dim = p1.size();
        ValueType distance = 0;

        for( IndexType d=0; d<dim; d++) {
            distance += std::pow( std::abs(p1[d]-p2[d]), 2 );
        }

        return std::pow( distance, 1.0/2.0);
    }


    /** Given a (global) index and the size for each dimension (numPpoints.size()=3) calculates the position
     * of the index in 3D. The return value is not the coordinates of the point!
     * */

    static std::tuple<IndexType, IndexType, IndexType> index2_3DPoint( const IndexType index, const  std::vector<IndexType> numPoints) {
        // a YxZ plane
        SCAI_ASSERT( numPoints.size()==3, "Wrong dimensions, should be 3");

        IndexType planeSize= numPoints[1]*numPoints[2];
        IndexType xIndex = index/planeSize;
        IndexType yIndex = (index % planeSize) / numPoints[2];
        IndexType zIndex = (index % planeSize) % numPoints[2];
        SCAI_ASSERT(xIndex >= 0, xIndex);
        SCAI_ASSERT(yIndex >= 0, yIndex);
        SCAI_ASSERT(zIndex >= 0, zIndex);
        assert(xIndex < numPoints[0]);
        assert(yIndex < numPoints[1]);
        assert(zIndex < numPoints[2]);
        return std::make_tuple(xIndex, yIndex, zIndex);
    }


    static std::tuple<IndexType, IndexType> index2_2DPoint( const IndexType index, const std::vector<IndexType> numPoints) {
        SCAI_ASSERT( numPoints.size()==2, "Wrong dimensions, should be 2");

        IndexType xIndex = index/numPoints[1];
        IndexType yIndex = index%numPoints[1];

        SCAI_ASSERT(xIndex >= 0, xIndex);
        SCAI_ASSERT(yIndex >= 0, yIndex);

        SCAI_ASSERT(xIndex < numPoints[0], xIndex << " for index: "<< index);
        SCAI_ASSERT(yIndex < numPoints[1], yIndex << " for index: "<< index);

        return std::make_tuple(xIndex, yIndex);
    }


    /** In this version, the second weight per PU is treated as an upper bound.
    The tree nodes (aka PUs) should have 2 weights, the first is treated as the computational
    power of this node and the second as the memory and memory is treated as an upper bound
    for the block size.

    @return A vector of size k, as the number of leaf nodes, with the feasible block size
    for every leaf/PE.
    */

    static std::vector<ValueType> blockSizesForMemory(
        const std::vector<std::vector<ValueType>> &inBlockSizes,
        const IndexType inputSize,
        const IndexType maxMemoryCapacity=0 );


//------------------------------------------------------------------------------

    /** Redistribute all data according to the given a partition.
    	This basically equivalent to:
    	scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues());
    	graph.redistribute( distFromPartition, noDist );
    	...

    	The partition itself is redistributed.
    	Afterwards, partition[i]=comm->getRank(), i.e., every PE gets its owned data.
    	It can also be done using a redistributor object.

    	@param[in,out] partition The partition according to which we redistribute.
    	@param[out] graph The graph to be redistributed.
    	@param[out] coordinates The coordinates of the graoh to be redistributed.
    	@param[out] nodeWeights The node weights to be redistributed.
    	@param[in] useRedistributor Flag if we should use or not a redistributor object.
    	@param[in] renumberPEs Flag if we should renumber some PE if this reduces the communication volume.
    	@return The distribution pointer of the created distribution.
    **/

    static scai::dmemo::DistributionPtr redistributeFromPartition(
        DenseVector<IndexType>& partition,
        CSRSparseMatrix<ValueType>& graph,
        std::vector<DenseVector<ValueType>>& coordinates,
        std::vector<DenseVector<ValueType>>& nodeWeights,
        Settings settings,
        bool useRedistributor = true,
        bool renumberPEs = true );


    static void redistributeInput(
        const scai::dmemo::DistributionPtr targetDistribution,
        scai::lama::DenseVector<IndexType>& partition,
        scai::lama::CSRSparseMatrix<ValueType>& graph,
        std::vector<scai::lama::DenseVector<ValueType>>& coordinates,
        std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights);


    static void redistributeInput(
        const scai::dmemo::RedistributePlan redistributor,
        scai::lama::DenseVector<IndexType>& partition,
        scai::lama::CSRSparseMatrix<ValueType>& graph,
        std::vector<scai::lama::DenseVector<ValueType>>& coordinates,
        std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights);


    /** Function to convert lama data structures to raw pointers as used
        by the metis and parmetis interface. All const arguments are the
        input and the rest are output parameters. Returns the number of
        local vertices/rows.

        \warning Edge weights not supported.
    **/
    static IndexType toMetisInterface(
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const struct Settings &settings,
        std::vector<IndexType>& vtxDist, 
        std::vector<IndexType>& xadj,
        std::vector<IndexType>& adjncy,
        std::vector<ValueType>& vwgt,
        std::vector<double>& tpwgts,
        IndexType &wgtFlag,
        IndexType &numWeights,
        std::vector<double>& ubvec,
        std::vector<double>& xyzLocal,
        std::vector<IndexType>& options);

    /** @brief Overloaded version with commTree
        \overload
    */
    static IndexType toMetisInterface(
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const ITI::CommTree<IndexType,ValueType> &commTree,
        const struct Settings &settings,
        std::vector<IndexType>& vtxDist, 
        std::vector<IndexType>& xadj,
        std::vector<IndexType>& adjncy,
        std::vector<ValueType>& vwgt,
        std::vector<double>& tpwgts,
        IndexType &wgtFlag,
        IndexType &numWeights,
        std::vector<double>& ubvec,
        std::vector<double>& xyzLocal,
        std::vector<IndexType>& options);

    /**
     * Iterates over the local part of the adjacency matrix and counts local edges.
     * If an inconsistency in the graph is detected, it tries to find the inconsistent edge and throw a runtime error.
     * Not guaranteed to find inconsistencies. Iterates once over the edge list.
     *
     * @param[in] input Adjacency matrix
     */
    static void checkLocalDegreeSymmetry(const CSRSparseMatrix<ValueType> &input);

	static bool checkConsistency(
		const CSRSparseMatrix<ValueType> &input,
		const std::vector<DenseVector<ValueType>> &coordinates,
		const std::vector<DenseVector<ValueType>> &nodeWeights,
		const Settings settings);
	
	
    /**@brief Check if distributions align and redistribute if not. Return if redistribution occurred.
    */

    static bool alignDistributions(
        scai::lama::CSRSparseMatrix<ValueType>& graph,
        std::vector<scai::lama::DenseVector<ValueType>>& coords,
        std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights,
        scai::lama::DenseVector<IndexType>& partition,
        const Settings settings);

    static IndexType absDiff(const IndexType& a, const IndexType& b) {
        return (a > b) ? (a - b) : (b - a);
    }


private:

//------------------------------------------------------------------------------
//taken from: https://stackoverflow.com/questions/236129/how-do-i-iterate-over-the-words-of-a-string
    template<typename Out>
    static void split(const std::string &s, char delim, Out result) {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
    }

}; //class aux

}// namespace ITI

