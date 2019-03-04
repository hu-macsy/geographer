/**
 * A collection of several output and mesh functions.
 * TODO: maybe split, move the mesh-related functions to MeshGenerator?
 */

#pragma once

#include <chrono>
#include <fstream>
#include <chrono>

#include <scai/lama.hpp>
#include <scai/lama/DenseVector.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/sparsekernel/openmp/OpenMPCSRUtils.hpp>
#include <scai/dmemo/RedistributePlan.hpp>

#include "GraphUtils.h"
#include "Settings.h"


using namespace scai::lama;

namespace ITI{

template <typename IndexType, typename ValueType>
class aux{
public:

//------------------------------------------------------------------------------   

static void timeMeasurement(std::chrono::time_point<std::chrono::high_resolution_clock> timeStart){

    std::chrono::duration<ValueType,std::ratio<1>> time = std::chrono::high_resolution_clock::now() - timeStart;
    ValueType elapTime = time.count() ;

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType thisPE = comm->getRank();
    const IndexType numPEs = comm->getSize();

    std::vector<ValueType> allTimes(numPEs,0);

    //set local time in your position
    allTimes[thisPE] = elapTime;

    //gather all times in root (=0) PE
    //std::vector<ValueType> allTimesLocal(numPEs);
    //comm->gatherImpl(allTimesLocal.data(), numPEs, 0 , allTimes.data(), scai::common::TypeTraits<ValueType>::stype);
    comm->sumImpl(allTimes.data(), allTimes.data(), numPEs, scai::common::TypeTraits<ValueType>::stype);

    //PRINT0(allTimes.size() << " : " << allTimesLocal.size() );

    if( thisPE==0 ){
        if( numPEs <33 ){
            for(int i=0; i<numPEs; i++){
                std::cout << i << ": " << allTimes[i] << " _ ";
            }
            std::cout << std::endl;
        }
        typename std::vector<ValueType>::iterator maxTimeIt = std::max_element( allTimes.begin(), allTimes.end() );
        IndexType maxTimePE = std::distance( allTimes.begin(), maxTimeIt );
        typename std::vector<ValueType>::iterator minTimeIt = std::min_element( allTimes.begin(), allTimes.end() );
        IndexType minTimePE = std::distance( allTimes.begin(), minTimeIt );

        IndexType slowPEs5=0, slowPEs8=0;
        for(int i=0; i<numPEs; i++ ){
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

static void writeHeatLike_local_2D(std::vector<IndexType> input,IndexType sideLen, IndexType dim, const std::string filename){
    std::ofstream f(filename);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");
    
    f<< "$map2 << EOD" << std::endl;
    
    for(IndexType i=0; i<sideLen; i++){
        for(IndexType j=0; j<sideLen; j++){
            //for(IndexType d=0; d<dim; d++){
            f<< j << " " << i << " " << input[i*sideLen+j] << std::endl;
            //PRINT( i/dim<< " " << i%dim << " " << input[i*dim +dim] );
        }
        f<< std::endl;
    }
    f<< "EOD"<< std::endl;
    f<< "set title \"Pixeled partition for file " << filename << "\" " << std::endl;
    f << "plot '$map2' using 2:1:3 with image" << std::endl;
}    
//------------------------------------------------------------------------------


static void writeHeatLike_local_2D(scai::hmemo::HArray<IndexType> input,IndexType sideLen, IndexType dim, const std::string filename){
    std::ofstream f(filename);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");
    
    f<< "$map2 << EOD" << std::endl;
    scai::hmemo::ReadAccess<IndexType> rInput( input );
    
    for(IndexType i=0; i<sideLen; i++){
        for(IndexType j=0; j<sideLen; j++){
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


static void print2DGrid(const scai::lama::CSRSparseMatrix<ValueType>& adjM, scai::lama::DenseVector<IndexType>& partition  ){
    
    IndexType N= adjM.getNumRows();
    
    IndexType numX = std::sqrt(N);
    IndexType numY = numX;
    SCAI_ASSERT_EQ_ERROR(N , numX*numY, "Input not a grid" );
    
    if( numX>65 ){
        PRINT("grid too big to print, aborting.");
        return;
    }
        
    //get the border nodes
    scai::lama::DenseVector<IndexType> border(adjM.getColDistributionPtr(), 0);
    border = ITI::GraphUtils<IndexType, ValueType>::getBorderNodes( adjM , partition);
    
    IndexType partViz[numX][numY];   
    IndexType bordViz[numX][numY]; 
    for(int i=0; i<numX; i++)
        for(int j=0; j<numY; j++){
            partViz[i][j]=partition.getValue(i*numX+j);
            bordViz[i][j]=border.getValue(i*numX+j);
        }

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    comm->synchronize();

    if(comm->getRank()==0 ){            
        std::cout<<"----------------------------"<< " Partition  "<< *comm << std::endl;    
        for(int i=0; i<numX; i++){
            for(int j=0; j<numY; j++){
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

template<typename T>
static void printVector( std::vector<T> v){
    for(int i=0; i<v.size(); i++){
        std::cout<< v[i] << ", ";
    }
    std::cout<< "\b\b\n" << std::endl;
}

//------------------------------------------------------------------------------

/*  From pixel (int) coords, either in 2S or 3D, to a 1D index. 
 * Only for cubes, where every side has the same length, maxLen;
 */
//static IndexType pixel2Index(IndexType pixel1, IndexType maxLen, IndexType dimension){
//}
 
 /** The l1 distance of two pixels in 2D if their given as a 1D distance.
  * @param[in] pixel1 The index of the first pixel.
  * @param[in] pixel1 The index of the second pixel.
  * @param[in] sideLen The length of the side of the cube.
  * 
  * @return The l1 distance of the pixels.
  */

static IndexType pixelL1Distance2D(IndexType pixel1, IndexType pixel2, IndexType sideLen){
     
     IndexType col1 = pixel1/sideLen;
     IndexType col2 = pixel2/sideLen;
     
     IndexType row1 = pixel1%sideLen;
     IndexType row2 = pixel2%sideLen;
     
     return std::abs(col1-col2) + std::abs(row1-row2);;
}


static ValueType pixelL2Distance2D(IndexType pixel1, IndexType pixel2, IndexType sideLen){
     
     IndexType col1 = pixel1/sideLen;
     IndexType col2 = pixel2/sideLen;
     
     IndexType row1 = pixel1%sideLen;
     IndexType row2 = pixel2%sideLen;
     
     return std::pow( ValueType (std::pow(std::abs(col1-col2),2) + std::pow(std::abs(row1-row2),2)) , 0.5);
}

//template<T>
static ValueType pointDistanceL2( std::vector<ValueType> p1, std::vector<ValueType> p2){
	const IndexType dim = p1.size();
	ValueType distance = 0;

	for( IndexType d=0; d<dim; d++){
		distance += std::pow( std::abs(p1[d]-p2[d]), 2 );
	}
	
	return std::pow( distance, 1.0/2.0);
}

//------------------------------------------------------------------------------

/* Given a (global) index and the size for each dimension (numPpoints.size()=3) calculates the position
 * of the index in 3D. The return value is not the coordinates of the point!
 * */

static std::tuple<IndexType, IndexType, IndexType> index2_3DPoint(IndexType index,  std::vector<IndexType> numPoints){
    // a YxZ plane
    SCAI_ASSERT( numPoints.size()==3 , "Wrong dimensions, should be 3");
    
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


static std::tuple<IndexType, IndexType> index2_2DPoint(IndexType index,  std::vector<IndexType> numPoints){
    SCAI_ASSERT( numPoints.size()==2 , "Wrong dimensions, should be 2");
    
    IndexType xIndex = index/numPoints[1];
    IndexType yIndex = index%numPoints[1];

    SCAI_ASSERT(xIndex >= 0, xIndex);
    SCAI_ASSERT(yIndex >= 0, yIndex);

    SCAI_ASSERT(xIndex < numPoints[0], xIndex << " for index: "<< index);
    SCAI_ASSERT(yIndex < numPoints[1], yIndex << " for index: "<< index);

    return std::make_tuple(xIndex, yIndex);
} 
//------------------------------------------------------------------------------

/** Redistribute all data according to the given a partition.
	This basically equivallen to:
	scai::dmemo::DistributionPtr distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues());
	graph.redistribute( distFromPartition, noDist );
	...

	The partititon itself is redistributed.

	Afterwards, partition[i]=comm->getRank(), i.e., every PE gets its owned data.

	It can also be done using a redistributor object.

	@param[in,out] partition The partition according to which we redistribute.
	@param[out] graph 

	@return The distribution pointer of the created distribution.
**/

static scai::dmemo::DistributionPtr redistributeFromPartition( 
                DenseVector<IndexType>& partition,
                CSRSparseMatrix<ValueType>& graph,
                std::vector<DenseVector<ValueType>>& coordinates,
                DenseVector<ValueType>& nodeWeights,
                Settings settings, 
                bool useRedistributor = true ){

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType numPEs = comm->getSize();
    //const IndexType thisPE = comm->getRank();
    const IndexType globalN = coordinates[0].getDistributionPtr()->getGlobalSize();
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));

    SCAI_ASSERT_EQ_ERROR( graph.getNumRows(), globalN, "Mismatch in graph and coordinates size" );
    SCAI_ASSERT_EQ_ERROR( nodeWeights.getDistributionPtr()->getGlobalSize(), globalN , "Mismatch in nodeWeights vector" );
    SCAI_ASSERT_EQ_ERROR( partition.size(), globalN, "Mismatch in partition size");
    SCAI_ASSERT_EQ_ERROR( partition.min(), 0, "Minimum entry in partition should be 0" );
    SCAI_ASSERT_EQ_ERROR( partition.max(), numPEs-1, "Maximum entry in partition must be equal the number of processors.")

    //possible optimization: go over your local partition, calculate size of each local block and claim the PE rank of the majority block

    {
		scai::hmemo::ReadAccess<IndexType> rPart( partition.getLocalValues() );
		//std::vector< std::pair<IndexType,IndexType> > blockSize
		//std::map<IndexType,IndexType> blockSizes;
		scai::lama::SparseVector<IndexType> blockSizes( numPEs, 0 );
		for (IndexType i = 0; i < localN; i++) {
			//blockSizes.insert( std::pair<IndexType,IndexType>( ))
			blockSizes[ rPart[i] ]++;
		}

		//sort block IDs based on their local size
		std::vector<IndexType> indices( numPEs );
  		std::iota( indices.begin(), indices.end(), 0);
  		std::sort( indices.begin(), indices.end(), [&v](IndexType i1, IndexType i2) {return blockSizes[i1] < blockSizes[i2];});

	}



    scai::dmemo::DistributionPtr distFromPartition;

    if( useRedistributor ){
        scai::dmemo::RedistributePlan resultRedist = scai::dmemo::redistributePlanByNewOwners(partition.getLocalValues(), partition.getDistributionPtr());
        //auto resultRedist = scai::dmemo::redistributePlanByNewOwners(partition.getLocalValues(), partition.getDistributionPtr());

        partition = DenseVector<IndexType>(resultRedist.getTargetDistributionPtr(), comm->getRank());
        scai::dmemo::RedistributePlan redistributor = scai::dmemo::redistributePlanByNewDistribution(resultRedist.getTargetDistributionPtr(), graph.getRowDistributionPtr());
        
        for (IndexType d=0; d<settings.dimensions; d++) {
            coordinates[d].redistribute(redistributor);
        }
        nodeWeights.redistribute(redistributor);    
        graph.redistribute( redistributor, noDist );

        distFromPartition = resultRedist.getTargetDistributionPtr();
    }else{
        // create new distribution from partition
        distFromPartition = scai::dmemo::generalDistributionByNewOwners( partition.getDistribution(), partition.getLocalValues());

        partition.redistribute( distFromPartition );
        graph.redistribute( distFromPartition, noDist );
        nodeWeights.redistribute( distFromPartition );

        // redistribute coordinates
        for (IndexType d = 0; d < settings.dimensions; d++) {
            //assert( coordinates[dim].size() == globalN);
            coordinates[d].redistribute( distFromPartition );
        }
    }

    const scai::dmemo::DistributionPtr inputDist = graph.getRowDistributionPtr();
    SCAI_ASSERT_ERROR( nodeWeights.getDistribution().isEqual(*inputDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( coordinates[0].getDistribution().isEqual(*inputDist), "Distribution mismatch" );
    SCAI_ASSERT_ERROR( partition.getDistribution().isEqual(*inputDist), "Distribution mismatch" );

    return distFromPartition;
}

}; //class aux

template class aux<IndexType, ValueType>;
}// namespace ITI
