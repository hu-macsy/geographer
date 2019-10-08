//#include "Wrappers.h"
//#include "Mapping.h"
#include "AuxiliaryFunctions.h"
#include "zoltanWrapper.h"

//for zoltan
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_InputTraits.hpp>


namespace ITI {

//---------------------------------------------------------
//                      zoltan
//---------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::zoltanPartition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    bool nodeWeightsFlag,
    std::string algo,
    struct Settings &settings,
    Metrics<ValueType> &metrics) {

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    PRINT0("\t\tStarting the zoltan wrapper for partition with "<< algo);

    bool repart = false;

    return zoltanWrapper<IndexType, ValueType>::zoltanCore( coords, nodeWeights, nodeWeightsFlag, algo, repart, settings, metrics);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::zoltanRepartition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    bool nodeWeightsFlag,
    std::string algo,
    struct Settings &settings,
    Metrics<ValueType> &metrics) {

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    PRINT0("\t\tStarting the zoltan wrapper for repartition with " << algo);

    bool repart = true;

    return zoltanWrapper<IndexType, ValueType>::zoltanCore( coords, nodeWeights, nodeWeightsFlag, algo, repart, settings, metrics);
}
//---------------------------------------------------------------------------------------

//relevant code can be found in zoltan, in Trilinos/packages/zoltan2/test/partition

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::zoltanCore (
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    bool nodeWeightsFlag,
    std::string algo,
    bool repart,
    struct Settings &settings,
    Metrics<ValueType> &metrics) {


/*
    typedef Zoltan2::BasicUserTypes<ValueType, IndexType, IndexType> myTypes;
    typedef Zoltan2::BasicVectorAdapter<myTypes> inputAdapter_t;

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = coords[0].getDistributionPtr();
    const IndexType thisPE = comm->getRank();
    const IndexType numBlocks = settings.numBlocks;

    IndexType dimensions = settings.dimensions;
    IndexType localN= dist->getLocalSize();

    //TODO: point directly to the localCoords data and save time and space for zoltanCoords
    // the local part of coordinates for zoltan
    ValueType *zoltanCoords = new ValueType [dimensions * localN];

    std::vector<scai::hmemo::HArray<ValueType>> localPartOfCoords( dimensions );
    for(unsigned int d=0; d<dimensions; d++) {
        localPartOfCoords[d] = coords[d].getLocalValues();
    }
    IndexType coordInd = 0;
    for(int d=0; d<dimensions; d++) {
        //SCAI_ASSERT_LE_ERROR( dimensions*(i+1), dimensions*localN, "Too large index, localN= " << localN );
        for(IndexType i=0; i<localN; i++) {
            SCAI_ASSERT_LT_ERROR( coordInd, localN*dimensions, "Too large coordinate index");
            zoltanCoords[coordInd++] = localPartOfCoords[d][i];
        }
    }

    std::vector<const ValueType *>coordVec( dimensions );
    std::vector<int> coordStrides(dimensions);

    coordVec[0] = zoltanCoords;     // coordVec[d] = localCoords[d].data(); or something
    coordStrides[0] = 1;

    for( int d=1; d<dimensions; d++) {
        coordVec[d] = coordVec[d-1] + localN;
        coordStrides[d] = 1;
    }

    ///////////////////////////////////////////////////////////////////////
    // Create parameters

    ValueType tolerance = 1+settings.epsilon;

    if (thisPE == 0)
        std::cout << "Imbalance tolerance is " << tolerance << std::endl;

    Teuchos::ParameterList params("test params");
    //params.set("debug_level", "basic_status");
    params.set("debug_level", "no_status");
    params.set("debug_procs", "0");
    params.set("error_check_level", "debug_mode_assertions");

    params.set("algorithm", algo);
    params.set("imbalance_tolerance", tolerance );
    params.set("num_global_parts", (int)numBlocks );

    params.set("compute_metrics", false);

    // chose if partition or repartition
    if( repart ) {
        params.set("partitioning_approach", "repartition");
    } else {
        params.set("partitioning_approach", "partition");
    }

    //TODO: params.set("partitioning_objective", "minimize_cut_edge_count");
    //      or something else, check at
    //      https://trilinos.org/docs/r12.12/packages/zoltan2/doc/html/z2_parameters.html

    // Create global ids for the coordinates.
    IndexType *globalIds = new IndexType [localN];
    IndexType offset = thisPE * localN;

    //TODO: can also be taken from the distribution?
    for (size_t i=0; i < localN; i++)
        globalIds[i] = offset++;

    //set node weights
    //see also: Trilinos/packages/zoltan2/test/partition/MultiJaggedTest.cpp, ~line 590
    const IndexType numWeights = nodeWeights.size();
    std::vector<std::vector<ValueType>> localWeights( numWeights, std::vector<ValueType>( localN, 1.0) );
    //localWeights[i][j] is the j-th weight of the i-th vertex (i is local ID)

    if( nodeWeightsFlag ) {
        for( unsigned int w=0; w<numWeights; w++ ) {
            scai::hmemo::ReadAccess<ValueType> rLocalWeights( nodeWeights[w].getLocalValues() );
            for(unsigned int i=0; i<localN; i++) {
                localWeights[w][i] = rLocalWeights[i];
            }
        }
    } else {
        //all weights are initiallized with unit weight
    }

    std::vector<const ValueType *>weightVec( numWeights );
    for( unsigned int w=0; w<numWeights; w++ ) {
        weightVec[w] = localWeights[w].data();
    }

    //if it is stride.size()==0, it assumed that all strides are 1
    std::vector<int> weightStrides; //( numWeights, 1);

    //create the problem and solve it
    inputAdapter_t *ia= new inputAdapter_t(localN, globalIds, coordVec,
                                           coordStrides, weightVec, weightStrides);

    Zoltan2::PartitioningProblem<inputAdapter_t> *problem =
        new Zoltan2::PartitioningProblem<inputAdapter_t>(ia, &params);

    if( comm->getRank()==0 )
        std::cout<< "About to call zoltan, algo " << algo << std::endl;

    int repeatTimes = settings.repeatTimes;
    double sumPartTime = 0.0;
    int r=0;

    for( r=0; r<repeatTimes; r++) {
        std::chrono::time_point<std::chrono::system_clock> beforePartTime =  std::chrono::system_clock::now();
        problem->solve();

        std::chrono::duration<double> partitionTmpTime = std::chrono::system_clock::now() - beforePartTime;
        double partitionTime= comm->max(partitionTmpTime.count() );
        sumPartTime += partitionTime;
        if( comm->getRank()==0 ) {
            std::cout<< "Running time for run number " << r << " is " << partitionTime << std::endl;
        }
        if( sumPartTime>HARD_TIME_LIMIT) {
            std::cout<< "Stopping runs because of excessive running total running time: " << sumPartTime << std::endl;
            break;
        }
    }

    if( r!=repeatTimes) {       // in case it has to break before all the runs are completed
        repeatTimes = r+1;
    }
    if(comm->getRank()==0 ) {
        std::cout<<"Number of runs: " << repeatTimes << std::endl;
    }

    metrics.MM["timeFinalPartition"] = sumPartTime/(ValueType)repeatTimes;

    //
    // convert partition to a DenseVector
    //
    scai::lama::DenseVector<IndexType> partitionZoltan(dist, IndexType(0));

    //std::vector<IndexType> localBlockSize( numBlocks, 0 );

    const Zoltan2::PartitioningSolution<inputAdapter_t> &solution = problem->getSolution();
    const int *partAssignments = solution.getPartListView();
    for(unsigned int i=0; i<localN; i++) {
        IndexType thisBlock = partAssignments[i];
        SCAI_ASSERT_LT_ERROR( thisBlock, numBlocks, "found wrong vertex id");
        SCAI_ASSERT_GE_ERROR( thisBlock, 0, "found negetive vertex id");
        partitionZoltan.getLocalValues()[i] = thisBlock;
        //localBlockSize[thisBlock]++;
    }
    for(int i=0; i<localN; i++) {
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        SCAI_ASSERT_EQ_ERROR( partitionZoltan.getLocalValues()[i], partAssignments[i], "Wrong conversion to DenseVector");
    }

    delete[] globalIds;
    delete[] zoltanCoords;

    return partitionZoltan;
    */

return scai::lama::DenseVector<IndexType> (coords[0].getDistributionPtr(), IndexType(0));
}

//---------------------------------------------------------------------------------------

template class zoltanWrapper<IndexType, double>;

}//namespace ITI