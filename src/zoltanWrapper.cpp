//#include "Wrappers.h"
//#include "Mapping.h"
#include "AuxiliaryFunctions.h"
#include "zoltanWrapper.h"

//for zoltan
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_Adapter.hpp>
#include <Zoltan2_InputTraits.hpp>
#include <Zoltan2_XpetraCrsGraphAdapter.hpp>

//from zoltan, needed to convert to graph for pulp
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_Array.hpp>

namespace ITI {

//---------------------------------------------------------
//                      zoltan
//---------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::partition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const Tool tool,
    const struct Settings &settings,
    Metrics<ValueType> &metrics) {

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    bool repart = false;
    const std::string algo= tool2String(tool);
    assert( algo!="" );

    PRINT0("\t\tStarting the zoltan wrapper for partition with "<< algo);

    if(algo=="pulp"){
        return zoltanWrapper<IndexType, ValueType>::zoltanCoreGraph( graph, nodeWeights, nodeWeightsFlag, algo, repart, settings, metrics);
    }else{
        return zoltanWrapper<IndexType, ValueType>::zoltanCoreCoords( coords, nodeWeights, nodeWeightsFlag, algo, repart, settings, metrics);
    }
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::repartition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const Tool tool,
    const struct Settings &settings,
    Metrics<ValueType> &metrics) {

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    bool repart = true;
    std::string algo= tool2String(tool);
    assert( algo!="" );

    PRINT0("\t\tStarting the zoltan wrapper for repartition with " << algo);

    return zoltanWrapper<IndexType, ValueType>::zoltanCoreCoords( coords, nodeWeights, nodeWeightsFlag, algo, repart, settings, metrics);
}
//---------------------------------------------------------------------------------------

//relevant code can be found in zoltan, in Trilinos/packages/zoltan2/test/partition

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::zoltanCoreCoords (
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const std::string algo,
    const bool repart,
    const struct Settings &settings,
    Metrics<ValueType> &metrics) {

    typedef Zoltan2::BasicUserTypes<ValueType, IndexType, IndexType> myTypes;
    typedef Zoltan2::BasicVectorAdapter<myTypes> inputAdapter_t;

    const scai::dmemo::DistributionPtr dist = coords[0].getDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    const IndexType thisPE = comm->getRank();

    const IndexType dimensions = settings.dimensions;
    const IndexType localN= dist->getLocalSize();

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

    comm->synchronize();

    // Create global ids for the coordinates.
    IndexType *globalIds = new IndexType [localN];
    IndexType offset = thisPE * localN;

    //TODO: can also be taken from the distribution?
    for (size_t i=0; i < localN; i++)
        globalIds[i] = offset++;

    //
    //set node weights
    //see also: Trilinos/packages/zoltan2/test/partition/MultiJaggedTest.cpp, ~line 590
    //

    const IndexType numWeights = nodeWeights.size();

    std::vector<std::vector<ValueType>> localWeights = extractLocalNodeWeights(nodeWeights, nodeWeightsFlag);

    std::vector<const ValueType *>weightVec( numWeights );
    for( unsigned int w=0; w<numWeights; w++ ) {
        weightVec[w] = localWeights[w].data();
    }

    //if it is stride.size()==0, it assumed that all strides are 1
    std::vector<int> weightStrides; //( numWeights, 1);

    //create the problem and solve it
    inputAdapter_t *ia= new inputAdapter_t(localN, globalIds, coordVec,
                                           coordStrides, weightVec, weightStrides);

    ///////////////////////////////////////////////////////////////////////
    // Create parameters

    Teuchos::ParameterList params = setParams( algo, settings, repart, numWeights, thisPE);    
	params.set("objects_to_partition", "coordinates");

    Zoltan2::PartitioningProblem<inputAdapter_t> *problem =
        new Zoltan2::PartitioningProblem<inputAdapter_t>(ia, &params);

    if( comm->getRank()==0 )
        std::cout<< "About to call zoltan, algo " << algo << std::endl;

    scai::lama::DenseVector<IndexType> partitionZoltan = runZoltanAlgo( problem, dist, settings, comm, metrics );

    delete[] globalIds;
    delete[] zoltanCoords;
 
    return partitionZoltan;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::zoltanCoreGraph (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const std::string algo,
    const bool repart,
    const struct Settings &settings,
    Metrics<ValueType> &metrics){

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::Array;
    using Teuchos::ArrayView;

    auto tpetraComm = Tpetra::getDefaultComm();

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr scaiComm = dist->getCommunicatorPtr();
    const IndexType thisPE = scaiComm->getRank();
    const IndexType localN= dist->getLocalSize();
    const Tpetra::global_size_t globalN= dist->getGlobalSize();

    //start indexing with 0,(C-style)
    const IndexType indexBase = 0;

    //the locally owned, global indices
    std::vector<IndexType> localGlobalIndices;
    {
        scai::hmemo::HArray<IndexType> myGlobalIndexes;
        dist->getOwnedIndexes(myGlobalIndexes);
        scai::hmemo::ReadAccess<IndexType> rInd(myGlobalIndexes);
        SCAI_ASSERT_EQ_ERROR( rInd.size(), localN, "Wrong distribution size?");
        localGlobalIndices.insert( localGlobalIndices.begin(), rInd.get(), rInd.get()+localN );
    }

    Array<IndexType> elementList( localGlobalIndices );
    SCAI_ASSERT_EQ_ERROR( elementList.size(), localN, "Wrong distribution size?");

    typedef Tpetra::Map<IndexType,IndexType> map_type;
    RCP<const map_type> mapFromDist = rcp( new map_type(globalN, elementList, indexBase, tpetraComm) );

    TEUCHOS_TEST_FOR_EXCEPTION(
        tpetraComm->getSize () > 1 && mapFromDist->isContiguous (),
        std::logic_error,
        "The cyclic Map claims to be contiguous."
    );

    //this is also used later
    const scai::lama::CSRStorage<ValueType>& storage = graph.getLocalStorage();

    //the allocated value per row is the degree of each vertex
    std::vector<unsigned long> degrees(localN);
    {
        const scai::hmemo::ReadAccess<IndexType> ia(storage.getIA());
        assert(ia.size()==localN+1);

        for( int i=0; i<localN; i++){
            degrees[i] = ia[i+1]-ia[i];
            assert(degrees[i]>0);
        }
    }
    ArrayView<unsigned long> numEntriesPerRow(degrees);
    degrees.clear();

    typedef Tpetra::CrsGraph<IndexType,IndexType> crs_graph_type;
    RCP<crs_graph_type> zoltanGraph( new crs_graph_type (mapFromDist, numEntriesPerRow) );

    {
        const scai::hmemo::ReadAccess<IndexType> ia(storage.getIA());
        const scai::hmemo::ReadAccess<IndexType> ja(storage.getJA());
        for( IndexType row=0; row<localN; row++){
            Array<IndexType> rowIndices(numEntriesPerRow[row]);
            int colInd = 0;
            for(int j = ia[row]; j<ia[row+1]; j++) {
                rowIndices[colInd++] = ja[j];
            }

            zoltanGraph->insertGlobalIndices( dist->local2Global(row), rowIndices() );
        }
    }

    //finalize creation of graph
    zoltanGraph->fillComplete();

    const IndexType numWeights = nodeWeights.size();
    const IndexType numEdgeWeights = 0;

    typedef Zoltan2::XpetraCrsGraphAdapter<crs_graph_type> SparseGraphAdapter;
    SparseGraphAdapter grAdapter(zoltanGraph, numWeights, numEdgeWeights );

    //
    //set vertex weights
    //see trilinos/packages/zoltan2/test/helpers/AdapterForTest.hpp
    //

    std::vector<std::vector<ValueType>> localWeights = extractLocalNodeWeights(nodeWeights, nodeWeightsFlag);

    typedef typename Zoltan2::BaseAdapter<ValueType>::scalar_t zscalar;
    std::vector<const zscalar*>weightVec( numWeights );

    for( unsigned int w=0; w<numWeights; w++ ) {
        weightVec[w] = (zscalar *) localWeights[w].data();
    }
    std::vector<int> weightStrides( numWeights, 1);

    for( unsigned int w=0; w<numWeights; w++ ) {
        grAdapter.setVertexWeights( weightVec[w], weightStrides[w], w);
    }

    ///////////////////////////////////////////////////////////////////////
    // Create parameters
    Teuchos::ParameterList params = setParams( algo, settings, repart, numWeights, thisPE);
    //seems that it does not affect much
    params.set("pulp_minimize_maxcut", true);

    scaiComm->synchronize();

    Zoltan2::PartitioningProblem<SparseGraphAdapter> *problem =
        new Zoltan2::PartitioningProblem<SparseGraphAdapter>( &grAdapter, &params);

    if( scaiComm->getRank()==0 )
        std::cout<< "About to call zoltan, algo " << algo << std::endl;

    scai::lama::DenseVector<IndexType> partitionZoltan = runZoltanAlgo( problem, dist, settings, scaiComm, metrics );
    return partitionZoltan;

}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
template<typename Adapter>
scai::lama::DenseVector<IndexType> zoltanWrapper<IndexType, ValueType>::runZoltanAlgo(
    Zoltan2::PartitioningProblem<Adapter> *problem,
    const scai::dmemo::DistributionPtr dist,
    const Settings& settings,
    const scai::dmemo::CommunicatorPtr comm,
    Metrics<ValueType> &metrics ){

	std::chrono::time_point<std::chrono::steady_clock> beforePartTime =  std::chrono::steady_clock::now();
    problem->solve();

    std::chrono::duration<double> partitionTmpTime = std::chrono::steady_clock::now() - beforePartTime;
    double partitionTime= comm->max(partitionTmpTime.count() );

    metrics.MM["timeTotal"] = partitionTime;

    //
    // convert partition to a DenseVector
    //
    scai::lama::DenseVector<IndexType> partitionZoltan(dist, IndexType(0));

    const Zoltan2::PartitioningSolution<Adapter> &solution = problem->getSolution();
    const int *partAssignments = solution.getPartListView();
	const IndexType localN= dist->getLocalSize();
	
    for(unsigned int i=0; i<localN; i++) {
        IndexType thisBlock = partAssignments[i];
        SCAI_ASSERT_LT_ERROR( thisBlock, settings.numBlocks, "found wrong vertex id");
        SCAI_ASSERT_GE_ERROR( thisBlock, 0, "found negetive vertex id");
        partitionZoltan.getLocalValues()[i] = thisBlock;
        //localBlockSize[thisBlock]++;
    }
    for(int i=0; i<localN; i++) {
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        SCAI_ASSERT_EQ_ERROR( partitionZoltan.getLocalValues()[i], partAssignments[i], "Wrong conversion to DenseVector");
    }

    return partitionZoltan;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
Teuchos::ParameterList zoltanWrapper<IndexType, ValueType>::setParams( 
    const std::string algo,
    const Settings settings,
    const bool repart,
	const IndexType numWeights, 
    const IndexType thisPE){

	//zoltan does not accept float 
    const double tolerance = 1+settings.epsilon;
    if (thisPE == 0)
        std::cout << "Imbalance tolerance is " << tolerance << std::endl;

    Teuchos::ParameterList params("zoltan params");
    //TODO: params.set("partitioning_objective", "minimize_cut_edge_count");
    //      or something else, check at
    //      https://docs.trilinos.org/latest-release/packages/zoltan2/doc/html/z2_parameters.html

	//if more than one vertex weights, emphasize in balancing
	if(numWeights>1){
		params.set("partitioning_objective", "balance_object_weight");
	}else{
		params.set("partitioning_objective", "minimize_cut_edge_count");
	}

    params.set("debug_level", "basic_status");
    //params.set("debug_level", "verbose_detailed_status");
    params.set("debug_procs", "0");
    params.set("error_check_level", "debug_mode_assertions");

    params.set("algorithm", algo);
    params.set("imbalance_tolerance", tolerance );
    params.set("num_global_parts", (int)settings.numBlocks );

    params.set("compute_metrics", false);

    // chose if partition or repartition
    if( repart ) {
        params.set("partitioning_approach", "repartition");
    } else {
        params.set("partitioning_approach", "partition");
    }
    //params.print();

    return params;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<std::vector<ValueType>> zoltanWrapper<IndexType, ValueType>::extractLocalNodeWeights(
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag){

    assert(nodeWeights.size()>0);
    const IndexType localN = nodeWeights[0].getDistributionPtr()->getLocalSize();
    const IndexType numWeights = nodeWeights.size();
    std::vector<std::vector<ValueType>> localWeights( numWeights, std::vector<ValueType>( localN, 1.0) );
    
    if( nodeWeightsFlag){
        for( unsigned int w=0; w<numWeights; w++ ) {
            scai::hmemo::ReadAccess<ValueType> rLocalWeights( nodeWeights[w].getLocalValues() );
            assert( rLocalWeights.size()==localN);
            for(unsigned int i=0; i<localN; i++) {
                localWeights[w][i] = rLocalWeights[i];
            }
        }
    }
    return localWeights;
}

//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::string zoltanWrapper<IndexType, ValueType>::tool2String( ITI::Tool tool){
    std::string algo="";
    if( tool==Tool::zoltanRIB )
        algo="rib";
    else if( tool==Tool::zoltanRCB )
        algo="rcb";
    else if (tool==Tool::zoltanMJ)
        algo="multijagged";
    else if (tool==Tool::zoltanXPulp)
        algo="pulp";
    else if (tool==Tool::zoltanSFC)
        algo="hsfc";
    else{
        throw std::runtime_error("ERROR:, given tool " + ITI::to_string(tool) + " does not exist.");
    }
    return algo;
}

//---------------------------------------------------------------------------------------

template class zoltanWrapper<IndexType, double>;
template class zoltanWrapper<IndexType, float>;

}//namespace ITI
