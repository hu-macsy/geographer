#include "AuxiliaryFunctions.h"
#include "parhipWrapper.h"


namespace ITI {

typedef IndexType idxtype;

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> parhipWrapper<IndexType, ValueType>::partition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const Tool tool,
    const ITI::CommTree<IndexType,ValueType> &commTree,
    const struct Settings &settings,
    Metrics<ValueType> &metrics){

    const scai::dmemo::CommunicatorPtr comm = graph.getRowDistributionPtr()->getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const IndexType localN= dist->getLocalSize();

    // vtxDist is an array of size numPEs and is replicated in every processor
    std::vector<idxtype> vtxDist; 
    std::vector<idxtype> xadj;
    std::vector<idxtype> adjncy;
    std::vector<ValueType> ivwgt;                // the weights of vertices.

    // tpwgts: array that is used to specify the fraction of
    // vertex weight that should be distributed to each sub-domain for each balance constraint.
    // Here we want equal sizes, so every value is 1/nparts; size = ncons*nparts 
    std::vector<ValueType> tpwgts;

    // the xyz array for coordinates of size dim*localN contains the local coords
    [[maybe_unused]] std::vector<ValueType> xyzLocal;    

    // ubvec: array of size ncon to specify imbalance for every vertex weigth.
    // 1 is perfect balance and nparts perfect imbalance. Here 1 for now
    std::vector<ValueType> ubvec;
    //local number of edges; number of node weights; flag about edge and vertex weights 
    idxtype numWeights=0, wgtFlag=0;
    std::vector<idxtype> options;             // options: array of integers for passing arguments.

    aux<idxtype,ValueType>::toMetisInterface(
        graph, coordinates, nodeWeights, settings, vtxDist, xadj, adjncy,
        ivwgt, tpwgts, wgtFlag, numWeights, ubvec, xyzLocal, options );

    //parhip requires weights to be integers
    std::vector<idxtype> vwgt( ivwgt.begin(), ivwgt.end() );

    int nparts= settings.numBlocks;       // nparts: the number of parts to partition (=k)
    //IndexType ndims = settings.dimensions;      // ndims: the number of dimensions    
    idxtype* adjwgt= NULL;                    // edges weights not supported

    MPI_Comm parhipComm;                         // comm: the MPI communicator
    MPI_Comm_dup(MPI_COMM_WORLD, &parhipComm);

    //parhip related parameters
    [[maybe_unused]] ValueType imbalance;
    int edgecut;                          // edgecut: the size of cut
    bool suppress_output = false;
    int seed = 0;
    int mode = 0;                               //partitioning mode

    switch( tool ){
        case Tool::parhipUltraFastMesh:
            mode =0;
            break;
        case Tool::parhipFastMesh:
            mode = 1;
            break;
        case Tool::parhipEcoMesh:
            mode = 2;
            break;
        case Tool::parhipUltraFastSocial:
            mode =3;
            break;
        case Tool::parhipFastSocial:
            mode = 4;
            break;
        case Tool::parhipEcoSocial:
            mode = 5;
            break;
        default:
            throw std::invalid_argument("Error, wrong mode/tool: "+ to_string(tool) + " provided in parhipWrapper.\nAborting...");
    }
    
    {
        [[maybe_unused]] double memIuse, freeRam, totalMemUse;
        std::tie(memIuse, totalMemUse) = getFreeRam(comm, freeRam, true);
        MSG0("Total mem usage before calling ParHIPPartitionKWay() " << totalMemUse );
    }

    // partition array of size localN, contains the block every vertex belongs
    //IndexType *partKway = new idx_t[ localN ];
    idxtype *partKway = new idxtype[ localN ];

    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

    ParHIPPartitionKWay( 
        vtxDist.data(), xadj.data(), adjncy.data(), vwgt.data(), 
        adjwgt, &nparts, &imbalance, suppress_output, seed, mode,
        &edgecut, partKway, &parhipComm );
    
    std::chrono::duration<double> partitionTime = std::chrono::steady_clock::now() - startTime;
    double partTime= comm->max(partitionTime.count() );
    metrics.MM["timeTotal"] = partTime;

    //
    // convert partition to a DenseVector
    //
    scai::lama::DenseVector<IndexType> partitionKway(dist, IndexType(0));
    for(unsigned int i=0; i<localN; i++) {
        partitionKway.getLocalValues()[i] = partKway[i];
    }

    // check correct transformation to DenseVector
    for(int i=0; i<localN; i++) {
        assert( partKway[i]== partitionKway.getLocalValues()[i]);
    }

    delete[] partKway;

    return partitionKway;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> parhipWrapper<IndexType, ValueType>::repartition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coordinates,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const Tool tool,
    const struct Settings &settings,
    Metrics<ValueType> &metrics){

    return scai::lama::DenseVector<IndexType>( graph.getRowDistributionPtr(), 0 );
}

//---------------------------------------------------------------------------------------

template class parhipWrapper<IndexType, double>;
//template class parhipWrapper<IndexType, float>;

}//namespace ITI