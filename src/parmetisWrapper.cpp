#include <parmetis.h>

#include "parmetisWrapper.h"
#include "AuxiliaryFunctions.h"
#include "Mapping.h"

namespace ITI {

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> parmetisWrapper<IndexType, ValueType>::refine(
        const scai::lama::CSRSparseMatrix<ValueType> &graph,
        const std::vector<scai::lama::DenseVector<ValueType>> &coords,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
        const scai::lama::DenseVector<IndexType> &partition,
        const ITI::CommTree<IndexType,ValueType> &commTree,
        struct Settings &settings,
        Metrics<ValueType> &metrics
    ){

    //probably we gonna have problems if the distribution does not have 
    //a consecutive numbering. Fix here or outside
    if( not std::is_same<ValueType,real_t>::value ){
        PRINT("*** Warning, ValueType and real_t do not agree");
    }

    if( sizeof(ValueType)!=sizeof(real_t) ) {
        std::cout<< "WARNING: IndexType size= " << sizeof(IndexType) << " and idx_t size=" << sizeof(idx_t) << "  do not agree, this may cause problems " << std::endl;
    }
    

    SCAI_ASSERT_DEBUG( graph.isConsistent(), graph << " input graph is not consistent" );
    //const scai::dmemo::DistributionPtr graphDist = graph.getRowDistributionPtr();

    // vtxDist is an array of size numPEs and is replicated in every processor
    std::vector<IndexType> vtxDist; 

    std::vector<IndexType> xadj;
    std::vector<IndexType> adjncy;
    // vwgt , adjwgt stores the weights of vertices.
    std::vector<ValueType> vVwgt;

    // tpwgts: array that is used to specify the fraction of
    // vertex weight that should be distributed to each sub-domain for each balance constraint.
    // Here we want equal sizes, so every value is 1/nparts; size = ncons*nparts 
    std::vector<real_t> tpwgts;

    // the xyz array for coordinates of size dim*localN contains the local coords
    std::vector<double> xyzLocal;    

    // ubvec: array of size ncon to specify imbalance for every vertex weigth.
    // 1 is perfect balance and nparts perfect imbalance. Here 1 for now
    std::vector<real_t> ubvec;

    //local number of edges; number of node weights; flag about edge and vertex weights 
    IndexType numWeights=0, wgtFlag=0;

    // options: array of integers for passing arguments.
    std::vector<IndexType> options;

    aux<IndexType,ValueType>::toMetisInterface(
        graph, coords, nodeWeights, commTree, settings, vtxDist, xadj, adjncy,
        vVwgt, tpwgts, wgtFlag, numWeights, ubvec, xyzLocal, options );

    SCAI_ASSERT_EQ_ERROR( tpwgts.size(), numWeights*settings.numBlocks, "Wrong tpwgts size" );
    {
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        SCAI_ASSERT_EQ_ERROR( vtxDist.size(), comm->getSize()+1, "Wrong vtxDist size" );
        const ValueType localWeightSum = std::accumulate( vVwgt.begin(), vVwgt.end(), 0.0);
        SCAI_ASSERT_GT_ERROR( localWeightSum, 0, "Sum of local vertex weights should not be 0" );
    }

    // nparts: the number of parts to partition (=k)
    IndexType nparts= settings.numBlocks;


    // numflag: 0 for C-style (start from 0), 1 for Fortran-style (start from 1)
    IndexType numflag= 0;          
    // edges weights not supported
    IndexType* adjwgt= NULL;

    // output parameters
    //
    // edgecut: the size of cut
    IndexType edgecut;

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    const IndexType localN= dist->getLocalSize();   

    //parmetis requires weights to be integers
    std::vector<IndexType> vwgt( vVwgt.begin(), vVwgt.end() );
    SCAI_ASSERT_EQ_ERROR( vwgt.size(), localN*numWeights, "Wrong weights size" );
    
    //partition and the graph rows must have the same distribution
    SCAI_ASSERT( dist->isEqual( partition.getDistribution()), "Distributions must agree" );
    
    // partition array of size localN, contains the block every vertex belongs
    std::vector<idx_t> partKway( localN );
    scai::hmemo::ReadAccess<IndexType> rLocalPart( partition.getLocalValues() );
    SCAI_ASSERT_EQ_ERROR( rLocalPart.size(), localN , "Wrong partition size" );

    for(int i=0; i<localN; i++){
        partKway[i]= rLocalPart[i];
    }    
    rLocalPart.release();

    // comm: the MPI communicator
    MPI_Comm metisComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &metisComm);
    //int metisRet;    

    PRINT0("About to call ParMETIS_V3_RefineKway in Wrappers::refine");

    //overwrite the default options because parmetis by default neglects the
    //partition if k=p
    std::vector<IndexType>options2(4,1);
    options2[1] = 0; //verbosity
    options2[3] = PARMETIS_PSR_UNCOUPLED; //if k=p (coupled) or not (uncoupled); 2 is always uncoupled

    std::chrono::time_point<std::chrono::steady_clock> startTime =  std::chrono::steady_clock::now();

    ParMETIS_V3_RefineKway(
        vtxDist.data(), xadj.data(), adjncy.data(), vwgt.data(), adjwgt, &wgtFlag, &numflag, &numWeights, &nparts, tpwgts.data() , ubvec.data(), options2.data(), &edgecut, partKway.data(), &metisComm );

    std::chrono::duration<double> partitionKwayTime = std::chrono::steady_clock::now() - startTime;
    double partKwayTime= comm->max(partitionKwayTime.count() );
    metrics.MM["timeLocalRef"] = partKwayTime;

    //
    // convert partition to a DenseVector
    //

    scai::lama::DenseVector<IndexType> partitionKway(dist, scai::hmemo::HArray<IndexType>(localN, partKway.data()) );

    return partitionKway;
}//refine
//-----------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> parmetisWrapper<IndexType, ValueType>::partition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const Tool tool,
    const ITI::CommTree<IndexType,ValueType> &commTree,
    const struct Settings &settings,
    Metrics<ValueType> &metrics) {

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    const IndexType localN= dist->getLocalSize();

    //PRINT0("\t\tStarting the metis wrapper");

    if( comm->getRank()==0 ) {
        std::cout << "\033[1;31m";
        //std::cout << "IndexType size: " << sizeof(IndexType) << " , ValueType size: "<< sizeof(ValueType) << std::endl;
        if( int(sizeof(IndexType)) != int(sizeof(idx_t)) ) {
            std::cout<< "WARNING: IndexType size= " << sizeof(IndexType) << " and idx_t size=" << sizeof(idx_t) << "  do not agree, this may cause problems (even if this print looks OK)." << std::endl;
        }
        if( sizeof(ValueType)!=sizeof(real_t) ) {
            std::cout<< "WARNING: IndexType size= " << sizeof(IndexType) << " and idx_t size=" << sizeof(idx_t) << "  do not agree, this may cause problems " << std::endl;
        }
        std::cout<<"\033[0m";
    }

    assert( commTree.checkTree() );

    //-----------------------------------------------------
    //
    // convert to parMetis data types
    //

    // partition array of size localN, contains the block every vertex belongs
    idx_t *partKway = new idx_t[ localN ];

    // vtxDist is an array of size numPEs and is replicated in every processor
    std::vector<IndexType> vtxDist;

    std::vector<IndexType> xadj;
    std::vector<IndexType> adjncy;
    // vwgt , adjwgt stores the weights of vertices.
    std::vector<ValueType> vVwgt;

    // tpwgts: array that is used to specify the fraction of
    // vertex weight that should be distributed to each sub-domain for each balance constraint.
    // Here we want equal sizes, so every value is 1/nparts; size = ncons*nparts 
    std::vector<double> tpwgts;

    // the xyz array for coordinates of size dim*localN contains the local coords
    std::vector<double> xyzLocal;

    // ubvec: array of size ncon to specify imbalance for every vertex weigth.
    // 1 is perfect balance and nparts perfect imbalance. Here 1 for now
    std::vector<double> ubvec;

    //local number of edges; number of node weights; flag about edge and vertex weights 
    IndexType numWeights=0, wgtFlag=0;

    // options: array of integers for passing arguments.
    std::vector<IndexType> options;

    IndexType newLocalN = aux<IndexType,ValueType>::toMetisInterface(
        graph, coords, nodeWeights, commTree, settings, vtxDist, xadj, adjncy,
        vVwgt, tpwgts, wgtFlag, numWeights, ubvec, xyzLocal, options );

    if( newLocalN==-1){
        return scai::lama::DenseVector<IndexType>(0,0);
    }

    // nparts: the number of parts to partition (=k)
    IndexType nparts= settings.numBlocks;
    // ndims: the number of dimensions
    IndexType ndims = settings.dimensions;      
    // numflag: 0 for C-style (start from 0), 1 for Fortran-style (start from 1)
    IndexType numflag= 0;          
    // edges weights not supported
    IndexType* adjwgt= NULL;

    //parmetis requires weights to be integers
    std::vector<IndexType> vwgt( vVwgt.begin(), vVwgt.end() );

    //
    // OUTPUT parameters
    //

    // edgecut: the size of cut
    IndexType edgecut;

    // comm: the MPI comunicator
    MPI_Comm metisComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &metisComm);

    //PRINT(*comm<< ": xadj.size()= "<< sizeof(xadj) << "  adjncy.size=" <<sizeof(adjncy) );
    //PRINT(*comm << ": "<< sizeof(xyzLocal)/sizeof(real_t) << " ## "<< sizeof(partKway)/sizeof(idx_t) << " , localN= "<< localN);

    //if(comm->getRank()==0){
    //  PRINT("dims=" << ndims << ", nparts= " << nparts<<", ubvec= "<< ubvec << ", options="<< *options << ", wgtflag= "<< wgtflag );
    //}

    //
    // get the partitions with parMetis
    //

    std::chrono::time_point<std::chrono::steady_clock> beforePartTime =  std::chrono::steady_clock::now();

    if( tool==Tool::parMetisGraph ) {
        if( comm->getRank()==0 ) 
            std::cout<< "About to call ParMETIS_V3_PartKway" << std::endl;

        ParMETIS_V3_PartKway( 
            vtxDist.data(), xadj.data(), adjncy.data(), vwgt.data(), adjwgt, &wgtFlag, &numflag, &numWeights, &nparts, tpwgts.data(), ubvec.data(), options.data(), &edgecut, partKway, &metisComm );
    } else if( tool==Tool::parMetisGeom ) {
        if( comm->getRank()==0 )
            std::cout<< "About to call ParMETIS_V3_PartGeom" << std::endl;

        ParMETIS_V3_PartGeomKway( vtxDist.data(), xadj.data(), adjncy.data(), vwgt.data(), adjwgt, &wgtFlag, &numflag, &ndims, xyzLocal.data(), &numWeights, &nparts, tpwgts.data(), ubvec.data(), options.data(), &edgecut, partKway, &metisComm );
    } else if( tool==Tool::parMetisSFC ) {
        if( comm->getRank()==0 ) 
            std::cout<< "About to call ParMETIS_V3_PartSfc" << std::endl;

        ParMETIS_V3_PartGeom( vtxDist.data(), &ndims, xyzLocal.data(), partKway, &metisComm );
    } else {
        //repartition

        //TODO: check if vsize is correct
        idx_t* vsize = new idx_t[localN];
        for(unsigned int i=0; i<localN; i++) {
            vsize[i] = 1;
        }

        /*
        //TODO-CHECK: does repartition requires edge weights?
        IndexType localM = graph.getLocalNumValues();
        adjwgt =  new idx_t[localM];
        for(unsigned int i=0; i<localM; i++){
            adjwgt[i] = 1;
        }
        */
        real_t itr = 1000;  //TODO: check other values too
        if( comm->getRank()==0 ) 
            std::cout<< "About to call ParMETIS_V3_AdaptiveRepart" << std::endl;

        ParMETIS_V3_AdaptiveRepart( vtxDist.data(), xadj.data(), adjncy.data(), vwgt.data(), vsize, adjwgt, &wgtFlag, &numflag, &numWeights, &nparts, tpwgts.data(), ubvec.data(), &itr, options.data(), &edgecut, partKway, &metisComm );

        delete[] vsize;
    }
    if( comm->getRank()==0 and settings.verbose ){
        std::cout << "\tedge cut returned by parMetis: " << edgecut << std::endl;
    }

    std::chrono::duration<double> partitionKwayTime = std::chrono::steady_clock::now() - beforePartTime;
    double partKwayTime= comm->max(partitionKwayTime.count() );
        

    metrics.MM["timeTotal"] = partKwayTime;

    //
    // convert partition to a DenseVector
    //
    scai::lama::DenseVector<IndexType> partitionKway(dist, IndexType(0));
    for(unsigned int i=0; i<localN; i++) {
        partitionKway.getLocalValues()[i] = partKway[i];
    }

    // check correct transformation to DenseVector
    for(int i=0; i<localN; i++) {
        //PRINT(*comm << ": "<< part[i] << " _ "<< partition.getLocalValues()[i] );
        assert( partKway[i]== partitionKway.getLocalValues()[i]);
    }

    delete[] partKway;

    //possible mapping at the end
    if( settings.mappingRenumbering ) {
        PRINT0("Applying renumbering of blocks based on the SFC index of their centers.");
        std::vector<scai::lama::DenseVector<ValueType>> copyCoords;
        if( not partitionKway.getDistribution().isEqual(coords[0].getDistribution()) ) {
            PRINT0("WARNING:\nCoordinates and partition do not have the same distribution.\nRedistributing coordinates to match distribution");
            for( int d=0; d<ndims; d++) {
                copyCoords[d] = coords[d];
                copyCoords[d].redistribute( partitionKway.getDistributionPtr() );
            }
        }
        Mapping<IndexType,ValueType>::applySfcRenumber( coords, nodeWeights, partitionKway, settings );
    }

    return partitionKway;

}

//-----------------------------------------------------------------------------------------

//
//TODO: parMetis assumes that vertices are stored in a consecutive manner. This is not true for a
//      general distribution. Must reindex vertices for parMetis
//
template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> parmetisWrapper<IndexType, ValueType>::repartition (
    const scai::lama::CSRSparseMatrix<ValueType> &graph,
    const std::vector<scai::lama::DenseVector<ValueType>> &coords,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights,
    const bool nodeWeightsFlag,
    const Tool tool,
    const struct Settings &settings,
    Metrics<ValueType> &metrics) {

    // copy graph and reindex
    scai::lama::CSRSparseMatrix<ValueType> copyGraph = graph;
    GraphUtils<IndexType, ValueType>::reindex(copyGraph);

    /*
    {// check that indices are consecutive, TODO: maybe not needed, remove?

        const scai::dmemo::DistributionPtr dist( copyGraph.getRowDistributionPtr() );
        //scai::hmemo::HArray<IndexType> myGlobalIndexes;
        //dist.getOwnedIndexes( myGlobalIndexes );
        const IndexType globalN = graph.getNumRows();
        const IndexType localN= dist->getLocalSize();
        const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

        std::vector<IndexType> myGlobalIndexes(localN);
        for(IndexType i=0; i<localN; i++){
            myGlobalIndexes[i] = dist->local2global( i );
        }

        std::sort( myGlobalIndexes.begin(), myGlobalIndexes.end() );
        SCAI_ASSERT_GE_ERROR( myGlobalIndexes[0], 0, "Invalid index");
        SCAI_ASSERT_LE_ERROR( myGlobalIndexes.back(), globalN, "Invalid index");

        for(IndexType i=1; i<localN; i++){
            SCAI_ASSERT_EQ_ERROR( myGlobalIndexes[i], myGlobalIndexes[i-1]+1, *comm << ": Invalid index for local index " << i);
        }

        //PRINT(*comm << ": min global ind= " <<  myGlobalIndexes.front() << " , max global ind= " << myGlobalIndexes.back() );
    }
    */

    //trying Moritz version that also redistributes coordinates
    const scai::dmemo::DistributionPtr dist( copyGraph.getRowDistributionPtr() );
    //SCAI_ASSERT_NE_ERROR(dist->getBlockDistributionSize(), nIndex, "Reindexed distribution should be a block distribution.");
    SCAI_ASSERT_EQ_ERROR(graph.getNumRows(), copyGraph.getNumRows(), "Graph sizes must be equal.");

    std::vector<scai::lama::DenseVector<ValueType>> copyCoords = coords;
    std::vector<scai::lama::DenseVector<ValueType>> copyNodeWeights = nodeWeights;

    // TODO: use constuctor to redistribute or a Redistributor
    for (IndexType d = 0; d < settings.dimensions; d++) {
        copyCoords[d].redistribute(dist);
    }

    if (nodeWeights.size() > 0) {
        for( unsigned int i=0; i<nodeWeights.size(); i++ ) {
            copyNodeWeights[i].redistribute(dist);
        }
    }

const ITI::CommTree<IndexType,ValueType> commTree;
    scai::lama::DenseVector<IndexType> partition = parmetisWrapper<IndexType, ValueType>::partition( copyGraph, copyCoords, copyNodeWeights, nodeWeightsFlag, tool, commTree, settings, metrics);

    //because of the reindexing, we must redistribute the partition
    partition.redistribute( graph.getRowDistributionPtr() );

    return partition;
}

template class parmetisWrapper<IndexType, real_t>;
template class parmetisWrapper<IndexType, float>;


}//namespace ITI
