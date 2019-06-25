#include "gtest/gtest.h"

#include "Mapping.h"
#include "GraphUtils.h"
#include "FileIO.h"
#include "Metrics.h"
//#include "Settings.h"

namespace ITI {

class MappingTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

TEST_F(MappingTest, testTorstenMapping) {

    //std::string fileName = "bigtrace-00000.graph";
    std::string fileName = "Grid4x4";
    std::string file = graphPath + fileName;
    Settings settings;
    settings.dimensions = 2;
    settings.numBlocks = 1;
    bool executed = true;

    //TODO:probably there is a better way to do that: exit tests as failed if p>7. Regular googletest tests
    // do not exit the test. Maybe by using death test, but I could not figure it out.

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if( comm->getSize()>6 ) {
        PRINT0("\n\tWARNING:File too small to read. If p>6 this test is not executed\n" );
        executed=false;
    } else {

        scai::lama::CSRSparseMatrix<ValueType> blockGraph = FileIO<IndexType, ValueType>::readGraph(file );
        const IndexType N = blockGraph.getNumRows();

        //This tests works only when np=1. Replicate the graph
        //TODO: of add error message to execute the test only whne np=1
        scai::dmemo::DistributionPtr noDist (new scai::dmemo::NoDistribution( N ));
        blockGraph.redistribute( noDist, noDist );

        scai::lama::CSRSparseMatrix<ValueType> PEGraph (blockGraph);


        blockGraph.setValue( 4, 8, 2 );
        blockGraph.setValue( 9, 10, 3 );
        blockGraph.setValue( 3, 7, 2.2 );

        PEGraph.setValue( 14, 15, 3.7);
        PEGraph.setValue( 6, 10, 4.3);

        std::vector<IndexType> mapping = Mapping<IndexType, ValueType>::torstenMapping_local(
                                             blockGraph, PEGraph );

        bool valid = Mapping<IndexType,ValueType>::isValid(blockGraph, PEGraph, mapping);
        EXPECT_TRUE( valid );

        PRINT("torsten mapping");
        //WARNING: if we call it as "Metrics metrics();" it throws an error
        Metrics metrics(settings);
        metrics.getMappingMetrics( blockGraph, PEGraph, mapping );

        PRINT("identity mapping");
        std::vector<IndexType> identityMapping(N,0);
        std::iota( identityMapping.begin(), identityMapping.end(), 0);
        metrics.getMappingMetrics( blockGraph, PEGraph, identityMapping );

        //print mapping
        const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        if( N<65 and comm->getRank()==0 ) {
            std::cout << "Mapping:" << std::endl;
            for(int i=0; i<N; i++) {
                std::cout << i << " <--> " << mapping[i] << std::endl;
            }
        }
        //metrics.print( std::cout );
    }
    EXPECT_TRUE(executed ) << "too many PEs, must be <7 for this test" ;
}

TEST_F(MappingTest, testSfcMapping) {

    //std::string fileName = "bigtrace-00000.graph";
    std::string fileName = "Grid32x32";
    std::string file = graphPath + fileName;

    std::ifstream f(file);
    IndexType N;
    f >> N;

    Settings settings;
    settings.dimensions = 2;

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, settings.dimensions);

    const scai::dmemo::DistributionPtr dist = coords[0].getDistributionPtr();
    scai::lama::DenseVector<IndexType> partition(dist, 1);

    const IndexType localN = partition.getDistributionPtr()->getLocalSize();
    SCAI_ASSERT_EQ_ERROR( localN, coords[0].getDistributionPtr()->getLocalSize(), "Size mismatch" );
    SCAI_ASSERT_EQ_ERROR( partition.size(), coords[0].size(), "Size mismatch");

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const IndexType thisPE = comm->getRank();
    const IndexType numPEs = comm->getSize();
    settings.numBlocks = numPEs;

    //set the local part of the partition vector
    //TODO: reconsider the initial partition; maybe get a sfc or ms partition?
    {
        //set local part: 0 sets 1, 1 sets 2, ...., last PE sets 0
        const IndexType	initPart = (thisPE+1)%numPEs;
        scai::hmemo::WriteAccess<IndexType> wPart( partition.getLocalValues() );
        for( int i=0; i<localN; i++) {
            wPart[i] = initPart;
        }
    }

    //set node weights
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(1);
    //nodeWeights[0] = scai::lama::DenseVector<IndexType>(dist, 1.0);
    nodeWeights[0].assign( scai::hmemo::HArray<ValueType>(std::vector<ValueType>(N,1.0)) );

    //get some metrics before renumbering
    std::vector<ValueType> optBlockSizes( settings.numBlocks, N/settings.numBlocks ); //uniform size blocks

    ValueType imbalanceBefore = GraphUtils<IndexType,ValueType>::computeImbalance( partition, settings.numBlocks);

    std::vector<IndexType> mapping = Mapping<IndexType,ValueType>::applySfcRenumber( coords, nodeWeights, partition, settings );

    SCAI_ASSERT_EQ_ERROR( mapping.size(), settings.numBlocks, "Size mismatch" );

    for(IndexType i=0; i<settings.numBlocks; i++) {
        PRINT0("block " << mapping[i] <<" is renumbered to " << i );
        if( settings.numBlocks==4 or settings.numBlocks==8) {
            //in this case, newBlock = oldBlock-1
            SCAI_ASSERT_EQ_ERROR( (mapping[i]+1)%numPEs, i%numPEs, "Wrong renumbering" );
        }
    }

    ValueType imbalanceAfter = GraphUtils<IndexType,ValueType>::computeImbalance( partition, settings.numBlocks);
    IndexType newK = *std::max_element( mapping.begin(), mapping.end() ) +1 ;

    IndexType mapSum = std::accumulate(mapping.begin(), mapping.end(), 0);

    SCAI_ASSERT_EQ_ERROR( mapSum, newK*(newK-1)/2, "Some block id is missing" );
    SCAI_ASSERT_EQ_ERROR( newK, settings.numBlocks, "Number of blocks changed(!)" );
    SCAI_ASSERT_EQ_ERROR( imbalanceBefore, imbalanceAfter, "Imbalance mismatch, some blocks has different size after renumbering" );

}


}//namespace ITI
