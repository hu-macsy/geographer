#include "FileIO.h"
#include "Wrappers.h"

#include "gtest/gtest.h"

using namespace scai;

namespace ITI {

template<typename T>
class WrappersTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    std::string graphPath = projectRoot+"/meshes/";
};

using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(WrappersTest, testTypes);

//-----------------------------------------------

TYPED_TEST( WrappersTest, testRefine ){
    using ValueType = TypeParam;

    std::string fileName = "Grid8x8";
    std::string file = WrappersTest<ValueType>::graphPath + fileName;
    std::ifstream f(file);
    const IndexType dimensions= 2;
    IndexType N, edges;
    f >> N >> edges;

    const dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(file );

    std::vector<DenseVector<ValueType>> coordinates = FileIO<IndexType, ValueType>::readCoords( std::string(file + ".xyz"), N, dimensions);

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const IndexType localN = dist->getLocalSize();
    EXPECT_TRUE(coordinates[0].getDistributionPtr()->isEqual(*dist));

    std::vector<DenseVector<ValueType>> uniformWeights(1, DenseVector<ValueType>( dist, 1.0) );
    //partition

    struct Settings settings;
    const IndexType k = 6;
    settings.numBlocks = k;  

    srand(comm->getRank());

    DenseVector<IndexType> firstPartition(dist, 0);
    for (IndexType i = 0; i < localN; i++) {
        //IndexType blockId = ( (rand() % k) % (comm->getRank()+1) )%k; //heavily imbalanced partition
        IndexType blockId = (rand() % k);
        firstPartition.getLocalValues()[i] = blockId;
    PRINT(comm->getRank() << ": " << blockId);        
    }

    ValueType cut = GraphUtils<IndexType,ValueType>::computeCut( graph ,firstPartition );

    PRINT0("First cut is " << cut );

    DenseVector<IndexType> refinedPartition = Wrappers<IndexType,ValueType>::refine( graph, coordinates, uniformWeights, firstPartition, settings );

    cut = GraphUtils<IndexType,ValueType>::computeCut( graph ,refinedPartition );

    PRINT0("Refined cut is " << cut );

}
}//ITI