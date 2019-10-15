#include "FileIO.h"
#include "Wrappers.h"
#include "AuxiliaryFunctions.h"
#include "parmetisWrapper.h"

#include "gtest/gtest.h"


using namespace scai;

namespace ITI {

//template<typename T>
class WrappersTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    std::string graphPath = projectRoot+"/meshes/";
};

//warning: currently, the wrappers wotk only for ValueType double 
//using testTypes = ::testing::Types<double,float>;
//TYPED_TEST_SUITE(WrappersTest, testTypes);

//-----------------------------------------------

TEST_F( WrappersTest, testRefine ){
    using ValueType = double;

    std::string fileName = "Grid32x32";
    std::string file = WrappersTest::graphPath + fileName;
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

    std::vector<DenseVector<ValueType>> uniformWeights(3, DenseVector<ValueType>( dist, 1.41) );


    //partition

    struct Settings settings;
    const IndexType k = comm->getSize()-1;
    settings.numBlocks = k;  
    settings.dimensions = dimensions;
    Metrics<ValueType> metrics(settings);

    srand(comm->getRank());

    DenseVector<IndexType> firstPartition(dist, 0);
    for (IndexType i = 0; i < localN; i++) {
        //heavily imbalanced partition
        //IndexType blockId = ( (rand() % k) % (comm->getRank()+1) )%k; 
        IndexType blockId = (rand() % k);
        firstPartition.getLocalValues()[i] = blockId;
    }

    aux<IndexType,ValueType>::print2DGrid( graph, firstPartition );

    const ValueType cut = GraphUtils<IndexType,ValueType>::computeCut( graph , firstPartition );
	ValueType imbalance = GraphUtils<IndexType,ValueType>::computeImbalance( firstPartition, settings.numBlocks );

    PRINT0("First cut is " << cut << " and imbalance " << imbalance );

    Wrappers<IndexType,ValueType>* parMetis = new parmetisWrapper<IndexType,ValueType>;
    DenseVector<IndexType> refinedPartition = parMetis->refine( graph, coordinates, uniformWeights, firstPartition, settings, metrics );

    aux<IndexType,ValueType>::print2DGrid( graph, refinedPartition );

    const ValueType refCut = GraphUtils<IndexType,ValueType>::computeCut( graph ,refinedPartition );
	imbalance = GraphUtils<IndexType,ValueType>::computeImbalance( refinedPartition, settings.numBlocks );   

    PRINT0("Refined cut is " << refCut << " and imbalance " << imbalance );
	
	EXPECT_LE( refCut, cut);

}
}//ITI
