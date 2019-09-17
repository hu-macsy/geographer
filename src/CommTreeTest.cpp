#include "gtest/gtest.h"

#include "CommTree.h"


namespace ITI {

template<typename T>
class CommTreeTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    std::string graphPath = projectRoot+"/meshes/";
};
using testTypes = ::testing::Types<double,float>;
TYPED_TEST_SUITE(CommTreeTest, testTypes);

//-----------------------------------------------

typedef typename CommTree<IndexType,ValueType>::commNode cNode;

TYPED_TEST(CommTreeTest, testTreeFromLeaves) {

    std::vector<cNode> leaves1 = {
        // 				{hierachy ids}, numCores, mem, speed
        cNode( std::vector<unsigned int>{0,0}, std::vector<ValueType> {0.4, 8, 50} ),
        cNode( std::vector<unsigned int>{0,1}, std::vector<ValueType> {0.4, 8, 90} ),

        cNode( std::vector<unsigned int>{1,0}, std::vector<ValueType> {0.6, 10, 80} ),
        cNode( std::vector<unsigned int>{1,1}, std::vector<ValueType> {0.3, 10, 90} ),
        cNode( std::vector<unsigned int>{1,2}, std::vector<ValueType> {0.5, 10, 70} ),

        cNode( std::vector<unsigned int>{2,0}, std::vector<ValueType> {0.8, 12, 80} ),
        cNode( std::vector<unsigned int>{2,1}, std::vector<ValueType> {0.58, 12, 90} ),
        cNode( std::vector<unsigned int>{2,2}, std::vector<ValueType> {0.2, 12, 90} ),
        cNode( std::vector<unsigned int>{2,3}, std::vector<ValueType> {0.41, 12, 100} ),
        cNode( std::vector<unsigned int>{2,4}, std::vector<ValueType> {0.11, 12, 90} )
    };
    std::vector<bool> isProp = { true, false, false };

    //re-initialize leaf counter
    ITI::CommTree<IndexType,ValueType>::commNode::leafCount = 0;

    //with one more level
    std::vector<cNode> leaves2 = {
        // 				{hierachy ids}, numCores, mem, speed  	//leafID
        cNode( std::vector<unsigned int>{0,0,0}, {4, 8, 50} ),	//0
        cNode( std::vector<unsigned int>{0,0,1}, {4, 0.8, 90} ),
        cNode( std::vector<unsigned int>{0,0,2}, {4, 0.8, 60} ),

        cNode( std::vector<unsigned int>{0,1,0}, {4, 0.8, 90} ),//3
        cNode( std::vector<unsigned int>{0,1,1}, {4, 0.8, 90} ),
        //deliberately wrong order
        cNode( std::vector<unsigned int>{1,0,1}, {6, 0.10, 80} ),//5
        cNode( std::vector<unsigned int>{1,0,2}, {6, 0.10, 80} ),
        cNode( std::vector<unsigned int>{1,0,0}, {6, 0.10, 80} ),

        cNode( std::vector<unsigned int>{1,2,0}, {6, 0.10, 70} ),//8
        cNode( std::vector<unsigned int>{1,2,1}, {6, 0.10, 70} ),

        cNode( std::vector<unsigned int>{1,1,0}, {6, 0.10, 90} ),//10
        cNode( std::vector<unsigned int>{1,1,1}, {6, 0.10, 90} ),

        cNode( std::vector<unsigned int>{2,0,0}, {8, 0.12, 80} ),//12
        cNode( std::vector<unsigned int>{2,0,3}, {8, 0.12, 80} ),
        cNode( std::vector<unsigned int>{2,0,2}, {8, 0.12, 80} ),
        cNode( std::vector<unsigned int>{2,0,1}, {8, 0.12, 80} ),

        cNode( std::vector<unsigned int>{2,1,0}, {8, 0.12, 90} ),//16
        cNode( std::vector<unsigned int>{2,1,1}, {8, 0.12, 90} ),
    };


    for( std::vector<cNode> leaves: {
                leaves1, leaves2
            } ) {

        ITI::CommTree<IndexType,ValueType> cTree( leaves, isProp );

        {   //check getLeaves() function
            std::vector<cNode> tmpLeaves = cTree.getLeaves();
            EXPECT_EQ( leaves.size(), tmpLeaves.size());

            for(unsigned int i=0; i<leaves.size(); i++ ) {
                if( leaves[i]!=tmpLeaves[i] ) {
                    throw std::logic_error( "getLeaves function error" );
                }
            }
        }
        //cTree.print();

        EXPECT_TRUE( cTree.checkTree(true) );
        EXPECT_EQ( cTree.getNumLeaves(), leaves.size() );
        EXPECT_EQ( cTree.getHierLevel(0).size(), 1 ); //top level is just the root

        //accumulate weights for all leaves
        std::vector<ValueType> sumWeights( 3, 0.0 );

        for( cNode l: leaves) {
            for(int i=0; i<3; i++) {
                sumWeights[i] += l.weights[i];
            }
        }

        cNode root = cTree.getRoot();//cTree.tree[0][0];
        for(int i=0; i<3; i++) {
            EXPECT_FLOAT_EQ( root.weights[i], sumWeights[i] ) << " for weight " << i;
        }

        EXPECT_EQ( root.children.size(), cTree.getNumLeaves() );
    }
}// TEST_F(CommTreeTest, testTreeFromLeaves)

//------------------------------------------------------------------------

TYPED_TEST(CommTreeTest, testTreeFlatHomogeneous) {

    IndexType k= 12;

    ITI::CommTree<IndexType,ValueType> cTree;
    IndexType numNodes = cTree.createFlatHomogeneous( k );

    //cTree.print();
    EXPECT_TRUE( cTree.checkTree(true) );
    EXPECT_EQ( cTree.getNumLeaves(), k );
    EXPECT_EQ( cTree.getHierLevel(0).size(), 1 );
    EXPECT_EQ( numNodes, cTree.getNumNodes() );
    EXPECT_EQ( numNodes, k+1);
}//TYPED_TEST(CommTreeTest, testTreeFlatHomogeneous)

//------------------------------------------------------------------------

TYPED_TEST(CommTreeTest, testTreeNonFlatHomogeneous) {

    IndexType k= 2*3*4*5;

    ITI::CommTree<IndexType,ValueType> cTree( {2,3,4,5}, 3);

    //cTree.print();
    EXPECT_TRUE( cTree.checkTree(true) );
    EXPECT_EQ( cTree.getNumLeaves(), k );
    EXPECT_EQ( cTree.getHierLevel(0).size(), 1 );

}//TYPED_TEST(CommTreeTest, testTreeNonFlatHomogeneous)

//------------------------------------------------------------------------

TYPED_TEST(CommTreeTest, testLabelDistance) {

    std::vector<cNode> nodes = {
        // 				      {hierachy ids}, numCores, mem, speed
        cNode( std::vector<unsigned int>{0,0,0,0,0}, {4, 8, 60} ),//0
        cNode( std::vector<unsigned int>{0,0,1,2,3}, {4, 8, 60} ),//1
        cNode( std::vector<unsigned int>{2,0,2,1,3}, {4, 8, 60} ),//2
        cNode( std::vector<unsigned int>{2,0,2,1,0}, {4, 8, 60} ),//3
        cNode( std::vector<unsigned int>{2,1,0,0,0}, {4, 8, 60} ),//4
        cNode( std::vector<unsigned int>{2,1,0,0,0}, {4, 8, 60} )
    };


    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[0], nodes[1])), 3 );
    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[0], nodes[2])), 5 );
    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[2], nodes[3])), 1 );
    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[1], nodes[3])), 5 );
    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[3], nodes[4])), 4 );
    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[2], nodes[4])), 4 );
    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[4], nodes[4])), 0 );
    //this throws a warning too because nodes 4 and 5 have identical hierarchy label
    //but have different leaf IDs
    EXPECT_EQ( (CommTree<IndexType,ValueType>::distance(nodes[4], nodes[5])), 0 );


}//TYPED_TEST(CommTreeTest, testLabelDistance)
//------------------------------------------------------------------------

TYPED_TEST(CommTreeTest, testExportGraph) {

    std::vector<cNode> leaves = {
        // 				{hierachy ids}, numCores, mem, speed , //leaf IDs
        cNode( std::vector<unsigned int>{0,0,0}, {4, 8, 50} ),//0
        cNode( std::vector<unsigned int>{0,0,1}, {4, 8, 90} ),
        cNode( std::vector<unsigned int>{0,0,2}, {4, 8, 60} ),

        cNode( std::vector<unsigned int>{0,1,0}, {4, 8, 90} ),//3
        cNode( std::vector<unsigned int>{0,1,1}, {4, 8, 90} ),
        //deliberately wrong order
        cNode( std::vector<unsigned int>{1,0,1}, {6, 10, 80} ),//5
        cNode( std::vector<unsigned int>{1,0,2}, {6, 10, 80} ),
        cNode( std::vector<unsigned int>{1,0,0}, {6, 10, 80} ),

        cNode( std::vector<unsigned int>{1,2,0}, {6, 10, 70} ),//8
        cNode( std::vector<unsigned int>{1,2,1}, {6, 10, 70} ),

        cNode( std::vector<unsigned int>{1,1,0}, {6, 10, 90} ),//10
        cNode( std::vector<unsigned int>{1,1,1}, {6, 10, 90} ),

        cNode( std::vector<unsigned int>{2,0,0}, {8, 12, 80} ),//12
        cNode( std::vector<unsigned int>{2,0,3}, {8, 12, 80} ),
        cNode( std::vector<unsigned int>{2,0,2}, {8, 12, 80} ),
        cNode( std::vector<unsigned int>{2,0,1}, {8, 12, 80} ),

        cNode( std::vector<unsigned int>{2,1,0}, {8, 12, 90} ),//16
        cNode( std::vector<unsigned int>{2,1,1}, {8, 12, 90} ),
    };

    const ITI::CommTree<IndexType,ValueType> cTree( leaves, {0,0,0} );

    const scai::lama::CSRSparseMatrix<ValueType> PEgraph = cTree.exportAsGraph_local();

    const IndexType N = PEgraph.getNumRows();
    EXPECT_EQ( N, cTree.getNumLeaves() );

    //complete graph
    EXPECT_EQ( PEgraph.getNumValues(), N*(N-1) );

    SCAI_ASSERT_LE_ERROR( \
                          scai::utilskernel::HArrayUtils::max(PEgraph.getLocalStorage().getIA()),\
                          PEgraph.getNumValues(), "some ia value is too large" );

    const scai::lama::CSRStorage<ValueType>& PEstorage = PEgraph.getLocalStorage();
    const scai::hmemo::HArray<ValueType> values = PEstorage.getValues();
    const ValueType max = scai::utilskernel::HArrayUtils::max( values );
    //This tree has depth 3 so this should be the maximum edge weight in the graph.
    EXPECT_EQ( max, 3 );

    //test specific edge weights
    EXPECT_EQ( PEstorage.getValue(0,2), 1 );
    EXPECT_EQ( PEstorage.getValue(12,13), 1 );

    EXPECT_EQ( PEstorage.getValue(2,4), 2 );
    EXPECT_EQ( PEstorage.getValue(8,10), 2 );

    EXPECT_EQ( PEstorage.getValue(3,12), 3 );
    EXPECT_EQ( PEstorage.getValue(0,17), 3 );

    const std::vector<cNode> getLeaves = cTree.getLeaves();
    EXPECT_EQ( leaves, getLeaves );


TYPED_TEST//------------------------------------------------------------------------

TYPED_TEST(CommTreeTest, testAdaptWeights) {

    std::vector<cNode> leaves = {
        // 				{hierachy ids}, numCores, mem, speed
        cNode( std::vector<unsigned int>{0,0}, {46, 0.2, 50} ),
        cNode( std::vector<unsigned int>{0,1}, {22, 0.1, 90} ),

        cNode( std::vector<unsigned int>{1,0}, {68, 0.01, 80} ),
        cNode( std::vector<unsigned int>{1,1}, {109, 0.12, 90} ),
        cNode( std::vector<unsigned int>{1,2}, {55, 0.13, 70} ),

        cNode( std::vector<unsigned int>{2,0}, {86, 0.3, 80} ),
        cNode( std::vector<unsigned int>{2,1}, {81, 0.04, 90} ),
        cNode( std::vector<unsigned int>{2,2}, {18, 0.05, 90} ),
        cNode( std::vector<unsigned int>{2,3}, {181, 0.11, 100} ),
        cNode( std::vector<unsigned int>{2,4}, {53, 0.23, 90} )
    };

    //1st and 3rd weights are proportional
    ITI::CommTree<IndexType,ValueType> cTree( leaves, { false, true, false } );

    IndexType N = 500; //the nodes of the supposed graph
    std::vector<scai::lama::DenseVector<ValueType>> nodeWeights(3);

    for( int i=0; i<3; i++ ) {
        nodeWeights[i].setRandom( N, 1 );
    }

    cTree.adaptWeights( nodeWeights );

    EXPECT_TRUE( cTree.checkTree( true ) );
}

}//namespace ITI