#include "gtest/gtest.h"

#include "CommTree.h"


namespace ITI {

class CommTreeTest : public ::testing::Test {
    protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";
};

typedef typename CommTree<IndexType,ValueType>::commNode cNode;

TEST_F(CommTreeTest, testTreeFromLeaves){

	std::vector<cNode> leaves1 = {
		// 				{hierachy ids}, numCores, mem, speed
		cNode( std::vector<unsigned int>{0,0}, 4, 8, 50),
		cNode( std::vector<unsigned int>{0,1}, 4, 8, 90),

		cNode( std::vector<unsigned int>{1,0}, 6, 10, 80),
		cNode( std::vector<unsigned int>{1,1}, 6, 10, 90),
		cNode( std::vector<unsigned int>{1,2}, 6, 10, 70),

		cNode( std::vector<unsigned int>{2,0}, 8, 12, 80),
		cNode( std::vector<unsigned int>{2,1}, 8, 12, 90),
		cNode( std::vector<unsigned int>{2,2}, 8, 12, 90),
		cNode( std::vector<unsigned int>{2,3}, 8, 12, 100),
		cNode( std::vector<unsigned int>{2,4}, 8, 12, 90)
	};

	//re-initialize leaf counter
	ITI::CommTree<IndexType,ValueType>::commNode::leafCount = 0;

	//with one more level
	std::vector<cNode> leaves2 = {
		// 				{hierachy ids}, numCores, mem, speed
		cNode( std::vector<unsigned int>{0,0,0}, 4, 8, 50),//0
		cNode( std::vector<unsigned int>{0,0,1}, 4, 8, 90),
		cNode( std::vector<unsigned int>{0,0,2}, 4, 8, 60),

		cNode( std::vector<unsigned int>{0,1,0}, 4, 8, 90),//3
		cNode( std::vector<unsigned int>{0,1,1}, 4, 8, 90),
		//deliberately wrong order
		cNode( std::vector<unsigned int>{1,0,1}, 6, 10, 80),//5
		cNode( std::vector<unsigned int>{1,0,2}, 6, 10, 80),
		cNode( std::vector<unsigned int>{1,0,0}, 6, 10, 80),

		cNode( std::vector<unsigned int>{1,2,0}, 6, 10, 70),//8
		cNode( std::vector<unsigned int>{1,2,1}, 6, 10, 70),

		cNode( std::vector<unsigned int>{1,1,0}, 6, 10, 90),//10
		cNode( std::vector<unsigned int>{1,1,1}, 6, 10, 90),

		cNode( std::vector<unsigned int>{2,0,0}, 8, 12, 80),//12
		cNode( std::vector<unsigned int>{2,0,3}, 8, 12, 80),
		cNode( std::vector<unsigned int>{2,0,2}, 8, 12, 80),
		cNode( std::vector<unsigned int>{2,0,1}, 8, 12, 80),

		cNode( std::vector<unsigned int>{2,1,0}, 8, 12, 90),//16
		cNode( std::vector<unsigned int>{2,1,1}, 8, 12, 90),
		
	};


	for( std::vector<cNode> leaves: {leaves1, leaves2} ){

		ITI::CommTree<IndexType,ValueType> cTree( leaves );

		{//check getLeaves() function
			std::vector<cNode> tmpLeaves = cTree.getLeaves();
			EXPECT_EQ( leaves.size(), tmpLeaves.size());

			for(unsigned int i=0; i<leaves.size(); i++ ){
				if( leaves[i]!=tmpLeaves[i] ){
					throw std::logic_error( "getLeaves function error" );
				}
			}
		}
		//cTree.print();

		EXPECT_TRUE( cTree.checkTree() );

		EXPECT_EQ( cTree.numLeaves, leaves.size() );
		EXPECT_EQ( cTree.tree[0].size(), 1 ); //top level is just the root
		
		//accumulate numCores, meme and speed for all leaves
		IndexType totNumCores = 0;
		IndexType totMem = 0;
		ValueType totSpeed = 0;
		for( cNode l: leaves){
			totNumCores += l.numCores;
			totMem += l.memMB;
			totSpeed += l.relatSpeed;
		}
		totSpeed /= leaves.size();

		cNode root = cTree.getRoot();//cTree.tree[0][0];
		EXPECT_EQ( root.numCores, totNumCores );
		EXPECT_EQ( root.memMB, totMem );
		//this is wrong, not sure how to test relative speed
		//EXPECT_EQ( root.relatSpeed, totSpeed);
		EXPECT_EQ( root.children.size(), cTree.numLeaves );
	}
}// TEST_F(CommTreeTest, testTreeFromLeaves)


TEST_F(CommTreeTest, testTreeforHomogeneous){

	IndexType k= 12;
	IndexType mem = 1;
	IndexType speed = 1;
	IndexType cores = 1;

	//re-initialize leaf counter
	ITI::CommTree<IndexType,ValueType>::commNode::leafCount = 0;

	std::vector<cNode> leaves;
	for(int i=0; i<k; i++){
	 	leaves.push_back( cNode(std::vector<unsigned int>{0}, cores, mem, speed) );
	}
	
	ITI::CommTree<IndexType,ValueType> cTree( leaves );

	//cTree.print();
	EXPECT_TRUE( cTree.checkTree() );
	EXPECT_EQ( cTree.numLeaves, leaves.size() );
	EXPECT_EQ( cTree.tree[0].size(), 1 ); 
}//TEST_F(CommTreeTest, testTreeforHomogeneous)

}//namespace ITI