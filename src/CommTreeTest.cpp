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

	//ITI::CommTree<IndexType,ValueType>::commNode::leafCount = 0;
	std::vector<cNode> leaves = {
		// 				{hierachy ids}, numCores, mem, speed
		cNode( std::vector<unsigned int>{0,0}, 4, 8, 50),
		cNode( std::vector<unsigned int>{0,1}, 4, 8, 100),

		cNode( std::vector<unsigned int>{1,0}, 6, 10, 80),
		cNode( std::vector<unsigned int>{1,1}, 6, 10, 90),
		cNode( std::vector<unsigned int>{1,2}, 6, 10, 70),

		cNode( std::vector<unsigned int>{2,0}, 8, 12, 80),
		cNode( std::vector<unsigned int>{2,1}, 8, 12, 90),
		cNode( std::vector<unsigned int>{2,2}, 8, 12, 90),
		cNode( std::vector<unsigned int>{2,3}, 8, 12, 100),
		cNode( std::vector<unsigned int>{2,4}, 8, 12, 90)
	};

	ITI::CommTree<IndexType,ValueType> cTree( leaves );
	cTree.print();

}

}//namespace ITI