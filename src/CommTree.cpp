/*
 * ComMTree.cpp
 *
 *  Created on: 11.10.2018
 *      Author: tzovas
 */

#include "CommTree.h"


namespace ITI {


//initialize static leaf counter
template <typename IndexType, typename ValueType>
unsigned int ITI::CommTree<IndexType,ValueType>::commNode::leafCount = 0;


//constructor to create tree from a vector of leaves
template <typename IndexType, typename ValueType>
CommTree<IndexType, ValueType>::CommTree(std::vector<commNode> leaves){
	
	//for example, if the vector hierarchy of a leaf is [0,3,2]
	//then the size is 3, but with the implied root, there are 4 levels,
	//thus the +1
	hierarchyLevels = leaves.front().hierarchy.size()+1;
	numLeaves = leaves.size();

	//sanity check
	for( commNode l: leaves){
		SCAI_ASSERT_EQ_ERROR( l.hierarchy.size(), hierarchyLevels-1, "Every leaf must have the same size hierarchy vector");
	}
	numNodes = createTreeFromLeaves( leaves );
}
//------------------------------------------------------------------------


template <typename IndexType, typename ValueType>
IndexType CommTree<IndexType, ValueType>::createTreeFromLeaves( const std::vector<commNode> leaves){

	//bottom level are the leaves
	std::vector<commNode> levelBelow = leaves;
	tree.insert( tree.begin(), levelBelow );
	IndexType size = levelBelow.size();

	IndexType hierarchyLevels = leaves.front().hierarchy.size();
	PRINT("There are " << hierarchyLevels << " levels of hierarchy and " << leaves.size() << " leaves");

	for(int h = hierarchyLevels-1; h>=0; h--){
		PRINT("starting level " << h);
		std::vector<commNode> levelAbove = createLevelAbove(levelBelow);
		//add the newly created level to the tree
		tree.insert(tree.begin(), levelAbove );
		size += levelAbove.size();
		levelBelow = levelAbove;
		PRINT("Size of level above (lvl " << h << ") is " << levelAbove.size() );
	}
	return size;
}//createTreeFromLeaves
//------------------------------------------------------------------------


//WARNING: Needed that 'typename' to compile...
template <typename IndexType, typename ValueType>
std::vector<typename CommTree<IndexType,ValueType>::commNode> CommTree<IndexType, ValueType>::createLevelAbove( const std::vector<commNode> levelBelow ){

	//a hierarchy prefix is the hierarchy vector without the last element
	//commNodes that have the same prefix, belong to the same father node
	typedef std::vector<unsigned int> hierPrefix;

	unsigned int levelBelowsize = levelBelow.size();
	std::vector<bool> seen(levelBelowsize, false);
	PRINT("level below has size " << levelBelowsize );

	//will be used later to normalize the speed
	ValueType maxRelatSpeed = 0;

	std::vector<commNode> aboveLevel;

	//assume nodes with the same parent are not necessarilly adjacent 
	//so we need to for loop
	for( unsigned int i=0; i<levelBelowsize; i++){
		//if this node is already accounted for
		if( seen[i] )
			continue;

		commNode thisNode = levelBelow[i];
		commNode fatherNode = thisNode;
		fatherNode.numChildren = 1;		//direct children are 0

		//get the prefix of this node and store it
		hierPrefix thisPrefix(thisNode.hierarchy.size()-1);
		std::copy(thisNode.hierarchy.begin(), thisNode.hierarchy.end()-1, thisPrefix.begin());

		//update the hierarchy vector of the father
		fatherNode.hierarchy = thisPrefix;

		//num of children in the level below. This is needed to calculate
		//the relative speed correctly
		IndexType numChildren = 1;

		//for debugging, remove or add macro or ...
		//PRINT(i << ": thisNode.id is " << thisNode.leafID << " prefix size " << thisPrefix.size() );

		for( unsigned int j=i+1; j<levelBelowsize; j++){
			commNode otherNode = levelBelow[j];
			//hierarchy prefix of the other node
			hierPrefix otherPrefix(otherNode.hierarchy.size()-1);
			std::copy(otherNode.hierarchy.begin(), otherNode.hierarchy.end()-1, otherPrefix.begin());
			//same prefix means that have the same father
			if( thisPrefix==otherPrefix ){
				fatherNode += otherNode;		//operator += overloading
				seen[j] = true;
				numChildren++;
			}
		}
		//update: changing the way we sum up the relative speeds.
		//speed is relative, so take the average
		//fatherNode.relatSpeed /= numChildren;
		if( fatherNode.relatSpeed>maxRelatSpeed ){
			maxRelatSpeed = fatherNode.relatSpeed;
		}

		aboveLevel.push_back(fatherNode);
	}

	//TODO:maybe this is not necessarily needed. This is how the input is given
	//see also in computePartition the  optWeightAllBlocks vector
	//normalize speeds so all are between 0 and 1
	for( unsigned int i=0; i<aboveLevel.size(); i++){
		aboveLevel[i].relatSpeed /= maxRelatSpeed;
	}

	return aboveLevel;
}//createLevelAbove
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
std::vector<unsigned int> CommTree<IndexType, ValueType>::getGrouping(const std::vector<commNode> thisLevel){

	std::vector<unsigned int> groupSizes;
	unsigned int numNewTotalNodes;//for debugging, printing

	std::vector<cNode> prevLevel = createLevelAbove(thisLevel);

	for( cNode c: prevLevel){
		groupSizes.push_back( c.getNumChildren() );
	}
	//the number of old blocks from the previous, provided partition
	
	numNewTotalNodes = std::accumulate(groupSizes.begin(), groupSizes.end(), 0);
	
	SCAI_ASSERT_EQ_ERROR( numNewTotalNodes, thisLevel.size(), "Vector size mismatch" );
	SCAI_ASSERT_EQ_ERROR( groupSizes.size(), prevLevel.size(), "Vector size mismatch" );
	
	return groupSizes;
}//getGrouping
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
void CommTree<IndexType, ValueType>::print(){

	if( checkTree() ){
		std::cout << "tree has " << hierarchyLevels << " hierarchy levels with total " << numNodes << " nodes and " << numLeaves << " number of leaves" <<std::endl;
		for(int i=0; i<tree.size(); i++){
			PRINT("hierarchy "<< i << " with size " << tree[i].size() );
			for(int j=0; j<tree[i].size(); j++){
				tree[i][j].print();
			}
		}
	}else{
		std::cout<<"Something is wrong" << std::endl;
	}

}//print()
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
bool CommTree<IndexType, ValueType>::checkTree(){

	SCAI_ASSERT_EQ_ERROR( hierarchyLevels, tree.size(), "Mismatch for hierachy levels and tree size");
	SCAI_ASSERT_EQ_ERROR( numLeaves, tree.back().size(), "Mismatch for number of leaves and the size of the nottom hierechy level");
	//check sum of sizes for every level
	SCAI_ASSERT_EQ_ERROR( tree.front().size(), 1 , "Top level of the tree should have size 1, only the root");
	SCAI_ASSERT_EQ_ERROR( numLeaves, tree.front()[0].children.size(), "The root should contain all leaves as children");

	return true;
}

//to force instantiation
template class CommTree<IndexType, ValueType>;
}//ITI