#pragma once

#include "Settings.h"
#include "Metrics.h"


namespace ITI {

template <typename IndexType, typename ValueType>
class CommTree{

public:

struct commNode{
	std::vector<unsigned int> hierarchy;
	std::vector<unsigned int> children;
	unsigned int numCores;
	unsigned int memMB;
	ValueType relatSpeed;
	bool isLeaf;
	static unsigned int leafCount;
	unsigned int leafID = std::numeric_limits<unsigned int>::max();

	commNode( std::vector<unsigned int> hier,
		// std::vector<unsigned int> children,
		 unsigned int c, unsigned int m, ValueType rSp, bool isLeaf=true)
	:	hierarchy(hier),
		//children(children),
		numCores(c),
		memMB(m),
		relatSpeed(rSp),
		isLeaf(isLeaf)
	{
		leafID = leafCount;
		leafCount++;
		//convention: leaf node have their ID as theri only child
		// this will speed up the += operator
		children.resize(1,leafID);
	}

	//this constructor is supposed to be used for non-leaf nodes
	commNode()
	:	hierarchy( std::vector<unsigned int>(1,1)), //TODO: is this right?
		numCores(0),
		memMB(0),
		relatSpeed(0.0),
		isLeaf(false)
	{
		//leafID = leafCount;
		//leafCount++;
	}

	commNode& operator+=( const commNode& c){
		this->numCores += c.numCores;
		this->memMB += c.memMB;
		this->relatSpeed += c.relatSpeed;
		/*//TODO: if leaf count and leafID workds properly then this 
		//check is not needed
		//if it a leaf then it has no children
		if(c.isLeaf){
			this->children.push_back(c.leafID);
		}else{
			this->children.insert( this->children.begin(), c.children.begin(), c.children.end() );
		}
		*/
		this->children.insert( this->children.begin(), c.children.begin(), c.children.end() );
		return *this;
	}

	void print(){
		std::cout 	<< "numCores= " << numCores \
					<< ", memory= " << memMB 	\
					<< ", speed= " << relatSpeed << std::endl;
		std::cout << "hierarchy vector: ";
		for(unsigned int i=0; i<hierarchy.size(); i++){
			std::cout<< hierarchy[i] << ", ";
		}
		std::cout<< std::endl;
		if( isLeaf ){
			std::cout<< "this is a leaf node with id "<< leafID  << std::endl;
		}else{
			std::cout << "contains children: ";
			for(unsigned int i=0; i<children.size(); i++){
				std::cout << children[i] <<  ", " ;
			}
			std::cout<< std::endl;
		}
	}

}; //struct commNode{



//structure used to store the communication tree. used for hierarchical
// partitioning
// tree.size==hierarchyLevels
// tree[i] is the vector of nodes for hierarchy level i
// tree.back() is the vector of all the leaves
std::vector<std::vector<commNode>> tree;

//must be known how many levels the tree has
//(well, it can infered but it is just easier)
IndexType hierarchyLevels;
IndexType numNodes;


/*@brief constructor to create tree from a vector of leaves
*/
CommTree(std::vector<commNode> leaves);

/* @brief Takes a vector of leaves and creates the tree
*/
IndexType createTreeFromLeaves( const std::vector<commNode> leaves);

/* @brief takes a level of the tree and creates the level above it by 
grouping together nodes that have the same last hierarchy index
*/
std::vector<commNode> createLevelAbove( const std::vector<commNode> levelBelow, IndexType hierLevel);



//------------------------------------------------------------------------


};//class CommGraph
}//namespace ITI