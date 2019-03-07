#pragma once

#include "Settings.h"
#include <numeric>

namespace ITI {


template <typename IndexType, typename ValueType>
class CommTree{

public:

/** Hierarchy is a vector of size ewual to the levels of the tree. Every possition
	indicates the tree-node in which this leaf belongs to.
	For example, if hierarchy={1, 0, 2} it means that this PE belongs to tree-node 1
	in the first level; then, within tree-node 1, it belongs to tree-node 0 and inside 0,
	it is leaf-node 2.
	implicit level 0    o
                      / | \
	level 1         o   o   o .... in 1
                       /|\
	level 2           o o o ....   in 0
                      ... | ...
    level 3               o        in 2, leaves

**/    


struct commNode{
	std::vector<unsigned int> hierarchy;
	//TODO: probably, keeping all chidren is not necessary and uses a lot of space
	// replace by keeping only the number of children
	std::vector<unsigned int> children;
	//this is the number of direct children this nodes has
	unsigned int numChildren;
	unsigned int numCores;
	ValueType memMB;
	ValueType relatSpeed;
	bool isLeaf;
	static unsigned int leafCount;
	unsigned int leafID = std::numeric_limits<unsigned int>::max();

	commNode( std::vector<unsigned int> hier,
		unsigned int c, ValueType m, ValueType rSp, bool isLeaf=true)
	:	hierarchy(hier),
		numChildren(0),
		numCores(c),
		memMB(m),
		relatSpeed(rSp),
		isLeaf(isLeaf)
	{
		leafID = leafCount;
		leafCount++;
		//convention: leaf node have their ID as their only child
		// this will speed up the += operator
		children.resize(1,leafID);
	}

	//this constructor is supposed to be used for non-leaf nodes
	commNode()
	:	hierarchy( std::vector<unsigned int>(1,1)), //TODO: is this right?
		numChildren(0), //TODO: check
		numCores(0),
		memMB(0.0),
		relatSpeed(0.0),
		isLeaf(false)
	{
		//TODO/check: if this is not a lef node why it has a leaf id?
		//leafID = leafCount;
		//leafCount++;
	}

	//used to construct the father node from the children
	commNode& operator+=( const commNode& c ){
		this->numCores += c.numCores;
		this->memMB += c.memMB;
		this->relatSpeed += c.relatSpeed;
		this->numChildren++;
		// by convention, leaf nodes have their id as their only child
		this->children.insert( this->children.begin(), c.children.begin(), c.children.end() );
		//nodes are added to form the upper level, so the result of
		//the addition is not a leaf node
		this->isLeaf = false;

		return *this;
	}

	bool operator==( const commNode& c ){
		if( this->hierarchy != c.hierarchy ){
			return false;
		}
		if( this->numChildren != c.numChildren ){
			return false;
		}
		if( this->numCores != c.numCores ){
			return false;
		}
		if( this->memMB != c.memMB ){
			return false;
		}
		if( this->relatSpeed != c.relatSpeed ){
			return false;
		}
		return true;
	}

	bool operator!=( const commNode& c ){
		return not (*this==c);
	}

	/* @brief Return the number of children this node has
	*/
	//TODO: probably children should be removed but this function is needed
	//to know in how many new blocks each block will be partitioned.
	IndexType numAncestors() const{
		return children.size();
	}

	/** @brief The number of direct children of this node
	*/
	IndexType getNumChildren() const{
		return numChildren;
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
		std::cout<< "---" << std::endl;
	}

}; //struct commNode


//structure used to store the communication tree. used for hierarchical
// partitioning
// tree.size==hierarchyLevels
// tree[i] is the vector of nodes for hierarchy level i
// tree.back() is the vector of all the leaves
std::vector<std::vector<commNode>> tree;

//must be known how many levels the tree has
//(well, it can infered but it is just easier)
IndexType hierarchyLevels; //hierarchyLevels = tree.size()
IndexType numNodes;
IndexType numLeaves;

/* @brief Return the root, i.e., hierarchy level 0.
*/
commNode getRoot() const{ 
	//TODO: check if tree is not initialized
	return tree[0][0]; 
}

/* @brief Return the requested hierarchy level
*/
std::vector<commNode> getHierLevel( int level) const {
	SCAI_ASSERT_LE_ERROR( level, hierarchyLevels, "Tree has less levels than requested" );
	return tree[level];
}

/** @brief Return the leaves of the tree
*/
std::vector<commNode> getLeaves() const {
	return tree.back();
}

/*@brief constructor to create tree from a vector of leaves
*/
CommTree(std::vector<commNode> leaves);

/* @brief Takes a vector of leaves and creates the tree
*/
IndexType createTreeFromLeaves( const std::vector<commNode> leaves);

/* @brief Takes a level of the tree and creates the level above it by 
grouping together nodes that have the same last hierarchy index
*/
static std::vector<commNode> createLevelAbove( const std::vector<commNode> levelBelow);

/** How nodes of a hierarchy level are grouped together. A hierarchy level
 is just a vector of nodes. Using the hierarchy prefix of a node, this
 function computes how nodes of this level are grouped together.
 ret.size() = the size of the previous level
 ret.accumulate() = the size of this level
 Example, if ret[0]=3, ret[1]=2 and ret[2]=3 that means that thisLevel[0,1,2] 
 belonged  to the same node in the previous level, i.e., have the same father,
 thisLevel[3,4] belonged to the same node, the same for thisLevel[5,6,7]  etc.

@param[in] thisLevel The input hierarchy level of the the tree.
@return A vector with the number of nodes for each group.
*/
static std::vector<unsigned int> getGrouping(const std::vector<commNode> thisLevel);

/** @brief Calculates the distance of two nodes using their hierarchy labels.
	We assume that leaves with the same father have distance 1. 
	Comparing two hierarchy labels, the distance is their first mismatch.
	In other words, the height of their least common ancestor.
	For example:
	hierarchy1 = { 3, 3, 1, 4, 2}
	hierarchy2 = { 3, 3, 0, 0, 1}
	hierarchy3 = { 0, 3, 1, 4, 2}
	distances(1,2)=3, distacne(1,3)=5, distance(2,3)=5
*/
static ValueType distance( const commNode node1, const commNode node2 );


/** Export the tree as a weighted graph. The edge weigth between to nodes
	is the distance of the nodes in the tree as it is calculates by the
	function distance.
	Remember: only leaves are nodes in the graph. This means that the
	number of nodes in the graph is equal the number of leaves and the
	graph is complete: it inludes all possible edges.

	This is not distributed, it works only localy in every PE.
*/
//TODO: since this a complete matrix, the CSRSparsematrix is not very efficient

static scai::lama::CSRSparseMatrix<ValueType> exportAsGraph_local(const std::vector<commNode> leaves);

scai::lama::CSRSparseMatrix<ValueType> exportAsGraph_local();

/*@brief Print information for the tree
*/
void print();

/* @brief Basic sanity checks for the tree.
*/
bool checkTree();


//------------------------------------------------------------------------


};//class CommTree

typedef typename ITI::CommTree<IndexType,ValueType>::commNode cNode;

}//namespace ITI