#pragma once

#include "Settings.h"
#include <numeric>

namespace ITI {


template <typename IndexType, typename ValueType>
class CommTree{

public:

/** Hierarchy is a vector of size equal to the levels of the tree. Every position
	indicates the tree-node this leaf belongs to.
	For example, if hierarchy={1, 0, 2}, it means that this PE belongs to tree-node 1
	in the first level; then, within tree-node 1, it belongs to tree-node 0 and inside 0,
	it is leaf-node 2.
	implicit level 0    o
                       /|\
	level 1           o o o .... in 1
                       /|\
	level 2           o o o ....   in 0
                      ... | ...
    level 3               o        in 2, leaves

**/    


struct commNode{
	std::vector<unsigned int> hierarchy;
	//TODO: probably, keeping all children is not necessary and uses a lot of space
	// replace by keeping only the number of children
	std::vector<unsigned int> children;
	//this is the number of direct children this node has
	unsigned int numChildren;

	unsigned int numCores;
	//ValueType memMB;
	//ValueType relatSpeed;

	//replace specific variables, such as memMB, with a vector of weights
	std::vector<ValueType> weights;

	bool isLeaf;
	static unsigned int leafCount;
	unsigned int leafID = std::numeric_limits<unsigned int>::max();

/*
	commNode( std::vector<unsigned int> hier,
		unsigned int c, ValueType m, ValueType rSp, bool isLeaf=true)
	:	hierarchy(hier),
		numChildren(0),
		numCores(c),
		//memMB(m),
		//relatSpeed(rSp),
		isLeaf(isLeaf)
	{
		leafID = leafCount;
		leafCount++;
		//convention: leaf node have their ID as their only child
		// this will speed up the += operator
		children.resize(1,leafID);

		weights.assign(1,0); //one weight of value 0
	}
*/

	commNode( std::vector<unsigned int> hier, std::vector<ValueType> leafWeights, bool isLeaf=true)
	:	hierarchy(hier),
		weights(leafWeights),
		isLeaf(isLeaf),
		numChildren(0)
	{
		leafID = leafCount;
		leafCount++;
		//convention: leaf node have their ID as their only child
		// this will speed up the += operator
		children.resize(1,leafID);
	}


	//this constructor is supposed to be used for non-leaf nodes
	//TODO: how to enforce that?
	commNode()
	:	hierarchy( std::vector<unsigned int>(1,1)), //TODO: is this right?
		numChildren(0), //TODO: check
		numCores(0),
		//memMB(0.0),
		//relatSpeed(0.0),
		isLeaf(false)
	{
		weights.assign(1,0); //one weight of value 0
		//TODO/check: If this is not a leaf node, why does it have a leaf id?
		//leafID = leafCount;
		//leafCount++;
	}

	//used to construct the father node from the children
	commNode& operator+=( const commNode& c ){
		this->numCores += c.numCores;

		//sum up the weights of the node
		for(unsigned int i=0; i<this->weights.size(); i++){
			this->weights[i] += c.weights[i];
		}

		this->numChildren++;
		// by convention, leaf nodes have their id as their only child
		this->children.insert( this->children.begin(), c.children.begin(), c.children.end() );
		//nodes are added to form the upper level, so the result of
		//the addition is not a leaf node
		this->isLeaf = false;

		return *this;
	}

	/** @brief Check if two nodes are the same, i.e., if all their values
	are the same except their leaf ID
	*/
	bool operator==( const commNode& c ) const{
		if( this->hierarchy != c.hierarchy ){
			return false;
		}
		if( this->numChildren != c.numChildren ){
			return false;
		}
		if( this->numCores != c.numCores ){
			return false;
		}
		
		for(unsigned int i=0; i<this->weights.size(); i++){
			if( this->weights[i] != c.weights[i])
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

	/** @brief The number of weights that its node has.
	*/
	IndexType getNumWeights() const{
		return weights.size();
	}

	void print() const{
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
		std::cout << weights.size() << " weights:";
		for(int i=0; i<weights.size(); i++){
			std::cout << weights[i] << ", ";
		}
		std::cout << std::endl << "---" << std::endl;
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
IndexType numWeights;
bool areWeightsAdapted = false;
std::vector<bool> isProportional;

/*@brief Default constructor.
*/
CommTree();

/*	@brief Constructor to create tree from a vector of leaves.
	@param[in] leaves The leaf nodes of the tree
	@param[in] isWeightProp A vector of size equal the number of weights
		that each tree node has. It is used to indicate if the
		corresponding weight is proportional or an absolute value.
		For example, node weights can be {64, 0.01} and isWeightProp={ false, true}. An interpretation could be that this node has
		64GB of memory and 1% of the total FLOPS of the system.
*/
CommTree( std::vector<commNode> leaves,
 std::vector<bool> isWeightProp );

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
std::vector<commNode> getLeaves() const {//careful here, this creates a copy, but you treat it like it were a reference
	return tree.back();
}

/** @brief The number of leaves of the tree.
*/
IndexType getNumLeaves() const {
	return tree.back().size();
}

/** @brief The number of weights that its node has.
*/
IndexType getNumWeights() const{
	return numWeights;
}

/* @brief Takes a vector of leaves and creates the tree
*/
IndexType createTreeFromLeaves( const std::vector<commNode> leaves);

/** @brief Creates an artificial flat tree with only one hierarchy level.
This mainly used when no communication is provided. All leaf nodes have
the same weight.
*/
IndexType createFlatHomogeneous( const IndexType numLeaves, const IndexType numNodeWeights = 1 );

/** @brief Given the desired sizes of the blocks, we construct a flat 
tree with one level where every leaf node has different weight.
This mainly used when  only block sizes are provided for partitioning.
**/
IndexType createFlatHeterogeneous( const std::vector<std::vector<ValueType>> &leafSizes );

/** Creates a vector of leaves with only one hierarchy level, i.e., a flat
tree. There can be multilpe weights for each leaf.
**/
std::vector<commNode> createLeaves( const std::vector<std::vector<ValueType>> &sizes);

/* @brief Takes a level of the tree and creates the level above it by 
grouping together nodes that have the same last hierarchy index
*/
static std::vector<commNode> createLevelAbove( const std::vector<commNode> &levelBelow);

/** Weights of leaf nodes can be given as relative values. Given specific
node weights, adapt them so now, leaf weights are calculated according 
to the provided node weights.
*/

void adaptWeights( const std::vector<scai::lama::DenseVector<ValueType>> &leafSizes );

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
std::vector<unsigned int> getGrouping(const std::vector<commNode> thisLevel) const;

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

scai::lama::CSRSparseMatrix<ValueType> exportAsGraph_local(const std::vector<commNode> leaves) const;

scai::lama::CSRSparseMatrix<ValueType> exportAsGraph_local() const;

/** Overloaded version that takes as input the communication tree
*/

//TODO: turn to static? move to other class?
std::vector<ValueType> computeImbalance(
    const scai::lama::DenseVector<IndexType> &part,
    IndexType k,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeight);


/** @brief Given a hierarchy level, it extracts the relative speed
of every PE (remember: every node in a hierarchy level is either a single
PE, if the level is the leaves, or a group of PEs) and calculates what
is the optimum weight each PE should have. Mainly used to comoute imbalance.
*/
//TODO: leave as static?
//std::vector<ValueType> getOptBlockWeights(
//    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights) const;

/** Returns a vector for every balance constrain.
	return.size()==the number of constrains
	return[i].size()==number of leaves, for all i
*/
//	03/19: We have two constrains: memory and cpu speed for every PE.
//	These two vectors are returned.
//First is the memory and then the cpu speed
std::vector<std::vector<ValueType>> getBalanceVectors() const;

/*@brief Print information for the tree
*/
void print() const;

/* @brief Basic sanity checks for the tree.
*/
bool checkTree( bool all=false ) const;



//------------------------------------------------------------------------


};//class CommTree

typedef typename ITI::CommTree<IndexType,ValueType>::commNode cNode;

}//namespace ITI