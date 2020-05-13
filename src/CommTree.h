#pragma once

#include <numeric>

#include "Settings.h"

namespace ITI {

/** @brief A tree structure to store the physical network in the form of a tree.

 A communication tree for storing information about the physical network if this can represented
in a hierarchical fashion. The actual compute nodes, the PEs, are only the leaves and they contain
a vector of weights to model different properties like CPU speed, memory, number of cores etc.
Intermediate node model some aggregation of PEs, like racks, chips, islands etc, and the weights
are added as we move up. So, the root node holds the sum for every weight.
*/

template <typename IndexType, typename ValueType>
class CommTree {

public:

    /** @brief A node of the communication tree.

     	Hierarchy is a vector of size equal to the levels of the tree. Every position
    	indicates the tree-node this leaf belongs to.
    	For example, if hierarchy={1, 0, 2}, it means that this PE belongs to tree-node 1
    	in the first level; then, within tree-node 1, it belongs to tree-node 0 and inside 0,
    	it is leaf-node 2.

    	\verbatim
    	implicit level 0    o
                           /|\
    	level 1           o o o ....   in child 0 (there is only the root)
                       ... /|\ ...
    	level 2           o o o ....   in child 1
                          ... | ...
        level 3               o        in child 2. These are the leaves
    	\endverbatim
    **/


    struct commNode {
        /** A hierarchy label for this node; size=number of levels. For example, if hierarchy={1,0,1,4} means that
        this node belongs to the 2nd node in level 0, to the 1st in level 1 , to the 2nd in level 2 and to the 5th
        in level 3. In general, if hierarchy[i]=x means that this node belong to  node i in level x.
        */
        std::vector<unsigned int> hierarchy;
        //TODO: probably, keeping all children is not necessary and uses a lot of space
        // replace by keeping only the number of children
        std::vector<unsigned int> children;

        unsigned int numChildren; 		///< this is the number of direct children this node has

        unsigned int numCores;			///< number of cores this processor has
        std::vector<ValueType> weights;	///< each node can have multiple weights
        bool isLeaf;					///< if this is leaf node or not
        static unsigned int leafCount;	///< number of leafs;
        /// a unique id only for leaf nodes
        unsigned int leafID = std::numeric_limits<unsigned int>::max();


        /** Constructor.
        @param[in] hier The hierarchy label for this node.
        @param[in] allWeights Each node can have several weights. These can represent, for example, computational speed
        of the node, memory capacity, number of cores e.t.c.
        @param[in] isLeaf If this node is a leaf or not
        */
        commNode( std::vector<unsigned int> hier, std::vector<ValueType> allWeights, bool isLeaf=true)
            :	hierarchy(hier),
              numChildren(0),
              weights(allWeights),
              isLeaf(isLeaf)

        {
            leafID = leafCount;
            leafCount++;
            //convention: leaf node have their ID as their only child
            // this will speed up the += operator
            children.resize(1,leafID);
        }


        //this constructor is supposed to be used for non-leaf nodes
        //TODO: how to enforce that?
        /** Default constructor for non-leaf nodes.
        */
        commNode()
            :	hierarchy( std::vector<unsigned int>(1,1)), //TODO: is this right?
              numChildren(0), //TODO: check
              numCores(0),
              isLeaf(false)
        {
            weights.assign(1,0); //one weight of value 0
            //TODO/check: If this is not a leaf node, why does it have a leaf id?
            //leafID = leafCount;
            //leafCount++;
        }

        /** Used to construct the father node from the children. The weights of the two nodes are added
        and their children are merged
        */
        commNode& operator+=( const commNode& c ) {
            this->numCores += c.numCores;

            //sum up the weights of the node
            for(unsigned int i=0; i<this->weights.size(); i++) {
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

        /** @brief Check if two nodes are the same, i.e., if all their values are the same except their leaf ID
        */
        bool operator==( const commNode& c ) const {
            if( this->hierarchy != c.hierarchy ) {
                return false;
            }
            if( this->numChildren != c.numChildren ) {
                return false;
            }
            if( this->numCores != c.numCores ) {
                return false;
            }

            for(unsigned int i=0; i<this->weights.size(); i++) {
                if( this->weights[i] != c.weights[i])
                    return false;
            }
            return true;
        }

        bool operator!=( const commNode& c ) {
            return not (*this==c);
        }

//TODO: these are only leaf nodes or all the nodes of the subtree?
        /* @brief Return the number of all ancestors this node has.
        */
        //TODO: probably children should be removed but this function is needed
        //to know how many new blocks each block will be partitioned.
        IndexType numAncestors() const {
            return children.size();
        }

        /** @brief The number of direct children of this node
        */
        IndexType getNumChildren() const {
            return numChildren;
        }

        /** @brief The number of weights that its node has.
        */
        IndexType getNumWeights() const {
            return weights.size();
        }

        /**@brief Print a node.
        */
        void print() const {
            std::cout << "hierarchy vector: ";
            for(unsigned int i=0; i<hierarchy.size(); i++) {
                std::cout<< hierarchy[i] << ", ";
            }
            std::cout<< std::endl;
            if( isLeaf ) {
                std::cout<< "this is a leaf node with id "<< leafID  << std::endl;
            } else {
                std::cout << "contains children: ";
                for(unsigned int i=0; i<children.size(); i++) {
                    std::cout << children[i] <<  ", " ;
                }
                std::cout<< std::endl;
            }
            std::cout << weights.size() << " weights:";
            for(int i=0; i<weights.size(); i++) {
                std::cout << weights[i] << ", ";
            }
            std::cout << std::endl << "---" << std::endl;
        }

    }; //struct commNode




    /** @brief Default constructor.
    */
//TODO: remove?
    CommTree();

    /**	@brief Constructor to create tree from a vector of leaves.
    	@param[in] leaves The leaf nodes of the tree
    	@param[in] isWeightProp A vector of size equal the number of weights
    		that each tree node has. It is used to indicate if the
    		corresponding weight is proportional or an absolute value.
    		For example, node weights can be {64, 0.01} and isWeightProp={ false, true}. An interpretation could be that this node has
    		64GB of memory and 1% of the total FLOPS of the system.
    */
    CommTree( const std::vector<commNode> &leaves, const std::vector<bool> isWeightProp );

    /** This creates a homogeneous but not flat tree. The tree has levels.size() number of levels
    	and number of leaves=levels[0]*levels[1]*...*levels.back(). Each leaf node has the given
    	number of weights set to 1 and all weights are proportional.
    	Example: leaves = {3,4,5,6}, the first level has 3 children, each node in the next level
    	has 4 children, each node in the next 5 and the nodes before the leaves has 6 children each.
    	In total, 4 levels and 3*4*5*6 = 360 leaves.

    	@param[in] levels The number of children that each node has in each level. If levels[i]=x, then
    	each node of the i-th level has x children.
    	@param[in] numWeights The number of weights that each node has. Node weights are set to 1 and
    	set to proportional.
    */
    CommTree( const std::vector<IndexType> &levels, const IndexType numWeights );

    /** @brief Return the root, i.e., hierarchy level 0.
    */
    commNode getRoot() const {
        //TODO: check if tree is not initialized
        return tree[0][0];
    }

    /** @brief Return the requested hierarchy level
    @param[in] level The requested hierarchy level
    @return A vector with the nodes of level.
    */
    std::vector<commNode> getHierLevel( int level) const {
        SCAI_ASSERT_LE_ERROR( level, hierarchyLevels, "Tree has fewer levels than requested" );
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

    /** @brief The number of all the nodes of the tree.
    */
    IndexType getNumNodes() const {
        return numNodes;
    }

    /** @brief The number of weights that its node has.
    */
    IndexType getNumWeights() const {
        return numWeights;
    }

    /**@brief The number of hierarchy levels.
    */
    IndexType getNumHierLevels() const {
        return hierarchyLevels;
    }

    bool areWeightsAdapted(){
        return areWeightsAdaptedV;
    }


    /** Creates a homogeneous tree with levels.size() number of levels
    and each node in level i has levels[i] children. \sa CommTree()
    */
    void createFromLevels( const std::vector<IndexType> &levels, const IndexType numWeights );

    /** Takes a vector of leaves and creates the tree. The hierarchy vector in every node is used
    to construct the level above until we reach the root.

    @param[in] leaves A vector with all the leaf nodes.
    @return The size of the tree, i.e., numNodes.
    */
    IndexType createTreeFromLeaves( const std::vector<commNode> leaves);

    /** Creates an artificial flat tree with only one hierarchy level.
    This mainly used when no communication is provided. All leaf nodes have the same weight.

    @param[in] numLeaves The number of the leaves.
    @param[in] numNodeWeights Number of weights that each node has.
    @return The size of the tree, i.e., numNodes. Since this is a flat tree with one hierarchy level,
    numNodes=numLeaves+1, where +1 is for the root node.
    */
    IndexType createFlatHomogeneous( const IndexType numLeaves, const IndexType numNodeWeights = 1 );

    /** Given the desired sizes of the blocks, we construct a flat
    tree with one level where every leaf node has different weight.
    This mainly used when only block sizes are provided for partitioning.

    @param[in] leafSizes The size of leaves: leafSizes.size()= numWeights, leafSizes[i].size()=numLeaves
    leafSizes[i][j] holds the i-th weight for the j-th leaf.
    @return The size of the tree, i.e., numNodes. Since this is a flat tree with one hierarchy level,
    numNodes=numLeanes+1, where +1 is for the root node.
    **/
    IndexType createFlatHeterogeneous( const std::vector<std::vector<ValueType>> &leafSizes );

    /** Overloaded with vector of size equal the number of weights to indicate which 
    nodes weights are proportional and which are to be taken as absolute values.
    */
    IndexType createFlatHeterogeneous( 
        const std::vector<std::vector<ValueType>> &leafSizes,
        const std::vector<bool> &isWeightProp);


    IndexType createHierHeterogeneous(
        const std::vector<std::vector<ValueType>> &leafSizes,
        const std::vector<bool> &isWeightProp,
        const std::vector<IndexType> &levels);


    /** Creates a vector of leaves with only one hierarchy level, i.e., a flat
    tree. There can be multiple weights for each leaf.

    @param[in] sizes The sizes of the leaves. sizes.size() is the number of different
    weights and sizes[i].size() is the number of leaves.
    @return A vector with all the leaves.
    **/
    std::vector<commNode> createLeaves( const std::vector<std::vector<ValueType>> &sizes);

    /** Creates leaves according to the provided levels.
    */
    std::vector<commNode> createLeaves( 
        const std::vector<std::vector<ValueType>> &sizes,
        const std::vector<IndexType> &levels);

    /* Takes a level of the tree and creates the level above it by grouping together nodes that
     have the same last hierarchy index.

     @param[in] levelBelow A vector of leaves
     @return A vector of leaves. ret.size()<=levelBelow.size() as the upper level cannot have
     more nodes from the level below.
    */
    static std::vector<commNode> createLevelAbove( const std::vector<commNode> &levelBelow);

    /** Weights of leaf nodes can be given as relative values. Given specific
    node weights, adapt them so now, leaf weights are calculated according
    to the provided node weights. Weights of the tree that are not proportional are
    not affected; only proportional weights are converted to absolute values.
    If the i-th weight is proportional, then the adapted weight for a node x is
    x.weight[i] = e*(sum(leafSizes[i])/sum(allNodes.weight[i]))

    @param[in] leafSizes This contains the absolute weights for the leaves.
    leafSizes.size()=numWeights, leafSizes[i].size()=numLeaves
    leafSizes[i][j]: the i-th weight for node j
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

    @param[in] thisLevel The input hierarchy level of the tree.
    @return A vector with the number of nodes for each group.
    */
    std::vector<unsigned int> getGrouping(const std::vector<commNode> thisLevel) const;

    /** Calculates the distance of two nodes using their hierarchy labels.
    	We assume that leaves with the same father have distance 1.
    	Comparing two hierarchy labels, the distance is their first mismatch.
    	In other words, the height of their least common ancestor.

    	For example:
    	@verbatim
    	1.hierarchy = { 3, 3, 1, 4, 2}
    	2.hierarchy = { 3, 3, 0, 0, 1}
    	3.hierarchy = { 0, 3, 1, 4, 2}
    	distances(1,2)=3, distacne(1,3)=5, distance(2,3)=5
    	@endverbatim

    	@param[in] node1 The first node
    	@param[in] node2 The second node
    	@return Their distance in the tree.
    */
    static ValueType distance( const commNode &node1, const commNode &node2 );

    /** Export the tree as a weighted graph. The edge weight between two nodes
    	is the distance of the nodes in the tree as it is calculates by the function distance.
    	Remember: only leaves are nodes in the graph. This means that the
    	number of nodes in the graph is equal the number of leaves of the tree and the
    	graph is complete: it includes all possible edges.

    	This is not distributed, it works only locally in every PE.

    	@return A (not distributed) graph.
    */
//TODO: since this a complete matrix, the CSRSparsematrix is not very efficient
    scai::lama::CSRSparseMatrix<ValueType> exportAsGraph_local() const;


    /** Compute the imbalance of a partition (of k blocks) where the leaves of the tree provide the
    balance constrain for every weight. Note that the actual graph is not needed to calculate the
    imbalance.

    @param[in] part A partition of the nodes of a graph. part.size() is the number of nodes,
    part[i]=x then node i belongs to block x.
    @param[in] k The number of blocks,  numLeaves=k, part.max()=k+1
    @param[in] nodeWeight The weights for every node.

    @return The maximum imbalance for every weight. ret.size()=nodeWeight.size()=numWeights
    */

//TODO: turn to static? move to other class?
    std::vector<ValueType> computeImbalance(
        const scai::lama::DenseVector<IndexType> &part,
        const IndexType k,
        const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeight);


    /** Returns a vector for every balance constrain, i.e., the weights of the nodes, for the specific hierarchy level.
    	return.size()==the number of constrains
    	return[i].size()==number of leaves, for all i
    	If level==-1 it will return the the constraints of the leaves.

    	@param[in] level The hierarchy level. If level=-1 then calculate for the leaves.
    	@return A vector of size numWeight that holds the balance/weight for all node of the level.
    	ret.size()=numWeight, ret[i].size()=hierarch[level].size()
    	ret[i][j] is the i-th weight for the j-th node in given hierarchy level.
    */

    std::vector<std::vector<ValueType>> getBalanceVectors( const IndexType level=-1) const;


    /** @brief Print information for the tree
    */
    void print() const;

    /** @brief Basic sanity checks for the tree.
    @param[in] allTests If true, do additional, more expensive tests
    */
    bool checkTree( bool allTests=false ) const;



private:

    scai::lama::CSRSparseMatrix<ValueType> exportAsGraph_local(const std::vector<commNode> leaves) const;

    /** @brief The part of the function that does not set the boolean vector of
    whether a wight is proportional or not
    */
    IndexType createFlatHeterogeneousCore( const std::vector<std::vector<ValueType>> &leafSizes );

    /**The root of the communication tree; used for hierarchical partitioning

    - tree.size==hierarchyLevels
    - tree[i] is the vector of nodes for hierarchy level i
    - tree.back() is the vector of all the leaves
    */
    std::vector<std::vector<commNode>> tree;

//must be known how many levels the tree has
//(well, it can inferred but it is just easier)
    IndexType hierarchyLevels; 			///< how many hierarchy levels exist, hierarchyLevels = tree.size()
    IndexType numNodes;					///< all the nodes of the tree
    IndexType numLeaves;				///< the leafs of the tree
    IndexType numWeights;				///< how many weights each node has
    bool areWeightsAdaptedV = false;		///< if relative weights are adapted, \sa adaptWeights
/// if isProportional[i] is true, then weight i is proportional and if false, weight i is absolute; isProportional.size()=numWeights
    std::vector<bool> isProportional;


//------------------------------------------------------------------------

};//class CommTree

template <typename IndexType, typename ValueType>
using cNode = typename ITI::CommTree<IndexType,ValueType>::commNode;
//typedef typename ITI::CommTree<IndexType,ValueType>::commNode cNode;

}//namespace ITI