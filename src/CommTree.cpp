/*
 * ComMTree.cpp
 *
 *  Created on: 11.10.2018
 *      Author: tzovas
 */

#include <scai/lama.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>
#include <scai/lama/matrix/CSRSparseMatrix.hpp>

#include "CommTree.h"
#include "GraphUtils.h"

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
	{
		scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		if( comm->getRank()==0 ){
			PRINT("There are " << hierarchyLevels << " levels of hierarchy and " << leaves.size() << " leaves");
		}
	}

	for(int h = hierarchyLevels-1; h>=0; h--){
		//PRINT("starting level " << h);
		std::vector<commNode> levelAbove = createLevelAbove(levelBelow);
		//add the newly created level to the tree
		tree.insert(tree.begin(), levelAbove );
		size += levelAbove.size();
		levelBelow = levelAbove;
		//PRINT("Size of level above (lvl " << h << ") is " << levelAbove.size() );
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
	//PRINT("level below has size " << levelBelowsize );

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
std::vector<std::vector<ValueType>> CommTree<IndexType, ValueType>::getBalanceVectors(){

	const std::vector<cNode> leaves = getLeaves();
	const IndexType numLeaves = leaves.size();

	std::vector<std::vector<ValueType>> constrains(2, std::vector<ValueType>(numLeaves,0.0) );

	for(IndexType i=0; i<numLeaves; i++){
		cNode c = leaves[i];
		constrains[0][i] = c.memMB;
		constrains[1][i] = c.relatSpeed;
	}

	return constrains;

}//getBalanceVectors
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
ValueType CommTree<IndexType, ValueType>::distance( const commNode node1, const commNode node2 ){

	const std::vector<unsigned int> hier1 = node1.hierarchy;
	const std::vector<unsigned int> hier2 = node2.hierarchy;
	const IndexType labelSize = hier1.size();

	SCAI_ASSERT_EQ_ERROR( labelSize, hier2.size(), "Hierarchy label size mismatch" );
	
	IndexType i=0;
	for( i=0; i<labelSize; i++){
		if( hier1[i]!=hier2[i] ){
			break;
		}
	}
	
	//TODO?: turn that to an error?
	if( i==labelSize and node1.leafID!=node2.leafID ){
		PRINT("WARNING: labels are identical but nodes have different leafIDs: " << node1.leafID <<"!="<<node2.leafID );
	}

	return labelSize-i;
}//distance
//------------------------------------------------------------------------

//TODO: since this a complete matrix, the CSRSparsematrix is not very efficient
template <typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> CommTree<IndexType, ValueType>::exportAsGraph_local(const std::vector<commNode> leaves) const{

	const IndexType numLeaves = leaves.size();

	//TODO: since this should be a complete graph we already know the size of ja and values
	std::vector<IndexType> ia(numLeaves+1, 0);
    std::vector<IndexType> ja;
    std::vector<ValueType> values;

    for( IndexType i=0; i<numLeaves; i++ ){
    	const commNode thisLeaf = leaves[i];
    	//to keep matrix symmetric
    	for( IndexType j=0; j<numLeaves; j++ ){
    		if( i==j )	//explicitly avoid self loops
    			continue;

    		const commNode otherLeaf = leaves[j];
    		const ValueType dist = distance( thisLeaf, otherLeaf );

    		ja.push_back(j);
    		values.push_back(dist);
		}
		//edges to all other nodes
		ia[i+1] = ia[i]+numLeaves-1;
    }
    assert(ja.size() == ia[numLeaves]);

    SCAI_ASSERT_EQ_ERROR( ia.size(), numLeaves+1, "Wrong ia size" );
    SCAI_ASSERT_EQ_ERROR( ja.size(), values.size(), "ja and values sizes must agree" );
    SCAI_ASSERT_EQ_ERROR( values.size(), numLeaves*(numLeaves-1), "It should be a complete graph" );

    //assign matrix
    scai::lama::CSRStorage<ValueType> myStorage(numLeaves, numLeaves, 
            scai::hmemo::HArray<IndexType>(ia.size(), ia.data()),
    		scai::hmemo::HArray<IndexType>(ja.size(), ja.data()),
    		scai::hmemo::HArray<ValueType>(values.size(), values.data()));

    return scai::lama::CSRSparseMatrix<ValueType>( myStorage );
}//exportAsGraph
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> CommTree<IndexType, ValueType>::exportAsGraph_local() const {
	std::vector<commNode> leaves = this->getLeaves();

	return exportAsGraph_local(leaves);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::pair<ValueType,ValueType> CommTree<IndexType,ValueType>::computeImbalance(
    const scai::lama::DenseVector<IndexType> &part,
    IndexType k,
    const scai::lama::DenseVector<ValueType> &nodeWeights) const{

    //cNode is typedefed in CommTree.h
    const std::vector<cNode> leaves = this->getLeaves();
    const IndexType numLeaves = leaves.size();
    SCAI_ASSERT_EQ_ERROR( numLeaves, k, "Number of blocks of the partition and number of leaves of the tree do not agree" );

    //extract the memory and speed of every node in a vector each
    std::vector<ValueType> memory( numLeaves );
    std::vector<ValueType> relatSpeed( numLeaves );
    for(unsigned int i=0; i<numLeaves; i++){
        memory[i] = leaves[i].memMB;
        relatSpeed[i] = leaves[i].relatSpeed;
    }

    std::vector<ValueType> optBlockWeight = getOptBlockWeights( /*leaves,*/ nodeWeights);
    SCAI_ASSERT_EQ_ERROR( optBlockWeight.size(), numLeaves, "Size mismatch");

    ValueType speedImbalance = GraphUtils<IndexType, ValueType>::computeImbalance( part, k, nodeWeights, optBlockWeight );

    //to measure imbalance according to memory constraints, we assume that
    // every points has unit weight (this can be changed, for example,
    //if we measure memory in MB, we can give each point a weight 
    //of few bytes)
    scai::lama::DenseVector<ValueType> unitWeights (nodeWeights.getDistributionPtr(), 1);

    ValueType sizeImbalance = GraphUtils<IndexType, ValueType>::computeImbalance( part, k, unitWeights, memory );

    return std::make_pair(speedImbalance, sizeImbalance);

}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> CommTree<IndexType,ValueType>::getOptBlockWeights(
	const scai::lama::DenseVector<ValueType> &nodeWeights) const {

	const std::vector<commNode> hierLevel = this->getLeaves();
    const IndexType numBlocks = hierLevel.size();

    //the total node weight of the input
    ValueType totalWeightSum;
    {
        scai::hmemo::ReadAccess<ValueType> rW( nodeWeights.getLocalValues() );
        ValueType localW = 0;
        for(int i=0; i<nodeWeights.getLocalValues().size(); i++ ){
            localW += rW[i];
        }
        const scai::dmemo::CommunicatorPtr comm = nodeWeights.getDistributionPtr()->getCommunicatorPtr();
        totalWeightSum = comm->sum(localW);
    }

    //the sum of the realtive speeds for all nodes
    ValueType speedSum = 0;
    for( cNode c: hierLevel){
        speedSum += c.relatSpeed;
    }

    //the optimum size for every block
    std::vector<ValueType> optBlockWeight( numBlocks );

    //for each PE, we are given its relative speed compare to the fastest
    //PE, a number between 0 and 1. The optimum weight it should have 
    //is this:
    // relative speed*( sum of input node weights / sum of all relative speeds)
    for( int i=0; i<numBlocks; i++){
        optBlockWeight[i] = totalWeightSum*(hierLevel[i].relatSpeed/speedSum);
    }

    return optBlockWeight;
}
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

	//TODO: add more "expensive" checks like, all leaf IDs are unique, or all hierarchy labels are unique
	//	or check the size of every subtree according to its label...

	return true;
}

//to force instantiation
template class CommTree<IndexType, ValueType>;
}//ITI