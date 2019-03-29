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


//default constructor
template <typename IndexType, typename ValueType>
CommTree<IndexType, ValueType>::CommTree(){
	hierarchyLevels=0;
	numNodes=0;
	numLeaves=0;
	numWeights=0;
}

//constructor to create tree from a vector of leaves
template <typename IndexType, typename ValueType>
CommTree<IndexType, ValueType>::CommTree( const std::vector<commNode> &leaves, std::vector<bool> isWeightProp ){
	
	isProportional = isWeightProp;

	//for example, if the vector hierarchy of a leaf is [0,3,2]
	//then the size is 3, but with the implied root, there are 4 levels,
	//thus the +1
	this->hierarchyLevels = leaves.front().hierarchy.size()+1;
	this->numLeaves = leaves.size();
	this->numWeights = leaves[0].getNumWeights();

	//sanity check, TODO: remove?
	for( commNode l: leaves){
		SCAI_ASSERT_EQ_ERROR( l.hierarchy.size(), hierarchyLevels-1, "Every leaf must have the same size hierarchy vector");
		SCAI_ASSERT_EQ_ERROR( l.getNumWeights(), numWeights, "Found leaf that does not have the same number of weights as the others before" );
	}
	this->numNodes = createTreeFromLeaves( leaves );
}
//------------------------------------------------------------------------


template <typename IndexType, typename ValueType>
IndexType CommTree<IndexType, ValueType>::createTreeFromLeaves( const std::vector<commNode> leaves){

	hierarchyLevels = leaves.front().hierarchy.size()+1;
	numLeaves = leaves.size();

	//bottom level are the leaves
	std::vector<commNode> levelBelow = leaves;
	tree.insert( tree.begin(), levelBelow );
	IndexType size = levelBelow.size();

	IndexType hierarchyLevels = leaves.front().hierarchy.size();//TODO: this declaration shadows the earlier one. Is this intentional?
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

template <typename IndexType, typename ValueType>
IndexType CommTree<IndexType, ValueType>::createFlatHomogeneous( 
	const IndexType numLeaves, const IndexType numNodeWeights ){

	//in the homogeneous case we may have several weights, but all leaves have the same weight
	std::vector<std::vector<ValueType>> sizes(numNodeWeights, std::vector<ValueType>(numLeaves, 1) ); 

	std::vector<cNode> leaves = createLeaves( sizes );
	SCAI_ASSERT_EQ_ERROR( leaves.size(), numLeaves, "Wrong number of leaves" );
	SCAI_ASSERT_EQ_ERROR( leaves[0].getNumWeights(), numNodeWeights, "Wrong number of weights");

	this->numNodes = createTreeFromLeaves(leaves);
	this->numWeights = numNodeWeights;
	this->isProportional = std::vector<bool>(numNodeWeights, true);

	return leaves.size();
}//createFlatHomogeneous
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
IndexType CommTree<IndexType, ValueType>::createFlatHeterogeneous( const std::vector<std::vector<ValueType>> &leafSizes ){
	//leafSizes.size() = number of weights
	std::vector<cNode> leaves = createLeaves( leafSizes );
	SCAI_ASSERT_EQ_ERROR( leaves.size(), leafSizes[0].size(), "Wrong number of leaves" );
	SCAI_ASSERT_EQ_ERROR( leaves[0].getNumWeights(), leafSizes.size(), "Wrong number of weights");

	this->numNodes = createTreeFromLeaves(leaves);
	this->numWeights = leafSizes.size();

	return leaves.size();
}//createFlatHeterogeneous
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
std::vector<typename CommTree<IndexType,ValueType>::commNode> CommTree<IndexType,ValueType>::createLeaves( const std::vector<std::vector<ValueType>> &sizes){
	//sizes.size() = number of weights
	const IndexType numWeights = sizes.size();
	SCAI_ASSERT_GT_ERROR( numWeights, 0, "Provided sizes vector is empty, there are no weights" );

	const IndexType numLeaves = sizes[0].size();
	SCAI_ASSERT_GT_ERROR( numLeaves, 0, "Provided sizes vector is empty, there are no block size" );


	std::vector<cNode> leaves( numLeaves );

	for( unsigned int i=0; i<numLeaves; i++ ){
		std::vector<ValueType> leafWeights(numWeights);
		for( unsigned int w=0; w<numWeights; w++){
			leafWeights[w] = sizes[w][i];
		}
		cNode leafNode( std::vector<unsigned int>{i}, leafWeights );
		leaves[i] = leafNode;
	}

	return leaves;
}
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
void CommTree<IndexType, ValueType>::adaptWeights( const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights ){

	//throw std::logic_error("The method adaptWeights has multiple issues creating segfaults later.");

	if( areWeightsAdapted ){
		scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		if( comm->getRank()==0 )
			std::cout<< " Tree node weights are already adapted, skipping adaptWeights " << std::endl;
	}else{
		//we need to access leaves like that because getLeaves is const
		std::vector<commNode> hierLevel = this->tree.back();

	    const IndexType numBlocks = hierLevel.size();
	    const IndexType numWeights = this->getNumWeights();
	    SCAI_ASSERT_EQ_ERROR( numWeights, nodeWeights.size(), "Given weights vector size and tree number of weights do not agree" );

//TODO: is 1 correct?
const std::vector<std::vector<ValueType>> &hierWeights = getBalanceVectors( -1 );
		SCAI_ASSERT_EQ_ERROR( numWeights, hierWeights.size(), "Number of weights in tree do not agree" );
		SCAI_ASSERT_EQ_ERROR( numWeights, isProportional.size(), "Number of weights and proportionality information do not agree" );

	    for( unsigned int i=0; i<numWeights; i++){

	    	const scai::lama::DenseVector<ValueType> &thisNodeWeight = nodeWeights[i];
	    	const std::vector<ValueType> &thisHierWeight = hierWeights[i];

		    //the total node weight of the input graph
		    ValueType sumNodeWeights = thisNodeWeight.sum();
		    ValueType sumHierWeights = std::accumulate( thisHierWeight.begin(), thisHierWeight.end(), 0.0 );
		    const ValueType scalingFactor = sumNodeWeights / sumHierWeights;
scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
		    if( not isProportional[i] ){
		    	SCAI_ASSERT_GE_ERROR( sumHierWeights, sumNodeWeights, "Provided node weights do not fit in the given tree for weight " << i );
		    }else{
		    	//go over the nodes and adapt the weights
		    	for( commNode &cNode : hierLevel ){//this creates a copy and only adjusts the weights in the copy, which is then discarded.
		    		cNode.weights[i] *= scalingFactor;
//PRINT0( cNode.weights[i] );			    		
		    	}
		    }
		}

		//clear tree and rebuild. This will correctly construct the intemediate levels
		tree.clear();
		IndexType size = createTreeFromLeaves( hierLevel );

		areWeightsAdapted = true;
	}
}//adaptWeights

//------------------------------------------------------------------------

//WARNING: Needed that 'typename' to compile...
template <typename IndexType, typename ValueType>
std::vector<typename CommTree<IndexType,ValueType>::commNode> CommTree<IndexType, ValueType>::createLevelAbove( const std::vector<commNode> &levelBelow ){

	//a hierarchy prefix is the hierarchy vector without the last element
	//commNodes that have the same prefix, belong to the same father node
	typedef std::vector<unsigned int> hierPrefix;

	unsigned int levelBelowsize = levelBelow.size();
	std::vector<bool> seen(levelBelowsize, false);
	//PRINT("level below has size " << levelBelowsize );

	//will be used later to normalize the speed
	ValueType maxRelatSpeed = 0;

	std::vector<commNode> aboveLevel;

	//assume nodes with the same parent are not necessarily adjacent 
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

		//for debugging, remove or add macro or ...
		//PRINT(i << ": thisNode.id is " << thisNode.leafID << " prefix size " << thisPrefix.size() );

		IndexType numWeights=thisNode.getNumWeights();
		//father.weights.resize( numWeights, 0.0 );
		SCAI_ASSERT_EQ_ERROR( fatherNode.getNumWeights(), numWeights, "Number of weights mismatch");
//not really needed
SCAI_ASSERT_EQ_ERROR( 
	std::accumulate( fatherNode.weights.begin(), fatherNode.weights.end(), 0.0),
	std::accumulate( thisNode.weights.begin(), thisNode.weights.end(), 0.0), "Weights are not copied?"
	);

		for( unsigned int j=i+1; j<levelBelowsize; j++){
			commNode otherNode = levelBelow[j];
			//hierarchy prefix of the other node
			hierPrefix otherPrefix(otherNode.hierarchy.size()-1);
			std::copy(otherNode.hierarchy.begin(), otherNode.hierarchy.end()-1, otherPrefix.begin());
			//same prefix means that have the same father
			if( thisPrefix==otherPrefix ){
				fatherNode += otherNode;		//operator += overloading
				seen[j] = true;
				//numChildren++;
			}
		}

		aboveLevel.push_back(fatherNode);
	}

	return aboveLevel;
}//createLevelAbove
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
std::vector<unsigned int> CommTree<IndexType, ValueType>::getGrouping(const std::vector<commNode> thisLevel) const {

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
std::vector<std::vector<ValueType>> CommTree<IndexType, ValueType>::getBalanceVectors( const IndexType level) const{

	//for -1, return the leaves
	const std::vector<cNode>& hierLvl = level==-1 ? tree.back() : tree[level];
	const IndexType numNodes = hierLvl.size();
	const IndexType numWeights = getNumWeights();

	std::vector<std::vector<ValueType>> constraints( numWeights, std::vector<ValueType>(numNodes,0.0) );

	for(IndexType i=0; i<numNodes; i++){
		cNode c = hierLvl[i];
		for( unsigned int w=0; w<numWeights; w++){
			constraints[w][i] = c.weights[w];
		}
	}

	return constraints;

}//getBalanceVectors
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
ValueType CommTree<IndexType, ValueType>::distance( const commNode &node1, const commNode &node2 ){

	const std::vector<unsigned int> &hier1 = node1.hierarchy;
	const std::vector<unsigned int> &hier2 = node2.hierarchy;
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
	const std::vector<commNode>& leaves = this->getLeaves();

	return exportAsGraph_local(leaves);
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<ValueType> CommTree<IndexType,ValueType>::computeImbalance(
    const scai::lama::DenseVector<IndexType> &part,
    IndexType k,
    const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights){

    //cNode is typedefed in CommTree.h
    const std::vector<cNode> leaves = this->getLeaves();
    const IndexType numLeaves = leaves.size();
    SCAI_ASSERT_EQ_ERROR( numLeaves, k, "Number of blocks of the partition and number of leaves of the tree do not agree" );

    if( not areWeightsAdapted ){
    	std::cout<<"Warning, tree weights are not adapted according to the input graph node weights. Will adapt first and then calculate imbalances." << std::endl;
    	//this line can change the tree. Without it the function can be const
    	this->adaptWeights( nodeWeights );
    }

    const IndexType numWeights = getNumWeights();
    SCAI_ASSERT_EQ_ERROR( numWeights, nodeWeights.size(), "Given weights vector size and tree number of weights do not agree" );
//TODO: is 1 correct?
    //get leaf balance vectors
const std::vector<std::vector<ValueType>> allConstrains = getBalanceVectors( -1 );
    std::vector<ValueType> imbalances( numWeights );

    //compute imbalance for all weights
    for( int i=0; i<numWeights; i++){

	    std::vector<ValueType> optBlockWeight = allConstrains[i];
	    SCAI_ASSERT_EQ_ERROR( optBlockWeight.size(), numLeaves, "Size mismatch");

	    ValueType imba = GraphUtils<IndexType, ValueType>::computeImbalance( part, k, nodeWeights[i], optBlockWeight );

    	imbalances[i] = imba;
	}

    return imbalances;
}
//---------------------------------------------------------------------------------------
/*
template<typename IndexType, typename ValueType>
std::vector<ValueType> CommTree<IndexType,ValueType>::getOptBlockWeights(
	const std::vector<scai::lama::DenseVector<ValueType>> &nodeWeights) const {

	const std::vector<commNode> hierLevel = this->getLeaves();
    const IndexType numBlocks = hierLevel.size();
    const IndexType numWeights = getNumWeights();
    SCAI_ASSERT_EQ_ERROR( numWeights, nodeWeights.size(), "Given weights vector size and tree number of weights do not agree" );

	const std::vector<std::vector<ValueType>> allLeafWeights = getBalanceVectors();
	SCAI_ASSERT_EQ_ERROR( numWeights, allLeafWeights.size, "Number of weigths in tree do not agree" );

    for( unsigned int i=0; i<numWeights; i++){

    	scai::lama::DenseVector<ValueType> &thisNodeWeight = nodeWeights[i];
    	scai::lama::DenseVector<ValueType> &thisLeafWeight = allLeafWeights[i];

	    //the total node weight of the input graph
	    ValueType sumNodeWeights = thisWeight.sum();
	    ValueType sumLeafWeights = std::Accumulate( thisLeafWeight.begin(), thisLeafWeight.end(), 0.0 );

	    if( not isProportional[i] ){
	    	SCAI_ASSERT_GE_ERROR( sumLeafWeights, sumNodeWeights, "Provided node weights are do not fit in the given tree for weight " << i );
	    }else{

	    }
	    
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
	}

    return optBlockWeight;
}
*/
//------------------------------------------------------------------------

template <typename IndexType, typename ValueType>
void CommTree<IndexType, ValueType>::print() const {

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
bool CommTree<IndexType, ValueType>::checkTree( bool all ) const{

	SCAI_ASSERT_EQ_ERROR( hierarchyLevels, tree.size(), "Mismatch for hierachy levels and tree size");
	SCAI_ASSERT_EQ_ERROR( numLeaves, tree.back().size(), "Mismatch for number of leaves and the size of the nottom hierechy level");
	//check sum of sizes for every level
	SCAI_ASSERT_EQ_ERROR( tree.front().size(), 1 , "Top level of the tree should have size 1, only the root");
	SCAI_ASSERT_EQ_ERROR( numLeaves, tree.front()[0].children.size(), "The root should contain all leaves as children");
	//basically, this is the same as the above
	SCAI_ASSERT_EQ_ERROR( getRoot().children.size(), numLeaves, "The root should contain all leaves as children" );

	//TODO: add more "expensive" checks like, all leaf IDs are unique, or all hierarchy labels are unique
	//	or check the size of every subtree according to its label...

	//add more "expensive" checks 

	if( all ){
		const IndexType numWeights = getNumWeights();
//TODO: is 1 correct?
const std::vector<std::vector<ValueType>> balanceVec = getBalanceVectors(-1);
		SCAI_ASSERT_EQ_ERROR( balanceVec.size(), numWeights, "Number of weights mismatch" );

		std::vector<ValueType> weightSums(numWeights, 0.0);
		for( int w=0; w<numWeights; w++ ){
			weightSums[w] = std::accumulate( balanceVec[w].begin(), balanceVec[w].end(), 0.0 );
		}
	
		for( int h=0; h<tree.size(); h++ ){
			const std::vector<commNode> &hierLvl = tree[h];
			std::vector<ValueType> weightSumsLvL(numWeights, 0.0);
			for(auto nodeIt=hierLvl.begin(); nodeIt!=hierLvl.end(); nodeIt++ ){
				SCAI_ASSERT_EQ_ERROR( nodeIt->getNumWeights(), numWeights, "Tree node has wrong number of weights" );
				SCAI_ASSERT_EQ_ERROR( nodeIt->hierarchy.size(), h, "Tree node has wrong label size" );
				for( int w=0; w<numWeights; w++ ){
					weightSumsLvL[w] += nodeIt->weights[w];
				}
			}
			for( int w=0; w<numWeights; w++ ){
				SCAI_ASSERT( weightSums[w]-weightSumsLvL[w]<1e-8, "Weight sums must agree in all levels");
			}
		}
	}


	return true;
}

//to force instantiation
template class CommTree<IndexType, ValueType>;
}//ITI