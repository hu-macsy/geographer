/*
 * IO.cpp
 *
 *  Created on: 15.02.2017
 *      Author: moritzl
 */

#include "FileIO.h"

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/Scalar.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/common/Math.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>
#include <scai/tracing.hpp>

#include <assert.h>
#include <cmath>
#include <set>
#include <climits>
#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <map>
#include <tuple>

using scai::lama::CSRStorage;
using scai::lama::Scalar;

namespace ITI {

//-------------------------------------------------------------------------------------------------
/*Given the adjacency matrix it writes it in the file "filename" using the METIS format. In the
 * METIS format the first line has two numbers, first is the number on vertices and the second
 * is the number of edges. Then, row i has numbers e1, e2, e3, ... notating the edges:
 * (i, e1), (i, e2), (i, e3), ....
 *
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeGraph (const CSRSparseMatrix<ValueType> &adjM, const std::string filename){
    SCAI_REGION( "FileIO.writeGraph" )
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    //PRINT(*comm << " In writeInFileMetisFormat");
    
    IndexType root =0;
    IndexType rank = comm->getRank();
    IndexType size = comm->getSize();
    scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
        
    IndexType globalN = distPtr->getGlobalSize();
    
    // Create a noDistribution and redistribute adjM. This way adjM is replicated in every PE
    // TODO: use gather (or something) to gather in root PE and print there, not replicate everywhere
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( globalN ));
    
    // in order to keep input array unchanged, create new tmp array by coping
    // adjM.redistribute( noDist , noDist);
    
    scai::lama::CSRSparseMatrix<ValueType> tmpAdjM ( adjM.getLocalStorage(),
                                                     adjM.getRowDistributionPtr(),
                                                     adjM.getColDistributionPtr()
                                           );
    tmpAdjM.redistribute( noDist , noDist);

    if(comm->getRank()==root){
        SCAI_REGION("FileIO.writeGraph.newVersion.writeInFile");
        std::ofstream fNew;
        std::string newFile = filename;
        fNew.open(newFile);

        const scai::lama::CSRStorage<ValueType>& localAdjM = tmpAdjM.getLocalStorage();
        const scai::hmemo::ReadAccess<IndexType> rGlobalIA( localAdjM.getIA() );
        const scai::hmemo::ReadAccess<IndexType> rGlobalJA( localAdjM.getJA() );
        
        // first line is number of nodes and edges
        IndexType cols= tmpAdjM.getNumColumns();
        fNew << cols <<" "<< tmpAdjM.getNumValues()/2 << std::endl;

        // globlaIA.size() = globalN+1
        SCAI_ASSERT_EQ_ERROR( rGlobalIA.size() , globalN+1, "Wrong globalIA size.");
        for(IndexType i=0; i< globalN; i++){        // for all local nodes
            for(IndexType j= rGlobalIA[i]; j<rGlobalIA[i+1]; j++){             // for all the edges of a node
                SCAI_ASSERT( rGlobalJA[j]<= globalN , rGlobalJA[j] << " must be < "<< globalN );
                fNew << rGlobalJA[j]+1 << " ";
            }
            fNew << std::endl;
        }
        fNew.close();
    }
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeGraphDistributed (const CSRSparseMatrix<ValueType> &adjM, const std::string filename){
    SCAI_REGION("FileIO.writeGraphDistributed")

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    std::string fileTo = filename + std::to_string(comm->getRank());
    std::ofstream f(fileTo);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    // notice that numValues is twice the number of edges of the graph
    assert(((int) adjM.getNumValues())%2 == 0); // even number of edges

    IndexType localNumNodes= adjM.getLocalNumRows();
    f<< localNumNodes <<" "<< adjM.getLocalNumValues()/2 << std::endl; // first line is number of nodes and edges

    // get local part
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    const scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
    const scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());

    for(IndexType i=0; i< ia.size()-1; i++){                  // for all local nodes
    	for(IndexType j=ia[i]; j<ia[i+1]; j++){             // for all the edges of a node
            f << ja[j]+1 << " ";
    	}
    	f << std::endl;
    }
    f.close();
}
//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and their dimension, writes them in file "filename".
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeCoords (const std::vector<DenseVector<ValueType>> &coords, const std::string filename){
    SCAI_REGION( "FileIO.writeCoords" );

    const IndexType dimension = coords.size();
    const IndexType n = coords[0].size();
    scai::dmemo::DistributionPtr dist = coords[0].getDistributionPtr();
	scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( n ));
	scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();

    /**
	 * If the input is replicated, we can write it directly from the root processor.
	 * If it is not, we need to create a replicated copy.
	 */

    std::vector<DenseVector<ValueType>> maybeCopy;
    const std::vector<DenseVector<ValueType>> &targetReference = dist->isReplicated() ? coords : maybeCopy;

	if (!dist->isReplicated()) {
		maybeCopy.resize(dimension);
		for (IndexType d = 0; d < dimension; d++) {
			maybeCopy[d] = DenseVector<ValueType>(coords[d], noDist);
		}

	}
	assert(targetReference[0].getDistributionPtr()->isReplicated());
	assert(targetReference[0].size() == n);

	if (comm->getRank() == 0) {
		std::ofstream filehandle(filename);
		filehandle.precision(15);
		if (filehandle.fail()) {
			throw std::runtime_error("Could not write to file " + filename);
		}
		for (IndexType i = 0; i < n; i++) {
			for (IndexType d = 0; d < dimension; d++) {
				filehandle << coords[d].getLocalValues()[i] << " ";
			}
			filehandle << std::endl;
		}
    }
}

//-------------------------------------------------------------------------------------------------
/*Given the vector of the coordinates and their dimension, writes them in file "filename".
 */
template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writeCoordsDistributed_2D (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const std::string filename){
    SCAI_REGION( "FileIO.writeCoordsDistributed" )

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr distPtr = coords[0].getDistributionPtr();
    
    std::string thisPEFilename = filename +'_'+ std::to_string(comm->getRank()) + ".xyz";
    std::ofstream f(thisPEFilename);
    if(f.fail())
        throw std::runtime_error("File "+ thisPEFilename+ " failed.");

    IndexType i, j;
    IndexType dimension= coords.size();

    assert(coords.size() == dimension );
    assert(coords[0].size() == numPoints);
    
    IndexType localN = distPtr->getLocalSize();
    
    scai::hmemo::ReadAccess<ValueType> coordAccess0( coords[0].getLocalValues() );
    scai::hmemo::ReadAccess<ValueType> coordAccess1( coords[1].getLocalValues() );
    
    for(i=0; i<localN; i++){
        f<< std::setprecision(15)<< coordAccess0[i] << " " << coordAccess1[i] << std::endl;
    }
    
}

template<typename IndexType, typename ValueType>
void FileIO<IndexType, ValueType>::writePartition(const DenseVector<IndexType> &part, const std::string filename) {
	SCAI_REGION( "FileIO.writePartition" );

	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	scai::dmemo::DistributionPtr dist = part.getDistributionPtr();
	scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( part.size() ));

	/**
	 * If the input partition is replicated, we can write it directly from the root processor.
	 * If it is not, we need to create a replicated copy.
	 */
	DenseVector<IndexType> maybeCopy;
	DenseVector<IndexType> &targetReference = dist->isReplicated() ? part : maybeCopy;
	if (!dist->isReplicated()) {
		maybeCopy = DenseVector<IndexType>(part, noDist);
	}
	assert(maybeCopy.getDistributionPtr()->isReplicated());
	assert(maybeCopy.size() == part.size());

	if (comm->getRank() == 0) {
		std::ofstream filehandle(filename);
		if (filehandle.fail()) {
			throw std::runtime_error("Could not write to file " + filename);
		}
		scai::hmemo::ReadAccess<IndexType> access(maybeCopy);
		for (IndexType i = 0; i < access.size(); i++) {
			filehandle << access[i] << std::endl;
		}
	}
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains a graph in the METIS format. The function reads that graph and transforms
 * it to the adjacency matrix as a CSRSparseMatrix.
 */

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraph(const std::string filename) {
	SCAI_REGION("FileIO.readGraph");

	std::ifstream file(filename);

	if (file.fail()) {
		throw std::runtime_error("Reading graph from " + filename + " failed.");
	}

	IndexType globalN, globalM;

	file >> globalN >> globalM;

	const ValueType avgDegree = ValueType(2*globalM) / globalN;

	//get distribution and local range
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( globalN ));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll to begin of local range. Neighbors of node i are in line i+1
    std::string line;
    std::getline(file, line);
    for (IndexType i = 0; i < beginLocalRange; i++) {
    	std::getline(file, line);
    }

    std::vector<IndexType> ia(localN+1, 0);
    std::vector<IndexType> ja;

    //we don't know exactly how many edges we are going to have, but in a regular mesh the average degree times the local nodes is a good estimate.
    ja.reserve(localN*avgDegree*1.1);

    //now read in local edges
    for (IndexType i = 0; i < localN; i++) {
    	bool read = std::getline(file, line).good();
    	assert(read);//if we have read past the end of the file, the node count was incorrect
        std::stringstream ss( line );
        std::string item;
        std::vector<IndexType> neighbors;

        while (std::getline(ss, item, ' ')) {
        	if (item.size() == 0) {
        		//probably some whitespace at end of line
        		continue;
        	}
        	IndexType neighbor = std::stoi(item)-1;//-1 because of METIS format
        	if (neighbor >= globalN || neighbor < 0) {
        		throw std::runtime_error(std::string(__FILE__) +", "+std::to_string(__LINE__) + ": Found illegal neighbor " + std::to_string(neighbor) + " in line " + std::to_string(i+beginLocalRange));
        	}
        	//std::cout << "Converted " << item << " to " << neighbor << std::endl;
        	neighbors.push_back(neighbor);
        }

        //set Ia array
        ia[i+1] = ia[i] + neighbors.size();
        //copy neighbors to Ja array
        std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(ja));
    }

    //TODO: maybe check that file is not longer than expected

    file.close();

    scai::utilskernel::LArray<ValueType> values(ja.size(), 1);//unweighted edges
    assert(ja.size() == ia[localN]);
    SCAI_ASSERT(comm->sum(localN) == globalN, "Sum " << comm->sum(localN) << " should be " << globalN);

    if (comm->sum(ja.size()) != 2*globalM) {
    	throw std::runtime_error("Expected " + std::to_string(2*globalM) + " edges, got " + std::to_string(comm->sum(ja.size())));
    }

    //assign matrix
    scai::lama::CSRStorage<ValueType> myStorage(localN, globalN, ja.size(), scai::utilskernel::LArray<IndexType>(ia.size(), ia.data()),
	scai::utilskernel::LArray<IndexType>(ja.size(), ja.data()), values);

    return scai::lama::CSRSparseMatrix<ValueType>(myStorage, dist, noDist);
}


//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads these coordinates and returns a vector of DenseVectors, one for each dimension
 */
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoords( std::string filename, IndexType numberOfPoints, IndexType dimension){
    SCAI_REGION( "FileIO.readCoords" )
    IndexType globalN= numberOfPoints;
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;

    //scroll forward to begin of local range
    std::string line;
    for (IndexType i = 0; i < beginLocalRange; i++) {
    	std::getline(file, line);
    }

    //create result vector
    std::vector<scai::utilskernel::LArray<ValueType> > coords(dimension);
    for (IndexType dim = 0; dim < dimension; dim++) {
    	coords[dim] = scai::utilskernel::LArray<ValueType>(localN, 0);
    }

    //read local range
    for (IndexType i = 0; i < localN; i++) {
		bool read = std::getline(file, line).good();
		if (!read) {
			throw std::runtime_error("Unexpected end of coordinate file. Was the number of nodes correct?");
		}
		std::stringstream ss( line );
		std::string item;

		IndexType dim = 0;
		while (dim < dimension) {
			bool read = std::getline(ss, item, ' ');
			if (!read or item.size() == 0) {
				throw std::runtime_error("Unexpected end of line. Was the number of dimensions correct?");
			}
			ValueType coord = std::stod(item);
			coords[dim][i] = coord;
			dim++;
		}
		if (dim < dimension) {
			throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimension) + " expected in line '" + line + "'");
		}
    }

    if (endLocalRange == globalN) {
    	bool eof = std::getline(file, line).eof();
    	if (!eof) {
    		throw std::runtime_error(std::to_string(numberOfPoints) + " coordinates read, but file continues.");
    	}
    }

    std::vector<DenseVector<ValueType> > result(dimension);

    for (IndexType i = 0; i < dimension; i++) {
        result[i] = DenseVector<ValueType>(dist, coords[i] );
    }

    return result;
}

template<typename IndexType, typename ValueType>
DenseVector<IndexType> FileIO<IndexType, ValueType>::readPartition(const std::string filename) {
	std::ifstream file(filename);

	if(file.fail())
		throw std::runtime_error("File "+ filename+ " failed.");

	std::vector<IndexType> part;
	std::string line;
	while (std::getline(file, line)) {
		part.push_back(std::stoi(line));
	}

	DenseVector<IndexType> result(part.size(), part.data());

	return result;
}

template<typename IndexType, typename ValueType>
std::pair<std::vector<ValueType>, std::vector<ValueType>> FileIO<IndexType, ValueType>::getBoundingCoords(std::vector<ValueType> centralCoords, IndexType level) {
	const IndexType dimension = centralCoords.size();
	const ValueType offset = 0.5 * (IndexType(1) << IndexType(level));
	std::vector<ValueType> minCoords(dimension);
	std::vector<ValueType> maxCoords(dimension);

	for (IndexType i = 0; i < dimension; i++) {
		minCoords[i] = centralCoords[i] - offset;
		maxCoords[i] = centralCoords[i] + offset;
	}
	return {minCoords, maxCoords};
}

template<typename IndexType, typename ValueType>
CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readQuadTree( std::string filename, std::vector<DenseVector<ValueType>> &coords ) {
	SCAI_REGION( "FileIO.readQuadTree" );

	const IndexType dimension = 3;
	const IndexType valuesPerLine = 1+2*dimension + 2*dimension*dimension;

    std::ifstream file(filename);

    if(file.fail())
            throw std::runtime_error("Reading file "+ filename+ " failed.");

    std::map<std::vector<ValueType>, std::shared_ptr<SpatialCell>> nodeMap;
    std::map<std::vector<ValueType>, std::set<std::vector<ValueType>>> pendingEdges;
    std::map<std::vector<ValueType>, std::set<std::vector<ValueType>>> confirmedEdges;
    std::set<std::shared_ptr<SpatialCell> > roots;

    IndexType duplicateNeighbors = 0;

    std::string line;
    while (std::getline(file, line)) {
    	std::vector<ValueType> values;
    	std::stringstream ss( line );
		std::string item;

		std::string comparison("timestep");
		std::string prefix(line.begin(), line.begin()+comparison.size());
		if (prefix == comparison) {
			std::cout << "Caught other timestep. Skip remainder of file." << std::endl;
			break;
		}

		IndexType i = 0;

		while (std::getline(ss, item, ' ')) {
			if (item.size() == 0) {
				continue;
			}

			try {
				values.push_back(std::stod(item));
			} catch (...) {
				std::cout << item << " could not be resolved as number." << std::endl;
				throw;
			}

			i++;
		}

		if (i == 0) {
			//empty line
			continue;
		} else if (i != valuesPerLine) {
			throw std::runtime_error("Expected "+std::to_string(valuesPerLine)+" values, but got "+std::to_string(i)+".");
		}

		//process quadtree node
		const std::vector<ValueType> ownCoords = {values[0], values[1], values[2]};
		const ValueType level = values[3];
		const std::vector<ValueType> parentCoords = {values[4], values[5], values[6]};

		assert(ownCoords != parentCoords);

		assert(*std::min_element(ownCoords.begin(), ownCoords.end()) >= 0);

		std::vector<ValueType> minCoords, maxCoords;
		std::tie(minCoords, maxCoords) = getBoundingCoords(ownCoords, level);

		//create own cell and add to node map
		std::shared_ptr<QuadNodeCartesianEuclid> quadNodePointer(new QuadNodeCartesianEuclid(minCoords, maxCoords));
		assert(nodeMap.count(ownCoords) == 0);
		nodeMap[ownCoords] = quadNodePointer;
		assert(confirmedEdges.count(ownCoords) == 0);
		confirmedEdges[ownCoords] = {};

		//check for pending edges
		if (pendingEdges.count(ownCoords) > 0) {
			std::set<std::vector<ValueType>> thisNodesPendingEdges = pendingEdges.at(ownCoords);
			for (std::vector<ValueType> otherCoords : thisNodesPendingEdges) {
				confirmedEdges[ownCoords].insert(otherCoords);
				confirmedEdges[otherCoords].insert(ownCoords);
			}
			pendingEdges.erase(ownCoords);
		}

		//check for parent pointer
		if (parentCoords[0] != -1 && nodeMap.count(parentCoords) == 0) {
			std::tie(minCoords, maxCoords) = getBoundingCoords(parentCoords, level+1);
			std::shared_ptr<QuadNodeCartesianEuclid> parentPointer(new QuadNodeCartesianEuclid(minCoords, maxCoords));
			nodeMap[parentCoords] = parentPointer;
			assert(confirmedEdges.count(parentCoords) == 0);
			confirmedEdges[parentCoords] = {};
			roots.insert(parentPointer);

			//check for pending edges of parent
			if (pendingEdges.count(parentCoords) > 0) {
				//std::cout << "Found pending edges for parent " << parentCoords[0] << ", " << parentCoords[1] << ", " << parentCoords[2] << std::endl;
				std::set<std::vector<ValueType>> thisNodesPendingEdges = pendingEdges.at(parentCoords);
				for (std::vector<ValueType> otherCoords : thisNodesPendingEdges) {
					confirmedEdges[parentCoords].insert(otherCoords);
					confirmedEdges[otherCoords].insert(parentCoords);
				}
				pendingEdges.erase(parentCoords);
			}
		}

		if (parentCoords[0] != -1) {
			roots.erase(quadNodePointer);
		}

		assert(nodeMap.count(parentCoords));//Why does this assert work? Why can't it happen that the parentCoords are -1?
		nodeMap[parentCoords]->addChild(quadNodePointer);
		assert(nodeMap[parentCoords]->height() > 1);

		//check own edges, possibly add to pending
		for (IndexType i = 0; i < 2*dimension; i++) {
			const IndexType beginIndex = 2*dimension+1+i*dimension;
			const IndexType endIndex = beginIndex+dimension;
			assert(endIndex <= values.size());
			if (i == 2*dimension -1) {
				assert(endIndex == values.size());
			}
			const std::vector<ValueType> possibleNeighborCoords(values.begin()+beginIndex, values.begin()+endIndex);
			assert(possibleNeighborCoords.size() == dimension);

			if (possibleNeighborCoords[0] == -1) {
				assert(possibleNeighborCoords[1] == -1);
				assert(possibleNeighborCoords[2] == -1);
				continue;
			} else {
				assert(possibleNeighborCoords[1] != -1);
				assert(possibleNeighborCoords[2] != -1);
			}

			if (nodeMap.count(possibleNeighborCoords)) {
				// this is actually not necessary, since if the other node was before this one,
				// the edges were added to the pending list and processed above - except if it was an implicitly referenced parent node.
				confirmedEdges[ownCoords].insert(possibleNeighborCoords);
				confirmedEdges[possibleNeighborCoords].insert(ownCoords);
			} else {
				//create empty set if not yet done
				if (pendingEdges.count(possibleNeighborCoords) == 0) {
					pendingEdges[possibleNeighborCoords] = {};
				}
				//target doesn't exist yet, can't have confirmed edges
				assert(confirmedEdges.count(possibleNeighborCoords) == 0);

				//if edge is already there, it was duplicate
				if (pendingEdges[possibleNeighborCoords].count(ownCoords)) {
					duplicateNeighbors++;
				}

				//finally, add pending edge
				pendingEdges[possibleNeighborCoords].insert(ownCoords);
			}
		}
    }

    file.close();
    std::cout << "Read file, found or created " << nodeMap.size() << " nodes and pending edges for " << pendingEdges.size() << " ghost nodes." << std::endl;
    if (duplicateNeighbors > 0) {
    	std::cout << "Found " << duplicateNeighbors << " duplicate neighbors." << std::endl;
    }

    assert(confirmedEdges.size() == nodeMap.size());

    for (auto pendingSets : pendingEdges) {
    	assert(nodeMap.count(pendingSets.first) == 0);
    }

    IndexType nodesInForest = 0;
    for (auto root : roots) {
    	nodesInForest += root->countNodes();
    }

    std::cout << "Found " << roots.size() << " roots with " << nodesInForest << " nodes hanging from them." << std::endl;

    assert(nodesInForest == nodeMap.size());

    //check whether all nodes have either no or the full amount of children
    for (std::pair<std::vector<ValueType>, std::shared_ptr<SpatialCell>> elems : nodeMap) {
    	bool consistent = elems.second->isConsistent();
    	if (!consistent) {
    		std::vector<ValueType> coords = elems.first;
    		//throw std::runtime_error("Node at " + std::to_string(coords[0]) + ", " + std::to_string(coords[1]) + ", " + std::to_string(coords[2]) + " inconsistent.");
    	}
    	//assert(elems.second->isConsistent());
    	assert(pendingEdges.count(elems.first) == 0);//list of pending edges was erased when node was handled, new edges should not be added to pending list
    }

    IndexType i = 0;
    IndexType totalEdges = 0;
    IndexType numLeaves = 0;
    IndexType leafEdges = 0;
    std::vector<std::set<std::shared_ptr<SpatialCell> > > result(nodeMap.size());
    for (std::pair<std::vector<ValueType>, std::set<std::vector<ValueType> > > edgeSet : confirmedEdges) {
    	result[i] = {};
    	for (std::vector<ValueType> neighbor : edgeSet.second) {
    		assert(nodeMap.count(neighbor) > 0);
    		result[i].insert(nodeMap[neighbor]);
    		totalEdges++;
    	}
    	assert(result[i].size() == edgeSet.second.size());

    	if (nodeMap[edgeSet.first]->height() == 1) {
    		numLeaves++;
    		leafEdges += result[i].size();
    		if (result[i].size() == 0) {
				//only parent nodes are allowed to have no edges.
				throw std::runtime_error("Node at " + std::to_string(edgeSet.first[0]) + ", " + std::to_string(edgeSet.first[1]) + ", " + std::to_string(edgeSet.first[2])
				+ " is isolated leaf node.");
			}
    	}

    	i++;
    }
    assert(nodeMap.size() == i++);
    std::cout << "Read " << totalEdges << " confirmed edges, among them " << leafEdges << " edges between " << numLeaves << " leaves." << std::endl;

    /**
     * now convert into CSRSparseMatrix
     */

    IndexType offset = 0;
	for (auto root : roots) {
		offset = root->indexSubtree(offset);
	}

    std::vector<std::shared_ptr<const SpatialCell> > rootVector(roots.begin(), roots.end());

	coords.clear();
	coords.resize(dimension);

	std::vector<std::vector<ValueType> > vCoords(dimension);
	std::vector< std::set<std::shared_ptr<const SpatialCell>>> graphNgbrsCells(nodesInForest);

	for (auto outgoing : confirmedEdges) {
		std::set<std::shared_ptr<const SpatialCell>> edgeSet;
		for (std::vector<ValueType> edgeTarget : outgoing.second) {
			edgeSet.insert(nodeMap[edgeTarget]);
		}
		graphNgbrsCells[nodeMap[outgoing.first]->getID()] = edgeSet;
	}

	scai::lama::CSRSparseMatrix<ValueType> matrix = SpatialTree::getGraphFromForest<IndexType, ValueType>( graphNgbrsCells, rootVector, vCoords);

	for (IndexType d = 0; d < dimension; d++) {
		assert(vCoords[d].size() == numLeaves);
		coords[d] = DenseVector<ValueType>(vCoords[d].size(), vCoords[d].data());
	}
    return matrix;
}

template void FileIO<int, double>::writeGraph (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeGraphDistributed (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeCoords (const std::vector<DenseVector<double>> &coords, const std::string filename);
template void FileIO<int, double>::writeCoordsDistributed_2D (const std::vector<DenseVector<double>> &coords, int numPoints, const std::string filename);
template CSRSparseMatrix<double> FileIO<int, double>::readGraph(const std::string filename);
template std::vector<DenseVector<double>> FileIO<int, double>::readCoords( std::string filename, int numberOfCoords, int dimension);
template CSRSparseMatrix<double>  FileIO<int, double>::readQuadTree( std::string filename, std::vector<DenseVector<double>> &coords );


} /* namespace ITI */
