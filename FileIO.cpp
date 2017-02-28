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
    
    IndexType globalN=0;
    IndexType root =0;
    IndexType rank = comm->getRank();
    IndexType size = comm->getSize();
    scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
        
    if( comm->getRank()==root){
        globalN = distPtr->getGlobalSize();
    }
    //scai::hmemo::HArray<IndexType> globalIA(globalN);
    //scai::hmemo::HArray<IndexType> globalJA(globalN);
    
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    scai::hmemo::HArray<IndexType> localIA_HA = localStorage.getIA();

    // copy HArray to ValueType[] for the gather
    // copy and gather IA array
    // not copy first element of ia array since it is always 0, only first PE writes the initial 0    
    IndexType localIAsize;
    IndexType startIndex;
    if(rank == 0){
        localIAsize = localIA_HA.size();
        startIndex = 0;
    }else{
        localIAsize = localIA_HA.size()-1;
        startIndex = 1;
    }

    scai::common::scoped_array<ValueType> localIA_ar( new ValueType[localIAsize]);
    {
        const scai::hmemo::ReadAccess<IndexType> readIA(localIA_HA);
        for(IndexType i=startIndex; i<localIA_HA.size(); i++){
            localIA_ar[i-startIndex] = readIA[i];
//PRINT(*comm <<": "<< i-startIndex << " $ " << localIA_ar[i-startIndex]);
        }
    } //readIA.release();
    
    scai::common::scoped_array<ValueType> tmpGlobalIA( new ValueType[globalN + size] );
    //scai::common::scoped_array<ValueType> globalIA( new ValueType[globalN+1] );
    std::vector<ValueType> globalIA;
         
    for( int i=0; i<globalN+size; i++){
        tmpGlobalIA[i]= -1;         // trash to clear up later
    }
    comm->gather(tmpGlobalIA.get(), localIAsize, root, localIA_ar.get());
 /*for(int i=0; i<globalN + size; i++){
    std::cout<<i <<":" <<tmpGlobalIA[i] <<"  ,  ";
}  */     
    // indices are not correct since every PE stores local indices in IA and there are
    // trash since gather expects same size of data from every partner (here takes as localSize
    // the local size of rank0). Copy to the correct form.
    IndexType trashCnt= 0;
    if(rank==root){
        IndexType prefix = 0;
        for(IndexType i=0; i<globalN+size; i++){
            if(tmpGlobalIA[i]!= -1){
                globalIA.push_back(tmpGlobalIA[i]+ prefix);
            }else{ //trash
                ++trashCnt;
            }
            if((i+1<globalN+size) and tmpGlobalIA[i+1]<tmpGlobalIA[i]){ 
                prefix = globalIA.back();
                //PRINT(i<<" :: " <<prefix);            
            }
        }      
        SCAI_ASSERT(trashCnt == comm->getSize()-1 , "Array from gather not in correct form");
    }else if(comm->getSize()==1){ //no distribution/communication
        globalIA.assign(localIA_ar.get(), localIA_ar.get()+ localIAsize);
    }

for(int i=0; i<globalIA.size(); i++){
    std::cout<<i <<":" <<globalIA[i] <<"  ,  ";
}
  

    //copy and gather JA array
    scai::hmemo::HArray<IndexType> localJA_HA = localStorage.getJA();
    scai::common::scoped_array<ValueType> localJA_ar( new ValueType[localJA_HA.size()]);
    {
        const scai::hmemo::ReadAccess<IndexType> readJA(localJA_HA);
        for(IndexType i=0; i<localJA_HA.size(); i++){
            localJA_ar[i] = readJA[i];
        }
    } //readJA.release();
PRINT(*comm << ": " << localJA_HA.size() );       
    IndexType globalJAsize = adjM.getNumValues();
PRINT(globalJAsize);    
    scai::common::scoped_array<ValueType> globalJA( new ValueType[globalJAsize] );
    for( int i=0; i<globalJAsize; i++){
        globalJA[i]= -1;         // trash to clear up later
    }
    //
    // size should be the same for all PEs ...
    comm->gather(globalJA.get(), 60/*localJA_HA.size()*/, root, localJA_ar.get());
    //
for(int i=0; i<globalN; i++){
    std::cout<< globalJA[i] <<" , ";
}


    // assertion on root
    if( rank==root){
        SCAI_ASSERT(globalIA.size()== globalN+1, *comm<< ": Global size "<< globalIA.size() << " is incorrect, should be " << globalN+1);
    }
    IndexType cols= adjM.getNumColumns() , rows= adjM.getNumRows();
    
    //assert( true == adjM.checkSymmetry() ); // this can be expensive
    assert(((int) adjM.getNumValues())%2==0); // even number of edges
    assert(cols==rows);

    std::cout << cols <<" "<< adjM.getNumValues()/2 << std::endl;

    
    if(comm->getRank()==root){
        SCAI_REGION("FileIO.writeGraph.newVersion.writeInFile");
        std::ofstream fNew;
        std::string newFile = filename;
        fNew.open(newFile);

        //const scai::hmemo::ReadAccess<IndexType> ia(globalIA);
        //scai::common::scoped_array<ValueType> ia = globalIA;
        //const scai::hmemo::ReadAccess<IndexType> ja(globalJA);
        // first line is number of nodes and edges
        fNew << cols <<" "<< adjM.getNumValues()/2 << std::endl;

        // globlaIA.size() = globalN+1
        for(IndexType i=0; i< globalN; i++){        // for all local nodes
            for(IndexType j=globalIA[i]; j<globalIA[i+1]; j++){             // for all the edges of a node
                SCAI_ASSERT( globalJA[j]<= globalN , globalJA[j] << " must be < "<< globalN );
                fNew << globalJA[j]+1 << " ";
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
void FileIO<IndexType, ValueType>::writeCoords (const std::vector<DenseVector<ValueType>> &coords, IndexType numPoints, const std::string filename){
    SCAI_REGION( "FileIO.writeCoords" )

    std::ofstream f(filename);
    if(f.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    IndexType i, j;
    IndexType dimension= coords.size();

    assert(coords.size() == dimension );
    assert(coords[0].size() == numPoints);
    for(i=0; i<numPoints; i++){
        for(j=0; j<dimension; j++)
            f<< std::setprecision(15)<< coords[j].getValue(i).Scalar::getValue<ValueType>() << " ";
        f<< std::endl;
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
    	bool read = std::getline(file, line);
    	assert(read);//if we have read past the end of the file, the node count was incorrect
        std::stringstream ss( line );
        std::string item;
        std::vector<IndexType> neighbors;

        while (std::getline(ss, item, ' ')) {
        	IndexType neighbor = std::stoi(item)-1;//-1 because of METIS format
        	if (neighbor >= globalN || neighbor < 0) {
        		throw std::runtime_error("Found illegal neighbor " + std::to_string(neighbor) + " in line " + std::to_string(i+beginLocalRange));
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
    assert(comm->sum(localN) == globalN);

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
		bool read = std::getline(file, line);
		assert(read);//if we have read past the end of the file, the node count was incorrect
		std::stringstream ss( line );
		std::string item;

		IndexType dim = 0;
		while (std::getline(ss, item, ' ') && dim < dimension) {
			ValueType coord = std::stod(item);
			coords[dim][i] = coord;
			dim++;
		}
		if (dim < dimension) {
			throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimension) + " expected in line '" + line + "'");
		}
    }

    std::vector<DenseVector<ValueType> > result(dimension);

    for (IndexType i = 0; i < dimension; i++) {
    	result[i] = DenseVector<ValueType>(coords[i], dist);
    }

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
std::vector<std::set<std::shared_ptr<SpatialCell> > > FileIO<IndexType, ValueType>::readQuadTree( std::string filename ) {
	SCAI_REGION( "FileIO.readQuadTree" );

	const IndexType dimension = 3;
	const IndexType valuesPerLine = 1+2*dimension + 2*dimension*dimension;

    std::ifstream file(filename);

    if(file.fail())
            throw std::runtime_error("File "+ filename+ " failed.");

    std::map<std::vector<ValueType>, std::shared_ptr<SpatialCell>> nodeMap;
    std::map<std::vector<ValueType>, std::set<std::vector<ValueType>>> pendingEdges;
    std::map<std::vector<ValueType>, std::set<std::vector<ValueType>>> confirmedEdges;

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
		assert(nodeMap.count(parentCoords));
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

    //check whether all nodes have either no or the full amount of children
    for (std::pair<std::vector<ValueType>, std::shared_ptr<SpatialCell>> elems : nodeMap) {
    	bool consistent = elems.second->isConsistent();
    	if (!consistent) {
    		std::vector<ValueType> coords = elems.first;
    		throw std::runtime_error("Node at " + std::to_string(coords[0]) + ", " + std::to_string(coords[1]) + ", " + std::to_string(coords[2]) + " inconsistent.");
    	}
    	assert(elems.second->isConsistent());
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
    return result;
}

template void FileIO<int, double>::writeGraph (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeGraphDistributed (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeCoords (const std::vector<DenseVector<double>> &coords, int numPoints, const std::string filename);
template CSRSparseMatrix<double> FileIO<int, double>::readGraph(const std::string filename);
template std::vector<DenseVector<double>> FileIO<int, double>::readCoords( std::string filename, int numberOfCoords, int dimension);
template std::vector<std::set<std::shared_ptr<SpatialCell> > >  FileIO<int, double>::readQuadTree( std::string filename );


} /* namespace ITI */
