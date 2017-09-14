/*
 * IO.cpp
 *
 *  Created on: 15.02.2017
 *      Author: moritzl
 */

#include "FileIO.h"
#include "AuxiliaryFunctions.h"

#include <scai/lama.hpp>
#include <scai/lama/matrix/all.hpp>
#include <scai/lama/Vector.hpp>
#include <scai/lama/Scalar.hpp>
#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/common/Math.hpp>
#include <scai/common/Settings.hpp>
#include <scai/lama/storage/MatrixStorage.hpp>
#include <scai/tracing.hpp>

#include <boost/algorithm/string.hpp>

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
    assert(dist->getGlobalSize() == n);
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
				filehandle << targetReference[d].getLocalValues()[i] << " ";
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
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraph(const std::string filename, Format format) {
	std::vector<DenseVector<ValueType>> dummyWeightContainer;
	return readGraph(filename, dummyWeightContainer, format);
}

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraph(const std::string filename, std::vector<DenseVector<ValueType>>& nodeWeights, Format format) {
	SCAI_REGION("FileIO.readGraph");

	if(format == Format::MATRIXMARKET){
            return FileIO<IndexType, ValueType>::readGraphMatrixMarket(filename);
        }
        
	if (!(format == Format::METIS or format == Format::AUTO)) {
		throw std::logic_error("Format not yet implemented.");
	}

	std::ifstream file(filename);

	if (file.fail()) {
		throw std::runtime_error("Reading graph from " + filename + " failed.");
	}
        
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
        
	//define variables
	std::string line;
	IndexType globalN, globalM;
	IndexType numberNodeWeights = 0;
	bool hasEdgeWeights = false;
	std::vector<ValueType> edgeWeights;//possibly of size 0

	//read first line to get header information
	std::getline(file, line);
	std::stringstream ss( line );
	std::string item;

	{
		//node count and edge count are mandatory. If these fail, std::stoi will raise an error. TODO: maybe wrap into proper error message
		std::getline(ss, item, ' ');
		globalN = std::stoi(item);
		std::getline(ss, item, ' ');
		globalM = std::stoi(item);

		bool readWeightInfo = !std::getline(ss, item, ' ').fail();
		if (readWeightInfo && item.size() > 0) {
			//three bits, describing presence of edge weights, vertex weights and vertex sizes
			int bitmask = std::stoi(item);
			hasEdgeWeights = bitmask % 10;
			if ((bitmask / 10) % 10) {
				bool readNodeWeightCount = !std::getline(ss, item, ' ').fail();
				if (readNodeWeightCount && item.size() > 0) {
					numberNodeWeights = std::stoi(item);
				} else {
					numberNodeWeights = 1;
				}
			}
		}

		if (comm->getRank() == 0) {
			std::cout << "Expecting " << globalN << " nodes and " << globalM << " edges, ";
			if (!hasEdgeWeights && numberNodeWeights == 0) {
				std::cout << "with no edge or node weights."<< std::endl;
			}
			else if (hasEdgeWeights && numberNodeWeights == 0) {
				std::cout << "with edge weights, but no node weights."<< std::endl;
			}
			else if (!hasEdgeWeights && numberNodeWeights > 0) {
				std::cout << "with no edge weights, but " << numberNodeWeights << " node weights."<< std::endl;
			}
			else {
				std::cout << "with edge weights and " << numberNodeWeights << " weights per node."<< std::endl;
			}
		}
	}

    const ValueType avgDegree = ValueType(2*globalM) / globalN;

    //get distribution and local range
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));
    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution( globalN ));

    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
    const IndexType localN = endLocalRange - beginLocalRange;
    SCAI_ASSERT_LE_ERROR(localN, std::ceil(ValueType(globalN) / comm->getSize()), "localN: " << localN << ", optSize: " << std::ceil(globalN / comm->getSize()));

    //std::cout << "Process " << comm->getRank() << " reading from " << beginLocalRange << " to " << endLocalRange << std::endl;

    //scroll to begin of local range. Neighbors of node i are in line i+1
    for (IndexType i = 0; i < beginLocalRange; i++) {
    	std::getline(file, line);
    }

    std::vector<IndexType> ia(localN+1, 0);
    std::vector<IndexType> ja;
    std::vector<ValueType> values;
    std::vector<std::vector<ValueType> > nodeWeightStorage(numberNodeWeights);
    for (IndexType i = 0; i < numberNodeWeights; i++) {
    	nodeWeightStorage[i].resize(localN);
    }

    //we don't know exactly how many edges we are going to have, but in a regular mesh the average degree times the local nodes is a good estimate.
    IndexType edgeEstimate = IndexType(localN*avgDegree*1.1);
    assert(edgeEstimate >= 0);
    ja.reserve(edgeEstimate);
    
    //std::cout << "Process " << comm->getRank() << " reserved memory for  " <<  edgeEstimate << " edges." << std::endl;

    //now read in local edges
    for (IndexType i = 0; i < localN; i++) {
    	bool read = !std::getline(file, line).fail();
    	//remove leading and trailing whitespace, since these can confuse the string splitter
    	boost::algorithm::trim(line);
    	assert(read);//if we have read past the end of the file, the node count was incorrect
        std::stringstream ss( line );
        std::string item;
        std::vector<IndexType> neighbors;
        std::vector<IndexType> weights;

        for (IndexType j = 0; j < numberNodeWeights; j++) {
        	bool readWeight = !std::getline(ss, item, ' ').fail();
        	if (readWeight && item.size() > 0) {
        		nodeWeightStorage[j][i] = std::stoi(item);
        	} else {
        		std::cout << "Could not parse " << item << std::endl;
        	}
        }

        while (!std::getline(ss, item, ' ').fail()) {
        	if (item.size() == 0) {
        		//probably some whitespace at end of line
        		continue;
        	}
        	IndexType neighbor = std::stoi(item)-1;//-1 because of METIS format
        	if (neighbor >= globalN || neighbor < 0) {
        		throw std::runtime_error(std::string(__FILE__) +", "+std::to_string(__LINE__) + ": Found illegal neighbor " + std::to_string(neighbor) + " in line " + std::to_string(i+beginLocalRange));
        	}

        	if (hasEdgeWeights) {
        		bool readEdgeWeight = !std::getline(ss, item, ' ').fail();
        		if (!readEdgeWeight) {
        			throw std::runtime_error("Edge weight for " + std::to_string(neighbor) + " not found in line " + std::to_string(beginLocalRange + i) + ".");
        		}
        		ValueType edgeWeight = std::stod(item);
        		values.push_back(edgeWeight);
        	}
        	//std::cout << "Converted " << item << " to " << neighbor << std::endl;
        	neighbors.push_back(neighbor);
        }

        //set Ia array
        ia[i+1] = ia[i] + neighbors.size();
        //copy neighbors to Ja array
        std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(ja));
        if (hasEdgeWeights) {
        	assert(ja.size() == values.size());
        }
    }

    //std::cout << "Process " << comm->getRank() << " read " << ja.size() << " local edges. " << std::endl;


	nodeWeights.resize(numberNodeWeights);
    //std::cout << "Process " << comm->getRank() << " allocated memory for " << numberNodeWeights << " node weights. " << std::endl;
	for (IndexType i = 0; i < numberNodeWeights; i++) {
		nodeWeights[i] = DenseVector<ValueType>(dist, scai::utilskernel::LArray<ValueType>(localN, nodeWeightStorage[i].data()));
	}

    //std::cout << "Process " << comm->getRank() << " converted node weights. " << std::endl;

    if (endLocalRange == globalN) {
		bool eof = std::getline(file, line).eof();
		if (!eof) {
			throw std::runtime_error(std::to_string(globalN) + " lines read, but file continues.");
		}
	}

    file.close();

    //std::cout << "Process " << comm->getRank() << " closed file. " << std::endl;


    if (!hasEdgeWeights) {
    	assert(values.size() == 0);
    	values.resize(ja.size(), 1);//unweighted edges
    }

    assert(ja.size() == ia[localN]);
    SCAI_ASSERT(comm->sum(localN) == globalN, "Sum " << comm->sum(localN) << " should be " << globalN);

    if (comm->sum(ja.size()) != 2*globalM) {
    	throw std::runtime_error("Expected " + std::to_string(2*globalM) + " edges, got " + std::to_string(comm->sum(ja.size())));
    }

    //assign matrix
    scai::lama::CSRStorage<ValueType> myStorage(localN, globalN, ja.size(), scai::utilskernel::LArray<IndexType>(ia.size(), ia.data()),
    		scai::utilskernel::LArray<IndexType>(ja.size(), ja.data()),
    		scai::utilskernel::LArray<ValueType>(values.size(), values.data()));

    //std::cout << "Process " << comm->getRank() << " created local storage " << std::endl;

    return scai::lama::CSRSparseMatrix<ValueType>(myStorage, dist, noDist);
}
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraphBinary(const std::string filename, std::vector<DenseVector<ValueType>>& nodeWeights){

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
typedef long int LI;
    // root PE reads header and broadcasts information to the other PEs
    std::vector<LI> header(3, 0);
    bool success=false;
    
    if( comm->getRank()==0 ){
        std::cout <<  "Reading binary graph ..."  << std::endl;
        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if(file) {
            success = true;
            file.read((char*)(&header[0]), 3*sizeof(LI));
        }
        file.close();
        SCAI_ASSERT( success, "Error while opening the file " << filename);
ITI::aux::printVector( header );        
    }            
        
    
    //broadcast the header info
    comm->bcast( header.data(), 3, 0 );
    
    IndexType version = header[0];
    IndexType N = header[1];
    IndexType M = header[2];
    
    
    PRINT( *comm << ": version= " << version << ", N= " << N << ", M= " << M );
    
}

//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> FileIO<IndexType, ValueType>::readGraphMatrixMarket(const std::string filename){
    SCAI_REGION( "FileIO.readGraphMatrixMarket" );
    std::ifstream file(filename);
    
    scai::common::Settings::putEnvironment( "SCAI_IO_TYPE_DATA", "_Pattern" );
    
    if(file.fail())
        throw std::runtime_error("Could not open file "+ filename + ".");
    
    //skip the first lines that have comments starting with '%'
    std::string line;
    std::getline(file, line);

    while( line[0]== '%'){
       std::getline(file, line);
    }
    std::stringstream ss;
    ss.str( line );
   
    IndexType numRows;
    IndexType numColumns;
    IndexType numValues;
    
    ss >> numRows>> numColumns >> numValues;
    
    SCAI_ASSERT( numRows==numColumns , "Number of rows should be equal to number o columns");

    scai::lama::CSRSparseMatrix<ValueType> graph;
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    /*
    IndexType numRows, numColumns, numValues;
    scai::common::scalar::ScalarType dataType;
    bool isVector;
    scai::lama::MatrixMarketIO::Symmetry sym;
    
    scai::lama::MatrixMarketIO::readMMHeader( filename, numRows, numColumns, numValues, dataType, isVector, sym);
    */
    const scai::dmemo::DistributionPtr rowDist(new scai::dmemo::BlockDistribution(numRows, comm));
    
    graph.readFromFile( filename, rowDist );
    
    //unsetenv( "SCAI_IO_TYPE_DATA" );
    return graph;
}
//-------------------------------------------------------------------------------------------------
  
    
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType> > FileIO<IndexType, ValueType>::readCoordsOcean(std::string filename, IndexType dimension) {
	SCAI_REGION( "FileIO.readCoords" );
	std::ifstream file(filename);

	if(file.fail())
		throw std::runtime_error("Could not open file "+ filename + ".");

	std::string line;
	bool read = !std::getline(file, line).fail();
	if (!read) {
		throw std::runtime_error("Could not read first line of " + filename + ".");
	}

	std::stringstream ss( line );
	std::string item;
	bool readLine = !std::getline(ss, item, ' ').fail();
	if (!readLine or item.size() == 0) {
		throw std::runtime_error("Unexpected end of first line.");
	}

	//now read in file size
	const IndexType globalN = std::stoi(item);
	if (!(globalN >= 0)) {
		throw std::runtime_error(std::to_string(globalN) + " is not a valid node count.");
	}

	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

	IndexType beginLocalRange, endLocalRange;
	scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, globalN, comm->getRank(), comm->getSize());
	const IndexType localN = endLocalRange - beginLocalRange;

	//scroll forward to begin of local range
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
		bool read = !std::getline(file, line).fail();
		if (!read) {
			throw std::runtime_error("Unexpected end of coordinate file. Was the number of nodes correct?");
		}
		std::stringstream ss( line );
		std::string item;

		//first column contains index
		bool readIndex = !std::getline(ss, item, ' ').fail();
		if (!readIndex or item.size() == 0) {
			throw std::runtime_error("Could not read first element of line " + std::to_string(i+1));
		}
		IndexType nodeIndex = std::stoi(item);

		if (!nodeIndex == i+1) {
			throw std::runtime_error("Found index " + std::to_string(nodeIndex) + " in line " + std::to_string(i+1));
		}

		//remaining columns contain coordinates
		IndexType dim = 0;
		while (dim < dimension) {
			bool read = !std::getline(ss, item, ' ').fail();
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
			throw std::runtime_error(std::to_string(globalN) + " coordinates read, but file continues.");
		}
	}

	std::vector<DenseVector<ValueType> > result(dimension);

	for (IndexType i = 0; i < dimension; i++) {
		result[i] = DenseVector<ValueType>(dist, coords[i] );
	}

	return result;
}

//-------------------------------------------------------------------------------------------------
/*File "filename" contains the coordinates of a graph. The function reads these coordinates and returns a vector of DenseVectors, one for each dimension
 */
template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoords( std::string filename, IndexType numberOfPoints, IndexType dimension, Format format){
    SCAI_REGION( "FileIO.readCoords" );

    if (format == Format::OCEAN) {
	return readCoordsOcean(filename, dimension);
    }

    IndexType globalN= numberOfPoints;
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(globalN, comm));

    if (format == Format::OCEAN) {
        PRINT0("Reading coordinates in OCEAN format");
        return readCoordsOcean(filename, dimension);
    }
    else if( format== Format::MATRIXMARKET){
        PRINT0("Reading coordinates in MATRIXMARKET format");
        return readCoordsMatrixMarket( filename );
    }
    
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
		bool read = !std::getline(file, line).fail();
		if (!read) {
			throw std::runtime_error("Unexpected end of coordinate file. Was the number of nodes correct?");
		}
		std::stringstream ss( line );
		std::string item;

		IndexType dim = 0;
		while (dim < dimension) {
			bool read = !std::getline(ss, item, ' ').fail();
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
//-------------------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
std::vector<DenseVector<ValueType>> FileIO<IndexType, ValueType>::readCoordsMatrixMarket ( const std::string filename){
    std::ifstream file(filename);

    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");
        
    //skip the first lines that have comments starting with '%'
    std::string line;
    std::getline(file, line);

    while( line[0]== '%'){
       std::getline(file, line);
    }
    std::stringstream ss;
    ss.str( line );
    
    IndexType numPoints ;
    IndexType dimensions ;
    
    ss >> numPoints >> dimensions;
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr dist(new scai::dmemo::BlockDistribution(numPoints, comm));

    PRINT0( "numPoints= "<< numPoints << " , " << dimensions);
    
    IndexType beginLocalRange, endLocalRange;
    scai::dmemo::BlockDistribution::getLocalRange(beginLocalRange, endLocalRange, numPoints, comm->getRank(), comm->getSize());
    
    // the local ranges for the MatrixMarket format
    const IndexType beginLocalRangeMM = beginLocalRange*dimensions;
    const IndexType endLocalRangeMM = endLocalRange*dimensions;
    const IndexType localN = endLocalRange - beginLocalRange;
    const IndexType localNMM = endLocalRangeMM - beginLocalRangeMM;
    
    //PRINT( *comm << ": localN= "<< localNMM << ", numPoints= "<< numPoints << ", beginLocalRange= "<< beginLocalRangeMM << ", endLocalRange= " << endLocalRangeMM );
    
    //scroll forward to begin of local range
    for (IndexType i = 0; i < beginLocalRangeMM; i++) {
    	std::getline(file, line);
    }
    
    //create result vector
    std::vector<scai::utilskernel::LArray<ValueType> > coords(dimensions);
    for (IndexType dim = 0; dim < dimensions; dim++) {
    	coords[dim] = scai::utilskernel::LArray<ValueType>(localN, 0);
    }
    
    //read local range
    for (IndexType i = 0; i < localNMM; i++) {
		bool read = !std::getline(file, line).fail();
		
                if (!read and i!=localNMM-1 ) {
			throw std::runtime_error("In FileIO.cpp, line " + std::to_string(__LINE__) +"Unexpected end of coordinate file. Was the number of nodes correct?");
		}
		
		std::stringstream ss;
		ss.str( line );
                         
                ValueType c;
                ss >> c;
                coords[i%dimensions][int(i/dimensions)] = c;
                
//PRINT( *comm << ": " << i%dimensions << " , " << i/dimensions << " :: " << coords[i%dimensions][int(i/dimensions)]  );
                /*
		IndexType dim = 0;
		while (dim < dimensions) {
			bool read = std::getline(ss, item, ' ');
			if (!read or item.size() == 0) {
				throw std::runtime_error("Unexpected end of line. Was the number of dimensions correct?");
			}
			ValueType coord = std::stod(item);
			coords[dim][i] = coord;
			dim++;
		}
		
		if (dim < dimensions) {
			throw std::runtime_error("Only " + std::to_string(dim - 1)  + " values found, but " + std::to_string(dimensions) + " expected in line '" + line + "'");
		}
		*/
    }

    if (endLocalRange == numPoints) {
    	bool eof = std::getline(file, line).eof();
    	if (!eof) {
    		throw std::runtime_error(std::to_string(numPoints) + " coordinates read, but file continues.");
    	}
    }

    std::vector<DenseVector<ValueType> > result(dimensions);

    for (IndexType i = 0; i < dimensions; i++) {
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
	while (!std::getline(file, line).fail()) {
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
    while (!std::getline(file, line).fail()) {
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

		while (!std::getline(ss, item, ' ').fail()) {
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

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
std::pair<IndexType, IndexType> FileIO<IndexType, ValueType>::getMatrixMarketCoordsInfos(const std::string filename){
        
    std::ifstream file(filename);
    
    if(file.fail())
        throw std::runtime_error("File "+ filename+ " failed.");
        
    //skip the first lines that have comments starting with '%'
    std::string line;
    std::getline(file, line);

    while( line[0]== '%'){
       std::getline(file, line);
    }
    std::stringstream ss;
    ss.str( line );
    
    IndexType numPoints ;
    IndexType dimensions ;
    
    ss >> numPoints >> dimensions;
    
    return std::make_pair( numPoints, dimensions);
}

//-------------------------------------------------------------------------------------------------
template<typename IndexType, typename ValueType>
std::vector<IndexType> FileIO<IndexType, ValueType>::readBlockSizes(const std::string filename , const IndexType numBlocks){
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    
    std::vector<IndexType> blockSizes(numBlocks, 0);
    
    if( comm->getRank()==0 ){
        std::ifstream file(filename);
        if(file.fail())
            throw std::runtime_error("File "+ filename+ " failed.");
        
        //read first line, has the number of blocks
        std::string line;
        std::getline(file, line);
        std::stringstream ss;
        ss.str( line );
        
        IndexType fileNumBlocks;
        ss >> fileNumBlocks;
        SCAI_ASSERT_EQ_ERROR( numBlocks, fileNumBlocks, "Number of blocks mismatch, given "<< numBlocks << " but the file has "<< fileNumBlocks );

        for(int i=0;i<numBlocks; i++){
            bool read = !std::getline(file, line).fail();
            
            if (!read and i!=numBlocks-1 ) {
                throw std::runtime_error("In FileIO.cpp, line " + std::to_string(__LINE__) +": Unexpected end of block sizes file " + filename + ". Was the number of blocks correct?");
            }
            std::stringstream ss;    
            ss.str( line );
            IndexType bSize;
            ss >> bSize;
            //blockSizes.push_back(bSize);
            blockSizes[i]= bSize;
        }
        SCAI_ASSERT( blockSizes.size()==numBlocks , "Wrong number of blocks: "  <<blockSizes.size() << " for file " << filename);
        file.close();
        
        bool eof = std::getline(file, line).eof();
        if (!eof) {
            throw std::runtime_error(std::to_string(numBlocks) + " blocks read, but file continues.");
        }
    }
    
    // this call causes a seg fault ??
    //comm->bcastImpl( blockSizes.data(), blockSizes.size(), 0, scai::common::TypeTraits<ValueType>::stype);
    comm->bcast( blockSizes.data(), numBlocks, 0);
    
    return blockSizes;
}

template void FileIO<int, double>::writeGraph (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeGraphDistributed (const CSRSparseMatrix<double> &adjM, const std::string filename);
template void FileIO<int, double>::writeCoords (const std::vector<DenseVector<double>> &coords, const std::string filename);
template void FileIO<int, double>::writeCoordsDistributed_2D (const std::vector<DenseVector<double>> &coords, int numPoints, const std::string filename);
template CSRSparseMatrix<double> FileIO<int, double>::readGraph(const std::string filename, Format format);
template scai::lama::CSRSparseMatrix<double> FileIO<int, double>::readGraphBinary(const std::string filename, std::vector<DenseVector<double>>& nodeWeights);

template std::vector<DenseVector<double>> FileIO<int, double>::readCoords( std::string filename, int numberOfCoords, int dimension, Format format);
template std::vector<DenseVector<double>> FileIO<int, double>::readCoordsOcean( std::string filename, int dimension );
template CSRSparseMatrix<double>  FileIO<int, double>::readQuadTree( std::string filename, std::vector<DenseVector<double>> &coords );
template std::pair<int, int> FileIO<int, double>::getMatrixMarketCoordsInfos(const std::string filename);
template std::vector<int> FileIO<int, double>::readBlockSizes(const std::string filename , const int numBlocks );

} /* namespace ITI */
