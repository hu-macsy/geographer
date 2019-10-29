#pragma once

/*
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <sys/stat.h>

*/
#include <cxxopts.hpp>

#include "AuxiliaryFunctions.h"
#include "FileIO.h"
#include "Settings.h"
#include "Metrics.h"
#include "MeshGenerator.h"
#include "parseArgs.h"

namespace ITI{

/** Read the needed parameters from the virtual machine and return the input data.
    @param[in] vm The virtual machine with the input parameters
    @param[in/out] settings Some input settings. Some of them might change
    @param[in] comm The communicator
    @param[out] graph The returned input matrix
    @param[out] coords The returned input coordinates
    @param[out] nodeWeights The returned input node weights
*/

template <typename ValueType>
IndexType readInput( 
    const cxxopts::ParseResult vm,
    const Settings settings,
    const scai::dmemo::CommunicatorPtr comm,
    scai::lama::CSRSparseMatrix<ValueType>& graph,
    std::vector<scai::lama::DenseVector<ValueType>>& coords,
    std::vector<scai::lama::DenseVector<ValueType>>& nodeWeights ){

    IndexType N;

    if (vm.count("graphFile")) {
        std::string graphFile =  vm["graphFile"].as<std::string>();
        std::string coordFile;

        if (vm.count("coordFile")) {
            coordFile = vm["coordFile"].as<std::string>();
        } else {
            coordFile = graphFile + ".xyz";
        }

        // read the graph
        if (vm.count("fileFormat")) {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, comm, settings.fileFormat );
        } else {
            graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, nodeWeights, comm );
        }

        N = graph.getNumRows();

        SCAI_ASSERT_EQUAL( graph.getNumColumns(),  graph.getNumRows(), "matrix not square");
        SCAI_ASSERT( graph.isConsistent(), "Graph not consistent");

        scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();

        // set the node weights
        IndexType numReadNodeWeights = nodeWeights.size();
        if (numReadNodeWeights == 0) {
            nodeWeights.resize(1);
            nodeWeights[0] = fill<DenseVector<ValueType>>(rowDistPtr, 1);
        }

        if (settings.numNodeWeights > 0) {
            if (settings.numNodeWeights < nodeWeights.size()) {
                nodeWeights.resize(settings.numNodeWeights);
                if (comm->getRank() == 0) {
                    std::cout << "Read " << numReadNodeWeights << " weights per node but " << settings.numNodeWeights << " weights were specified, thus discarding "
                              << numReadNodeWeights - settings.numNodeWeights << std::endl;
                }
            } else if (settings.numNodeWeights > nodeWeights.size()) {
                nodeWeights.resize(settings.numNodeWeights);
                for (IndexType i = numReadNodeWeights; i < settings.numNodeWeights; i++) {
                    nodeWeights[i] = fill<DenseVector<ValueType>>(rowDistPtr, 1);
                }
                if (comm->getRank() == 0) {
                    std::cout << "Read " << numReadNodeWeights << " weights per node but " << settings.numNodeWeights << " weights were specified, padding with "
                              << settings.numNodeWeights - numReadNodeWeights << " uniform weights. " << std::endl;
                }
            }
        }

        //read the coordinates file
        if (vm.count("coordFormat")) {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, comm, settings.coordFormat);
        } else if (vm.count("fileFormat")) {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, comm, settings.fileFormat);
        } else {
            coords = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, comm);
        }
        SCAI_ASSERT_EQUAL(coords[0].getLocalValues().size(), coords[1].getLocalValues().size(), "coordinates not of same size" );      

    }else if(vm.count("generate")) {

        N = settings.numX * settings.numY * settings.numZ;

        std::vector<ValueType> maxCoord(settings.dimensions); // the max coordinate in every dimensions, used only for 3D
        maxCoord[0] = settings.numX;
        maxCoord[1] = settings.numY;
        if(settings.dimensions==3) {
            maxCoord[2] = settings.numZ;
        }

        std::vector<IndexType> numPoints(3); // number of points in each dimension, used only for 3D

        for (IndexType i = 0; i < settings.dimensions; i++) {
            numPoints[i] = maxCoord[i];
        }

        if( comm->getRank()== 0) {
            std::cout<< "Generating for dim= "<< settings.dimensions << " and numPoints= "<< settings.numX << ", " << settings.numY << ", "<< settings.numZ << ", in total "<< N << " number of points" << std::endl;
            std::cout<< "\t\t and maxCoord= "; //<< maxCoord[0] << ", "<< maxCoord[1] << ", " << maxCoord[2]<< std::endl;
            for (IndexType i = 0; i < settings.dimensions; i++) {
                std::cout << maxCoord[i] << ", ";
            }
            std::cout << std::endl;
        }

        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph = scai::lama::zero<scai::lama::CSRSparseMatrix<ValueType>>( rowDistPtr, noDistPtr );

        scai::dmemo::DistributionPtr coordDist ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        for(IndexType i=0; i<settings.dimensions; i++) {
            coords[i].allocate(coordDist);
            coords[i] = static_cast<ValueType>( 0 );
        }

        // create the adjacency matrix and the coordinates
        ITI::MeshGenerator<IndexType, ValueType>::createStructuredMesh_dist( graph, coords, maxCoord, numPoints, settings.dimensions );

        IndexType nodes= graph.getNumRows();
        IndexType edges= graph.getNumValues()/2;
        if(comm->getRank()==0) {
            std::cout<< "Generated random 3D graph with "<< nodes<< " and "<< edges << " edges."<< std::endl;
        }

        nodeWeights.resize(1);
        nodeWeights[0] = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);

    } else if (vm.count("quadTreeFile")) {
        //if (comm->getRank() == 0) {
        graph = ITI::FileIO<IndexType, ValueType>::readQuadTree(vm["quadTreeFile"].as<std::string>(), coords);
        N = graph.getNumRows();
        //}

        //broadcast graph size from root to initialize distributions
        //IndexType NTransport[1] = {static_cast<IndexType>(graph.getNumRows())};
        //comm->bcast( NTransport, 1, 0 );
        //N = NTransport[0];

        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr(new scai::dmemo::NoDistribution(N));
        graph.redistribute(rowDistPtr, noDistPtr);
        for (IndexType i = 0; i < settings.dimensions; i++) {
            coords[i].redistribute(rowDistPtr);
        }

        nodeWeights.resize(1);
        nodeWeights[0] = scai::lama::fill<scai::lama::DenseVector<ValueType>>(graph.getRowDistributionPtr(), 1);

    }else{
        std::cout << "No input file was given. Call again with --graphFile, --quadTreeFile" << std::endl;
        return 126;        
    }

    if( not aux<IndexType,ValueType>::checkConsistency( graph, coords, nodeWeights, settings) ){
        PRINT0("Input not consistent.\nAborting...");
        return -1;
    }

    return N;
}

}