/**
 * @file analyzePartition.cpp
 *
 * A standalone executable to convert the coordinates of a graph into a heatmap.
 */

#include <cxxopts.hpp>

#include "../src/FileIO.h"

using scai::lama::CSRSparseMatrix;
using scai::lama::DenseVector;
using scai::lama::fill;
using ITI::Settings;
using ITI::IndexType;
using ITI::ValueType;
using ITI::version;

int main(int argc, char** argv) {
    using namespace cxxopts;
    cxxopts::Options options("graphToHeatmap", "Converting graph to grid, suitable for heat map plotting");

    struct Settings settings;

    IndexType numGridCells = 100;
    IndexType globalN = -1;

    options.add_options()
    ("help", "display options")
    ("version", "show version")
    //input and coordinates
    ("graphFile", "read graph from file", value<std::string>())
    ("coordFile", "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz", value<std::string>())
    ("fileFormat", "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.", value<ITI::Format>())
    ("coordFormat", "format of coordinate file: AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4 ", value<ITI::Format>())
    ("nodeWeightIndex", "index of node weight", value<IndexType>()->default_value(0))
    ("numVertices", "Number of vertices, in case no graph file is given", value<IndexType>())
    ("dimensions", "Number of dimensions", value<IndexType>())
    ("gridCells", "Number of grid cells in each dimension to use for the visualization.", value<IndexType>()->default_value(std::to_string(numGridCells)))
    ;

    cxxopts::ParseResult vm = options.parse(argc, argv);

    bool validOptions = true;

    if (vm.count("help")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    if (vm.count("version")) {
        std::cout << "Git commit " << ITI::version << std::endl;
        return 0;
    }

    if (!(vm.count("graphFile") || vm.count("numVertices"))) {
        std::cout << "Need either a graph file or the number of vertices." << std::endl;
        validOptions = false;
    }

    if (!(vm.count("graphFile") || vm.count("coordFile"))) {
        std::cout << "Coordinate file needed." << std::endl;
        validOptions = false;
    }

    if (!validOptions) {
        return 126;
    }

    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    if (comm->getSize() > 1) {
        std::cout << "This program does not profit from parallelism." << std::endl;
    }

    std::string coordFile;
    std::string graphFile;

    if (vm.count("graphFile")) {
        graphFile = vm["graphFile"].as<std::string>();
    }

    if (vm.count("coordFile")) {
        coordFile = vm["coordFile"].as<std::string>();
    } else {
        assert(vm.count("graphFile"));
        coordFile = graphFile + ".xyz";
    }

    if (vm.count("dimensions")) {
        settings.dimensions = vm["dimensions"].as<IndexType>();
    }

    if (settings.dimensions != 2) {
        throw std::logic_error("Only implemented for 2 dimensions.");
    }

    if (vm.count("fileFormat")) {
        settings.fileFormat = vm["fileFormat"].as<ITI::Format>();
    }

    if (vm.count("coordFormat")) {
        settings.coordFormat = vm["coordFormat"].as<ITI::Format>();
    }

    if (vm.count("gridCells")) {
        numGridCells = vm["gridCells"].as<IndexType>();
    }

    DenseVector<ValueType> nodeWeights;

    if (vm.count("graphFile")) {
        std::vector<DenseVector<ValueType> > vectorOfnodeWeights;
        CSRSparseMatrix<ValueType> graph = ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, vectorOfnodeWeights, settings.fileFormat );
        const IndexType numReadVertices = graph.getNumRows();
        if (!vm.count("numVertices")) {
            globalN = numReadVertices;
        } else {
            assert(vm["numVertices"].as<IndexType>() == numReadVertices);
        }

        if (vectorOfnodeWeights.size() > 0) {
            nodeWeights = vectorOfnodeWeights[vm["nodeWeightIndex"].as<IndexType>()];
        } else {
            nodeWeights = fill<DenseVector<ValueType>>(globalN, 1);
        }
    } else {
        nodeWeights = fill<DenseVector<ValueType>>(globalN, 1);
    }

    const scai::dmemo::DistributionPtr noDist(new scai::dmemo::NoDistribution(globalN));
    nodeWeights.redistribute(noDist);

    std::vector<DenseVector<ValueType>> coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, globalN, settings.dimensions, settings.coordFormat);

    std::vector<ValueType> minCoords(settings.dimensions);
    std::vector<ValueType> maxCoords(settings.dimensions);

    for (IndexType d = 0; d < settings.dimensions; d++) {
        minCoords[d] = coordinates[d].min();
        maxCoords[d] = coordinates[d].max();
    }

    const ValueType resolutionX = (maxCoords[0] - minCoords[0]) / (numGridCells-1);
    const ValueType resolutionY = (maxCoords[1] - minCoords[1]) / (numGridCells-1);

    const IndexType localN = globalN;
    std::vector<std::vector<ValueType>> convertedCoords(settings.dimensions);

    for (IndexType d = 0; d < settings.dimensions; d++) {
        coordinates[d].redistribute(noDist);//replicate. Alternatively, create heat map locally and add together later

        scai::hmemo::ReadAccess<ValueType> rAccess(coordinates[d].getLocalValues());
        convertedCoords[d] = std::vector<ValueType>(rAccess.get(), rAccess.get()+localN);

        minCoords[d] = *std::min_element(convertedCoords[d].begin(), convertedCoords[d].end());
        maxCoords[d] = *std::max_element(convertedCoords[d].begin(), convertedCoords[d].end());

        assert(convertedCoords[d].size() == localN);
    }

    std::vector<std::vector<ValueType>> weightsInCell(numGridCells, std::vector<ValueType>(numGridCells, 0));
    std::vector<std::vector<bool>> populated(numGridCells, std::vector<bool>(numGridCells, false));

    assert(convertedCoords.size() == 2);
    scai::hmemo::ReadAccess<ValueType> rWeights(nodeWeights.getLocalValues());
    assert(rWeights.size() == localN);
    for (IndexType i = 0; i < localN; i++) {
        const IndexType gridIndexX = (convertedCoords[0][i] - minCoords[0]) / resolutionX;
        SCAI_ASSERT_LT_ERROR( gridIndexX, numGridCells, "grid index out of bounds");
        const IndexType gridIndexY = (convertedCoords[1][i] - minCoords[1]) / resolutionY;
        SCAI_ASSERT_LT_ERROR( gridIndexY, numGridCells, "grid index out of bounds");

        weightsInCell[gridIndexX][gridIndexY] += rWeights[i];
        populated[gridIndexX][gridIndexY] = true;
    }

    //open output file
    std::string outFileName = coordFile + ".heat";
    std::ofstream file;
    file.open(outFileName.c_str());

    //write column labels
    file << "x" << "\t" << "y" << "\t" << "weight" << std::endl;

    //write heat map
    for (IndexType i = 0; i < weightsInCell.size(); i++) {
        double x = minCoords[0] + i*resolutionX;
        for (IndexType j = 0; j < weightsInCell[i].size(); j++) {
            double y = minCoords[1] + j*resolutionY;
            std::string heat = populated[i][j] ? std::to_string(weightsInCell[i][j]) : std::string("NaN");
            file << x << '\t' << y << '\t' << heat << std::endl;
        }
        file << std::endl;
    }
    file.close();
}
