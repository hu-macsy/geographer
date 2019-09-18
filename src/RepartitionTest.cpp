#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/lama/Vector.hpp>

#include <algorithm>
#include <memory>
#include <cstdlib>
#include <numeric>

#include "GraphUtils.h"
#include "FileIO.h"
#include "Repartition.h"
#include "AuxiliaryFunctions.h"

#include "gtest/gtest.h"

using namespace scai;

namespace ITI {

class RepartitionTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

TEST_F(RepartitionTest, testNodeWeights) {
    //std::string fileName = "bubbles-00010.graph";
    std::string fileName = "Grid16x16";
    std::string graphFile = graphPath + fileName;
    std::string coordFile = graphFile + ".xyz";
    const IndexType dimensions = 2;

    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    CSRSparseMatrix<ValueType> graph = FileIO<IndexType, ValueType>::readGraph(graphFile );

    const IndexType n = graph.getNumRows();

    std::vector<DenseVector<ValueType>> coords = FileIO<IndexType, ValueType>::readCoords( std::string(coordFile), n, dimensions);

    //scai::dmemo::DistributionPtr distPtr = coords[0].getDistributionPtr();


    //scai::dmemo::DistributionPtr distPtr(new scai::dmemo::BlockDistribution(n, comm));

    struct Settings settings;
    settings.dimensions = dimensions;
    IndexType seed =0;
    IndexType divergence = 1;
    scai::lama::DenseVector<ValueType> nodeWeights = Repartition<IndexType,ValueType>::setNodeWeights( coords, seed, divergence, dimensions);
}
//-----------------------------------------------------------------------------------------------------

} //namespace
