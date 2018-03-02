#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <algorithm>
#include <memory>
#include <cstdlib>
#include <numeric>

#include "GraphUtils.h"
#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "gtest/gtest.h"
#include "AuxiliaryFunctions.h"


using namespace scai;

namespace ITI {

class ParcoRepartTest : public ::testing::Test {
    protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";

};

TEST_F(RepartitionTest, testRepart){
    
	
	const IndexType n= 100;
	scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
	scai::dmemo::DistributionPtr distPtr(new scai::dmemo::BlockDistribution(n, comm));
	struct Settings settings;
	
	settings.dimensions = 2;
	
	scai::lama::DenseVector<ValueType> nodeWeights = Repartition<IndexType,ValueType>::sNW( distPtr, 0, settings);
}
//-----------------------------------------------------------------------------------------------------

} //namespace
