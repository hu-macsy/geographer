#include "KMeansCoreset.h"
#include "gtest/gtest.h"

namespace ITI {

class KMeansCorsetTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

TEST_F(KMeansCorsetTest, testMyPartition) {
	
	//do some tests
}

}//ITI
