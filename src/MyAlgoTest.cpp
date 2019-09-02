#include "MyAlgo.h"
#include "gtest/gtest.h"

namespace ITI {

class MyAlgoTest : public ::testing::Test {
protected:
    // the directory of all the meshes used
    // projectRoot is defined in config.h.in
    const std::string graphPath = projectRoot+"/meshes/";
};

TEST_F(MyAlgoTest, testMyPartition) {
	
	//do some tests
}

}//ITI
