#include "gtest/gtest.h"

#include "Mapping.h"
#include "GraphUtils.h"
#include "FileIO.h"
#include "Metrics.h"
//#include "Settings.h"

namespace ITI {

class MappingTest : public ::testing::Test {
    protected:
        // the directory of all the meshes used
        std::string graphPath = "./meshes/";
};

TEST_F(MappingTest, testTorstenMapping){

	//std::string fileName = "bigtrace-00000.graph";
	std::string fileName = "Grid4x4";
    std::string file = graphPath + fileName;
    Settings settings;
    settings.dimensions = 2;
    settings.numBlocks = 1;

    scai::lama::CSRSparseMatrix<ValueType> blockGraph = FileIO<IndexType, ValueType>::readGraph(file );
    //scai::lama::CSRSparseMatrix<ValueType> PEGraph = FileIO<IndexType, ValueType>::readGraph(file );
    scai::lama::CSRSparseMatrix<ValueType> PEGraph (blockGraph);
    const IndexType N = blockGraph.getNumRows();

    blockGraph.setValue( 4, 8, 2 );
    blockGraph.setValue( 9, 10, 3 );
    blockGraph.setValue( 3, 7, 2.2 );

    PEGraph.setValue( 14, 15, 3.7);
    PEGraph.setValue( 6, 10, 4.3);

	std::vector<IndexType> mapping = Mapping<IndexType, ValueType>::torstenMapping_local(
		blockGraph, PEGraph );

	bool valid = Mapping<IndexType,ValueType>::isValid(blockGraph, PEGraph, mapping);
	EXPECT_TRUE( valid );

	//WARNING: if we call it as "Metrics metrics();" it throws an error
	Metrics metrics(settings);
	metrics.getMappingMetrics( blockGraph, PEGraph, mapping );

	std::vector<IndexType> identityMapping(N,0);
	std::iota( identityMapping.begin(), identityMapping.end(), 0);
	metrics.getMappingMetrics( blockGraph, PEGraph, identityMapping );

	//print mapping
	if( N<65 ){
		std::cout << "Mapping:" << std::endl;
		for(int i=0; i<N; i++){
			std::cout << i << " <--> " << mapping[i] << std::endl;
		}
	}
}

}//namespace ITI
