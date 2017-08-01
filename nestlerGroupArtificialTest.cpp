#include <scai/lama.hpp>

#include <scai/lama/matrix/all.hpp>
#include <scai/lama/matutils/MatrixCreator.hpp>

#include <scai/dmemo/BlockDistribution.hpp>
#include <scai/dmemo/Distribution.hpp>

#include <scai/hmemo/Context.hpp>
#include <scai/hmemo/HArray.hpp>

#include <scai/utilskernel/LArray.hpp>
#include <scai/lama/Vector.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <memory>
#include <cstdlib>
#include <chrono>

//#include "MeshGenerator.h"
//#include "FileIO.h"
#include "ParcoRepart.h"
#include "Settings.h"
#include "MultiLevel.h"
//#include "LocalRefinement.h"
//#include "SpectralPartition.h"
#include "AuxiliaryFunctions.h"
#include "MultiSection.h"

typedef double ValueType;
typedef int IndexType;


int main(int argc, char** argv) {
	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;

        desc.add_options()
            ("help", "display options")
            ("version", "show version")
            ("graphFile", value<std::string>(), "read graph from file")
            ("coordFile", value<std::string>(), "coordinate file. If none given, assume that coordinates for graph arg are in file arg.xyz")
            ("generate", "generate random graph. Currently, only uniform meshes are supported.")
            ("weakScaling", "generate coordinates locally for weak scaling")
            ("dimensions", value<int>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
            ("numX", value<int>(&settings.numX)->default_value(settings.numX), "Number of points in x dimension of generated graph")
            ("numY", value<int>(&settings.numY)->default_value(settings.numY), "Number of points in y dimension of generated graph")
            ("numZ", value<int>(&settings.numZ)->default_value(settings.numZ), "Number of points in z dimension of generated graph")
            ("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
            ("numBlocks", value<IndexType>(&settings.numBlocks), "Number of blocks to partition to")
            ("minBorderNodes", value<int>(&settings.minBorderNodes)->default_value(settings.minBorderNodes), "Tuning parameter: Minimum number of border nodes used in each refinement step")
            ("stopAfterNoGainRounds", value<int>(&settings.stopAfterNoGainRounds)->default_value(settings.stopAfterNoGainRounds), "Tuning parameter: Number of rounds without gain after which to abort localFM. A value of 0 means no stopping.")
            ("initialPartition",  value<int> (&settings.initialPartition), "Parameter for different initial partition: 0 for the hilbert space filling curve, 1 for the pixeled method, 2 for spectral parition")
            ("bisect", value<bool>(&settings.bisect)->default_value(settings.bisect), "Used for the multisection method. If set to true the algorithm perfoms bisections (not multisection) until the desired number of parts is reached")
             ("cutsPerDim", value<std::vector<IndexType>>(&settings.cutsPerDim)->multitoken(), "If msOption=2, then provide d values that define the number of cuts per dimension.")
            ("pixeledSideLen", value<int>(&settings.pixeledSideLen)->default_value(settings.pixeledSideLen), "The resolution for the pixeled partition or the spectral")
            ("minGainForNextGlobalRound", value<int>(&settings.minGainForNextRound)->default_value(settings.minGainForNextRound), "Tuning parameter: Minimum Gain above which the next global FM round is started")
            ("gainOverBalance", value<bool>(&settings.gainOverBalance)->default_value(settings.gainOverBalance), "Tuning parameter: In local FM step, choose queue with best gain over queue with best balance")
            ("useDiffusionTieBreaking", value<bool>(&settings.useDiffusionTieBreaking)->default_value(settings.useDiffusionTieBreaking), "Tuning Parameter: Use diffusion to break ties in Fiduccia-Mattheyes algorithm")
            ("useGeometricTieBreaking", value<bool>(&settings.useGeometricTieBreaking)->default_value(settings.useGeometricTieBreaking), "Tuning Parameter: Use distances to block center for tie breaking")
            ("skipNoGainColors", value<bool>(&settings.skipNoGainColors)->default_value(settings.skipNoGainColors), "Tuning Parameter: Skip Colors that didn't result in a gain in the last global round")
            ("multiLevelRounds", value<int>(&settings.multiLevelRounds)->default_value(settings.multiLevelRounds), "Tuning Parameter: How many multi-level rounds with coarsening to perform")
            ("fileFormat", value<int>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for METIS format, 1 for MatrixMarket format. See FileIO for more details.")
            ("outputFile", value<std::string>()->default_value("rectanglesNestler"), "The name of the output file to write the partition")
        ;
                                
                     
    variables_map vm;
    store(command_line_parser(argc, argv).
    options(desc).run(), vm);
    notify(vm);
    
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    
    if (vm.count("version")) {
        std::cout << "Git commit " << version << std::endl;
        return 0;
    }
    
    if (vm.count("generate") && vm.count("file")) {
        std::cout << "Pick one of --file or --generate" << std::endl;
        return 0;
    }
    
    if (vm.count("generate") && (vm["dimensions"].as<int>() != 3)) {
        std::cout << "Mesh generation currently only supported for three dimensions" << std::endl;
        return 0;
    }
                        
    IndexType sideLen = 200;                            // length of grid side 
    IndexType edges= -1;                                // number of edges
    IndexType dim = settings.dimensions;
    IndexType N = std::pow( sideLen, dim);      	// total number of points
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    scai::dmemo::DistributionPtr distPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
    
    const IndexType localN = distPtr->getLocalSize();
    
    scai::lama::DenseVector<ValueType> nodeWeights( distPtr );               // node weights
    std::vector<ValueType> maxCoord(dim);                                       // the max coordinate in every dimensions
    
    IndexType initialPartition = settings.initialPartition;
    
    IndexType bigR = 3*sideLen/4;       // radious of big circle
    IndexType bigRSq = bigR*bigR;
    IndexType smallR = 2*sideLen/3;       // radious of small circle
    IndexType smallRSq = smallR*smallR;
    
    std::vector<IndexType> maxPoints= {sideLen, sideLen};
    ValueType totalLocalWeight = 0, totalGlobalWeight = 0;
    //
    // create random local weights   
    //
    {
        scai::hmemo::WriteOnlyAccess<ValueType> wLocalWeights( nodeWeights.getLocalValues() );
      
        std::ofstream fNew;
        std::string newFile = "meshes/nestler200.weights";
        fNew.open(newFile);
        
        fNew << sideLen << " " << 2 << std::endl;
        
        std::random_device rd;
        std::default_random_engine gen(rd());
        
        // points in the heavy zone have weight w: 5<w<10 and on the light zone: 0.5<w<2
        std::uniform_real_distribution<ValueType> distHeavy(4.0, 6.0);
        std::uniform_real_distribution<ValueType> distLight(0.5, 2.0);
PRINT0( localN );        
srand(time(NULL));
        for(IndexType i=0; i<localN; i++){  
            IndexType globalIndex = distPtr->local2global(i);
            
            std::tuple<IndexType, IndexType> point = ITI::aux::index2_2DPoint( i , maxPoints );
            IndexType pointNormSq = std::pow(std::get<0>(point), 2) + std::pow(std::get<1>(point), 2);  //x^2 + y^2
            
            if( pointNormSq-bigRSq<0 and pointNormSq-smallRSq>0 ){     //point is in the big circle but NOT in the small circle
                wLocalWeights[i] = std::round(distHeavy(gen)*1000.0)/1000.0;
                //wLocalWeights[i] = std::round(((ValueType)rand()/RAND_MAX+1)*5 *1000.0)/1000.0;
            }else{                                          //point is either in the small circle or not even in the big one
                wLocalWeights[i] = std::round(distLight(gen)*1000.0)/1000.0 ;
                //wLocalWeights[i] = std::round(((ValueType)rand()/RAND_MAX+1)*2 *1000.0)/1000.0;
            }
            totalLocalWeight += wLocalWeights[i];
            fNew<< std::get<0>(point) << " " << std::get<1>(point) << " " << wLocalWeights[i] << std::endl;
        }
    }
    totalGlobalWeight = comm->sum( totalLocalWeight );
    
    PRINT0("Created local part of weights");
    
    if (comm->getRank() == 0){
        std::cout<< "commit:"<< version<< /*", input:"<< ( vm.count("graphFile") ? vm["graphFile"].as<std::string>() :" generate") <<*/ std::endl;
    }
    
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();
    
    //
    // get partition into rectangles
    //
    std::shared_ptr<ITI::rectCell<IndexType,ValueType>> root = ITI::MultiSection<IndexType, ValueType>::getRectangles(nodeWeights, sideLen, settings);
    
    std::chrono::duration<double> partitionTime = std::chrono::system_clock::now() - startTime;
    
    //
    // get information for the rectangles
    //
    root->printLeavesInFile( vm["outputFile"].as<std::string>(), dim );
    
    std::vector<std::shared_ptr<ITI::rectCell<IndexType, ValueType>>> allLeaves = root->getAllLeaves();
    std::shared_ptr<ITI::rectCell<IndexType, ValueType>> thisLeaf;
    ValueType thisLeafWeight;
    
    ValueType totalLeafWeight=0, maxLeafWeight=0, minLeafWeight=totalGlobalWeight;
    struct ITI::rectangle maxRect, minRect;
    
    for(int l=0; l<allLeaves.size(); l++){
        thisLeaf = allLeaves[l];
        thisLeafWeight = thisLeaf->getLeafWeight();
        PRINT0("leaf " << ": "<< thisLeafWeight );        
        
        totalLeafWeight += thisLeafWeight;
        
        if( thisLeafWeight>maxLeafWeight ){
            maxLeafWeight = thisLeafWeight;
            maxRect = thisLeaf->getRect();
        }
        if( thisLeafWeight<minLeafWeight ){
            minLeafWeight = thisLeafWeight;
            minRect = thisLeaf->getRect();
        }
    }
    
    SCAI_ASSERT_LE_ERROR( totalLeafWeight-totalGlobalWeight, 0.00000001 , "Wrong weights sum.");
    
    ValueType optWeight = totalGlobalWeight/settings.numBlocks;
    
    PRINT0("maxWeight= " << maxLeafWeight << ", optWeight= "<< optWeight << " , minWeight= "<< minLeafWeight );
    std::cout<< "max rectangle is"<< std::endl;
    maxRect.print();
    std::cout<< "min rectangle is"<< std::endl;
    minRect.print();
    
    PRINT0("\nGot rectangles in time: " << partitionTime.count() << " - imbalance is " << (maxLeafWeight - optWeight)/optWeight);
}
