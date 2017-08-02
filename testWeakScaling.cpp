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

#include "MeshGenerator.h"
#include "FileIO.h"
#include "ParcoRepart.h"
#include "Settings.h"
#include "MultiLevel.h"
#include "LocalRefinement.h"
#include "SpectralPartition.h"
#include "MultiSection.h"
#include "AuxiliaryFunctions.h"

typedef double ValueType;
typedef int IndexType;


//----------------------------------------------------------------------------

namespace ITI {
	std::istream& operator>>(std::istream& in, Format& format)
	{
		std::string token;
		in >> token;
		if (token == "AUTO" or token == "0")
			format = ITI::Format::AUTO ;
		else if (token == "METIS" or token == "1")
			format = ITI::Format::METIS;
		else if (token == "ADCIRC" or token == "2")
			format = ITI::Format::ADCIRC;
		else if (token == "OCEAN" or token == "3")
			format = ITI::Format::OCEAN;
		else
			in.setstate(std::ios_base::failbit);
		return in;
	}

	std::ostream& operator<<(std::ostream& out, Format& method)
	{
		std::string token;

		if (method == ITI::Format::AUTO)
			token = "AUTO";
		else if (method == ITI::Format::METIS)
			token = "METIS";
		else if (method == ITI::Format::ADCIRC)
			token = "ADCIRC";
		else if (method == ITI::Format::OCEAN)
			token = "OCEAN";
		out << token;
		return out;
	}
}


int main(int argc, char** argv) {
	using namespace boost::program_options;
	options_description desc("Supported options");

        enum class pointDistribution { uniform, normal};
        std::string pointDist = "uniform";
        
	struct Settings settings;
        IndexType localN = -1; 		// total number of points
        IndexType ff = 1;
        
	desc.add_options()
				("dimensions", value<int>(&settings.dimensions)->default_value(settings.dimensions), "Number of dimensions of generated graph")
				("numX", value<int>(&settings.numX)->default_value(settings.numX), "Number of points in x dimension of generated graph")
				("numY", value<int>(&settings.numY)->default_value(settings.numY), "Number of points in y dimension of generated graph")
				("numZ", value<int>(&settings.numZ)->default_value(settings.numZ), "Number of points in z dimension of generated graph")
				("epsilon", value<double>(&settings.epsilon)->default_value(settings.epsilon), "Maximum imbalance. Each block has at most 1+epsilon as many nodes as the average.")
                                ("fileFormat", value<int>(&ff)->default_value(ff), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
                                ("distribution", value<std::string>(&pointDist)->default_value(pointDist), "The distribution of the points: can be normal or uniform")
                                ("numPoints", value<IndexType>(&localN) , "Number of local per PE to be generated.")
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
				;

        variables_map vm;
        store(command_line_parser(argc, argv).options(desc).run(), vm);
        notify(vm);
        
        if( localN<=0 ){
            std::cout << "Aborting, wrong number of points to be generated: " << localN << std::endl;
            return -1;
        }
        
        char machineChar[255];
        std::string machine;
        gethostname(machineChar, 255);
        if (machineChar) {
            machine = std::string(machineChar);
        } else {
            std::cout << "machine char not valid" << std::endl;
        }
        
        if( vm.count("cutsPerDim") ){
            IndexType tmpK = 1;
            for( const auto& i: settings.cutsPerDim){
                tmpK *= i;
            }
            settings.numBlocks= tmpK;
        }
        
        const IndexType initialPartition = settings.initialPartition;
        const IndexType dim = settings.dimensions;
        const IndexType k = settings.numBlocks;
            
        std::vector<ValueType> maxCoords( dim, 10);
        std::vector<ValueType> minCoords( dim, 0);
        
        scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
                
        if (comm->getRank() == 0){
            std::cout<< "input: weakScaling" << std::endl;
        }
        
        if( initialPartition!=4 ){
            std::cout << "Weak scaling works only for multisection (for now)" << std::endl;
            std::terminate();
        }
        /*
        if( comm->getSize()!=settings.numBlocks ){
            PRINT("Setting numBlocks= comm->getSize()= "<< comm->getSize() << " since is needed for redistribution to measure cut");
            settings.numBlocks = comm->getSize();
        }
        */
         
        const IndexType N = localN * comm->getSize();                   // total number of points
        PRINT0("localN= "<< localN << ", globalN= " << N);        
        
        std::random_device rnd_dvc;
        std::mt19937 mersenne_engine(rnd_dvc());
        
        //
        //create structured local part of graph
        //
        std::vector<DenseVector<ValueType>> coordinates(dim);           // the coordinates of the graph
        
        scai::dmemo::DistributionPtr rowDistPtr ( scai::dmemo::Distribution::getDistributionPtr( "BLOCK", comm, N) );
        scai::dmemo::DistributionPtr noDistPtr( new scai::dmemo::NoDistribution( N ));
        scai::lama::CSRSparseMatrix<ValueType> graph( rowDistPtr, noDistPtr);     // the adjacency matrix of the graph
        
        PRINT0("\"Created\" local part of graph. (for MultiSection the adjacency graph is not needed and it is empty)");
        
        //
        //create random local weights
        //
        scai::lama::DenseVector<ValueType> nodeWeights;                 // node weights
        
        std::uniform_real_distribution<ValueType> dist(1.0, 5.0);
        auto gen = std::bind(dist, mersenne_engine);
        
        std::vector<ValueType> tmpLocalWeights(localN);
        std::generate( begin(tmpLocalWeights), end(tmpLocalWeights), gen);

        scai::hmemo::HArray<ValueType> tmpWeights( tmpLocalWeights.size(), tmpLocalWeights.data() );

        nodeWeights.swap( tmpWeights, rowDistPtr);
        
        ValueType totalLocalWeight = std::accumulate( tmpLocalWeights.begin(), tmpLocalWeights.end(), 0.0);
        long long totalGlobalWeight = comm->sum(totalLocalWeight);
        
        PRINT0("Created local part of weights, totalGlobalWeight= " << totalGlobalWeight);
        
        //
        // create random local coordinates   
        //
        
        std::vector< std::vector<IndexType> > localPoints( localN, std::vector<IndexType>(dim,0) );
        
        std::vector<ValueType> scaledMin(dim, std::numeric_limits<ValueType>::max());
        std::vector<ValueType> scaledMax(dim, std::numeric_limits<ValueType>::lowest());
        
        ValueType scale = std::pow( N -1 , 1.0/dim);
        PRINT0( scale );
        
        for(IndexType d=0; d<dim; d++){  
            std::vector<ValueType> tmpLocalCoords(localN);
            
            if( pointDist=="uniform" ){
                std::uniform_real_distribution<ValueType> dist(0.0, maxCoords[d]);
                auto gen = std::bind(dist, mersenne_engine);
                std::generate( begin(tmpLocalCoords), end(tmpLocalCoords), gen);
            }else if( pointDist=="normal"){
                std::normal_distribution<ValueType> dist(0.0, maxCoords[d]);
                auto gen = std::bind(dist, mersenne_engine);
                std::generate( begin(tmpLocalCoords), end(tmpLocalCoords), gen);
            }else{
                PRINT0("Aborting, distribution " << pointDist << " not available");
                return -1;
            }
            
            scai::hmemo::HArray<ValueType> tmpHarray ( tmpLocalCoords.size(), tmpLocalCoords.data() ) ;
            coordinates[d].swap( tmpHarray, rowDistPtr );
        }
        PRINT0("Created local part of coordinates");
        
        // get a multisection partition
        //scai::lama::DenseVector<IndexType> multiSectionPartition =  ITI::MultiSection<IndexType, ValueType>::getPartitionNonUniform( graph, coordinates, nodeWeights, settings);
        
        // scale the coordinates. Done in a separate loop to mimic the running time of MultiSection::getPartition better
        
        std::chrono::time_point<std::chrono::system_clock>  beforeInitialTime =  std::chrono::system_clock::now();
        
        for(IndexType d=0; d<dim; d++){  
            scai::hmemo::ReadAccess<ValueType> localPartOfCoords( coordinates[d].getLocalValues() );
            
            scaledMax[d] = int(scale) ;
            scaledMin[d] = 0;
            
            ValueType thisDimScale = scale/(maxCoords[d]-minCoords[d]);
            ValueType thisDimMin = minCoords[d];
        
            for (IndexType i = 0; i < localN; i++) {
                ValueType normalizedCoord = localPartOfCoords[i] - thisDimMin;
                IndexType scaledCoord =  normalizedCoord * thisDimScale; 
                
                localPoints[i][d] = scaledCoord;
                
                SCAI_ASSERT( scaledCoord >=0 and scaledCoord<=scale, "Wrong scaled coordinate " << scaledCoord << " is either negative or more than "<< scale);
            }
        }

        std::shared_ptr<ITI::rectCell<IndexType,ValueType>> root = ITI::MultiSection<IndexType, ValueType>::getRectanglesNonUniform( graph, localPoints, nodeWeights, scaledMin, scaledMax, settings);
        scai::lama::DenseVector<IndexType> multiSectionPartition = ITI::MultiSection<IndexType, ValueType>::setPartition( root, rowDistPtr, localPoints);
        
        std::chrono::duration<double> partitionTime =  std::chrono::system_clock::now() - beforeInitialTime;

        //
        // prints - assertions
        //

        assert( multiSectionPartition.size() == N);
        assert( coordinates[0].size() == N);
                
        //if(dimensions==2){
        //   ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"finalWithMS");
        //}

        std::string destPath = "./partResults/weakScaling/";
        boost::filesystem::create_directories( destPath );   
        std::string logFile = destPath + "resultsWS_" +std::to_string(settings.numBlocks)+".log";
        std::ofstream logF(logFile);
        
        std::vector<std::shared_ptr<ITI::rectCell<IndexType, ValueType>>> allLeaves = root->getAllLeaves();
        std::shared_ptr<ITI::rectCell<IndexType, ValueType>> thisLeaf;
        ValueType thisLeafWeight;
        
        long long totalLeafWeight=0, maxLeafWeight=0, minLeafWeight=totalGlobalWeight;
        struct ITI::rectangle maxRect, minRect;
        
        for(int l=0; l<allLeaves.size(); l++){
            thisLeaf = allLeaves[l];
            thisLeafWeight = thisLeaf->getLeafWeight();
            PRINT0("leaf " << l << " weight: "<< thisLeafWeight );        
            
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
        if( comm->getRank()==0){
            std::cout<< "max rectangle is"<< std::endl;
            maxRect.print();
            std::cout<< "min rectangle is"<< std::endl;
            minRect.print();
        }
        
        //ValueType cut = ITI::ParcoRepart<IndexType, ValueType>::computeCut( graph, multiSectionPartition);
        //ValueType imbalance = ITI::ParcoRepart<IndexType, ValueType>::computeImbalance( multiSectionPartition, k);
        
        scai::lama::CSRSparseMatrix<ValueType> blockGraph = ITI::MultiSection<IndexType, ValueType>::getBlockGraphFromTree_local(root);
        
        IndexType maxComm = ITI::GraphUtils::getGraphMaxDegree<IndexType, ValueType>( blockGraph);
        IndexType totalComm = blockGraph.getNumValues()/2;
        ValueType imbalance = (maxLeafWeight - optWeight)/optWeight;

        if(comm->getRank()==0){
            if( settings.bisect==1 ){
                logF << "--  Initial bisection, total time: " << partitionTime.count() << std::endl;
            }else{
                logF << "--  Initial multisection, total time: " << partitionTime.count() << std::endl;
            }
            logF << "\tfinal imbalance= "<< imbalance;
            logF  << std::endl  << std::endl  << std::endl; 
            std::cout << "\033[1;32m--Initial multisection, total time: " << partitionTime.count() << std::endl;
            std::cout << "\t imbalance= "<< imbalance << " , maxComm= "<< maxComm << " , totalComm= " << totalComm <<"\033[0m";
            std::cout << std::endl  << std::endl  << std::endl;
        }
        //PRINT0("\nGot rectangles in time: " << partitionTime.count() << " - imbalance is " << maxLeafWeight/optWeight -1);
        /*
        i f(*dimensions==2){
        ITI::FileIO<IndexType, ValueType>::writeCoordsDistributed_2D( coordinates, N, destPath+"multisectPart");
            }
            */        

        return 0;
}