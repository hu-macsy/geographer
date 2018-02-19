#pragma once

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define BUILD_COMMIT_STRING STR_VALUE(BUILD_COMMIT)
#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

const std::string version = BUILD_COMMIT_STRING;

typedef long int IndexType;
typedef double ValueType;

namespace ITI{
enum class Format {AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4, TEEC = 5, BINARY = 6, EDGELIST = 7, EDGELISTDIST = 8};

inline std::istream& operator>>(std::istream& in, Format& format){
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
    else if (token == "MATRIXMARKET" or token == "4")
		format = ITI::Format::MATRIXMARKET;
    else if (token == "TEEC" or token == "5")
        format = ITI::Format::TEEC;
    else if (token == "BINARY" or token == "6")
        format = ITI::Format::BINARY;
	else if (token == "EDGELIST" or token == "7")
        format = ITI::Format::EDGELIST;
	else if (token == "EDGELISTDIST" or token == "8")
	    format = ITI::Format::EDGELISTDIST;
	else
		in.setstate(std::ios_base::failbit);
	return in;
}

inline std::ostream& operator<<(std::ostream& out, Format method){
	std::string token;

	if (method == ITI::Format::AUTO)
		token = "AUTO";
	else if (method == ITI::Format::METIS)
		token = "METIS";
	else if (method == ITI::Format::ADCIRC)
		token = "ADCIRC";
	else if (method == ITI::Format::OCEAN)
		token = "OCEAN";
    else if (method == ITI::Format::MATRIXMARKET)
        token = "MATRIXMARKET";
    else if (method == ITI::Format::TEEC)
        token = "TEEC";
    else if (method == ITI::Format::BINARY)
        token == "BINARY";
	else if (method == ITI::Format::EDGELISTDIST)
        token == "EDGELISTDIST";
	else if (method == ITI::Format::EDGELIST)
	    token == "EDGELIST";
	out << token;
	return out;
}
}

enum class InitialPartitioningMethods {SFC = 0, Pixel = 1, Spectral = 2, KMeans = 3, Multisection = 4, None = 5};

struct Settings{
    //partition settings
    IndexType numBlocks = 2;
    double epsilon = 0.05;
    bool repartition = false;
    
    //input data and other info
    IndexType dimensions= 2;
    std::string fileName = "-";
    std::string outFile = "-";
    ITI::Format fileFormat = ITI::Format::METIS;   // 0 for METIS, 4 for MatrixMarket
    bool useDiffusionCoordinates = false;
    IndexType diffusionRounds = 20;
    std::vector<IndexType> blockSizes;
    std::string machine;
    
    //mesh generation
    IndexType numX = 32;
    IndexType numY = 32;
    IndexType numZ = 32;
    
    //general tuning parameters
    InitialPartitioningMethods initialPartition = InitialPartitioningMethods::SFC;
    InitialPartitioningMethods initialMigration = InitialPartitioningMethods::SFC;
    
    //tuning parameters for local refinement
    IndexType minBorderNodes = 1;
    IndexType stopAfterNoGainRounds = 0;
    IndexType minGainForNextRound = 1;
    IndexType numberOfRestarts = 0;
    bool useDiffusionTieBreaking = false;
    bool useGeometricTieBreaking = false;
    bool gainOverBalance = false;
    bool skipNoGainColors = false;

    //tuning parameters for SFC
    IndexType sfcResolution = 17;

    //tuning parameters balanced K-Means
    IndexType minSamplingNodes = 100;
    double influenceExponent = 0.5;
    double influenceChangeCap = 0.1;
    IndexType balanceIterations = 20;
    IndexType maxKMeansIterations = 50;
    bool tightenBounds = false;
    bool freezeBalancedInfluence = false;
    bool erodeInfluence = false;
    bool manhattanDistance = false;

    //parameters for multisection
    bool bisect = false;    // 0: works for square k, 1: bisect, for k=power of 2
    bool useExtent = false;
    std::vector<IndexType> cutsPerDim;
    IndexType pixeledSideLen = 10;

    //tuning parameters for multiLevel heuristic
    IndexType multiLevelRounds = 0;
    IndexType coarseningStepsBetweenRefinement = 3;

    //debug and profiling parameters
    bool verbose = false;
    bool writeDebugCoordinates = false;
    bool writeInFile = false;
    bool storeInfo = false;
	int repeatTimes = 1;
    
    //
    // print settings
    //
    
    void print(std::ostream& out, const scai::dmemo::CommunicatorPtr comm){
        
        if( comm->getRank()==0){
                
            IndexType numPoints = numX* numY* numZ;
            
            
            out<< "Code git version: " << version << " and machine: "<< machine << std::endl;
            out<< "Setting: number of points= " << numPoints<< ", dimensions= "<< dimensions << ", filename: " << fileName << std::endl;
            if( outFile!="-" ){
                out<< "outFile: " << outFile << std::endl;
            }
    
            out<< "minBorderNodes= " << minBorderNodes << std::endl;
            out<< "stopAfterNoGainRounds= "<< stopAfterNoGainRounds << std::endl;
            out<< "minGainForNextRound= " << minGainForNextRound << std::endl;
            out<< "multiLevelRounds= " << multiLevelRounds << std::endl;
            out<< "coarseningStepsBetweenRefinement= "<< coarseningStepsBetweenRefinement << std::endl;
            out<< "parameters used:" <<std::endl;
            if( useDiffusionTieBreaking ){
                out<< "\tuseDiffusionTieBreaking"  <<std::endl;
            }
            if( useGeometricTieBreaking ){
                out<< "\tuseGeometricTieBreaking" <<std::endl;
            }
            if( gainOverBalance ){
                out<< "\tgainOverBalance"  << std::endl;
            }
            if( skipNoGainColors ){
                out<< "\tskipNoGainColors" << std::endl;
            }

            out<< "initial migration: " << static_cast<int>(initialMigration) << std::endl;
            
            if (initialPartition==InitialPartitioningMethods::SFC) {
                out<< "initial partition: hilbert curve" << std::endl;
                out<< "\tsfcResolution: " << sfcResolution << std::endl;
            } else if (initialPartition==InitialPartitioningMethods::Pixel) {
                out<< "initial partition: pixels" << std::endl;
                out<< "\tpixeledSideLen: "<< pixeledSideLen << std::endl;
            } else if (initialPartition==InitialPartitioningMethods::Spectral) {
                out<< "initial partition: spectral" << std::endl;
            } else if (initialPartition==InitialPartitioningMethods::KMeans) {
                out<< "initial partition: K-Means" << std::endl;
                out<< "\tminSamplingNodes: " << minSamplingNodes << std::endl;
                out<< "\tinfluenceExponent: " << influenceExponent << std::endl;
            } else if (initialPartition==InitialPartitioningMethods::Multisection) {
                out<< "initial partition: MultiSection" << std::endl;
                out<< "\tbisect: " << bisect << std::endl;
                out<< "\tuseExtent: "<< useExtent << std::endl;
            } else {
                out<< "initial partition undefined" << std::endl;
            }
            out << "epsilon= "<< epsilon << std::endl;
			out << "numBlocks= " << numBlocks << std::endl;
        }
    }
};

