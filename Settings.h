#pragma once

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define BUILD_COMMIT_STRING STR_VALUE(BUILD_COMMIT)
#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

const std::string version = BUILD_COMMIT_STRING;

enum class InitialPartitioningMethods {SFC = 0, Pixel = 1, Spectral = 2, KMeans = 3, Multisection = 4};
    

struct Settings{
	//partition settings
	IndexType numBlocks = 2;
	double epsilon = 0.05;

    //input data
    IndexType dimensions= 2;
    std::string fileName = "-";
    IndexType fileFormat = 1;   // 0 for METIS, 4 for MatrixMarket
    bool useDiffusionCoordinates = false;
    IndexType diffusionRounds = 20;
    std::vector<IndexType> blockSizes;

    //mesh generation
	IndexType numX = 32;
	IndexType numY = 32;
	IndexType numZ = 32;

    //general tuning parameters
    InitialPartitioningMethods initialPartition = InitialPartitioningMethods::SFC;

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

    //parameters for multisection
    bool bisect = 0;    // 0: works for square k, 1: bisect, for k=power of 2
    bool useExtent = false;
    std::vector<IndexType> cutsPerDim;

    //tuning parameters for multiLevel heuristic
    IndexType multiLevelRounds = 0;
    IndexType coarseningStepsBetweenRefinement = 3;
    IndexType pixeledSideLen = 10;

    //debug parameters
    bool writeDebugCoordinates = false;

    void print(std::ostream& out){
        IndexType numPoints = numX* numY* numZ;
        
        out<< "Setting: number of points= " << numPoints<< ", dimensions= "<< dimensions << ", minBorderNodes= "\
        << minBorderNodes << ", stopAfterNoGainRounds= "<< stopAfterNoGainRounds <<\
        ", minGainForNextRound= " << minGainForNextRound << ", sfcResolution= "<<\
        sfcResolution << ", epsilon= "<< epsilon << ", numBlocks= " << numBlocks << std::endl;
        out<< "multiLevelRounds: " << multiLevelRounds << std::endl;
        out<< "coarseningStepsBetweenRefinement: "<< coarseningStepsBetweenRefinement << std::endl;
        out<< "useDiffusionTieBreaking: " << useDiffusionTieBreaking <<std::endl;
        out<< "useGeometricTieBreaking: " << useGeometricTieBreaking <<std::endl;
        out<< "gainOverBalance: " << gainOverBalance << std::endl;
        out<< "skipNoGainColors: "<< skipNoGainColors << std::endl;
        out<< "pixeledSideLen: "<< pixeledSideLen << std::endl;
        if (initialPartition==InitialPartitioningMethods::SFC) {
            out<< "initial partition: hilbert curve" << std::endl;
        } else if (initialPartition==InitialPartitioningMethods::Pixel) {
            out<< "initial partition: pixels" << std::endl;
        } else if (initialPartition==InitialPartitioningMethods::Spectral) {
        	out<< "initial partition: spectral" << std::endl;
        } else if (initialPartition==InitialPartitioningMethods::KMeans) {
            out<< "initial partition: K-Means" << std::endl;
        } else if (initialPartition==InitialPartitioningMethods::Multisection) {
        	out<< "initial partition: MultiSection" << std::endl;
        } else {
            out<< "initial partition undefined" << std::endl;
        }
    }
};

