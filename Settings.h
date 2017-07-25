#pragma once

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define BUILD_COMMIT_STRING STR_VALUE(BUILD_COMMIT)
#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

const std::string version = BUILD_COMMIT_STRING;

struct Settings{
    IndexType dimensions= 3;
    IndexType numX = 32;
    IndexType numY = 32;
    IndexType numZ = 32;
    IndexType numBlocks = 2;
    IndexType minBorderNodes = 1;
    IndexType stopAfterNoGainRounds = 0;
    IndexType minGainForNextRound = 1;
    IndexType sfcResolution = 17;
    IndexType numberOfRestarts = 0;
    IndexType diffusionRounds = 20;
    IndexType multiLevelRounds = 0;
    IndexType coarseningStepsBetweenRefinement = 3;
    IndexType pixeledSideLen = 10;
    IndexType fileFormat = 0;   // 0 for METSI, 1 for MatrixMarket
    IndexType initialPartition = 0;
    bool useDiffusionTieBreaking = false;
    bool useGeometricTieBreaking = false;
    bool useDiffusionCoordinates = false;
    bool gainOverBalance = false;
    bool skipNoGainColors = false;
    bool writeDebugCoordinates = false;
    bool bisect = false;
    bool useExtent = false;
    double epsilon = 0.05;
    std::string fileName = "-";
    
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
        out<< "fileFormat: "<< fileFormat << std::endl;
        switch( initialPartition){
            case 0: {
                out<< "initial partition: hilbert curve" << std::endl;  break;
            } 
            case 1:{
                out<< "initial partition: pixels" << std::endl;     break;
            }
            case 2:{
                out<< "initial partition: spectral" << std::endl;   break;
            }
            case 3:{
                out<< "initial partition: k-means" << std::endl;   break;
            }
            case 4:{
                if (!bisect){
                    out<< "initial partition: multisection" << std::endl;
                }else{
                    out<< "initial partition: bisection" << std::endl;
                }
                break;
            }
            default:{
                out<< "initial partition undefined" << std::endl;   break;
            }
        }
    }
};

