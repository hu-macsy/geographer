#pragma once

struct Settings{
    IndexType dimensions= 3;
    IndexType numX = 32;
    IndexType numY = 32;
    IndexType numZ = 32;
    IndexType numBlocks = 2;
    IndexType borderDepth = 4;
    IndexType stopAfterNoGainRounds = 0;
    IndexType minGainForNextRound = 1;
    IndexType sfcResolution = 17;
    IndexType numberOfRestarts = 0;
    IndexType diffusionRounds = 20;
    IndexType multiLevelRounds = 0;
    IndexType coarseningStepsBetweenRefinement = 3;
    bool useDiffusionTieBreaking = false;
    bool useGeometricTieBreaking = false;
    bool gainOverBalance = false;
    bool skipNoGainColors = false;
    double epsilon = 0.05;
    
    void print(std::ostream& out){
        IndexType numPoints = numX* numY* numZ;
        
        out<< "Setting: number of points= " << numPoints<< ", dimensions= "<< dimensions << ", borderDepth= "\
        << borderDepth << ", stopAfterNoGainRounds= "<< stopAfterNoGainRounds <<\
        ", minGainForNextRound= " << minGainForNextRound << ", sfcResolution= "<<\
        sfcResolution << ", epsilon= "<< epsilon << ", numBlocks= " << numBlocks << std::endl;
        out<< "multiLevelRounds: " << multiLevelRounds << std::endl;
        out<< "coarseningStepsBetweenRefinement: "<< coarseningStepsBetweenRefinement << std::endl;
        out<< "useDiffusionTieBreaking: " << useDiffusionTieBreaking <<std::endl;
        out<< "useGeometricTieBreaking: " << useGeometricTieBreaking <<std::endl;
        out<< "gainOverBalance: " << gainOverBalance << std::endl;
        out<< "skipNoGainColors: "<< skipNoGainColors << std::endl;
    }
/*    
    void print3D(std::ostream& out){
        IndexType numPoints = numX* numY* numZ;
        
        out<< "Settings: number of points= " << numPoints<< ", dimensions= "<< dimensions << ", borderDepth= "\
        << borderDepth << ", stopAfterNoGainRounds= "<< stopAfterNoGainRounds <<\
        ", minGainForNextRound= " << minGainForNextRound << ", sfcResolution= "<<\
        sfcResolution << ", epsilon= "<< epsilon << ", numBlocks= " << numBlocks << std::endl;
    }
*/
};

