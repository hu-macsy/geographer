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
    IndexType sfcResolution = 0;
    IndexType numberOfRestarts = 0;
    IndexType diffusionRounds = 20;
    IndexType multiLevelRounds = 5;
    IndexType coarseningStepsBetweenRefinement = 3;
    bool useDiffusionTieBreaking = false;
    bool useGeometricTieBreaking = false;
    bool gainOverBalance = false;
    bool skipNoGainColors = false;
    double epsilon = 0.05;
    
    void print2D(std::ostream& out){
        IndexType numPoints = numX;
        
        out<< "Setting: number of points= " << numPoints<< ", dimensions= "<< dimensions << ", borderDepth= "\
        << borderDepth << ", stopAfterNoGainRounds= "<< stopAfterNoGainRounds <<\
        ", minGainForNextRound= " << minGainForNextRound << ", sfcResolution= "<<\
        sfcResolution << ", epsilon= "<< epsilon << ", numBlocks= " << numBlocks << std::endl;
    }
    
    void print3D(std::ostream& out){
        IndexType numPoints = numX* numY* numZ;
        
        out<< "Settings: number of points= " << numPoints<< ", dimensions= "<< dimensions << ", borderDepth= "\
        << borderDepth << ", stopAfterNoGainRounds= "<< stopAfterNoGainRounds <<\
        ", minGainForNextRound= " << minGainForNextRound << ", sfcResolution= "<<\
        sfcResolution << ", epsilon= "<< epsilon << ", numBlocks= " << numBlocks << std::endl;
    }
};

