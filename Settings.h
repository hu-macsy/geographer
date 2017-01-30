#pragma once

struct Settings{
    IndexType dimensions= 2;
    IndexType numX, numY, numZ;
    IndexType numBlocks = 2;
    IndexType borderDepth = 4;
    IndexType stopAfterNoGainRounds = 20;
    IndexType minGainForNextRound = 1;
    IndexType sfcResolution = 5;
    double epsilon = 0.2;
    
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

