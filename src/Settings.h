#pragma once

#include <iostream>
#include <scai/lama.hpp>
#include <assert.h>

#include "config.h"

#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl

namespace ITI{

	using scai::IndexType;

	/*The size of a point/vertex in the application. This is mainly (only)
	used for the mapping using the CommTree. Every node in the tree has a 
	memory variable that indicated the maximum allowed size of this PE or
	group of PEs. Remember, in the CommTree the leaves are the actual PEs
	and the other nodes are groups consisting of a number of PEs. Then,
	every PEs p, can contain at most p.memory/bytesPerVertex vertices.
	TODO: investigate the best value to use
	*/
	const IndexType bytesPerVertex = 8;

enum class Format {AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4, TEEC = 5, BINARY = 6, EDGELIST = 7, BINARYEDGELIST = 8, EDGELISTDIST = 9};

inline std::istream& operator>>(std::istream& in, Format& format){
	std::string token;//TODO: There must be a more elegant way to do this with a map!
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
	else if (token == "BINARYEDGELIST" or token == "8")
	    format = ITI::Format::BINARYEDGELIST;
	else if (token == "EDGELISTDIST" or token == "9")
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
	else if (method == ITI::Format::BINARYEDGELIST)
	    token == "BINARYEDGELIST";
	out << token;
	return out;
}

//-----------------------------------------------------------------------------------

enum class Tool{ geographer, geoKmeans, geoSFC, geoHierKM, geoHierRepart, geoMS, parMetisGraph, parMetisGeom, parMetisSFC, zoltanRIB, zoltanRCB, zoltanMJ, zoltanSFC, none};


std::istream& operator>>(std::istream& in, ITI::Tool& tool);

std::ostream& operator<<(std::ostream& out, const ITI::Tool tool);

std::string toString(const ITI::Tool& t);

ITI::Tool toTool(const std::string& s);

struct Settings{
	Settings();
	bool checkValidity();

    //partition settings
    IndexType numBlocks = 2;
    double epsilon = 0.03;
    bool repartition = false;
    
    //input data and other info
    IndexType dimensions= 2;
    std::string fileName = "-";
    std::string outFile = "-";
    std::string outDir = "-"; //this is used by the competitors main
    std::string PEGraphFile = "-";
    std::string blockSizesFile = "-";
    ITI::Format fileFormat = ITI::Format::AUTO;   // 0 for METIS, 4 for MatrixMarket
    ITI::Format coordFormat = ITI::Format::AUTO; 
    bool useDiffusionCoordinates = false;
    IndexType diffusionRounds = 20;
    IndexType numNodeWeights = -1;
    std::string machine;
    
    //mesh generation
    IndexType numX = 32;
    IndexType numY = 32;
    IndexType numZ = 32;
    
    //general tuning parameters
    ITI::Tool initialPartition = ITI::Tool::geoKmeans;
    static const ITI::Tool initialMigration = ITI::Tool::geoSFC;

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
    IndexType sfcResolution = 7;

    //tuning parameters balanced K-Means
//TODO?: in the heterogenous and hierarchical case, minSamplingNodes
//makes more sense to be a percentage of the nodes, not a number. Or not?
    IndexType minSamplingNodes = 100;	

    ValueType influenceExponent = 0.5;
    ValueType influenceChangeCap = 0.1;
    IndexType balanceIterations = 20;
    IndexType maxKMeansIterations = 50;
    bool tightenBounds = false;
    bool freezeBalancedInfluence = false;
    bool erodeInfluence = false;
    //bool manhattanDistance = false;
    std::vector<IndexType> hierLevels; //for hierarchial kMeans

    //parameters for multisection
    bool bisect = false;    // 0: works for square k, 1: bisect, for k=power of 2
    bool useExtent = false;
    std::vector<IndexType> cutsPerDim;
    IndexType pixeledSideLen = 10;

    //tuning parameters for multiLevel heuristic
    bool noRefinement = false;
    IndexType multiLevelRounds = 0;
    IndexType coarseningStepsBetweenRefinement = 3;
    IndexType thisRound=-1;

    //debug and profiling parameters
    bool verbose = false;
    bool writeDebugCoordinates = false;
    bool writePEgraph = false;
    bool writeInFile = false;
    bool storeInfo = false;
    //TODO: turn to false by default
    bool debugMode = false; //extra checks and prints
	IndexType repeatTimes = 1;
    
    //calculate expensive performance metrics?
    bool computeDiameter = false;
    IndexType maxDiameterRounds = 2;
    std::string metricsDetail = "no";

    //this is used by the competitors main to set the tools we are gonna use
    std::vector<std::string> tools;

    // variable to check if the settings given are valid or not
    bool isValid = true;

    //struct communicationTree commTree;
    //
    // print settings
    //
    
	void print(std::ostream& out) const {
		
		out<< "Git commit: " << version << " and machine: "<< machine << std::endl;
		
		IndexType numPoints = numX* numY* numZ;
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
		
		out<< "initial migration: " << initialMigration << std::endl;
		
		if (initialPartition==ITI::Tool::geoSFC) {
			out<< "initial partition: hilbert curve" << std::endl;
			out<< "\tsfcResolution: " << sfcResolution << std::endl;
		}
		else if (initialPartition==ITI::Tool::geoKmeans) {
			out<< "initial partition: K-Means" << std::endl;
			out<< "\tminSamplingNodes: " << minSamplingNodes << std::endl;
			out<< "\tinfluenceExponent: " << influenceExponent << std::endl;
		} else if (initialPartition==ITI::Tool::geoMS) {
			out<< "initial partition: MultiSection" << std::endl;
			out<< "\tbisect: " << bisect << std::endl;
			out<< "\tuseExtent: "<< useExtent << std::endl;
		} else {
			out<< "initial partition undefined" << std::endl;
		}
		out << "epsilon= "<< epsilon << std::endl;
		out << "numBlocks= " << numBlocks << std::endl;
	}
//--------------------------------------------------------------------------------------------

    void print(std::ostream& out, const scai::dmemo::CommunicatorPtr comm) const {
		if( comm->getRank()==0){
			 print( out );
		}
	}
    
}; //struct Settings

}// namespace ITI