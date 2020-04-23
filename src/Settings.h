#pragma once

#include <iostream>
#include <scai/lama.hpp>
#include <assert.h>

#include "config.h"

#define PRINT( msg ) std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl
#define PRINT0( msg ) if(comm->getRank()==0)  std::cout<< __FILE__<< ", "<< __LINE__ << ": "<< msg << std::endl //not happy with these macros

namespace ITI {

using scai::IndexType;

/*The size of a point/vertex in the application. This is mainly (only)
used for the mapping using the CommTree. Every node in the tree has a
memory variable that indicated the maximum allowed size of this PE or
group of PEs. Remember, in the CommTree the leaves are the actual PEs
and the other nodes are groups consisting of a number of PEs. Then,
every PEs p, can contain at most p.memory/bytesPerVertex vertices.
TODO: investigate the best value to use
*/
//TODO: is this needed?
const IndexType bytesPerVertex = 8;

/** Different file formats to store a graph. See also FileIO.

AUTO: The program tries to deduce the format automatically

METIS: First line has a number that represents the number of vertices of the graph and some flags for vertex and node weights.
Then, line i has the vertices that are neighbors of vertex i.
The METIS format is described in detail here <a href="glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf">metis manual</a>.

ADCIRC: Coordinate format of the ADCIRC sea simulation model

MATRIXMARKET: The matrixmarket format. Details <a href="https://math.nist.gov/MatrixMarket/formats.html">here</a> and
<a href="https://people.sc.fsu.edu/~jburkardt/data/mm/mm.html">here</a>.

TEEC:  Graphs stored in the TEEC file format //TODO: probably not used

BINARY: Graphs stored in a binary format, see <a href="http://algo2.iti.kit.edu/schulz/software_releases/kahipv2.00.pdf">here</a>.

EDGELIST: The graph is stored as sequence of edges.

BINARYEDGELIST The graph is stored as sequence of edges but stored in binary format.

EDGELISTDIST: An edge list that is stored in several files.
*/

enum class Format {AUTO, METIS, ADCIRC, MATRIXMARKET, TEEC, BINARY, EDGELIST, BINARYEDGELIST, EDGELISTDIST};


/** @brief Operator to convert an enum Format to a stream.
*/
inline std::istream& operator>>(std::istream& in, Format& format) {
    std::string token;//TODO: There must be a more elegant way to do this with a map!
    in >> token;
    if (token == "AUTO")
        format = ITI::Format::AUTO ;
    else if (token == "METIS")
        format = ITI::Format::METIS;
    else if (token == "ADCIRC")
        format = ITI::Format::ADCIRC;
    else if (token == "MATRIXMARKET")
        format = ITI::Format::MATRIXMARKET;
    else if (token == "TEEC")
        format = ITI::Format::TEEC;
    else if (token == "BINARY")
        format = ITI::Format::BINARY;
    else if (token == "EDGELIST")
        format = ITI::Format::EDGELIST;
    else if (token == "BINARYEDGELIST")
        format = ITI::Format::BINARYEDGELIST;
    else if (token == "EDGELISTDIST")
        format = ITI::Format::EDGELISTDIST;
    else
        in.setstate(std::ios_base::failbit);
    return in;
}

/** @brief Operator to convert a stream to an enum Format.
*/

inline std::ostream& operator<<(std::ostream& out, Format method) {
    std::string token;

    if (method == ITI::Format::AUTO)
        token = "AUTO";
    else if (method == ITI::Format::METIS)
        token = "METIS";
    else if (method == ITI::Format::ADCIRC)
        token = "ADCIRC";
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

/** Different tools, i.e., algorithmic approaches, that can be used to partition a input graph, point set or a mesh, i.e., a graph with coordinates.
	For geographer, typically these are predetermined combinations of the input settings.

- geographer Both graph and coordinates are required. First, we take an initial partition using only the coordinates and the balanced k-means algorithm
	and them, the initial partition is improved by using the graph and doing multilevel coarsening and local refinement.
- geoKmeans Partition a point set (no graph is needed) using the balanced k-means algorithm.
- geoHierKM Partition a point set (no graph is needed) using the hierarchical balanced k-means algorithm.
- geoHierRepart First step is same as using geoHierKM but we also do a post-processing repartition step to improve the cut more.
- geoSFC Partition a point set (no graph is needed) using the hilbert space filling curve.
- geoMS Partition a point set (no graph is needed) using the MultiSection algorithm.
### The tools below require the external libraries parmetis and zoltan2.
- parMetisGraph Partition a graph using parmetis
- parMetisGeom Partition a mesh using a version of parmetis that also uses coordinates for an initial partition.
- parMetisSFC Partition a point set (no graph is needed) using the space filling curve algorithm of parmetis.
- zoltanRIB Partition a point set (no graph is needed) using the Recursive Inertia Bisection of zoltan2.
- zoltanRCB Partition a point set (no graph is needed) using the Recursive Coordinate Bisection of zoltan2.
- zoltanMJ Partition a point set (no graph is needed) using the Multijagged algorithm of zoltan2.
- zoltanMJ Partition a point set (no graph is needed) using the space filling curves algorithm of zoltan2.
*/
enum class Tool { geographer, geoKmeans, geoKmeansBalance, geoHierKM, geoHierRepart, geoSFC, geoMS, parMetisGraph, parMetisGeom, parMetisSFC, parMetisRefine, zoltanRIB, zoltanRCB, zoltanMJ, zoltanXPulp, zoltanSFC, parhipFastMesh, parhipUltraFastMesh, parhipEcoMesh, myAlgo, none, unknown};


std::istream& operator>>(std::istream& in, ITI::Tool& tool);

std::ostream& operator<<(std::ostream& out, const ITI::Tool tool);

std::string to_string(const ITI::Tool& t);

std::string to_string(const ITI::Format& f);

ITI::Tool to_tool(const std::string& s);

std::string getCallingCommand( const int argc, char** argv );

/** @brief A structure that holds several options for partitioning, input, output, metrics e.t.c.
*/
struct Settings {
    Settings();
    bool checkValidity(const scai::dmemo::CommunicatorPtr comm );

    /** @name General partition settings
    */
    //@{
    IndexType numBlocks = 2; 	///< number of blocks to partition to
    double epsilon = 0.03;		///< maximum allowed imbalance of the output partition
    bool repartition = false; 	///< set to true to respect the initial partition

    ITI::Tool initialPartition = ITI::Tool::geoKmeans;			///< the tool to use to get the initial partition, \sa Tool
    //static const ITI::Tool initialMigration = ITI::Tool::geoSFC;///< pre-processing step to redistribute/migrate coordinates
    ITI::Tool initialMigration = ITI::Tool::geoSFC;
    //@}

    /** @name Input data and other info
    */
    //@{
    IndexType dimensions= 2;	///< the dimension of the point set
    std::string fileName = "-";	///< the name of the input file to read the graph from
    std::string outFile = "-";	///< name of the file to store metrics (if desired)
    std::string outDir = "-"; 	//this is used by the competitors main
    std::string PEGraphFile = "-"; //TODO: this should not be in settings
    std::string blockSizesFile = "-"; //TODO: this should not be in settings
    ITI::Format fileFormat = ITI::Format::AUTO;   	///< the format of the input file, \sa Format
    ITI::Format coordFormat = ITI::Format::AUTO; 	///< the format of the coordinated input file, \sa Format
    bool useDiffusionCoordinates = false;		///< if not coordinates are provided, we can use artificial coordinates
    IndexType diffusionRounds = 20;				///< number of rounds to create the diffusion coordinates
    IndexType numNodeWeights = 0;		///< number of vertex weights
    std::string machine;                ///< name of the machine that the executable is running
    double seed;                        ///< random seed used for some routines
    std::string callingCommand;         ///< the complete calling command used
    bool autoSetCpuMem = false;         ///< if set, geographer will gather cpu and memory info and use them for partitioning
    IndexType processPerNode = 24;      ///< the number of processes per compute node. Is used with autoSetCpuMem to determine the cpu ID
    //@}

    /** @name Mesh generation settings
    For the mesh generator, the number of points per dimension
    */
    //@{
    IndexType numX = 32;
    IndexType numY = 32;
    IndexType numZ = 1;
    //@}

    /** @name Tuning parameters for local refinement
     */
    //@{
    IndexType minBorderNodes = 1;			///< minimum number of border nodes for the local refinement
    double minBorderNodesPercent = 0.001;
    IndexType stopAfterNoGainRounds = 0; 	///< number of rounds to stop local refinement if no gain is achieved
    IndexType minGainForNextRound = 1;		///< minimum gain to be achieved so local refinement proceeds to next round
    IndexType numberOfRestarts = 0;
    bool useDiffusionTieBreaking = false;	///< if diffusion should be used for tie breaking
    bool useGeometricTieBreaking = false;	///< if distance from center should be used for tie braking
    bool gainOverBalance = false;
    bool skipNoGainColors = false;			///< if we should skip some rounds if there is no gain
    ITI::Tool localRefAlgo = ITI::Tool::geographer; ///< with which algorithm to do local refinement
    //@}

    /** @name Space filling curve parameters
    */
    //@{
    IndexType sfcResolution = 9; 			///<tuning parameters for SFC, the resolution depth for the curve
    //@}


    /** @name Tuning parameters balanced K-Means
    */
    //@{
//TODO?: in the heterogenous and hierarchical case, minSamplingNodes
//makes more sense to be a percentage of the nodes, not a number. Or not?
    IndexType minSamplingNodes = 100;		///< the starting number of sampled nodes. If set to -1, all nodes are considered from the start

    double influenceExponent = 0.5;
    double influenceChangeCap = 0.1;
    IndexType balanceIterations = 20;		///< maximum number of iteration to do in order to achieve balance
    IndexType maxKMeansIterations = 50;		///< maximum number of global k-means iterations
    bool tightenBounds = false;
    bool freezeBalancedInfluence = false;
    bool erodeInfluence = false;
    bool keepMostBalanced = false;
    //IndexType batchSize = 100;              ///< after how many moves we calculate the global sum in KMeans::rebalance()
    double batchPercent = 0.05;          ///< calculate the batch size as a percentage of the number of local points
    //bool manhattanDistance = false;
    std::vector<IndexType> hierLevels; 		///< for hierarchial kMeans, the number of blocks per level
    std::string kMeansMshipSort = "lex";    ///< used in KMeans::rebalance() to sort vertices. Possible values are "lex" and "sqImba"
    //@}

    /** @name Parameters for multisection
    */
    //@{
    bool bisect = false;    				///< if true, we perform a bisection ( false: works for square k, true: for k=power of 2)
    bool useIter = false;                   ///< use the iterative approach
    IndexType maxIterations = 20;           ///< maximum number of iterations for iterative approach
    std::vector<IndexType> cutsPerDim;		///< the cuts we must do per dimensions (size=dimensions)
    IndexType pixeledSideLen = 10;			///< the side length of a uniform grid
    //@}

    /** @name Tuning parameters for multiLevel heuristic
    */
    //@{
    bool noRefinement = false;				///< if we will do local refinement or not
    IndexType multiLevelRounds = 3;			///< number of multilevel rounds
    IndexType coarseningStepsBetweenRefinement = 3; ///< number of rounds every which we do coarsening
    bool nnCoarsening = false;              ///< when matching vertices, use the nearest neighbor to match (and contract with)
    //@}

    /** @name Debug and profiling parameters
    */
    //@{
    bool verbose = false;					///< print more output info
    bool debugMode = false; 				///< even more checks and prints
    bool writeDebugCoordinates = false;		///< store coordinates and block id
    bool writePEgraph = false;				///< store the processor graph
    //TODO: storeInfo is mostly ignore. remove?
    bool storeInfo = false;					///< store metrics info
    bool storePartition = false;            ///< store partition info
    IndexType repeatTimes = 1;				///< for benchmarking, how many times is the partition repeated
    IndexType thisRound=-1; //TODO: what is this? This has nothing to do with the settings.

    std::string metricsDetail = "no";		///< level of details, possible values are: no, easy, all
    //calculate expensive performance metrics?
    bool computeDiameter = false;			///< if the diameter should be computed (can be expensive)
    IndexType maxDiameterRounds = 2;		///< max number of rounds to approximate the diameter
    IndexType maxCGIterations = 300;        ///< max number of iterations of the CG solver in metrics
    double CGResidual = 1e-6;
    //@}

    /** @name Various parameters
    */
    //@{

    /// use some default settings; will overwrite other arguments given in the command line
    bool setAutoSettings;
    ///this is used by the competitors main to set the tools we are gonna use
    std::vector<std::string> tools;

    /// for mapping by renumbering the block centers according to their SFC index
    bool mappingRenumbering = false;

    /// variable to check if the settings given are valid or not
    bool isValid = true;
    //@}
	
	int myAlgoParam = 0;

    //
    // print settings
    //

    /** @brief Print the settings.
    * @param[in] out The stream to print to
    */
    void print(std::ostream& out) const {

        out<< "Git commit: " << version << " and machine: "<< machine << std::endl;

        out<< "Setting: dimensions= "<< dimensions << ", filename: " << fileName << std::endl;
        if( outFile!="-" ) {
            out<< "outFile: " << outFile << std::endl;
        }

        out<< "minBorderNodes= " << minBorderNodes << std::endl;
        out<< "stopAfterNoGainRounds= "<< stopAfterNoGainRounds << std::endl;
        out<< "minGainForNextRound= " << minGainForNextRound << std::endl;
        out<< "multiLevelRounds= " << multiLevelRounds << std::endl;
        out<< "coarseningStepsBetweenRefinement= "<< coarseningStepsBetweenRefinement << std::endl;
        out<< "parameters used:" <<std::endl;
        if( useDiffusionTieBreaking ) {
            out<< "\tuseDiffusionTieBreaking"  <<std::endl;
        }
        if( useGeometricTieBreaking ) {
            out<< "\tuseGeometricTieBreaking" <<std::endl;
        }
        if( gainOverBalance ) {
            out<< "\tgainOverBalance"  << std::endl;
        }
        if( skipNoGainColors ) {
            out<< "\tskipNoGainColors" << std::endl;
        }

        out<< "initial migration: " << initialMigration << std::endl;
        out<< "initial partition: " << initialPartition << std::endl;

        if(ITI::to_string(initialPartition).rfind("geoSFC",0)==0 ){
        //if (initialPartition==ITI::Tool::geoSFC) {
            out<< "\tsfcResolution: " << sfcResolution << std::endl;
        }
        //else if (initialPartition==ITI::Tool::geoKmeans) {
        else if(ITI::to_string(initialPartition).rfind("geoKmeans",0)==0 ){
            out<< "\tminSamplingNodes: " << minSamplingNodes << std::endl;
            out<< "\tinfluenceExponent: " << influenceExponent << std::endl;
        }
        else if(ITI::to_string(initialPartition).rfind("geoHier",0)==0 ){
            out<< "\tminSamplingNodes: " << minSamplingNodes << std::endl;
            out<< "\thier levels: ";
            for(unsigned int i=0; i<hierLevels.size(); i++) {
               out<< hierLevels[i] << ", ";
            }
            out<< std::endl;
        }
        // else if (initialPartition==ITI::Tool::geoMS) {
        else if(ITI::to_string(initialPartition).rfind("geoMS",0)==0 ){
            out<< "\tbisect: " << bisect << std::endl;
            out<< "\tuseIter "<< useIter << std::endl;
        } else {
            out<< "initial partition undefined" << std::endl;
        }
        out << "local refinement algo: ";
        if( noRefinement ){
            out << "none" << std::endl;
        }else{
            out << localRefAlgo << std::endl;
        }

        out << "epsilon= "<< epsilon << std::endl;
        out << "numBlocks= " << numBlocks << std::endl;
    }
//--------------------------------------------------------------------------------------------

    void print(std::ostream& out, const scai::dmemo::CommunicatorPtr comm) const {
        if( comm->getRank()==0) {
            print( out );
        }
    }

    template <typename ValueType>
    Settings setDefault( const scai::lama::CSRSparseMatrix<ValueType>& graph );

}; //struct Settings



struct int_pair {
    int32_t first;
    int32_t second;
    bool operator<(const int_pair& rhs ) const {
        return first < rhs.first || (first == rhs.first && second < rhs.second);
    }
    bool operator>(const int_pair& rhs ) const {
        return first > rhs.first || (first == rhs.first && second > rhs.second);
    }
    bool operator<=(const int_pair& rhs ) const {
        return !operator>(rhs);
    }
    bool operator>=(const int_pair& rhs ) const {
        return !operator<(rhs);
    }
};

/** @endcond INTERNAL
*/
}// namespace ITI
