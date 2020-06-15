#include <unistd.h>

#include <scai/lama/matrix/all.hpp>

#include "Settings.h"


std::ostream& ITI::operator<<( std::ostream& out, const ITI::Tool tool) {
    std::string token;
    using ITI::Tool;

    switch( tool ) {
    case Tool::geographer:
        token = "geographer";
        break;
    case Tool::geoKmeans:
        token = "geoKmeans";
        break;
    case Tool::geoSFC:
        token = "geoSFC";
        break;
    case Tool::geoHierKM:
        token = "geoHierKM";
        break;
    case Tool::geoHierRepart:
        token = "geoHierRepart";
        break;
    case Tool::geoKmeansBalance:
        token = "geoKmeansBalance";
        break;
    case Tool::geoMS:
        token = "geoMS";
        break;
    case Tool::geomRebalance:
        token = "geomRebalance";
        break;
    case Tool::parMetisGraph:
        token = "parMetisGraph";
        break;
    case Tool::parMetisGeom:
        token = "parMetisGeom";
        break;
    case Tool::parMetisSFC:
        token = "parMetisSFC";
        break;
    case Tool::parMetisRefine:
        token = "parMetisRefine";
        break;
    case Tool::zoltanRIB:
        token = "zoltanRIB";
        break;
    case Tool::zoltanRCB:
        token = "zoltanRCB";
        break;
    case Tool::zoltanMJ:
        token = "zoltanMJ";
        break;
    case Tool::zoltanXPulp:
        token = "zoltanXPulp";
        break;
    case Tool::zoltanSFC:
        token = "zoltanSFC";
        break;
    case Tool::parhipFastMesh:
        token = "parhipFastMesh";
        break;
    case Tool::parhipUltraFastMesh:
        token = "parhipUltraFastMesh";
        break;
    case Tool::parhipEcoMesh:
        token = "parhipEcoMesh";
        break;
	case Tool::myAlgo:
		token = "myAlgo";
		break;
    case Tool::none:
        token = "none";
        break;
    case Tool::unknown:
    default:
        token = "unknown";
    }

    out << token;
    return out;
}


std::string ITI::to_string(const ITI::Tool& t) {
    std::ostringstream out;
    out<< t;
    return out.str();
}

std::string ITI::to_string(const ITI::Format& f) {
    std::ostringstream out;
    out<< f;
    return out.str();
}

std::istream& ITI::operator>>(std::istream& in, ITI::Tool& tool) {
    std::string token;
    in >> token;
    std::string tokenLower=token;
    std::transform(token.begin(), token.end(), tokenLower.begin(), ::tolower);

    if( token=="Geographer" or tokenLower=="geographer" )
        tool = ITI::Tool::geographer;
    else if( token=="geoSFC" or tokenLower=="geosfc")
        tool = ITI::Tool::geoSFC;
    else if( token=="geoKmeans" or tokenLower=="geokmeans")
        tool = ITI::Tool::geoKmeans;
    else if( token=="geoHierKM" or tokenLower=="geohierkmeans" )
        tool = ITI::Tool::geoHierKM;
    else if( token=="geoHierRepart" or tokenLower=="geohierrepart")
        tool = ITI::Tool::geoHierRepart;
    else if( token=="geoKmeansBalance" or tokenLower=="geokmeansbalance")
        tool = ITI::Tool::geoKmeansBalance;
    else if( token=="geoMS" or tokenLower=="geoms")
        tool = ITI::Tool::geoMS;
    else if( token=="geomRebalance" or tokenLower=="geomrebalance")
        tool = ITI::Tool::geomRebalance;
    else if( token=="parMetisGraph" or tokenLower=="parmetisgraph")
        tool = ITI::Tool::parMetisGraph;
    else if( token=="parMetisGeom" or tokenLower=="parmetisgeom" )
        tool = ITI::Tool::parMetisGeom;
    else if( token=="parMetisSFC" or tokenLower=="parmetissfc")
        tool = ITI::Tool::parMetisSFC;
    else if( token=="parmetisRefine" or tokenLower=="parmetisrefine")
        tool = ITI::Tool::parMetisRefine;
    else if( token=="zoltanRIB" or tokenLower=="zoltanrib")
        tool = ITI::Tool::zoltanRIB;
    else if( token=="zoltanRCB" or tokenLower=="zoltanrcb")
        tool = ITI::Tool::zoltanRCB;
    else if( token=="zoltanMJ" or tokenLower=="zoltanmj")
        tool = ITI::Tool::zoltanMJ;
    else if( token=="zoltanXPulp" or tokenLower=="zoltanxpulp")
        tool = ITI::Tool::zoltanXPulp;
    else if( token=="zoltanSFC" or tokenLower=="zoltansfc")
        tool = ITI::Tool::zoltanSFC;
    else if( token=="parhipFastMesh" or tokenLower=="parhipfastmesh" )
        tool = ITI::Tool::parhipFastMesh;
    else if( token=="parhipUltraFastMesh" or tokenLower=="parhipultrafastmesh")
        tool = ITI::Tool::parhipUltraFastMesh;
    else if( token=="parhipEcoMesh" or tokenLower=="parhipecomesh")
        tool = ITI::Tool::parhipEcoMesh;
	else if( token=="myAlgo")
		tool = ITI::Tool::myAlgo;
    else if( token=="None" or tokenLower=="none")
        tool = ITI::Tool::none;
    else
        tool = ITI::Tool::unknown;

    return in;
}

ITI::Tool ITI::to_tool(const std::string& s) {
    std::stringstream ss;
    ss << s;
    ITI::Tool t;
    ss >> t;
    return t;
}

std::string ITI::getCallingCommand( const int argc, char** argv ){
    std::string callingCommand = "";
    for (IndexType i = 0; i < argc; i++) {
        callingCommand += std::string(argv[i]) + " ";
    }  
    return callingCommand;
}

ITI::Settings::Settings() {
    char machineChar[255];
    gethostname(machineChar, 255);

    this->machine = std::string(machineChar);
}

bool ITI::Settings::checkValidity(const scai::dmemo::CommunicatorPtr comm ) {
    if( this->storeInfo and this->outFile=="-" and this->outDir=="-" ){
        this->isValid = false;
        if( comm->getRank()==0){
            std::cout<< "ERROR: storeInfo argument was given but no outFile or outDir was given" << std::endl;
        }
        return false;
    }
    if( initialMigration==Tool::unknown or initialPartition==Tool::unknown){
        this->isValid = false;
        if( initialMigration==Tool::unknown ){
            throw std::runtime_error("provided tool for initialMigration is not known");
        }
        if( initialPartition==Tool::unknown ){
            throw std::runtime_error("provided tool for initialPartition is not known");
        }
        return false;
    }

    return true;
}

template <typename ValueType>
ITI::Settings ITI::Settings::setDefault( const scai::lama::CSRSparseMatrix<ValueType>& graph){
    Settings retSet = *this;

    const scai::dmemo::DistributionPtr dist = graph.getRowDistributionPtr();
    const long int localN = dist->getLocalSize();

    retSet.minBorderNodes = std::max( int(localN*minBorderNodesPercent), 1); //10% of local nodes
    const scai::dmemo::CommunicatorPtr comm = dist->getCommunicatorPtr();
    if(comm->getRank() == 0 ){
        std::cout << "\tsetting (in PE 0) minBorderNodes to " << retSet.minBorderNodes << std::endl;
    }

    retSet.stopAfterNoGainRounds = 2;

    //TODO: when we set the minSamplingNodes, kmeans hangs after roundsTillAll rounds
    //long int roundsTillAll = 6; //in how many rounds we get all local points
    //retSet.minSamplingNodes = localN/std::pow(2,roundsTillAll);

    //minGainForNextRound is set inside ParcoRepart::doLocalRefinement()

    return retSet;
}


void ITI::MSG0( const std::string message ){
    const scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    if( comm->getRank()==0) {
       std::cout<< message << std::endl;
    }
}


//instantiation
template ITI::Settings ITI::Settings::setDefault<double>( const scai::lama::CSRSparseMatrix<double>& graph );
template ITI::Settings ITI::Settings::setDefault( const scai::lama::CSRSparseMatrix<float>& graph );
