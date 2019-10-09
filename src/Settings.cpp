#include <unistd.h>

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
    case Tool::geoMS:
        token = "geoMS";
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
    case Tool::zoltanSFC:
        token = "zoltanSFC";
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

    if( token=="Geographer" or token=="geographer" )
        tool = ITI::Tool::geographer;
    else if( token=="geoSFC" or token=="geoSfc" or token=="SFC")
        tool = ITI::Tool::geoSFC;
    else if( token=="geoKmeans" or token=="geoKMeans" or token=="Kmeans")
        tool = ITI::Tool::geoKmeans;
    else if( token=="geoHierKM" or token=="geoHierKmeans" or token=="geoHierKMeans")
        tool = ITI::Tool::geoHierKM;
    else if( token=="geoHierRepart" or token=="geohierrepart" or token=="geoHieRepart")
        tool = ITI::Tool::geoHierRepart;
    else if( token=="geoMS" or token=="geoMultiSection" or token=="geoMultisection")
        tool = ITI::Tool::geoMS;
    else if( token=="parMetisGraph" or token=="parMetisgraph" or token=="parmetisGraph")
        tool = ITI::Tool::parMetisGraph;
    else if( token=="parMetisGeom" or token=="parMetisgeom" or token=="parmetisGeom")
        tool = ITI::Tool::parMetisGeom;
    else if( token=="parMetisSFC" or token=="parMetisSfc" or token=="parmetisSFC")
        tool = ITI::Tool::parMetisSFC;
    else if( token=="parMetisRefine" or token=="parMetisrfc" or token=="parmetisRefine")
        tool = ITI::Tool::parMetisRefine;
    else if( token=="zoltanRIB" or token=="zoltanRib" or token=="zoltanrib")
        tool = ITI::Tool::zoltanRIB;
    else if( token=="zoltanRCB" or token=="zoltanRcb" or token=="zoltanrcb")
        tool = ITI::Tool::zoltanRCB;
    else if( token=="zoltanMJ" or token=="zoltanMj" or token=="zoltanmj")
        tool = ITI::Tool::zoltanMJ;
    else if( token=="zoltanSFC" or token=="zoltanSfc" or token=="zoltansfc")
        tool = ITI::Tool::zoltanRIB;
	else if( token=="myAlgo")
		tool = ITI::Tool::myAlgo;
    else if( token=="None" or token=="none")
        tool = ITI::Tool::none;
    else
        tool = ITI::Tool::unknown;

    return in;
}

ITI::Tool ITI::toTool(const std::string& s) {
    std::stringstream ss;
    ss << s;
    ITI::Tool t;
    ss >> t;
    return t;
}

ITI::Settings::Settings() {
    char machineChar[255];
    gethostname(machineChar, 255);

    this->machine = std::string(machineChar);
}

bool ITI::Settings::checkValidity() {
    if( this->storeInfo && this->outFile=="-" ) {
        this->isValid = false;
        return false;
    }
    if( initialPartition==Tool::unknown or initialPartition==Tool::unknown){
        return false;
    }

    return isValid;
}

