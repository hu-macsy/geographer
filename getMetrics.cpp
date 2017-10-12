#include <scai/lama.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "Settings.h"
#include "FileIO.h"
#include "GraphUtils.h"
#include "Metrics.h"


namespace ITI {
	std::istream& operator>>(std::istream& in, Format& format)
	{
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
		else
			in.setstate(std::ios_base::failbit);
		return in;
	}

	std::ostream& operator<<(std::ostream& out, Format method)
	{
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
		out << token;
		return out;
	}
}




int main(int argc, char** argv) {
	using namespace boost::program_options;
	options_description desc("Supported options");

	struct Settings settings;
    
    std::string partFile;
    std::string graphFile;
    std::string coordFile;
    
    ITI::Format coordFormat;
    
    desc.add_options()
        ("partFile", value<std::string>(&partFile), "The file to read the partition from.")
        ("graphFile", value<std::string>(&graphFile), "The graph file.")
        ("coordFile", value<std::string>(&coordFile), "The coordinates file.")
        ("fileFormat", value<ITI::Format>(&settings.fileFormat)->default_value(settings.fileFormat), "The format of the file to read: 0 is for AUTO format, 1 for METIS, 2 for ADCRIC, 3 for OCEAN, 4 for MatrixMarket format. See FileIO.h for more details.")
        ("coordFormat", value<ITI::Format>(&coordFormat), "format of coordinate file: AUTO = 0, METIS = 1, ADCIRC = 2, OCEAN = 3, MATRIXMARKET = 4 ")
        ("dimensions", value<IndexType>(&settings.dimensions), "Number of dimensions of generated graph")
        //TODO: add option to read node weights
        //("nodeWeights")
    ;
         
    variables_map vm;
    store(command_line_parser(argc, argv).
    options(desc).run(), vm);
    notify(vm);
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();

    
    if( !vm.count("dimensions") and comm->getRank()==0 ){
        std::cout<< "Must specify number of dimensions d using --dimensions=d" << std::endl;
    }
    
    // the adjacency matrix of the graph
    scai::lama::CSRSparseMatrix<ValueType> graph =  ITI::FileIO<IndexType, ValueType>::readGraph( graphFile, settings.fileFormat );
    
    IndexType N = graph.getNumRows();
    
    const scai::dmemo::DistributionPtr rowDistPtr = graph.getRowDistributionPtr();
    
    std::vector<DenseVector<ValueType>> coordinates(settings.dimensions); // the coordinates of the graph
    coordinates = ITI::FileIO<IndexType, ValueType>::readCoords(coordFile, N, settings.dimensions, coordFormat);
     
    scai::lama::DenseVector<IndexType> partition = ITI::FileIO<IndexType, ValueType>::readPartition( partFile, N);
    
    scai::lama::DenseVector<ValueType> uniformWeights( rowDistPtr, 0);
    
    struct Metrics metrics( comm->getSize() );
    

    
    return 0;
}