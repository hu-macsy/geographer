#pragma once

#include <iostream>
#include <boost/program_options.hpp>

#include "Settings.h"

namespace ITI {

std::pair<boost::program_options::variables_map, Settings> parseInput(int argc, char** argv);

}