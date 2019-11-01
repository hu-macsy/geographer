#pragma once

#include <iostream>
#include <cxxopts.hpp>

#include "Settings.h"

namespace ITI {
cxxopts::Options populateOptions();
Settings interpretSettings(cxxopts::ParseResult result);

template< typename T>
std::vector<T> parseVector( cxxopts::ParseResult vm, std::string paramName );


}