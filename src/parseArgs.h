#pragma once

#include <iostream>
#include <cxxopts.hpp>

#include "Settings.h"

namespace ITI {
cxxopts::Options populateOptions();
cxxopts::ParseResult parseInput(cxxopts::Options options, int argc, char** argv);
Settings interpretSettings(cxxopts::ParseResult result);

}