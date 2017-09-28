# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.


project( DelaunayGraphs_example )

#CMAKE_MINIMUM_REQUIRED(VERSION 2.4.5)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

if ( COMMAND cmake_policy )
  cmake_policy( SET CMP0003 NEW )
endif()

find_package(CGAL QUIET COMPONENTS Core )

if ( CGAL_FOUND )

  include( ${CGAL_USE_FILE} )

  include( CGAL_CreateSingleSourceCGALProgram )

  # std c++11 needed only for the timng info in generate_dt_binary
  # not at all needed for generate_dt
  set (CMAKE_CXX_STANDARD 11)	# requires cmake version 3.1

  create_single_source_cgal_program( "generate_dt_binary.cpp" )
  create_single_source_cgal_program( "generate_dt.cpp" )

else()

    message(STATUS "This program requires the CGAL library, and will not be compiled.")

endif()

