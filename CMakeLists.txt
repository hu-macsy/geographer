cmake_minimum_required (VERSION 3.0.2)
project (Geographer)
include(CMakePackageConfigHelpers)
include(CTest)

# define library version
set(Geographer_VERSION_MAJOR 0 STRING "major version")
set(Geographer_VERSION_MINOR 9 STRING "minor version")
set(Geographer_VERSION ${Geographer_VERSION_MAJOR}.${Geographer_VERSION_MINOR} STRING "version")

#set(CMAKE_VERBOSE_MAKEFILE ON)

option(USE_NETWORKIT "Use NETWORKIT package." OFF)
if(USE_NETWORKIT)
  if (NOT DEFINED NETWORKIT_DIR)
     message( FATAL_ERROR "Install Directory of NETWORKIT not defined, specify with -DNETWORKIT_DIR" )
  endif (NOT DEFINED NETWORKIT_DIR)
  add_definitions(-DUSE_NETWORKIT)
endif(USE_NETWORKIT)

set(NETWORKIT_INCLUDE "${NETWORKIT_DIR}/include/") 
set(NETWORKIT_LIB "${NETWORKIT_DIR}/lib/")

### set compiler manually ###
#set(CMAKE_CXX_COMPILER /lrz/sys/compilers/gcc/7.3.0/bin/c++)


set(RBC_DIR "${PROJECT_SOURCE_DIR}/src/RBC")

### find MPI ###
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

### find GTest ###
find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})
if(GTEST_FOUND)
  enable_testing()  
endif(GTEST_FOUND)

option(USE_BOOST "Use boost library for number parsing" OFF)

### find Boost ###
if (USE_BOOST)
  find_package(Boost REQUIRED COMPONENTS program_options filesystem system)
  include_directories(${Boost_INCLUDE_DIRS})
  add_definitions(-DUSE_BOOST)
else(USE_BOOST)
  find_package(Boost COMPONENTS program_options filesystem system)
endif(USE_BOOST)

### find SCAI ###
set ( SCAI_PATH_HINTS "${SCAI_DIR}" "${SCAI_HOME}" "$ENV{SCAI_HOME}" )

if ( ${CMAKE_VERSION} VERSION_GREATER 3.11)
    ## use new policy to configure with SCAI_ROOT, suppress warning
    cmake_policy( SET CMP0074 NEW )
else ()
    ## simulate new policy by adding SCAI_ROOT to hints
    set ( SCAI_PATH_HINTS ${SCAI_PATH_HINTS} "${SCAI_ROOT}" "$ENV{SCAI_ROOT}" )
endif ()

find_package( SCAI REQUIRED CONFIG 
              NO_CMAKE_PATH 
              NO_SYSTEM_ENVIRONMENT_PATH 
              PATHS ${SCAI_PATH_HINTS} )

include_directories(${SCAI_INCLUDE_DIRS})
add_definitions(${SCAI_DEFINITIONS})
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCAI_CXX_FLAGS}" )
link_directories(${SCAI_ROOT}/lib/)

### include and link RBC ###
find_library( RBC_LIBRARY _rbc "${RBC_DIR}/build/lib" )

if((NOT RBC_LIBRARY) OR (NOT EXISTS ${RBC_LIBRARY}))
  message("RBC Library not found, cloning and building it now.")
  execute_process(COMMAND git submodule update --init
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  execute_process(COMMAND make WORKING_DIRECTORY ${RBC_DIR})
  find_library( RBC_LIBRARY _rbc "${RBC_DIR}/build/lib" )
endif()

### add networkit include and link directory ###
# link_directories has to happen before target definitions
if (USE_NETWORKIT)
  include_directories(${NETWORKIT_INCLUDE})
  link_directories(${NETWORKIT_LIB})
endif (USE_NETWORKIT)

### get git commit ###
execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions("-DBUILD_COMMIT=${GIT_COMMIT}")

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --openmp --std=c++11 -O3 -fPIC -Wall -Wno-sign-compare -Wno-unused-variable" )

### define install directories ###
set(BIN_DEST "bin")
set(INCLUDE_DEST "include")
set(HEADER_DEST "${INCLUDE_DEST}/geographer")
set(LIB_DEST "lib/geographer")
set(CMAKE_DEST "share/cmake/geographer")

### set runtime path ###
set ( CMAKE_SKIP_BUILD_RPATH FALSE )
set ( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_DEST}")
set ( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

### set value type ###
set(Geographer_VALUE_TYPE "double" CACHE STRING "variable type used for values")
configure_file("src/config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h" DESTINATION "${HEADER_DEST}")

### include and define targets ###
add_subdirectory(src)

if (Boost_FOUND)
  add_executable(analyze src/FileIO.cpp src/Settings.cpp src/GraphUtils.cpp src/CommTree.cpp src/Metrics.cpp tools/analyzePartition.cpp)
  target_link_libraries(analyze ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

  add_executable(heatMap src/FileIO.cpp src/Settings.cpp src/GraphUtils.cpp src/CommTree.cpp tools/graphToHeatmap.cpp)
  target_link_libraries(heatMap ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})
endif (Boost_FOUND)

### support for FindPackage ###
configure_file(GeographerConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfig.cmake @ONLY)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfigVersion.cmake
  VERSION 0.9
  COMPATIBILITY ExactVersion )

### install files ###
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfigVersion.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfig.cmake
        DESTINATION "${CMAKE_DEST}" )

### find zoltan ###
FIND_PACKAGE( Zoltan2 PATHS "/home/hpc/pr87si/di36qin/Code/Trilinos/build/packages/zoltan2/")

#TODO: check if libraries are indeed found?
find_library( METIS_LIB metis "/lrz/sys/libraries/metis/5.1.0/i8r8/lib" )
find_library( PARMETIS_LIB parmetis "/lrz/sys/libraries/parmetis/4.0.3/ibmmpi/i8r8/lib" )

### for more info see the Zoltan2 documentation and https://trilinos.org/oldsite/Finding_Trilinos.txt
### see also in Trilinos_install_path/install/include/Makefile.export.Zoltan2

if( Zoltan2_FOUND )
	message( "\n\tFound package Zoltan2\n")

	link_directories( ${Zoltan2_LIBRARY_DIRS} ) # not sure if needed, "/home/hpc/pr87si/di36qin/Code/Trilinos/install/lib/"
	# maybe ${Zoltan2_TPL_LIBRARY_DIRS} is needed but, at least in my installation, it is empty

	add_executable( allCompetitors Wrappers.cpp ${FILES_COMMON} allCompetitorsPart.cpp )
	target_include_directories( allCompetitors PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} )
	
	### include parmetis and metis ###
	target_include_directories( allCompetitors PUBLIC "/lrz/sys/libraries/metis/5.1.0/i8r8/include" )
	target_include_directories( allCompetitors PUBLIC "/lrz/sys/libraries/parmetis/4.0.3/ibmmpi/i8r8/include" )

	target_link_libraries( allCompetitors ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${Boost_LIBRARIES} ${Zoltan2_TPL_LIBRARIES} ${Zoltan2_LIBRARIES} ${METIS_LIB} ${PARMETIS_LIB} ${MPI_CXX_LIBRARIES} )
else()
	#message( "Package Zoltan2 was not found.\nSet it in the CMakeLists.txt file by editing the line 'FIND_PACKAGE(Zoltan2 ...)'")
endif()
