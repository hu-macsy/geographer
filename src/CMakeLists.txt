cmake_minimum_required (VERSION 2.8.8)
project (Geographer)
include(CMakePackageConfigHelpers)

# define library version
set(Geographer_VERSION_MAJOR 0 STRING "major version")
set(Geographer_VERSION_MINOR 9 STRING "minor version")
set(Geographer_VERSION ${Geographer_VERSION_MAJOR}.${Geographer_VERSION_MINOR} STRING "version")

if (NOT DEFINED SCAI_DIR)
	message( FATAL_ERROR "Install Directory of LAMA not defined, specify with -DSCAI_DIR" )
endif (NOT DEFINED SCAI_DIR)

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

### set files ###
set(FILES_HEADER ParcoRepart.h MultiLevel.h LocalRefinement.h HilbertCurve.h MeshGenerator.h FileIO.h SpectralPartition.h Diffusion.h GraphUtils.h MultiSection.h KMeans.h CommTree.h AuxiliaryFunctions.h HaloPlanFns.h Metrics.h Mapping.h Settings.h)
set(FILES_COMMON ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp SpectralPartition.cpp Diffusion.cpp GraphUtils.cpp MultiSection.cpp KMeans.cpp CommTree.cpp AuxiliaryFunctions.cpp HaloPlanFns.cpp Metrics.cpp Mapping.cpp Settings.cpp)
set(FILES_TEST test_main.cpp auxTest.cpp DiffusionTest.cpp FileIOTest.cpp GraphUtilsTest.cpp HilbertCurveTest.cpp KMeansTest.cpp LocalRefinementTest.cpp MeshGeneratorTest.cpp MultiLevelTest.cpp MultiSectionTest.cpp ParcoRepartTest.cpp quadtree/test/QuadTreeTest.cpp SpectralPartitionTest.cpp)

set(RBC_DIR "${PROJECT_SOURCE_DIR}/RBC")

### find MPI ###
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

### find GTest ###
find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})

### find Boost ###
find_package(Boost REQUIRED COMPONENTS program_options filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

### include and link Lama and RBC ###
include_directories(${SCAI_INCLUDE})
include_directories(${RBC_DIR})

link_directories(${SCAI_LIB})
link_directories(${RBC_DIR}/build/lib)

### get git commit ###
execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --openmp --std=c++11 -O3 -fPIC -Wall -Wno-sign-compare -Wno-unused-variable -DBUILD_COMMIT=${GIT_COMMIT}" )

### define install directories ###
set(BIN_DEST "bin")
set(INCLUDE_DEST "include/geographer")
set(LIB_DEST "lib/geographer")
set(CMAKE_DEST "share/cmake/geographer")

### set runtime path ###
set ( CMAKE_SKIP_BUILD_RPATH FALSE )
set ( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set ( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

### define targets ###
add_library(geographer SHARED ${FILES_COMMON})
target_link_libraries(geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(Geographer main.cpp)
target_link_libraries(Geographer geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(analyze FileIO.cpp Settings.cpp GraphUtils.cpp CommTree.cpp Metrics.cpp ../tools/analyzePartition.cpp)
target_link_libraries(analyze ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(heatMap FileIO.cpp Settings.cpp GraphUtils.cpp CommTree.cpp ../tools/graphToHeatmap.cpp)
target_link_libraries(heatMap ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

### add networkit library if found ###
if (USE_NETWORKIT)
  target_link_libraries(Geographer networkit)
endif (USE_NETWORKIT)

### define unit test target if GTest was found ###
if (GTEST_FOUND)
	add_executable(GeographerTest ${FILES_TEST})
	target_link_libraries(GeographerTest geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES})
endif (GTEST_FOUND)

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

install(TARGETS Geographer DESTINATION "${BIN_DEST}") # executable
install(TARGETS geographer EXPORT geographer DESTINATION "${LIB_DEST}") # library
install(FILES ${FILES_HEADER} DESTINATION "${INCLUDE_DEST}")
install(EXPORT geographer DESTINATION "${CMAKE_DEST}")

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
