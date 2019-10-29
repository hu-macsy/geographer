cmake_minimum_required (VERSION 3.0.2)
project (Geographer)
include(CMakePackageConfigHelpers)
include(CTest)

# define library version
set(Geographer_VERSION_MAJOR 0 STRING "major version")
set(Geographer_VERSION_MINOR 9 STRING "minor version")
set(Geographer_VERSION ${Geographer_VERSION_MAJOR}.${Geographer_VERSION_MINOR} STRING "version")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE) 
    #set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT CODE_COVERAGE)
    set(CODE_COVERAGE FALSE)
endif()

### add code coverage
if(CMAKE_COMPILER_IS_GNUCXX AND CODE_COVERAGE)
    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CmakeModules)
    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
    #setup_target_for_coverage(${PROJECT_NAME}_coverage ${PROJECT_TEST_NAME} coverage)
    setup_target_for_coverage_gcovr_html( 
        NAME ${PROJECT_NAME}_coverage 
        EXECUTABLE GeographerTest -j 4
        DEPANDENCIES GeographerTest)
    #setup_target_for_coverage(${PROJECT_NAME}_coverage GeographerTest coverage)
endif()

option(USE_NETWORKIT "Use NETWORKIT package." OFF)
if(USE_NETWORKIT)
  if (NOT DEFINED NETWORKIT_DIR)
     message( FATAL_ERROR "Install Directory of NETWORKIT not defined, specify with -DNETWORKIT_DIR" )
  endif (NOT DEFINED NETWORKIT_DIR)
  add_definitions(-DUSE_NETWORKIT)
endif(USE_NETWORKIT)

set(NETWORKIT_INCLUDE "${NETWORKIT_DIR}/include/") 
set(NETWORKIT_LIB "${NETWORKIT_DIR}/lib/")

set(CXXOPTS_DIR "${PROJECT_SOURCE_DIR}/external/cxxopts/include")

### find MPI ###
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
link_libraries(${MPI_CXX_LIBRARIES})

set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_REQUIRED_FLAGS ${MPI_CXX_LINK_FLAGS})

### include GTest ###
add_subdirectory("${PROJECT_SOURCE_DIR}/external/googletest" "googletest" EXCLUDE_FROM_ALL)
set(GTEST_LIBRARIES gtest)
enable_testing()
set(GTEST_FOUND TRUE)

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
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCAI_CXX_FLAGS}" )
link_directories(${SCAI_ROOT}/lib/)

### download KaDiS if not found. ###
find_path(KaDiS_CMAKE CMakeLists.txt external/KaDiS NO_DEFAULT_PATH)

if((NOT KaDiS_CMAKE) OR (NOT EXISTS "${KaDiS_CMAKE}/CMakeLists.txt"))
  message("KaDiS Library not found, cloning it now.")
  execute_process(COMMAND git submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
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

#add_definitions("-Wall -Wno-sign-compare -Wno-unused")
add_definitions("-Wall -Wno-sign-compare")

### define install directories ###
set(BIN_DEST "bin")
set(INCLUDE_DEST "include")
set(HEADER_DEST "${INCLUDE_DEST}/geographer")
set(LIB_DEST "lib/geographer")
set(CMAKE_DEST "share/cmake/geographer")

### set runtime path ###
cmake_policy(SET CMP0042 NEW) # using rpath also on Mac OS
set ( CMAKE_SKIP_BUILD_RPATH FALSE )
set ( CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${LIB_DEST} ${SCAI_ROOT}/lib/)
set ( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

add_subdirectory(external/KaDiS EXCLUDE_FROM_ALL)
set(RBC_LIBRARY rbcsorting)

### set value type ###
configure_file("src/config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h" DESTINATION "${HEADER_DEST}")

### include and define targets ###
add_subdirectory(src)

add_subdirectory(tools)


if( TARGET wrappers )
	set( WRAPPERS_FOUND TRUE)
	#set( EXTRA_LIBS "")
endif()

### support for FindPackage ###
configure_file(GeographerConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfig.cmake @ONLY)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfigVersion.cmake
  VERSION 0.9
  COMPATIBILITY ExactVersion )
  
  
### documentation using doxygen
option(BUILD_DOC "Build documentation, default=OFF" OFF)

# check if Doxygen is installed
find_package(Doxygen)

if (DOXYGEN_FOUND AND BUILD_DOC)
	# set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxConfigFile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile)
    
    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started, documentation to be written in " ${CMAKE_CURRENT_BINARY_DIR}/doc)
    message("Disable using BUILD_DOC=OFF")
    
	# note the option ALL which allows to build the docs together with the application
    add_custom_target( doxygendoc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND AND BUILD_DOC)
	if(NOT DOXYGEN_FOUND AND BUILD_DOC)
		message("Doxygen need to be installed to generate the doxygen documentation")
	endif()
	if(NOT BUILD_DOC)
		message("Building the documentation is deactivated, use BUILD_DOC=ON to activate it, or make doxygendoc to build only the documentation")
	endif()
endif (DOXYGEN_FOUND AND BUILD_DOC)


### install files ###
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfigVersion.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GeographerConfig.cmake
        DESTINATION "${CMAKE_DEST}" )
