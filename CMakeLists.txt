cmake_minimum_required (VERSION 2.6)
project (Geographer)

set(SCAI_DIR "/home/moritz/lama3/lama/install")
set(SCAI_INCLUDE "${SCAI_DIR}/include/") 
set(SCAI_LIB "${SCAI_DIR}/lib/")

### set files ###
set(FILES_COMMON ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp PrioQueueForInts.cpp SpectralPartition.cpp Diffusion.cpp GraphUtils.cpp MultiSection.cpp KMeans.cpp)
set(FILES_TEST test_main.cpp auxTest.cpp DiffusionTest.cpp FileIOTest.cpp GraphUtilsTest.cpp HilbertCurveTest.cpp KMeansTest.cpp LocalRefinementTest.cpp MeshGeneratorTest.cpp MultiLevelTest.cpp MultiSectionTest.cpp ParcoRepartTest.cpp quadtree/test/QuadTreeTest.cpp SpectralPartitionTest.cpp)

set(RBC_DIR "${PROJECT_SOURCE_DIR}/RBC")

set(SCAI_LIBLIST scai_lama scai_dmemo scai_common scai_hmemo scai_utilskernel scai_tracing scai_sparsekernel scai_solver scai_logging)

### find MPI ###
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

### find GTest ###
find_package(GTest)#TODO: make build of unit tests conditional on whether GTest was found
include_directories(${GTEST_INCLUDE_DIRS})

### include and link Lama and RBC ###
include_directories(${SCAI_INCLUDE})
include_directories(${RBC_DIR})

link_directories(${SCAI_LIB})
link_directories(${RBC_DIR}/build/lib)

### get git commit ###
execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --openmp --std=c++11 -DBUILD_COMMIT=${GIT_COMMIT}" )

### define targets ###
add_executable(Geographer ${FILES_COMMON} main.cpp)
target_link_libraries(Geographer ${SCAI_LIBLIST} boost_program_options boost_filesystem boost_system _rbc ${MPI_CXX_LIBRARIES})

add_executable(GeographerTest ${FILES_COMMON} ${FILES_TEST})
target_link_libraries(GeographerTest ${SCAI_LIBLIST} boost_program_options boost_filesystem boost_system _rbc ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES})