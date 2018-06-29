cmake_minimum_required (VERSION 2.6)
project (Geographer)

if (NOT DEFINED SCAI_DIR)
	message( FATAL_ERROR "Install Directory of LAMA not defined, specify with -DSCAI_DIR" )
endif (NOT DEFINED SCAI_DIR)

set(SCAI_INCLUDE "${SCAI_DIR}/include/") 
set(SCAI_LIB "${SCAI_DIR}/lib/")

### set files ###
set(FILES_COMMON ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp SpectralPartition.cpp Diffusion.cpp GraphUtils.cpp MultiSection.cpp KMeans.cpp /home/harry/courses/thesis/edge_coloring/Thesis_GraphColoring/Thesis_GraphColoring/ColoringAlgorithms.cpp /home/harry/courses/thesis/edge_coloring/Thesis_GraphColoring/Thesis_GraphColoring/HelpFunctions.cpp  /home/harry/courses/thesis/edge_coloring/Thesis_GraphColoring/Thesis_GraphColoring/Graph.cpp)

# for mec
#set(FILES_COMMON "${FILES_COMMON} /home/harry/courses/thesis/edge_coloring/Thesis_GraphColoring/Thesis_GraphColoring/ColoringAlgorithms.cpp /home/harry/courses/thesis/edge_coloring/Thesis_GraphColoring/Thesis_GraphColoring/HelpFunctions.cpp  /home/harry/courses/thesis/edge_coloring/Thesis_GraphColoring/Thesis_GraphColoring/Graph.cpp")

set(FILES_TEST test_main.cpp auxTest.cpp DiffusionTest.cpp FileIOTest.cpp GraphUtilsTest.cpp HilbertCurveTest.cpp KMeansTest.cpp LocalRefinementTest.cpp MeshGeneratorTest.cpp MultiLevelTest.cpp MultiSectionTest.cpp ParcoRepartTest.cpp quadtree/test/QuadTreeTest.cpp SpectralPartitionTest.cpp)

set(RBC_DIR "${PROJECT_SOURCE_DIR}/RBC")

set(SCAI_LIBLIST scai_lama scai_dmemo scai_common scai_hmemo scai_utilskernel scai_tracing scai_sparsekernel scai_solver scai_logging)

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

### link the mec code
link_directories(/home/harry/courses/thesis/edge_coloring/Thesis_GraphColoring/Thesis_GraphColoring)

### get git commit ###
execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --openmp --std=c++11 -DBUILD_COMMIT=${GIT_COMMIT}" )

### define targets ###
add_executable(Geographer ${FILES_COMMON} main.cpp)
target_link_libraries(Geographer ${SCAI_LIBLIST} ${Boost_LIBRARIES} _rbc ${MPI_CXX_LIBRARIES})

if (GTEST_FOUND)
	add_executable(GeographerTest ${FILES_COMMON} ${FILES_TEST})
	target_link_libraries(GeographerTest ${SCAI_LIBLIST} ${Boost_LIBRARIES} _rbc ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES})
endif (GTEST_FOUND)