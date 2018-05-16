cmake_minimum_required (VERSION 2.6)
project (Geographer)

set(SCAI_INCLUDE "/home/moritz/lama3/lama/install/include/") 
set(GTEST_INCLUDE "home/moritz/Includes/gtest-1.7.0/include")
set(RBC_DIR "${PROJECT_SOURCE_DIR}/RBC")
set(MPI_INCLUDE /usr/lib/openmpi/include/)

set(SCAI_LIB "/home/moritz/lama3/lama/install/lib/")
set(MPI_LIB "/usr/lib/openmpi/lib/")

include_directories(${SCAI_INCLUDE})
include_directories(${GTEST_INCLUDE})
include_directories(${RBC_DIR})
include_directories(${MPI_INCLUDE})

link_directories(${SCAI_LIB})
link_directories(${MPI_LIB})
link_directories(${RBC_DIR}/build/lib)

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --openmp --std=c++11" )

add_executable(Geographer ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp PrioQueueForInts.cpp 
SpectralPartition.cpp Diffusion.cpp GraphUtils.cpp MultiSection.cpp KMeans.cpp main.cpp)

target_link_libraries(Geographer scai_lama scai_dmemo scai_common scai_hmemo scai_utilskernel scai_tracing scai_sparsekernel scai_solver scai_logging boost_program_options boost_filesystem boost_system _rbc mpi_cxx mpi)