cmake_minimum_required (VERSION 3.0.2)

get_directory_property(hasParent PARENT_DIRECTORY)
if(NOT hasParent)
  message(FATAL_ERROR "Do not call this CMakeLists.txt directly. Instead, call the one in the parent directory.")
endif(NOT hasParent)

### set files ###
set(FILES_HEADER ParcoRepart.h MultiLevel.h LocalRefinement.h HilbertCurve.h MeshGenerator.h FileIO.h SpectralPartition.h Diffusion.h GraphUtils.h MultiSection.h KMeans.h CommTree.h AuxiliaryFunctions.h HaloPlanFns.h Metrics.h Mapping.h Settings.h)
set(FILES_COMMON ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp SpectralPartition.cpp Diffusion.cpp GraphUtils.cpp MultiSection.cpp KMeans.cpp CommTree.cpp AuxiliaryFunctions.cpp HaloPlanFns.cpp Metrics.cpp Mapping.cpp Settings.cpp)
set(FILES_TEST test_main.cpp auxTest.cpp DiffusionTest.cpp FileIOTest.cpp GraphUtilsTest.cpp HilbertCurveTest.cpp KMeansTest.cpp LocalRefinementTest.cpp MeshGeneratorTest.cpp MultiLevelTest.cpp MultiSectionTest.cpp ParcoRepartTest.cpp quadtree/test/QuadTreeTest.cpp SpectralPartitionTest.cpp)

### define targets ###
add_library(geographer SHARED ${FILES_COMMON})
target_link_libraries(geographer ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})
if (USE_BOOST)
	target_include_directories(geographer ${Boost_INCLUDE_DIRS})
	target_link_libraries(geographer ${Boost_LIBRARIES})
endif(USE_BOOST)

if (Boost_FOUND)
	add_executable(Geographer main.cpp parseArgs.cpp)
	target_include_directories(Geographer PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(Geographer geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
else (Boost_FOUND)
	message("Boost not found, cannot build standalone executable.")
endif (Boost_FOUND)

### add networkit library if found ###
if (USE_NETWORKIT)
  target_link_libraries(Geographer networkit)
endif (USE_NETWORKIT)

### define unit test target if GTest was found ###
if (GTEST_FOUND AND Boost_FOUND)
	add_executable(GeographerTest ${FILES_TEST})
	target_include_directories(Geographer PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(GeographerTest geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES})
endif (GTEST_FOUND AND Boost_FOUND)

install(TARGETS geographer EXPORT geographer DESTINATION "${LIB_DEST}") # library
install(FILES ${FILES_HEADER} DESTINATION "${HEADER_DEST}")
install(EXPORT geographer DESTINATION "${CMAKE_DEST}")

if (Boost_FOUND)
	install(TARGETS Geographer DESTINATION "${BIN_DEST}") # executable
endif (Boost_FOUND)