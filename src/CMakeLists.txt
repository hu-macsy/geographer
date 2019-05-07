### set files ###
set(FILES_HEADER ParcoRepart.h MultiLevel.h LocalRefinement.h HilbertCurve.h MeshGenerator.h FileIO.h SpectralPartition.h Diffusion.h GraphUtils.h MultiSection.h KMeans.h CommTree.h AuxiliaryFunctions.h HaloPlanFns.h Metrics.h Mapping.h Settings.h)
set(FILES_COMMON ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp SpectralPartition.cpp Diffusion.cpp GraphUtils.cpp MultiSection.cpp KMeans.cpp CommTree.cpp AuxiliaryFunctions.cpp HaloPlanFns.cpp Metrics.cpp Mapping.cpp Settings.cpp)
set(FILES_TEST test_main.cpp auxTest.cpp DiffusionTest.cpp FileIOTest.cpp GraphUtilsTest.cpp HilbertCurveTest.cpp KMeansTest.cpp LocalRefinementTest.cpp MeshGeneratorTest.cpp MultiLevelTest.cpp MultiSectionTest.cpp ParcoRepartTest.cpp quadtree/test/QuadTreeTest.cpp SpectralPartitionTest.cpp)

### add networkit include and link directory ###
# link_directories has to happen before target definitions
if (USE_NETWORKIT)
  include_directories(${NETWORKIT_INCLUDE})
  link_directories(${NETWORKIT_LIB})
endif (USE_NETWORKIT)

### define targets ###
add_library(geographer SHARED ${FILES_COMMON})
target_link_libraries(geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(Geographer main.cpp)
target_link_libraries(Geographer geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})

### add networkit library if found ###
if (USE_NETWORKIT)
  target_link_libraries(Geographer networkit)
endif (USE_NETWORKIT)

### define unit test target if GTest was found ###
if (GTEST_FOUND)
	add_executable(GeographerTest ${FILES_TEST})
	target_link_libraries(GeographerTest geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES})
endif (GTEST_FOUND)

install(TARGETS Geographer DESTINATION "${BIN_DEST}") # executable
install(TARGETS geographer EXPORT geographer DESTINATION "${LIB_DEST}") # library
install(FILES ${FILES_HEADER} DESTINATION "${INCLUDE_DEST}")
#install(EXPORT geographer DESTINATION "${CMAKE_DEST}")