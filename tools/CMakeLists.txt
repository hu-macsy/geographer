add_definitions(--openmp -pthread)
link_libraries(--openmp)

set(FILES_CORE ../src/FileIO.cpp ../src/Settings.cpp ../src/GraphUtils.cpp ../src/CommTree.cpp ../src/Metrics.cpp)

add_executable(analyze ${FILES_CORE} analyzePartition.cpp)
target_include_directories(analyze PUBLIC ${CXXOPTS_DIR})
target_link_libraries(analyze ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

if (Boost_FOUND)
	add_executable(heatMap ${FILES_CORE} graphToHeatmap.cpp)
	target_include_directories(heatMap PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(heatMap ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})
endif (Boost_FOUND)


target_include_directories(Geographer PUBLIC ${CXXOPTS_DIR})