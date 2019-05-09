add_definitions(--openmp -pthread)
link_libraries(--openmp)

set(FILES_CORE ../src/FileIO.cpp ../src/Settings.cpp ../src/GraphUtils.cpp ../src/CommTree.cpp ../src/Metrics.cpp)

add_executable(analyze ${FILES_CORE} analyzePartition.cpp)
target_link_libraries(analyze ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(heatMap ${FILES_CORE} graphToHeatmap.cpp)
target_link_libraries(heatMap ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})