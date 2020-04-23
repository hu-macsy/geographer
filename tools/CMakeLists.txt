get_directory_property(hasParent PARENT_DIRECTORY)
if(NOT hasParent)
  message(FATAL_ERROR "Do not call this CMakeLists.txt directly. Instead, call the one in the parent directory.")
endif(NOT hasParent)

add_definitions(--openmp -pthread)
link_libraries(--openmp)

set(FILES_CORE ../src/FileIO.cpp ../src/Settings.cpp ../src/GraphUtils.cpp ../src/CommTree.cpp)

add_executable(analyze ${FILES_CORE} ../src/parseArgs.cpp ../src/AuxiliaryFunctions.cpp  ../src/Metrics.cpp analyzePartition.cpp)
target_include_directories(analyze PUBLIC ${CXXOPTS_DIR})
target_link_libraries(analyze ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(heatMap ${FILES_CORE} graphToHeatmap.cpp)
target_include_directories(heatMap PUBLIC ${CXXOPTS_DIR})
target_link_libraries(heatMap ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(coords2binary ${FILES_CORE} coords2binary.cpp)
target_include_directories(coords2binary PUBLIC "../src/")
target_link_libraries(coords2binary ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(distEdgeList2CSR ${FILES_CORE} distEdgeList2CSR.cpp)
target_include_directories(distEdgeList2CSR PUBLIC "../src/")
target_link_libraries(distEdgeList2CSR ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(distEdgeList2nonDist ${FILES_CORE} distEdgeList2nonDist.cpp)
target_include_directories(distEdgeList2nonDist PUBLIC "../src/")
target_link_libraries(distEdgeList2nonDist ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

set(TOOLS_DIR "${BIN_DEST}/tools")
install(TARGETS analyze DESTINATION ${TOOLS_DIR})
install(TARGETS heatMap DESTINATION ${TOOLS_DIR})
install(TARGETS coords2binary DESTINATION ${TOOLS_DIR})
install(TARGETS distEdgeList2CSR DESTINATION ${TOOLS_DIR})
install(TARGETS distEdgeList2nonDist DESTINATION ${TOOLS_DIR})