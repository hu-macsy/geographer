cmake_minimum_required (VERSION 3.0.2)

get_directory_property(hasParent PARENT_DIRECTORY)
if(NOT hasParent)
  message(FATAL_ERROR "Do not call this CMakeLists.txt directly. Instead, call the one in the parent directory.")
endif(NOT hasParent)

add_definitions(-pthread)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	add_definitions(-qopenmp)
	link_libraries(-qopenmp)
else ()
	add_definitions(--openmp)
	link_libraries(--openmp)
endif()

### set files ###
set(FILES_HEADER ParcoRepart.h MultiLevel.h LocalRefinement.h HilbertCurve.h MeshGenerator.h FileIO.h Diffusion.h GraphUtils.h MultiSection.h KMeans.h CommTree.h AuxiliaryFunctions.h HaloPlanFns.h Metrics.h Mapping.h Settings.h)
set(FILES_COMMON ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp Diffusion.cpp GraphUtils.cpp MultiSection.cpp KMeans.cpp CommTree.cpp AuxiliaryFunctions.cpp HaloPlanFns.cpp Metrics.cpp Mapping.cpp Settings.cpp)
set(FILES_TEST test_main.cpp auxTest.cpp CommTreeTest.cpp DiffusionTest.cpp FileIOTest.cpp GraphUtilsTest.cpp HilbertCurveTest.cpp KMeansTest.cpp LocalRefinementTest.cpp MeshGeneratorTest.cpp MultiLevelTest.cpp MultiSectionTest.cpp ParcoRepartTest.cpp quadtree/test/QuadTreeTest.cpp )


###
### Check if external libraries metis, parmetis and zoltan2 are found. If they are found,
### create the allCompetitors executable and also, later, link them with libgeographer
###
     

### find zoltan ###
FIND_PACKAGE( Zoltan2 PATHS "/home/hpc/pr87si/di36qin/Code/Trilinos/build/packages/zoltan2/")

### find metis and parmetis

#this are the paths in supermuc; provide them using -DMETIS_BASE in the cmake call
#set(METIS_BASE "/lrz/sys/libraries/metis/5.1.0/i8r8")
#set(PARMETIS_BASE  "/lrz/sys/libraries/parmetis/4.0.3/ibmmpi/i8r8")

find_library( METIS_LIB metis "${METIS_BASE}/lib" )
find_library( PARMETIS_LIB parmetis "${PARMETIS_BASE}/lib" )

###
### define libraries and executables
###


### for more info see the Zoltan2 documentation and https://trilinos.org/oldsite/Finding_Trilinos.txt
### see also in Trilinos_install_path/install/include/Makefile.export.Zoltan2

if( Zoltan2_FOUND AND METIS_BASE AND PARMETIS_BASE )
	message( "\n\tFound package Zoltan2" )
	message( "\tFound library metis in " ${METIS_LIB} )
	message( "\tFound library parmetis in " ${PARMETIS_LIB} )
	message( "\n" )
	
	#WARNING: if we use ${Zoltan2_LIBRARIES}, it tries to link to the .a library and this creates an error. We need the .so library
	find_library( ZOLTAN2_LIB libzoltan2.so ${Zoltan2_LIBRARY_DIRS} )
	find_library( ZOLTAN_LIB libzoltan.so ${Zoltan_LIBRARY_DIRS} )

	#TODO: link_directories is probably not needed; there is a better way to do this; at least, this works for now
	link_directories( ${Zoltan2_LIBRARY_DIRS} ${METIS_BASE}/lib ${PARMETIS_BASE}/lib )
	
	add_library( wrappers SHARED  Wrappers.cpp)
	target_include_directories( wrappers PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} "${METIS_BASE}/include" "${PARMETIS_BASE}/include" )
	target_link_libraries( wrappers ${RBC_LIBRARY} ${Zoltan2_TPL_LIBRARIES} ${METIS_LIB} ${PARMETIS_LIB} ${ZOLTAN2_LIB}  ${ZOLTAN_LIB} ${Zoltan2_LIBRARIES} ) # libzoltan2.so metis parmetis
	
	set( EXTRA_LIBRARIES_FOUND "true" )
	set( EXTRA_LIBRARIES ${METIS_LIB} ${PARMETIS_LIB} ${ZOLTAN2_LIB} ${Zoltan2_LIBRARIES}  PARENT_SCOPE )

	add_executable( allCompetitors allCompetitorsPart.cpp parseArgs.cpp)
	target_include_directories( allCompetitors PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} )
	# include parmetis and metis ###
	target_include_directories( allCompetitors PUBLIC "${METIS_BASE}/include" "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
	target_link_libraries( allCompetitors geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${Zoltan2_TPL_LIBRARIES} ${METIS_LIB} ${PARMETIS_LIB} ${MPI_CXX_LIBRARIES} )# metis parmetis)
	
	install(TARGETS wrappers EXPORT wrappers DESTINATION "${LIB_DEST}") 
	install(TARGETS allCompetitors DESTINATION "${BIN_DEST}") # executable
	install(FILES Wrappers.h DESTINATION "${HEADER_DEST}")
else()
	#message( "Package Zoltan2 was not found.\nSet it in the CMakeLists.txt file by editing the line 'FIND_PACKAGE(Zoltan2 ...)'")
endif()


###
### geographer versions
###

add_library(geographer SHARED ${FILES_COMMON})
target_compile_options(geographer PRIVATE -fPIC)
target_link_libraries(geographer ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})
if (USE_BOOST)
	target_include_directories(geographer PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(geographer ${Boost_LIBRARIES})
endif(USE_BOOST)

add_executable(Geographer main.cpp parseArgs.cpp)
target_include_directories(Geographer PUBLIC ${CXXOPTS_DIR})
target_link_libraries(Geographer geographer ${SCAI_LIBRARIES} ${MPI_CXX_LIBRARIES})


### add networkit library if found ###
if (USE_NETWORKIT)
  target_link_libraries(Geographer networkit)
endif (USE_NETWORKIT)

### define unit test target if GTest was found ###
if (GTEST_FOUND AND Boost_FOUND)
	add_executable(GeographerTest ${FILES_TEST})
	target_include_directories(GeographerTest PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(GeographerTest geographer ${SCAI_LIBRARIES} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES})
	target_link_libraries(GeographerTest -pthread)
	add_test(NAME GeographerTest COMMAND GeographerTest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	install(TARGETS GeographerTest DESTINATION "${BIN_DEST}") # test executable
endif (GTEST_FOUND AND Boost_FOUND)

install(TARGETS geographer EXPORT geographer DESTINATION "${LIB_DEST}") # library
install(FILES ${FILES_HEADER} DESTINATION "${HEADER_DEST}")
install(TARGETS Geographer DESTINATION "${BIN_DEST}") # executable
