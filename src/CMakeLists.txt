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
set(FILES_COMMON ParcoRepart.cpp MultiLevel.cpp LocalRefinement.cpp HilbertCurve.cpp MeshGenerator.cpp FileIO.cpp Diffusion.cpp GraphUtils.cpp MultiSection_iter.cpp MultiSection.cpp KMeans.cpp CommTree.cpp AuxiliaryFunctions.cpp  HaloPlanFns.cpp Metrics.cpp Mapping.cpp Settings.cpp)
set(FILES_TEST test_main.cpp quadtree/test/QuadTreeTest.cpp    auxTest.cpp CommTreeTest.cpp DiffusionTest.cpp  FileIOTest.cpp GraphUtilsTest.cpp HilbertCurveTest.cpp KMeansTest.cpp LocalRefinementTest.cpp MappingTest.cpp MeshGeneratorTest.cpp MultiLevelTest.cpp MultiSectionTest.cpp ParcoRepartTest.cpp )

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


find_library( PARMETIS_LIB libparmetis.so "${PARMETIS_BASE}/lib"  NO_DEFAULT_PATH)

if( PARMETIS_LIB AND NOT METIS_INCLUDE)
    message( FATAL_ERROR "PARMETIS_BASE was specified but not METIS_INCLUDE. Use -DMETIS_INCLUDE=<path_to_directory_that_contains_metis.h>. Probably, a candidate directory is parmetis/metis/include")
endif()


###
### define libraries and executables
###


### for more info see the Zoltan2 documentation and https://trilinos.org/oldsite/Finding_Trilinos.txt
### see also in Trilinos_install_path/install/include/Makefile.export.Zoltan2

if(Zoltan2_FOUND AND PARMETIS_LIB )
	message( "\n\tFound package Zoltan2" )
	message( "\tFound metis inlude in " ${METIS_INCLUDE} )
	message( "\tFound parmetis library in " ${PARMETIS_LIB} )
	message( "\n" )
	
    add_definitions(-DZOLTAN_FOUND -DPARMETIS_FOUND)

	#WARNING: if we use ${Zoltan2_LIBRARIES}, it tries to link to the .a library and this creates an error. We need the .so library
	find_library( ZOLTAN2_LIB libzoltan2.so ${Zoltan2_LIBRARY_DIRS} )
	find_library( ZOLTAN_LIB libzoltan.so ${Zoltan_LIBRARY_DIRS} )

	#TODO: link_directories is probably not needed; there is a better way to do this; at least, this works for now
	link_directories( ${Zoltan2_LIBRARY_DIRS} ${PARMETIS_BASE}/lib )
	
	add_library( wrappers SHARED  Wrappers.cpp parmetisWrapper.cpp zoltanWrapper.cpp )
	target_include_directories( wrappers PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} ${METIS_INCLUDE} "${PARMETIS_BASE}/include" )
	target_link_libraries( wrappers ${RBC_LIBRARY} ${Zoltan2_TPL_LIBRARIES} ${PARMETIS_LIB} ${ZOLTAN2_LIB}  ${ZOLTAN_LIB} ${Zoltan2_LIBRARIES} ) # libzoltan2.so metis parmetis
	
	set( EXTRA_LIBRARIES_FOUND "true" )
	set( EXTRA_LIBRARIES ${PARMETIS_LIB} ${ZOLTAN2_LIB} ${Zoltan2_LIBRARIES}  PARENT_SCOPE )

	
    add_executable( allCompetitors allCompetitorsPart.cpp parseArgs.cpp)
	target_include_directories( allCompetitors PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} )
	# include parmetis and metis 
	target_include_directories( allCompetitors PUBLIC ${METIS_INCLUDE} "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
	target_link_libraries( allCompetitors geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${Zoltan2_TPL_LIBRARIES} ${PARMETIS_LIB} ${MPI_CXX_LIBRARIES} )# metis parmetis)


    add_executable( testMetisRefine testMetisRefinement.cpp parseArgs.cpp)
    target_include_directories( testMetisRefine PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} )
    # include parmetis and metis 
    target_include_directories( testMetisRefine PUBLIC ${METIS_INCLUDE} "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
    target_link_libraries( testMetisRefine geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${Zoltan2_TPL_LIBRARIES} ${PARMETIS_LIB} ${MPI_CXX_LIBRARIES} )# metis parmetis)

	
	install(TARGETS wrappers EXPORT wrappers DESTINATION "${LIB_DEST}") 
	install(TARGETS allCompetitors DESTINATION "${BIN_DEST}") # executable
    install(TARGETS testMetisRefine DESTINATION "${BIN_DEST}") # executable
	install(FILES Wrappers.h parmetisWrapper.h zoltanWrapper.h DESTINATION "${HEADER_DEST}")
elseif( PARMETIS_LIB )
    message( "\tFound metis inlude in " ${METIS_INCLUDE} )
    message( "\tFound parmetis library in " ${PARMETIS_LIB} )
    message( "\n" )

    add_definitions(-DPARMETIS_FOUND)

    #TODO: link_directories is probably not needed; there is a better way to do this; at least, this works for now
    link_directories( ${METIS_BASE}/lib ${PARMETIS_BASE}/lib )
    
    add_library( wrappers SHARED  Wrappers.cpp parmetisWrapper.cpp)
    target_include_directories( wrappers PUBLIC ${METIS_INCLUDE} "${PARMETIS_BASE}/include" )
    target_link_libraries( wrappers ${RBC_LIBRARY} ${PARMETIS_LIB} ${METIS_LIB}  ) 
    
    set( EXTRA_LIBRARIES_FOUND "true" )
    set( EXTRA_LIBRARIES ${PARMETIS_LIB} PARENT_SCOPE )

    
    add_executable( allCompetitors allCompetitorsPart.cpp parseArgs.cpp)
    # include parmetis and metis 
    target_include_directories( allCompetitors PUBLIC "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
    target_link_libraries( allCompetitors geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES} ) 


    add_executable( testMetisRefine testMetisRefinement.cpp parseArgs.cpp)
    # include parmetis and metis 
    target_include_directories( testMetisRefine PUBLIC "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
    target_link_libraries( testMetisRefine geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES} ) 

    
    install(TARGETS wrappers EXPORT wrappers DESTINATION "${LIB_DEST}") 
    install(TARGETS allCompetitors DESTINATION "${BIN_DEST}") # executable
    install(TARGETS testMetisRefine DESTINATION "${BIN_DEST}") # executable
    install(FILES Wrappers.h parmetisWrapper.h DESTINATION "${HEADER_DEST}")
else()
	message( "Some of the packages Zoltan2, metis or parmetis were not found.\nSet it in the CMakeLists.txt file by editing the line 'FIND_PACKAGE(Zoltan2 ...)'")
endif()


###
### geographer versions
###

add_library(geographer SHARED ${FILES_COMMON})
target_compile_options(geographer PRIVATE -fPIC)
target_link_libraries(geographer ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})

add_executable(GeographerStandalone main.cpp parseArgs.cpp)
target_include_directories(GeographerStandalone PUBLIC ${CXXOPTS_DIR})
target_link_libraries(GeographerStandalone geographer ${SCAI_LIBRARIES} ${MPI_CXX_LIBRARIES})

### add networkit library if found ###
if (USE_NETWORKIT)
  target_link_libraries(GeographerStandalone networkit)
endif (USE_NETWORKIT)


option(COMPILE_TESTS "Compile the tests using googletest." ON)

### define unit test target if GTest was found ###
if (GTEST_FOUND AND COMPILE_TESTS)

    # add the wrappers test if zoltan and parmetis is found
    if( EXTRA_LIBRARIES_FOUND )
        set(FILES_TEST ${FILES_TEST} WrappersTest.cpp )
    endif( EXTRA_LIBRARIES_FOUND )

	add_executable(GeographerTest ${FILES_TEST})
	
	target_compile_options(GeographerTest PRIVATE -fprofile-arcs -ftest-coverage)
	target_link_libraries(GeographerTest -fprofile-arcs -ftest-coverage)
	target_link_libraries(GeographerTest geographer ${SCAI_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES})
	target_link_libraries(GeographerTest -pthread)
	if( EXTRA_LIBRARIES_FOUND )
        target_link_libraries(GeographerTest wrappers)
    endif( EXTRA_LIBRARIES_FOUND )
    
	add_test(NAME GeographerTest COMMAND GeographerTest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	install(TARGETS GeographerTest DESTINATION "${BIN_DEST}" OPTIONAL) # test executable
endif (GTEST_FOUND AND COMPILE_TESTS)

### install library, header files and standalone executable ####
install(TARGETS geographer EXPORT geographer DESTINATION "${LIB_DEST}") # library
install(FILES ${FILES_HEADER} DESTINATION "${HEADER_DEST}")
install(TARGETS GeographerStandalone DESTINATION "${BIN_DEST}") # executable
