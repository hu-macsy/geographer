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

message("${METIS_BASE}/lib")

find_library( METIS_LIB metis "${METIS_BASE}/lib"  NO_DEFAULT_PATH)
find_library( PARMETIS_LIB libparmetis.so "${PARMETIS_BASE}/lib"  NO_DEFAULT_PATH)


###
### define libraries and executables
###


### for more info see the Zoltan2 documentation and https://trilinos.org/oldsite/Finding_Trilinos.txt
### see also in Trilinos_install_path/install/include/Makefile.export.Zoltan2

if( Zoltan2_FOUND AND METIS_LIB AND PARMETIS_LIB )
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
	target_include_directories( allCompetitors PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} )
	# include parmetis and metis 
	target_include_directories( allCompetitors PUBLIC "${METIS_BASE}/include" "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
	target_link_libraries( allCompetitors geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${Zoltan2_TPL_LIBRARIES} ${METIS_LIB} ${PARMETIS_LIB} ${MPI_CXX_LIBRARIES} )# metis parmetis)


    add_executable( testMetisRefine testMetisRefinement.cpp parseArgs.cpp)
    target_include_directories( testMetisRefine PUBLIC ${Zoltan2_INCLUDE_DIRS}  ${Zoltan2_TPL_INCLUDE_DIRS} )
    # include parmetis and metis 
    target_include_directories( testMetisRefine PUBLIC "${METIS_BASE}/include" "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
    target_link_libraries( testMetisRefine geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${Zoltan2_TPL_LIBRARIES} ${METIS_LIB} ${PARMETIS_LIB} ${MPI_CXX_LIBRARIES} )# metis parmetis)

	
	install(TARGETS wrappers EXPORT wrappers DESTINATION "${LIB_DEST}") 
	install(TARGETS allCompetitors DESTINATION "${BIN_DEST}") # executable
    install(TARGETS testMetisRefine DESTINATION "${BIN_DEST}") # executable
	install(FILES Wrappers.h DESTINATION "${HEADER_DEST}")
elseif( METIS_LIB AND PARMETIS_LIB )
    message( "\tFound library metis in " ${METIS_LIB} )
    message( "\tFound library parmetis in " ${PARMETIS_LIB} )
    message( "\n" )

    #TODO: link_directories is probably not needed; there is a better way to do this; at least, this works for now
    link_directories( ${METIS_BASE}/lib ${PARMETIS_BASE}/lib )
    
    add_library( wrappers SHARED  Wrappers.cpp)
    target_include_directories( wrappers PUBLIC "${METIS_BASE}/include" "${PARMETIS_BASE}/include" )
    target_link_libraries( wrappers ${RBC_LIBRARY} ${METIS_LIB} ${PARMETIS_LIB} ) # libzoltan2.so metis parmetis
    
    set( EXTRA_LIBRARIES_FOUND "true" )
    set( EXTRA_LIBRARIES ${METIS_LIB} ${PARMETIS_LIB} PARENT_SCOPE )

    
    add_executable( allCompetitors allCompetitorsPart.cpp parseArgs.cpp)
    # include parmetis and metis 
    target_include_directories( allCompetitors PUBLIC "${METIS_BASE}/include" "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
    target_link_libraries( allCompetitors geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES} ) # ${METIS_LIB} ${PARMETIS_LIB}  metis parmetis)


    add_executable( testMetisRefine testMetisRefinement.cpp parseArgs.cpp)
    # include parmetis and metis 
    target_include_directories( testMetisRefine PUBLIC "${METIS_BASE}/include" "${PARMETIS_BASE}/include" ${CXXOPTS_DIR})
    target_link_libraries( testMetisRefine geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES} ) # ${METIS_LIB} ${PARMETIS_LIB}  )# metis parmetis)

    
    install(TARGETS wrappers EXPORT wrappers DESTINATION "${LIB_DEST}") 
    install(TARGETS allCompetitors DESTINATION "${BIN_DEST}") # executable
    install(TARGETS testMetisRefine DESTINATION "${BIN_DEST}") # executable
    install(FILES Wrappers.h DESTINATION "${HEADER_DEST}")
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

### define unit test target if GTest was found ###
if (GTEST_FOUND )

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
endif (GTEST_FOUND)

### install library, header files and standalone executable ####
install(TARGETS geographer EXPORT geographer DESTINATION "${LIB_DEST}") # library
install(FILES ${FILES_HEADER} DESTINATION "${HEADER_DEST}")
install(TARGETS GeographerStandalone DESTINATION "${BIN_DEST}") # executable
