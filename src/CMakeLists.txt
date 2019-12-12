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
find_library( PARMETIS_LIB NAMES libparmetis.so libparmetis.a HINTS "${PARMETIS_BASE}/lib" )

if(PARMETIS_LIB AND NOT DEFINED METIS_INCLUDE)
    set( METIS_INCLUDE "${PARMETIS_BASE}/../../../metis/include")
    message(WARNING "PARMETIS_BASE was specified but not METIS_INCLUDE. Use -DMETIS_INCLUDE=<path_to_directory_that_contains_metis.h>. Setting path to ${METIS_INCLUDE}")   
endif()



### find parhip ###
find_library( PARHIP_LIB libparhip.a "${PARHIP_BASE}" NO_DEFAULT_PATH)


###
### define libraries and executables
###


### for more info see the Zoltan2 documentation and https://trilinos.org/oldsite/Finding_Trilinos.txt
### see also in Trilinos_install_path/install/include/Makefile.export.Zoltan2

if(Zoltan2_FOUND OR PARMETIS_LIB OR PARHIP_LIB )
    if(Zoltan2_FOUND)
        message( "\n\tFound package Zoltan2" )
        add_definitions(-DZOLTAN_FOUND)

        #WARNING: if we use ${Zoltan2_LIBRARIES}, it tries to link to the .a library and this creates an error. We need the .so library
        find_library( ZOLTAN2_LIB libzoltan2.so ${Zoltan2_LIBRARY_DIRS} )
        find_library( ZOLTAN_LIB libzoltan.so ${Zoltan_LIBRARY_DIRS} )

        #TODO: link_directories is probably not needed; there is a better way to do this; at least, this works for now
        link_directories( ${Zoltan2_LIBRARY_DIRS} ${PARMETIS_BASE}/lib ) 
        
        set(ZOLTAN_WRAP_CPP zoltanWrapper.cpp)
        set(ZOLTAN_WRAP_H zoltanWrapper.h)
        set(ZOLTAN_INCLUDES ${Zoltan2_INCLUDE_DIRS} ${Zoltan2_TPL_INCLUDE_DIRS})
        set(ZOLTAN_LIBS  ${Zoltan2_TPL_LIBRARIES} ${ZOLTAN2_LIB}  ${ZOLTAN_LIB} ${Zoltan2_LIBRARIES})
    else()
        message(WARNING "zoltan was not found")
    endif()
    if(PARMETIS_LIB)
        message( "\tFound metis inlude in " ${METIS_INCLUDE} )
        message( "\tFound parmetis library in " ${PARMETIS_LIB} )
        set(CMAKE_CXX_STANDARD 17)

        add_definitions(-DPARMETIS_FOUND)
        set(PARMETIS_WRAP_CPP parmetisWrapper.cpp)
        set(PARMETIS_WRAP_H parmetisWrapper.h)
        set(PARMETIS_INCLUDES ${METIS_INCLUDE} "${PARMETIS_BASE}/include")

        #find metis library
        if(NOT DEFINED METIS_LIB_DIR)
            message( WARNING "variable METIS_LIB_DIR was not set. Set this variable to point to the directory that libmetis exists. Will attempt to find it...")
        endif()
    
        find_library(METIS_LIB NAMES libmetis.so libmetis.a HINTS "/usr/local/lib" "/usr/lib" "${METIS_LIB_DIR}" "${METIS_LIB_DIR}/lib" "${METIS_LIB_DIR}/libmetis" "${METIS_INCLUDE}/../lib" "${METIS_INCLUDE}/../libmetis" "${PARMETIS_BASE}/../libmetis")
        message("\tFound metis lib in " ${METIS_LIB})
        
        set(PARMETIS_LIBS ${PARMETIS_LIB} ${METIS_LIB} )
    else()
        message(WARNING "parmetis was not found")        
    endif()
    if(PARHIP_LIB)
        message("\tFound ParHip base in " ${PARHIP_BASE})
        message("\t\tFound parhip lib in " ${PARHIP_LIB} )
        add_definitions(-DPARHIP_FOUND)

        set(PARHIP_WRAP_CPP parhipWrapper.cpp)
        set(PARHIP_WRAP_H parhipWrapper.h)
        set(PARHIP_INCLUDES ${PARHIP_BASE} )
        
        find_library( KAHIP_LIB libkahip.a "${PARHIP_BASE}/parallel" NO_DEFAULT_PATH)
        if(KAHIP_LIB)
            message("\t\tFound kahip lib in " ${KAHIP_LIB} )
        else()
            message(FATAL_ERROR "kahip library not found")
        endif()

        set(PARHIP_LIBS ${PARHIP_LIB} ${KAHIP_LIB})
    else()
        message(WARNING "parhip was not found")
    endif()
    message( "\n" )
    
    add_library( wrappers SHARED ${PARMETIS_WRAP_CPP} ${ZOLTAN_WRAP_CPP} ${PARHIP_WRAP_CPP} )
    target_include_directories( wrappers PUBLIC ${ZOLTAN_INCLUDES} ${PARMETIS_INCLUDES} ${PARHIP_INCLUDES} )
    target_link_libraries( wrappers ${RBC_LIBRARY} ${ZOLTAN_LIBS} ${PARMETIS_LIBS} ${PARHIP_LIBS} ) # libzoltan2.so metis parmetis
    
    
    add_executable( allCompetitors allCompetitorsPart.cpp parseArgs.cpp)
    target_include_directories( allCompetitors PUBLIC ${ZOLTAN_INCLUDES} ${PARMETIS_INCLUDES} ${CXXOPTS_DIR} )
    target_link_libraries( allCompetitors geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES} )

    if( PARMETIS_FOUND )
        add_executable( testMetisRefine testMetisRefinement.cpp parseArgs.cpp)
        target_include_directories( testMetisRefine PUBLIC ${ZOLTAN_INCLUDES} ${PARMETIS_INCLUDES} ${CXXOPTS_DIR})
        target_link_libraries( testMetisRefine geographer wrappers ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES} )
        install(TARGETS testMetisRefine DESTINATION "${BIN_DEST}") # executable
    endif()

    set( EXTRA_LIBRARIES_FOUND "true" )
    set( EXTRA_LIBRARIES ${PARMETIS_LIBS} ${ZOLTAN2_LIB} ${ZOLTAN_LIBS}  PARENT_SCOPE )
    
    install(TARGETS wrappers EXPORT wrappers DESTINATION "${LIB_DEST}") 
    install(TARGETS allCompetitors DESTINATION "${BIN_DEST}") # executable
    install(FILES Wrappers.h ${PARMETIS_WRAP_H} ${ZOLTAN_WRAP_H} ${PARHIP_H} DESTINATION "${HEADER_DEST}")   
else()
    #message( "Some of the packages Zoltan2, metis or parmetis were not found.\nSet it in the CMakeLists.txt file by editing the line 'FIND_PACKAGE(Zoltan2 ...)'")
endif()


###
### geographer versions
###

add_library(geographer SHARED ${FILES_COMMON})
target_compile_options(geographer PRIVATE -fPIC)
target_link_libraries(geographer ${SCAI_LIBRARIES} ${RBC_LIBRARY} ${MPI_CXX_LIBRARIES})
if(PARMETIS_LIB)
    target_include_directories( geographer PUBLIC ${PARMETIS_INCLUDES} )
    target_link_libraries(geographer wrappers)
endif()

add_executable(GeographerStandalone main.cpp parseArgs.cpp)
target_include_directories(GeographerStandalone PUBLIC ${CXXOPTS_DIR})
target_link_libraries(GeographerStandalone geographer ${SCAI_LIBRARIES} ${MPI_CXX_LIBRARIES})

### add networkit library if found ###
if (USE_NETWORKIT)
  target_link_libraries(GeographerStandalone networkit)
endif (USE_NETWORKIT)

###
### define unit test target if GTest was found ###
###

option(COMPILE_TESTS "Compile the tests using googletest." ON)

if (GTEST_FOUND AND COMPILE_TESTS)

    # add the wrappers test if zoltan and parmetis is found
    if( PARMETIS_FOUND )
        set(FILES_TEST ${FILES_TEST} WrappersTest.cpp )
    endif( PARMETIS_FOUND )

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
