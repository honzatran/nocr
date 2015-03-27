cmake_minimum_required(VERSION 2.8)
project ( NOCR )
find_package( OpenCV REQUIRED )
# find_package( Boost REQUIRED )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${NOCR_SOURCE_DIR}/bin" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${NOCR_SOURCE_DIR}/lib" )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${NOCR_SOURCE_DIR}/lib" )

# set( VLROOT ${NOCR_SOURCE_DIR}/vlfeat-0.9.18/ )
# set( VLROOT_INCLUDE_DIR ${NOCR_SOURCE_DIR}/vlfeat-0.9.18/ )
# set( VLROOT_LIB ${NOCR_SOURCE_DIR}/vlfeat-0.9.18/bin/glnx86/libvl.so )
set( NOCR_EXTERNAL_LIB ${NOCR_SOURCE_DIR}/lib )

# include_directories( ${VLROOT_INCLUDE_DIR} )
include_directories( ${NOCR_SOURCE_DIR}/External-libraries/libsvm-3.18/ )
include_directories( ${NOCR_SOURCE_DIR}/External-libraries/liblinear-1.94/ )
include_directories( ${NOCR_SOURCE_DIR}/External-libraries/pugixml/ )
    
if (UNIX)
    add_compile_options(-std=c++0x -Wall -ggdb -Wextra -msse4.2)
endif(UNIX)

add_subdirectory( ./External-libraries)

set (required_libraries 
        ${NOCR_EXTERNAL_LIB}/libpugi.so
        ${NOCR_EXTERNAL_LIB}/libLibSVM.so 
        ${NOCR_EXTERNAL_LIB}/libLibLinear.so)

add_subdirectory( NOCRLib )

include_directories( "${NOCR_SOURCE_DIR}/NOCRLib/include" )

add_subdirectory( ./NOCRMain )
add_subdirectory( ./Experiments-programs)
add_subdirectory( ./Training-programs)
add_subdirectory( ./PerformanceMeasuring-programs)



