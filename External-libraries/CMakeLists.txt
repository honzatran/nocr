# project ( NOCR )
# find_package( OpenCV REQUIRED )
# # find_package( Boost REQUIRED )
#
# set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${NOCR_SOURCE_DIR}/bin" )
# set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${NOCR_SOURCE_DIR}/lib" )
# set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${NOCR_SOURCE_DIR}/lib" )
#
# # set( VLROOT ${NOCR_SOURCE_DIR}/vlfeat-0.9.18/ )
# # set( VLROOT_INCLUDE_DIR ${NOCR_SOURCE_DIR}/vlfeat-0.9.18/ )
# # set( VLROOT_LIB ${NOCR_SOURCE_DIR}/vlfeat-0.9.18/bin/glnx86/libvl.so )
# set( NOCR_EXTERNAL_LIB ${NOCR_SOURCE_DIR}/lib )
#
# # include_directories( ${VLROOT_INCLUDE_DIR} )
# include_directories( ${NOCR_SOURCE_DIR}/libsvm-3.18/ )
# include_directories( ${NOCR_SOURCE_DIR}/liblinear-1.94/ )
# include_directories( ${NOCR_SOURCE_DIR}/pugixml/ )
#     
# add_compile_options(-std=c++0x -Wall -ggdb -Wextra)

add_subdirectory( ./pugixml)
add_subdirectory( ./libsvm-3.18)
# add_subdirectory( ./liblinear-1.94)





