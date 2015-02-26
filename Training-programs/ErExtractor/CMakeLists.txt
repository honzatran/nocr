find_package( Boost COMPONENTS program_options REQUIRED )

add_executable( ErExtractor main.cpp )
target_link_libraries( ErExtractor NOCRLib )
target_link_libraries( ErExtractor ${OpenCV_LIBS} )
target_link_libraries( ErExtractor ${Boost_PROGRAM_OPTIONS_LIBRARY} )
target_link_libraries( ErExtractor ${NOCR_EXTERNAL_LIB}/libLibSVM.a ) 
target_link_libraries( ErExtractor ${NOCR_EXTERNAL_LIB}/libLibLinear.a ) 
target_link_libraries( ErExtractor ${NOCR_EXTERNAL_LIB}/libpugi.a ) 
