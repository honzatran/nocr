find_package( Boost COMPONENTS program_options REQUIRED )

add_executable( ErExtractor main.cpp )
target_link_libraries( ErExtractor NOCRLib )
target_link_libraries( ErExtractor ${OpenCV_LIBS} )
target_link_libraries( ErExtractor ${Boost_PROGRAM_OPTIONS_LIBRARY} )
target_link_libraries( ErExtractor ${required_libraries})
