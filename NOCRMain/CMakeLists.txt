find_package( Boost COMPONENTS program_options REQUIRED )

add_executable( NOCR main.cpp xml_creator.cpp 
    ./translation_recorder.cpp )

target_link_libraries( NOCR NOCRLib )
target_link_libraries( NOCR ${OpenCV_LIBS} )
target_link_libraries( NOCR ${Boost_PROGRAM_OPTIONS_LIBRARY} )
target_link_libraries( NOCR ${NOCR_EXTERNAL_LIB}/libLibSVM.a ) 
target_link_libraries( NOCR ${NOCR_EXTERNAL_LIB}/libLibLinear.a ) 
target_link_libraries( NOCR ${NOCR_EXTERNAL_LIB}/libpugi.a )

 
  
