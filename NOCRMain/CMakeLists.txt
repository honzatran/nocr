
set (bin_exec nocr)

add_executable( ${bin_exec} main.cpp xml_creator.cpp 
    ./translation_recorder.cpp )

target_link_libraries( ${bin_exec} NOCRLib )
target_link_libraries( ${bin_exec} ${OpenCV_LIBS} )
# target_link_libraries( ${bin_exec} ${Boost_LIBRARIES})
target_link_libraries( ${bin_exec} ${required_libraries})
