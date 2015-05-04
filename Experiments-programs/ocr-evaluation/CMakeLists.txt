set (bin_exec ocr-evaluation)

add_executable( ${bin_exec} main.cpp ../common/segmentation_base.cpp )
target_link_libraries( ${bin_exec} NOCRLib )
target_link_libraries( ${bin_exec} ${OpenCV_LIBS} )
target_link_libraries( ${bin_exec} ${Boost_PROGRAM_OPTIONS_LIBRARY} )
target_link_libraries( ${bin_exec} ${required_libraries})
