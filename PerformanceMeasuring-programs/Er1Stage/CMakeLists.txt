find_package( Boost COMPONENTS program_options REQUIRED )

set (bin_exec er1stage)

add_executable( ${bin_exec} main.cpp measure_task.cpp )
target_link_libraries( ${bin_exec} NOCRLib )
target_link_libraries( ${bin_exec} ${OpenCV_LIBS} )
target_link_libraries( ${bin_exec} ${Boost_PROGRAM_OPTIONS_LIBRARY} )
target_link_libraries( ${bin_exec} ${required_libraries})
