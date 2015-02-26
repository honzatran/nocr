find_package( Boost COMPONENTS program_options REQUIRED )

set (exec_name 1stage-letter-evaluation)

add_executable( ${exec_name} main.cpp )
target_link_libraries( ${exec_name} NOCRLib )
target_link_libraries( ${exec_name} ${OpenCV_LIBS} )
target_link_libraries( ${exec_name} ${Boost_PROGRAM_OPTIONS_LIBRARY} )
target_link_libraries( ${exec_name} ${required_libraries})
