#cmake_minimum_required(VERSION 2.8)
#project ( desionTree )
#find_package( OpenCV REQUIRED )
#find_package( Boost REQUIRED )
#add_executable ( desionTree main_desTree.cpp ../headers/segment/component.cpp 
#  ../headers/findLetter/findLetter.cpp
#  ../headers/swt/swt.cpp ../headers/iooper.cpp ../headers/segment/segment.cpp 
#  ../headers/preprocesing.cpp
#  )

set (bin_exec TrainGenerator)

add_executable( ${bin_exec} main_desTree.cpp )
target_link_libraries( ${bin_exec} ${OpenCV_LIBS} )
target_link_libraries( ${bin_exec} NOCRLib )  
target_link_libraries( ${bin_exec} ${required_libraries})

target_link_libraries( ${bin_exec} ${Boost_PROGRAM_OPTIONS_LIBRARY} )
