cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set( SOURCES 
    NOCRGui/convertoropencv.cpp
    NOCRGui/imageworker.cpp
    NOCRGui/main.cpp
    NOCRGui/mainwindow.cpp
    NOCRGui/newwordsdialog.cpp
    NOCRGui/translationrecord.cpp
    NOCRGui/xmlcreator.cpp
)


set( Qt5Path /opt/Qt5.0.1/5.0.1/gcc )
set( CMAKE_PREFIX_PATH ${Qt5Path} )

find_package(Qt5Widgets REQUIRED )
find_package(Qt5Core REQUIRED )
find_package(Qt5Gui REQUIRED )
find_package(Qt5Xml REQUIRED )

qt5_wrap_ui( UIS_HDRS 
    NOCRGui/mainwindow.ui  
    NOCRGui/newwordsdialog.ui 
    )

set (bin_exec NOCRGui)

add_executable(${bin_exec} ${SOURCES} ${UIS_HDRS} )
qt5_use_modules(${bin_exec} Widgets)
qt5_use_modules(${bin_exec} Core)
qt5_use_modules(${bin_exec} Gui)
qt5_use_modules(${bin_exec} Xml)


target_link_libraries( ${bin_exec} NOCRLib )
target_link_libraries( ${bin_exec} ${OpenCV_LIBS} )
target_link_libraries( ${bin_exec} ${Boost_LIBRARIES})
target_link_libraries( ${bin_exec} ${required_libraries})


