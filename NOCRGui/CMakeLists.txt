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

add_executable(NOCRGui ${SOURCES} ${UIS_HDRS} )
qt5_use_modules(NOCRGui Widgets)
qt5_use_modules(NOCRGui Core)
qt5_use_modules(NOCRGui Gui)
qt5_use_modules(NOCRGui Xml)


target_link_libraries( NOCRGui NOCRLib )
target_link_libraries( NOCRGui ${OpenCV_LIBS} )
target_link_libraries( NOCRGui ${NOCR_EXTERNAL_LIB}/libLibSVM.a ) 
target_link_libraries( NOCRGui ${NOCR_EXTERNAL_LIB}/libLibLinear.a ) 


