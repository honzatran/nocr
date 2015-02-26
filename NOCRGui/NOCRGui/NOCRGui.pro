#-------------------------------------------------
#
# Project created by QtCreator 2014-06-17T02:32:02
#
#-------------------------------------------------

QT       += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NOCRGui
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    convertoropencv.cpp \
    imageworker.cpp \
    translationrecord.cpp \
    xmlcreator.cpp \
    newwordsdialog.cpp

HEADERS  += mainwindow.h \
    convertoropencv.h \
    imageworker.h \
    translationrecord.h \
    xmlcreator.h \
    newwordsdialog.h

FORMS    += mainwindow.ui \
    newwordsdialog.ui
CONFIG += console
CONFIG += release

QMAKE_CXXFLAGS += -std=c++0x




unix:!macx: LIBS += -L$$PWD/../../lib/ -lNOCRLib

INCLUDEPATH += $$PWD/../../NOCRLib/include
DEPENDPATH += $$PWD/../../NOCRLib/include

unix:!macx: PRE_TARGETDEPS += $$PWD/../../lib/libNOCRLib.a

unix:!macx: LIBS += -L$$PWD/../../lib/ -lLibSVM

INCLUDEPATH += $$PWD/../../libsvm-3.18
DEPENDPATH += $$PWD/../../libsvm-3.18/

unix:!macx: PRE_TARGETDEPS += $$PWD/../../lib/libLibSVM.a


unix:!macx: LIBS += -L$$PWD/../../lib/ -lLibLinear

INCLUDEPATH += $$PWD/../../liblinear-1.94/
DEPENDPATH += $$PWD/../../liblinear-1.94/

unix:!macx: PRE_TARGETDEPS += $$PWD/../../lib/libLibLinear.a

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv

