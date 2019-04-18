TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    HOGLib/fhog.cpp \
    KCFTracker/kcfTracker.cpp \
    Utilities/saliency.cpp

INCLUDEPATH += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

HEADERS += \
    HOGLib/fhog.hpp \
    KCFTracker/kcfTracker.h \
    Utilities/utils.h \
    Utilities/saliency.h \
