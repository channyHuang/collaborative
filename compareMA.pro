QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += $$PWD/../../thirdLibs/eigen-3.4.0/

# opencv
#INCLUDEPATH += $$PWD/../../thirdLibs/MinGW-Build-OpenCV-4.5.5-x64/include
#LIB_OPENCV = $$PWD/../../thirdLibs/MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/bin
#LIB_OPENCV_FILES = $$files($${LIB_OPENCV}/"*.dll", false)

# ncnn
#INCLUDEPATH += $$PWD/../../thirdLibs/ncnn/install/include
#LIB_NCNN = $$PWD/../../thirdLibs/ncnn/buildForQt/install/lib
#LIB_NCNN_FILES = $$files($${LIB_NCNN}/"*.a", false)

#LIBS += -L $$PWD/../../thirdLibs/MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/lib
# $$PWD/../../thirdLibs/MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/bin

#LIBS += $${LIB_NCNN_FILES}

LIBS += -fopenmp

HEADER_FILES = $$files("*.h", false)

SOURCE_FILES = $$files("*.cpp", false)

INCLUDEPATH += $$PWD/include

HEADERS += \
        $${HEADER_FILES}

SOURCES += \
        $${SOURCE_FILES}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
