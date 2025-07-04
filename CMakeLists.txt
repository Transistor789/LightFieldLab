cmake_minimum_required(VERSION 3.16)
project(LightFieldLab LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BIN_NAME "lightfieldlab")

if(WIN32) # Windows
    set(CMAKE_PREFIX_PATH
        "D:/Qt/6.7.2/msvc2019_64;D:/opencv/install/x64/vc17/lib")
else() # Linux/MacOS
    set(OpenCV_DIR "/usr/local/opencv/Release")
    set(Qt_DIR "~/Qt/6.8.3/macos")
endif()
set(CMAKE_PREFIX_PATH "${OpenCV_DIR};${Qt_DIR}")

message(${OpenCV_DIR})
message(${CMAKE_PREFIX_PATH})

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
# find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs dnn_superres)

add_subdirectory(test)

set(PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/mainwindow.cpp
    ${CMAKE_SOURCE_DIR}/src/ui.cpp
    ${CMAKE_SOURCE_DIR}/src/lfprocessor.cpp
    ${CMAKE_SOURCE_DIR}/src/lfload.cpp
    ${CMAKE_SOURCE_DIR}/src/lfrefocus.cpp
    ${CMAKE_SOURCE_DIR}/src/lfsuperres.cpp
    ${CMAKE_SOURCE_DIR}/include/mainwindow.h
    ${CMAKE_SOURCE_DIR}/include/ui.h
    ${CMAKE_SOURCE_DIR}/include/lfdata.h
    ${CMAKE_SOURCE_DIR}/include/lfprocessor.h
    ${CMAKE_SOURCE_DIR}/include/lfload.h
    ${CMAKE_SOURCE_DIR}/include/lfrefocus.h
    ${CMAKE_SOURCE_DIR}/include/lfsuperres.h
)

add_executable(${BIN_NAME} ${PROJECT_SOURCES})
target_include_directories(${BIN_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${BIN_NAME} PRIVATE
    ${OpenCV_LIBS}
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)
target_precompile_headers(${BIN_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include/pch.h)
# copy data
file(COPY "${CMAKE_SOURCE_DIR}/input" DESTINATION "${CMAKE_BINARY_DIR}")

# info output 
message(STATUS "===============================================================")
message(STATUS "Current system is ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_BINDIR: ${CMAKE_INSTALL_BINDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "QT_VERSION: ${QT_VERSION}")
message(STATUS "===============================================================")
