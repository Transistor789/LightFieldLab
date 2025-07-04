cmake_minimum_required(VERSION 3.10)
project(test)

# 定义资源目录
set(RESOURCE_DIR "${CMAKE_SOURCE_DIR}/data")
set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_PREFIX_PATH "/usr/local/opencv")

# find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
# find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
# find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# test
add_executable(test test.cpp)

# test_metal
add_executable(test_metal test_metal.cpp)
target_link_libraries(test_metal ${OpenCV_LIBS})

# test_window_base
add_executable(test_window_base
    test_window_base.cpp
    ${CMAKE_SOURCE_DIR}/src/lfload.cpp
    ${CMAKE_SOURCE_DIR}/include/window_base.h
    ${CMAKE_SOURCE_DIR}/include/lfload.h
)
target_include_directories(test_window_base PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_window_base PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})

# test_ui
add_executable(test_ui
    test_ui.cpp
    ${CMAKE_SOURCE_DIR}/src/ui.cpp
    ${CMAKE_SOURCE_DIR}/include/ui.h
)
target_include_directories(test_ui PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_ui PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# test_superres
add_executable(test_superres
    test_lfsuperres.cpp
)
target_include_directories(test_superres PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_superres PRIVATE
    ${OpenCV_LIBS}
    opencv_dnn_superres
)

# test_lfloader
# add_executable(test_lfloader
#     test_lfload.cpp
#     ${CMAKE_SOURCE_DIR}/src/lfload.cpp
#     ${CMAKE_SOURCE_DIR}/include/lfdata.h
#     ${CMAKE_SOURCE_DIR}/include/lfload.h
#     ${CMAKE_SOURCE_DIR}/include/window_base.h
# )
# target_include_directories(test_lfloader PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(test_lfloader PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})

# test_lfprocessor
# add_executable(test_lfprocessor
#     test_lfprocessor.cpp
#     ${CMAKE_SOURCE_DIR}/src/lfprocessor.cpp
#     ${CMAKE_SOURCE_DIR}/src/lfload.cpp
#     ${CMAKE_SOURCE_DIR}/src/ui.cpp
#     ${CMAKE_SOURCE_DIR}/include/lfprocessor.h
#     ${CMAKE_SOURCE_DIR}/include/lfload.h
#     ${CMAKE_SOURCE_DIR}/include/window_base.h
#     ${CMAKE_SOURCE_DIR}/include/ui.h
# )
# target_include_directories(test_lfprocessor PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(test_lfprocessor PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})

# test_refocuser
# add_executable(test_refocuser
#     test_refocuser.cpp
#     ${CMAKE_SOURCE_DIR}/src/lfrefocuser.cpp
#     ${CMAKE_SOURCE_DIR}/src/lfload.cpp
#     ${CMAKE_SOURCE_DIR}/include/lfrefocus.h
#     ${CMAKE_SOURCE_DIR}/include/lfload.h
#     ${CMAKE_SOURCE_DIR}/include/worker_interface.h
# )
# target_include_directories(test_refocuser PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(test_refocuser PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})


