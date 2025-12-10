cmake_minimum_required(VERSION 3.20)
project(LightFieldLab LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BIN_NAME "lightfieldlab")


if(WIN32) # Windows
    set(OpenCV_DIR "D:\\Libraries\\opencv-4.11.0-cuda-12.8")
    set(Qt6_DIR "D:\\Qt\\6.10.0\\msvc2022_64")
    set(OPENSSL_ROOT_DIR "D:\\Libraries\\OpenSSL-3.5.1")
    set(Boost_DIR "D:\\Libraries\\boost_1_89_0")
    set(Eigen_DIR "D:\\Libraries\\eigen-5.0.1")
    find_program(WINDEPLOYQT_EXECUTABLE
        NAMES windeployqt windeployqt.exe
        PATHS "${Qt6_DIR}/bin"
        NO_DEFAULT_PATH
    )
elseif(APPLE) # MacOS
    set(OpenCV_DIR "/usr/local/opencv/${CMAKE_BUILD_TYPE}")
    set(Qt6_DIR "~/Qt/6.8.3/macos")
    set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl@3/")
elseif(UNIX) # Linux
    set(OpenCV_DIR "/usr/local/opencv/${CMAKE_BUILD_TYPE}")
    set(Qt6_DIR "/home/jax/Qt/6.10.0/gcc_64")
endif()
set(CMAKE_PREFIX_PATH "${OpenCV_DIR};${Qt6_DIR};${Boost_DIR};${Eigen3_DIR}")

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs dnn_superres)
find_package(OpenSSL REQUIRED)
# find_package(Eigen3 REQUIRED)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/core)
set(UI_DIR ${PROJECT_SOURCE_DIR}/ui)
include_directories(${SOURCE_DIR})
set(SOURCE_FILES
    main.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfload.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/lfrefocus.cpp
    ${SOURCE_DIR}/lfsuperres.cpp
    ${UI_DIR}/mainwindow.cpp
    ${UI_DIR}/ui.cpp
    ${UI_DIR}/lightfieldlab.cpp
)
set(INCLUDE_FILES
    ${SOURCE_DIR}/utils.h
    ${SOURCE_DIR}/config.h
    ${SOURCE_DIR}/lfdata.h
    ${SOURCE_DIR}/lfload.h
    ${SOURCE_DIR}/lfrefocus.h
    ${SOURCE_DIR}/lfsuperres.h
    ${UI_DIR}/lightfieldlab.h
    ${UI_DIR}/mainwindow.h
    ${UI_DIR}/ui.h
    ${UI_DIR}/qlfapi.h
    ${UI_DIR}/lightfieldlab.h

)

add_executable(${BIN_NAME} ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(${BIN_NAME} PRIVATE
    "${SOURCE_DIR}"
    "${UI_DIR}"
    "${OPENSSL_INCLUDE_DIR}"
)
target_link_directories(${BIN_NAME} PRIVATE
    "${OPENSSL_ROOT_DIR}/lib"
)
target_link_libraries(${BIN_NAME} PRIVATE
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    OpenSSL::SSL OpenSSL::Crypto
    Qt6::Core Qt6::Gui Qt6::Widgets
)
# === DEBUG: Check if OpenSSL::Crypto propagates include dirs ===
get_target_property(ssl_inc OpenSSL::SSL INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(crypto_inc OpenSSL::Crypto INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "OpenSSL::SSL INTERFACE_INCLUDE_DIRECTORIES = ${ssl_inc}")
message(STATUS "OpenSSL::Crypto INTERFACE_INCLUDE_DIRECTORIES = ${crypto_inc}")

# Also check final target
get_target_property(final_inc ${BIN_NAME} INCLUDE_DIRECTORIES)
message(STATUS "Final target (${BIN_NAME}) INCLUDE_DIRECTORIES = ${final_inc}")
target_precompile_headers(${BIN_NAME} PRIVATE ${SOURCE_DIR}/pch.h)

# info output 
message(STATUS "===============================================================")
message(STATUS "Current system is ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "OpenCV found: ${OpenCV_FOUND}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Qt6 found: ${Qt6_FOUND}")
message(STATUS "Qt6 version: ${Qt6Core_VERSION}")
message(STATUS "Qt6 include dirs: ${Qt6Core_INCLUDE_DIRS}")
message(STATUS "Qt6 libraries: ${Qt6Core_LIBRARIES}")
message(STATUS "OpenSSL found: ${OPENSSL_FOUND}")
message(STATUS "OpenSSL version: ${OPENSSL_VERSION}")
message(STATUS "OpenSSL include dir: ${OPENSSL_INCLUDE_DIR}")
message(STATUS "OpenSSL libraries: ${OPENSSL_LIBRARIES}")
message(STATUS "===============================================================")

# copy data
file(COPY "${CMAKE_SOURCE_DIR}/data" DESTINATION "${CMAKE_BINARY_DIR}")
add_subdirectory(tests)

if(WIN32)
    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${BIN_NAME} POST_BUILD
            COMMAND ${WINDEPLOYQT_EXECUTABLE}
            $<TARGET_FILE:${BIN_NAME}>
            $<$<CONFIG:Debug>:--debug>$<$<NOT:$<CONFIG:Debug>>:--release>
            --no-translations
            --verbose=1
            COMMENT "Running windeployqt for $<CONFIG> configuration..."
            VERBATIM
        )
    else()
        message(WARNING "windeployqt not found in '${Qt6_DIR}/bin'. Auto-deployment disabled.")
    endif()

    # ===== Copy OpenSSL DLLs to output directory =====
    set(OPENSSL_DLLS
        "${OPENSSL_ROOT_DIR}/bin/libssl-3-x64.dll"
        "${OPENSSL_ROOT_DIR}/bin/libcrypto-3-x64.dll"
    )

    add_custom_command(TARGET ${BIN_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENSSL_DLLS}
        "$<TARGET_FILE_DIR:${BIN_NAME}>/"
        COMMENT "Copying OpenSSL DLLs to $<CONFIG> output directory..."
        VERBATIM
    )

    # ===== Copy OpenCV DLLs =====
    # 1. 遍历 ${OpenCV_LIBS}，筛选出有效的 CMake Target
    set(OPENCV_DLLS_TO_COPY "")
    foreach(cv_item ${OpenCV_LIBS})
        # 检查是否为 Target (忽略 'debug'/'optimized' 等关键字)
        if(TARGET "${cv_item}")
            # 获取目标类型，确保是动态库 (SHARED) 或未知类型 (UNKNOWN，通常是导入的库)
            get_target_property(target_type "${cv_item}" TYPE)
            if("${target_type}" STREQUAL "SHARED_LIBRARY" OR "${target_type}" STREQUAL "UNKNOWN_LIBRARY")
                # 使用生成器表达式自动获取对应配置(Debug/Release)的 DLL 路径
                list(APPEND OPENCV_DLLS_TO_COPY "$<TARGET_FILE:${cv_item}>")
            endif()
        endif()
    endforeach()

    # 2. 如果找到了 OpenCV DLLs，添加到构建后命令
    if(OPENCV_DLLS_TO_COPY)
        # --- Copy to Main Target Directory (${BIN_NAME}) ---
        add_custom_command(TARGET ${BIN_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS_TO_COPY}
            "$<TARGET_FILE_DIR:${BIN_NAME}>/"
            COMMENT "Copying OpenCV DLLs to ${BIN_NAME} output directory..."
            VERBATIM
        )
    endif()

endif()
