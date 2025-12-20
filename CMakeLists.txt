cmake_minimum_required(VERSION 3.20)
project(LightFieldLab LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BIN_NAME "lightfieldlab")

if(MSVC)
    # 1. 架构指令：Debug/Release 都开启 (为了能编译 SIMD 代码)
    add_compile_options(/arch:AVX2)

    # 2. 优化指令：分开写，每个参数单独判断
    # 仅在 Release 或 RelWithDebInfo 模式下生效
    add_compile_options($<$<CONFIG:Release>:/O2>)
    add_compile_options($<$<CONFIG:Release>:/Ob2>)

    add_compile_options($<$<CONFIG:RelWithDebInfo>:/O2>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:/Ob2>)

else()
    # Linux/GCC/Clang
    add_compile_options(-mavx2 -march=native)

    # 优化指令也分开写
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:-O3>)
endif()

if(WIN32) # Windows
    set(OpenCV_DIR "D:\\Libraries\\opencv-4.11.0-cuda-12.8")
    set(Qt6_DIR "D:\\Qt\\6.10.0\\msvc2022_64")
    set(OPENSSL_ROOT_DIR "D:\\Libraries\\OpenSSL-3.5.1")
    set(Boost_DIR "D:\\Libraries\\boost_1_89_0")
    set(Eigen_DIR "D:\\Libraries\\eigen-5.0.1")
    set(TENSORRT_DIR "D:\\Libraries\\TensorRT-10.12.0.36")
    set(ORT_DIR "D:\\Libraries\\onnxruntime-win-x64-gpu-1.22.0")
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
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
# find_package(Eigen3 REQUIRED)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/core)
set(UI_DIR ${PROJECT_SOURCE_DIR}/ui)
include_directories(${SOURCE_DIR})
set(SOURCE_FILES
    main.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/lfrefocus.cpp
    ${SOURCE_DIR}/lfsr.cpp
    ${SOURCE_DIR}/trtwrapper.cpp
    ${SOURCE_DIR}/distgssr.cpp
    ${SOURCE_DIR}/lfmbase.cpp
    ${SOURCE_DIR}/lfsr_base.cpp
    ${UI_DIR}/mainwindow.cpp
    ${UI_DIR}/ui.cpp
    ${UI_DIR}/lightfieldlab.cpp
)
set(INCLUDE_FILES
    ${SOURCE_DIR}/utils.h
    ${SOURCE_DIR}/config.h
    ${SOURCE_DIR}/lfdata.h
    ${SOURCE_DIR}/lfio.h
    ${SOURCE_DIR}/lfrefocus.h
    ${SOURCE_DIR}/lfsr.h
    ${SOURCE_DIR}/trtwrapper.h
    ${SOURCE_DIR}/distgssr.h
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
    "${TENSORRT_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_directories(${BIN_NAME} PRIVATE
    "${OPENSSL_ROOT_DIR}/lib"

)
target_link_libraries(${BIN_NAME} PRIVATE
    ${OpenCV_LIBS} opencv_dnn_superres opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    CUDA::cudart
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
# file(COPY "${CMAKE_SOURCE_DIR}/data" DESTINATION "${CMAKE_BINARY_DIR}")
add_subdirectory(tests)

if(WIN32)
    # ==============================================================================
    # 1. Qt Deployment (必须要绑定到具体的可执行文件 Target)
    # ==============================================================================
    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${BIN_NAME} POST_BUILD
            COMMAND ${WINDEPLOYQT_EXECUTABLE}
            $<TARGET_FILE:${BIN_NAME}>
            $<$<CONFIG:Debug>:--debug>$<$<NOT:$<CONFIG:Debug>>:--release>
            --no-translations
            --verbose=1
            --dir "$<TARGET_FILE_DIR:${BIN_NAME}>"
            COMMENT "Running windeployqt for $<CONFIG> configuration..."
            VERBATIM
        )
    else()
        message(WARNING "windeployqt not found in '${Qt6_DIR}/bin'. Auto-deployment disabled.")
    endif()

    # ==============================================================================
    # 2. 全局资源与依赖拷贝 (脱离具体 Target，独立管理)
    # ==============================================================================

    # 定义需要拷贝的目标文件夹列表: 
    # 1. 构建根目录 (例如 build/Release)
    # 2. 测试目录 (例如 build/tests/Release)
    # 注意：这里我们使用 CMAKE_BINARY_DIR 的根，如果你的 tests 输出在特定子目录，请根据实际调整
    set(DEST_DIRS
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/tests"
    )

    # 确保目标文件夹存在，否则拷贝可能会失败
    foreach(DIR ${DEST_DIRS})
        file(MAKE_DIRECTORY "${DIR}")
    endforeach()

    # 创建一个独立的伪目标，专门用于拷贝依赖
    # "ALL" 表示只要构建整个项目，这个 Target 就会自动运行
    add_custom_target(CopyDependencies ALL COMMENT "Copying Data and DLLs to build directories...")

    # --------------------------------------------------------
    # A. 复制 Data 文件夹
    # --------------------------------------------------------
    # 假设 data 文件夹在项目根目录 ${CMAKE_SOURCE_DIR}/data
    add_custom_command(TARGET CopyDependencies POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/data"
        "${CMAKE_BINARY_DIR}/data"
        COMMENT "Copying data folder to build root..."
        VERBATIM
    )

    # --------------------------------------------------------
    # B. 准备 DLL 列表
    # --------------------------------------------------------
    set(DLLS_TO_COPY "")

    # 1. 添加 OpenSSL DLLs
    list(APPEND DLLS_TO_COPY
        "${OPENSSL_ROOT_DIR}/bin/libssl-3-x64.dll"
        "${OPENSSL_ROOT_DIR}/bin/libcrypto-3-x64.dll"
    )

    # 2. 添加 OpenCV DLLs (自动解析)
    foreach(cv_item ${OpenCV_LIBS})
        if(TARGET "${cv_item}")
            get_target_property(target_type "${cv_item}" TYPE)
            if("${target_type}" STREQUAL "SHARED_LIBRARY" OR "${target_type}" STREQUAL "UNKNOWN_LIBRARY")
                # 使用生成器表达式获取 DLL 路径
                list(APPEND DLLS_TO_COPY "$<TARGET_FILE:${cv_item}>")
            endif()
        endif()
    endforeach()

    # 3. 添加 TensorRT/CUDA DLLs (如果你需要的话，手动添加)
    list(APPEND DLLS_TO_COPY "${TENSORRT_DIR}/lib/nvinfer_10.dll" ...)

    # --------------------------------------------------------
    # C. 执行拷贝 (分发到所有目录)
    # --------------------------------------------------------
    if(DLLS_TO_COPY)
        foreach(DEST_DIR ${DEST_DIRS})
            add_custom_command(TARGET CopyDependencies POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLLS_TO_COPY}
                "${DEST_DIR}/"
                COMMENT "Distributing DLLs to ${DEST_DIR}..."
                VERBATIM
            )
        endforeach()
    endif()

endif()


