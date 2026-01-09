cmake_minimum_required(VERSION 3.20)
project(LightFieldLab LANGUAGES CXX)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# ==============================================================================
# 1. 项目基础配置
# ==============================================================================
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BIN_NAME "lightfieldlab")

# ==============================================================================
# 2. 平台特定配置 (编译器选项 & 依赖路径)
# ==============================================================================
if(WIN32)
    message(STATUS "Configuring for Windows (MSVC)...")

    # --- 编译器选项 ---
    add_compile_options(/arch:AVX2)
    add_compile_options(/fp:fast)
    add_compile_options($<$<CONFIG:Release>:/O2>)
    add_compile_options($<$<CONFIG:Release>:/Ob2>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:/O2>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:/Ob2>)

    # --- 依赖库路径 ---
    set(OpenCV_DIR "D:\\Libraries\\opencv-4.11.0-cuda-12.8")
    set(Qt6_DIR "D:\\Qt\\6.10.0\\msvc2022_64")
    set(OPENSSL_ROOT_DIR "D:\\Libraries\\OpenSSL-3.5.1")
    set(Boost_DIR "D:\\Libraries\\boost_1_89_0")
    set(Eigen_DIR "D:\\Libraries\\eigen-5.0.1")
    set(ORT_DIR "D:\\Libraries\\onnxruntime-win-x64-gpu-1.22.0")
    set(TENSORRT_DIR "D:\\Libraries\\TensorRT-10.12.0.36")

    # 查找 windeployqt
    find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt PATHS "${Qt6_DIR}/bin" NO_DEFAULT_PATH)

elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux...")

    # --- 编译器选项 ---
    add_compile_options(-mavx2 -march=native)
    add_compile_options($<$<CONFIG:Release>:-O3>)

    # --- 依赖库路径 ---
    set(OpenCV_DIR "/usr/local/opencv/${CMAKE_BUILD_TYPE}")
    set(Qt6_DIR "/home/jax/Qt/6.10.0/gcc_64")
    set(ORT_DIR "/home/jax/Libraries/onnxruntime-linux-x64-gpu-1.22.0")
    set(TENSORRT_DIR "/usr") # apt 安装通常在系统目录
endif()

# 将所有自定义路径加入前缀，方便 find_package 查找
set(CMAKE_PREFIX_PATH "${OpenCV_DIR};${Qt6_DIR};${Boost_DIR};${Eigen_DIR}")

# ==============================================================================
# 3. 查找依赖包 & 验证 (增强可读性)
# ==============================================================================
message(STATUS "--------------------------------------------------")
message(STATUS "Searching for dependencies...")

# --- Qt6 ---
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
message(STATUS "  [OK] Qt6 Version: ${Qt6_VERSION} (Dir: ${Qt6_DIR})")

# --- OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs dnn_superres)
message(STATUS "  [OK] OpenCV Version: ${OpenCV_VERSION} (Dir: ${OpenCV_DIR})")

# --- OpenSSL ---
find_package(OpenSSL REQUIRED)
message(STATUS "  [OK] OpenSSL Version: ${OPENSSL_VERSION}")

# --- OpenMP ---
find_package(OpenMP REQUIRED)
message(STATUS "  [OK] OpenMP found")

# --- CUDA ---
find_package(CUDAToolkit REQUIRED)
message(STATUS "  [OK] CUDA Toolkit found (Dir: ${CUDAToolkit_INCLUDE_DIRS})")

# --- Eigen3 (Linux only check) ---
if(UNIX)
    find_package(Eigen3 REQUIRED)
    message(STATUS "  [OK] Eigen3 found (Dir: ${EIGEN3_INCLUDE_DIR})")
endif()

# --- TensorRT Libraries (Manual Find) ---
find_library(NVINFER_LIB NAMES nvinfer nvinfer_10 HINTS "${TENSORRT_DIR}/lib" "${TENSORRT_DIR}/lib/x86_64-linux-gnu")
if(NVINFER_LIB)
    message(STATUS "  [OK] TensorRT nvinfer: ${NVINFER_LIB}")
else()
    message(FATAL_ERROR "Could NOT find nvinfer library. Please check TENSORRT_DIR.")
endif()

find_library(NVONNXPARSER_LIB NAMES nvonnxparser nvonnxparser_10 HINTS "${TENSORRT_DIR}/lib" "${TENSORRT_DIR}/lib/x86_64-linux-gnu")
if(NVONNXPARSER_LIB)
    message(STATUS "  [OK] TensorRT nvonnxparser: ${NVONNXPARSER_LIB}")
else()
    message(WARNING "Could NOT find nvonnxparser library. Tests might fail.")
endif()
message(STATUS "--------------------------------------------------")

# ==============================================================================
# 4. 源文件管理
# ==============================================================================
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/core)
set(UI_DIR ${PROJECT_SOURCE_DIR}/ui)

# 核心算法模块
set(SRCS_CORE
    main.cpp
    core/utils.cpp
    core/config.cpp
    core/lfio.cpp
    core/lfmbase.cpp
    core/lfdisp_base.cpp
    core/lfsr_base.cpp
    core/raw_decode.cpp
    core/hexgrid_fit.cpp
    core/centers_extract.cpp
    core/centers_sort.cpp
    core/lfcapture.cpp
    core/lfcalibrate.cpp
    core/lfrefocus.cpp
    core/lfdepth.cpp
    core/distgdisp.cpp
    core/lfsr.cpp
    core/distgssr.cpp
    core/lfisp.cpp
    core/colormatcher.cpp
    core/trtwrapper.cpp
    core/califinder.cpp
)

# UI 界面模块
set(SRCS_UI
    ui/qlogger.h
    ui/lfcontrol.cpp ui/lfcontrol.h
    ui/mainwindow.cpp ui/mainwindow.h ui/mainwindow.ui
    ui/widgetcontrol.cpp ui/widgetcontrol.h ui/widgetcontrol.ui
    ui/widgetlogger.cpp ui/widgetlogger.h ui/widgetlogger.ui
    ui/widgetimage.cpp ui/widgetimage.h ui/widgetimage.ui

    ui/dialogccm.cpp ui/dialogccm.h ui/dialogccm.ui
    ui/interactiveview.cpp ui/interactiveview.h
    ui/smartimageviewer.cpp ui/smartimageviewer.h ui/smartimageviewer.ui
)

# ==============================================================================
# 5. 目标构建
# ==============================================================================
add_executable(${BIN_NAME} ${SRCS_CORE} ${SRCS_UI})

# --- 头文件路径 ---
target_include_directories(${BIN_NAME} PRIVATE
    "${SOURCE_DIR}"
    "${UI_DIR}"
    "${OpenCV_DIR}/include"
    "${TENSORRT_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${Eigen_DIR}" # Windows下的Eigen路径
)

if(UNIX)
    # Linux下apt安装的Eigen通常位置
    target_include_directories(${BIN_NAME} PRIVATE "/usr/include/eigen3")
endif()

# --- 库文件路径 ---
target_link_directories(${BIN_NAME} PRIVATE
    "${SOURCE_DIR}"
    "${OPENSSL_ROOT_DIR}/lib"
)

# --- 链接库 ---
target_link_libraries(${BIN_NAME} PRIVATE
    Qt6::Core Qt6::Gui Qt6::Widgets
    ${OpenCV_LIBS} opencv_dnn_superres opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    OpenSSL::SSL OpenSSL::Crypto
    OpenMP::OpenMP_CXX
    CUDA::cudart
    ${NVINFER_LIB}
)

# 平台特定链接
if(WIN32)
    target_link_libraries(${BIN_NAME} PRIVATE USBConfiguration)
elseif(UNIX)
    target_link_libraries(${BIN_NAME} PRIVATE Eigen3::Eigen)
endif()

# 预编译头 (加速编译)
target_precompile_headers(${BIN_NAME} PRIVATE ${SOURCE_DIR}/pch.h)

# 添加测试子目录
add_subdirectory(tests)

# ==============================================================================
# 6. 资源拷贝 (跨平台：Linux/Windows 都会执行)
# ==============================================================================

# 1. 创建一个全局的伪目标，用于触发拷贝任务
#    ALL 表示每次构建都会检查运行
add_custom_target(CopyDependencies ALL COMMENT "Checking and copying runtime resources...")

# 2. [通用] 拷贝 Data 目录
#    现在这段逻辑在 if(WIN32) 之外，所有平台都会把源码下的 data 拷到 build/data
add_custom_command(TARGET CopyDependencies POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/data" "${CMAKE_BINARY_DIR}/data"
    COMMENT "Copying data folder to build directory..."
)

# ==============================================================================
# 7. Windows 专属部署 (DLLs & Windeployqt)
# ==============================================================================
if(WIN32)
    # A. 运行 windeployqt (Qt 部署)
    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${BIN_NAME} POST_BUILD
            COMMAND ${WINDEPLOYQT_EXECUTABLE}
            $<TARGET_FILE:${BIN_NAME}>
            $<$<CONFIG:Debug>:--debug>$<$<NOT:$<CONFIG:Debug>>:--release>
            --no-translations --verbose=1
            --dir "$<TARGET_FILE_DIR:${BIN_NAME}>"
            COMMENT "Running windeployqt..."
        )
    endif()

    # B. 拷贝 DLLs (仅 Windows 需要)
    #    注意：我们继续使用上面定义的 CopyDependencies 目标来挂载 DLL 拷贝命令

    # 收集 DLLs
    set(DLLS_TO_COPY
        "${OPENSSL_ROOT_DIR}/bin/libssl-3-x64.dll"
        "${OPENSSL_ROOT_DIR}/bin/libcrypto-3-x64.dll"
        "${SOURCE_DIR}/ft602.dll"
        "${SOURCE_DIR}/USBConfiguration.dll"
        "${TENSORRT_DIR}/lib/nvinfer_10.dll"
    )

    # 自动收集 OpenCV DLL
    foreach(cv_item ${OpenCV_LIBS})
        if(TARGET "${cv_item}")
            get_target_property(target_type "${cv_item}" TYPE)
            if("${target_type}" STREQUAL "SHARED_LIBRARY" OR "${target_type}" STREQUAL "UNKNOWN_LIBRARY")
                list(APPEND DLLS_TO_COPY "$<TARGET_FILE:${cv_item}>")
            endif()
        endif()
    endforeach()

    # 定义 DLL 拷贝的目标目录 (Build根目录 和 Tests目录)
    set(DEST_DIRS "${CMAKE_BINARY_DIR}" "${CMAKE_BINARY_DIR}/tests")
    foreach(DIR ${DEST_DIRS})
        file(MAKE_DIRECTORY "${DIR}")
    endforeach()

    # 执行拷贝 (挂载到 CopyDependencies)
    foreach(DEST_DIR ${DEST_DIRS})
        add_custom_command(TARGET CopyDependencies POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLLS_TO_COPY} "${DEST_DIR}/"
            COMMENT "Copying DLLs to ${DEST_DIR}..."
        )
    endforeach()
endif()
