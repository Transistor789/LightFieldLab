# ==============================================================================
# 1. 定义工具名称与源文件
# ==============================================================================
set(TOOL_BIN "inria_dataset_runner")

# 复用父作用域定义的核心算法源文件
# 注意：我们需要过滤掉主程序的 main.cpp，因为工具自带 main()
set(TOOL_SRCS
    inria_dataset_runner.cpp
)

# 遍历父作用域的 SRCS_CORE，转换为绝对路径并过滤 main.cpp
foreach(src_file ${SRCS_CORE})
    if(NOT "${src_file}" MATCHES "main.cpp")
        # 由于我们在子目录，需指向项目根目录下的 core 文件夹
        list(APPEND TOOL_SRCS "${PROJECT_SOURCE_DIR}/${src_file}")
    endif()
endforeach()

# ==============================================================================
# 2. 构建可执行文件
# ==============================================================================
add_executable(${TOOL_BIN} ${TOOL_SRCS})

# 设置 CUDA 架构（与主程序一致）
set_property(TARGET ${TOOL_BIN} PROPERTY CUDA_ARCHITECTURES native)

# ==============================================================================
# 3. 包含路径与链接库
# ==============================================================================

# 包含路径：直接复用主程序的逻辑
target_include_directories(${TOOL_BIN} PRIVATE
    "${PROJECT_SOURCE_DIR}/core"
    "${OpenCV_DIR}/include"
    "${TENSORRT_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${Eigen_DIR}"
)

if(UNIX)
    target_include_directories(${TOOL_BIN} PRIVATE "/usr/include/eigen3")
endif()

target_link_directories(${TOOL_BIN} PRIVATE
    "${PROJECT_SOURCE_DIR}/core"
    "${OPENSSL_ROOT_DIR}/lib"
)

# 链接库：工具通常不需要全部 Qt 模块，只需 Core 处理字符串和文件系统
target_link_libraries(${TOOL_BIN} PRIVATE
    Qt6::Core
    ${OpenCV_LIBS} opencv_dnn_superres opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    OpenSSL::SSL OpenSSL::Crypto
    OpenMP::OpenMP_CXX
    CUDA::cudart
    ${NVINFER_LIB}
)

# 如果你的 runner.cpp 涉及特定的 Windows 库
if(WIN32)
    target_link_libraries(${TOOL_BIN} PRIVATE USBConfiguration)
elseif(UNIX)
    target_link_libraries(${TOOL_BIN} PRIVATE Eigen3::Eigen)
endif()


# ==============================================================================
# 4. 运行时输出位置 (方便读取 Data 和 DLL)
# ==============================================================================
# 将工具编译结果直接放在 build 根目录，这样它能直接读取同级的 data 文件夹
set_target_properties(${TOOL_BIN} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(calibration_evaluator
    calibration_evaluator.cpp
    "${PROJECT_SOURCE_DIR}/core/lfcalibrate.cpp"
    "${PROJECT_SOURCE_DIR}/core/centers_extract.cpp"
    "${PROJECT_SOURCE_DIR}/core/centers_sort.cpp"
    "${PROJECT_SOURCE_DIR}/core/hexgrid_fit.cpp"
    "${PROJECT_SOURCE_DIR}/core/utils.cpp"
    "${PROJECT_SOURCE_DIR}/core/lfio.cpp"
    "${PROJECT_SOURCE_DIR}/core/raw_decode.cpp"
    "${PROJECT_SOURCE_DIR}/core/califinder.cpp"
)
target_include_directories(calibration_evaluator PRIVATE
    "${PROJECT_SOURCE_DIR}/core"
    "${Eigen_DIR}"
    "${OPENSSL_ROOT_DIR}\\lib"
)
target_link_libraries(calibration_evaluator PRIVATE
    ${OpenCV_LIBS}
    OpenSSL::SSL OpenSSL::Crypto

)
