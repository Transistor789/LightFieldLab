# ==============================================================================
# tests/CMakeLists.txt - 测试模块构建脚本 (Fixed)
# ==============================================================================

# 【关键修复】: 必须包含 Core 目录，否则所有测试都找不到 .h 头文件
include_directories(${SOURCE_DIR})

# 1. 定义通用源文件集合 (减少重复代码)
# ------------------------------------------------------------------------------
# 基础模块: Utils, Config, IO, RawDecode
set(SRCS_COMMON
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/raw_decode.cpp
)

# 核心算法模块: ISP, Calibrate, Centers, HexGrid
set(SRCS_ALGO
    ${SOURCE_DIR}/lfisp.cpp
    ${SOURCE_DIR}/lfcalibrate.cpp
    ${SOURCE_DIR}/centers_extract.cpp
    ${SOURCE_DIR}/centers_sort.cpp
    ${SOURCE_DIR}/hexgrid_fit.cpp
)

# 深度学习/推理模块: TensorRT Wrapper, LFMBase, LFSRBase, LFDispBase
set(SRCS_INFER
    ${SOURCE_DIR}/trtwrapper.cpp
    ${SOURCE_DIR}/lfmbase.cpp
    ${SOURCE_DIR}/lfsr_base.cpp
    ${SOURCE_DIR}/lfdisp_base.cpp
)

# 2. 定义通用依赖库 (减少链接重复)
# ------------------------------------------------------------------------------
# 基础库
set(LIBS_BASIC
    OpenSSL::SSL OpenSSL::Crypto
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)

# CUDA 加速的 CV 库 (部分测试需要)
set(LIBS_CUDA_CV
    opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    CUDA::cudart
)

# ==============================================================================
# 3. 基础功能测试
# ==============================================================================

# Test: Basic Config
add_executable(test_config test_config.cpp ${SRCS_COMMON})
target_link_libraries(test_config PRIVATE ${LIBS_BASIC})

# Test: LF Load
add_executable(test_lfload test_lfload.cpp ${SRCS_COMMON})
target_link_libraries(test_lfload PRIVATE ${LIBS_BASIC} ${LIBS_CUDA_CV})

# Test: Read Raw (Simple)
add_executable(test_readraw test_readraw.cpp)
target_link_libraries(test_readraw PRIVATE ${OpenCV_LIBS})

# Test: OpenSSL
add_executable(test_openssl test_openssl.cpp ${SRCS_COMMON})
target_link_libraries(test_openssl PRIVATE ${LIBS_BASIC})

# Test: JSON
add_executable(test_json test_json.cpp)
# 不需要额外链接库，只需要 include (已通过全局 include_directories 处理)

# Test: Refocus
add_executable(test_lfrefocus test_lfrefocus.cpp ${SOURCE_DIR}/lfrefocus.cpp ${SRCS_COMMON})
target_link_libraries(test_lfrefocus PRIVATE ${LIBS_BASIC})

# ==============================================================================
# 4. 传统算法测试 (Calibration, ISP, Resample)
# ==============================================================================

# Test: Calibration
add_executable(test_lfcalibrate test_lfcalibrate.cpp ${SRCS_COMMON} ${SRCS_ALGO})
target_link_libraries(test_lfcalibrate PRIVATE ${LIBS_BASIC} ${LIBS_CUDA_CV})

# Test: ISP Pipeline
add_executable(test_isp test_isp.cpp ${SRCS_COMMON} ${SRCS_ALGO})
target_link_libraries(test_isp PRIVATE ${LIBS_BASIC} ${LIBS_CUDA_CV})

# Test: Resample
add_executable(test_lfresample test_lfresample.cpp ${SRCS_COMMON} ${SRCS_ALGO})
target_link_libraries(test_lfresample PRIVATE ${LIBS_BASIC} opencv_cudaimgproc opencv_cudaarithm CUDA::cudart)

# Linux 下这几个算法通常需要 Eigen3
if(UNIX)
    target_link_libraries(test_lfcalibrate PRIVATE Eigen3::Eigen)
    target_link_libraries(test_isp PRIVATE Eigen3::Eigen)
    target_link_libraries(test_lfresample PRIVATE Eigen3::Eigen)
    # 顺便把 Eigen 的头文件路径也加上，防止万一没用 Eigen3::Eigen 自动处理
    target_include_directories(test_lfcalibrate PRIVATE "/usr/include/eigen3")
    target_include_directories(test_isp PRIVATE "/usr/include/eigen3")
    target_include_directories(test_lfresample PRIVATE "/usr/include/eigen3")
endif()

# ==============================================================================
# 5. 深度学习与推理测试 (TensorRT, ONNX)
# ==============================================================================

# Test: TensorRT CLI Basic
add_executable(test_trtcli test_trtcli.cpp)
target_include_directories(test_trtcli PRIVATE "${TENSORRT_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(test_trtcli PRIVATE ${NVINFER_LIB} ${NVONNXPARSER_LIB} CUDA::cudart)

# Test: ONNX Runtime (DistgSSR)
add_executable(test_distgssr_onnx test_distgssr_onnx.cpp ${SOURCE_DIR}/utils.cpp)
target_include_directories(test_distgssr_onnx PRIVATE ${ORT_DIR}/include)
target_link_directories(test_distgssr_onnx PRIVATE ${ORT_DIR}/lib)
target_link_libraries(test_distgssr_onnx PRIVATE onnxruntime ${OpenCV_LIBS})

# Windows: Copy ONNX Runtime DLLs
if(WIN32)
    add_custom_command(TARGET test_distgssr_onnx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:test_distgssr_onnx>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime_providers_shared.dll" "$<TARGET_FILE_DIR:test_distgssr_onnx>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime_providers_cuda.dll" "$<TARGET_FILE_DIR:test_distgssr_onnx>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime_providers_tensorrt.dll" "$<TARGET_FILE_DIR:test_distgssr_onnx>"
        COMMENT "Copying ONNX Runtime DLLs..."
    )
endif()

# Test: TensorRT Super Resolution (DistgSSR)
add_executable(test_distgssr_trt test_distgssr_trt.cpp ${SOURCE_DIR}/distgssr.cpp ${SRCS_COMMON} ${SRCS_INFER})
target_link_libraries(test_distgssr_trt PRIVATE ${LIBS_BASIC} ${NVINFER_LIB} CUDA::cudart)

# Test: TensorRT Disparity (DistgDisp)
add_executable(test_distgdisp test_distgdisp.cpp ${SOURCE_DIR}/distgdisp.cpp ${SRCS_COMMON} ${SRCS_INFER})
target_link_libraries(test_distgdisp PRIVATE ${LIBS_BASIC} ${NVINFER_LIB} CUDA::cudart)

# Test: Epipolar Image Transform (EPI)
add_executable(test_epit test_epit.cpp ${SOURCE_DIR}/epit.cpp ${SRCS_COMMON} ${SRCS_INFER})
target_link_libraries(test_epit PRIVATE ${LIBS_BASIC} ${NVINFER_LIB} CUDA::cudart)

# Test: Depth Estimation High Level
add_executable(test_lfdepth test_lfdepth.cpp ${SOURCE_DIR}/lfdepth.cpp ${SOURCE_DIR}/distgdisp.cpp ${SRCS_COMMON} ${SRCS_INFER})
target_link_libraries(test_lfdepth PRIVATE ${LIBS_BASIC} ${NVINFER_LIB} CUDA::cudart)

# Test: Super Resolution High Level (包含 OpenCV DNN)
add_executable(test_lfsuperres
    test_lfsuperres.cpp
    ${SOURCE_DIR}/lfsr.cpp
    ${SOURCE_DIR}/distgssr.cpp
    ${SRCS_COMMON}
    ${SRCS_INFER}
)
target_link_libraries(test_lfsuperres PRIVATE ${LIBS_BASIC} opencv_dnn_superres ${NVINFER_LIB} CUDA::cudart)

# ==============================================================================
# 6. 统一补充路径 (兜底配置)
# ==============================================================================

# 补充 TensorRT 和 CUDA Include (针对用到它们的 Target)
foreach(TGT test_trtcli test_distgssr_trt test_distgdisp test_epit test_lfdepth test_lfsuperres)
    target_include_directories(${TGT} PRIVATE
        "${TENSORRT_DIR}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
    )
endforeach()

# 补充 OpenSSL Include & Link Dir (所有用到 SRCS_COMMON 的都需要 OpenSSL)
foreach(TGT test_openssl test_lfload test_lfrefocus test_config test_lfcalibrate test_isp test_lfresample test_distgssr_trt test_distgdisp test_epit test_lfdepth test_lfsuperres)
    target_include_directories(${TGT} PRIVATE ${OPENSSL_INCLUDE_DIR})
    target_link_directories(${TGT} PRIVATE "${OPENSSL_ROOT_DIR}/lib")
endforeach()
