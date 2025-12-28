cmake_minimum_required(VERSION 3.10)
project(test)

# test
add_executable(test test.cpp)

# test_ui
# add_executable(test_ui
#     test_ui.cpp
#     ${UI_DIR}/ui.cpp
#     ${UI_DIR}/ui.h
# )
# target_include_directories(test_ui PRIVATE ${CMAKE_SOURCE_DIR}/ui)
# target_link_libraries(test_ui PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# test_superres
add_executable(test_lfsuperres
    test_lfsuperres.cpp
    ${SOURCE_DIR}/lfsr.cpp
    ${SOURCE_DIR}/trtwrapper.cpp
    ${SOURCE_DIR}/distgssr.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfmbase.cpp
    ${SOURCE_DIR}/lfsr_base.cpp
)
target_include_directories(test_lfsuperres PRIVATE
    ${SOURCE_DIR}
    "${TENSORRT_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_lfsuperres PRIVATE
    ${OpenCV_LIBS}
    opencv_dnn_superres
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    OpenSSL::SSL OpenSSL::Crypto
    CUDA::cudart
)

# test_load
add_executable(test_lfload
    test_lfload.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/utils.cpp
)
target_include_directories(test_lfload PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_directories(test_lfload PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_lfload PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    OpenMP::OpenMP_CXX
)


# test_lfrefocus
add_executable(test_lfrefocus
    test_lfrefocus.cpp
    ${SOURCE_DIR}/lfrefocus.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/utils.cpp
)
target_include_directories(test_lfrefocus PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_libraries(test_lfrefocus PRIVATE ${OpenCV_LIBS} OpenSSL::SSL OpenSSL::Crypto OpenMP::OpenMP_CXX)

# add_executable(test_future test_future.cpp)

add_executable(test_readraw test_readraw.cpp)
target_link_libraries(test_readraw PRIVATE ${OpenCV_LIBS})
# message(STATUS "OpenCV LIBS: ${OpenCV_LIBS}")


add_executable(test_openssl
    test_openssl.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/utils.cpp
)
find_package(OpenSSL REQUIRED)
target_include_directories(test_openssl PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_directories(test_openssl PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_openssl PRIVATE OpenSSL::SSL OpenSSL::Crypto ${OpenCV_LIBS})

add_executable(test_config test_config.cpp ${SOURCE_DIR}/config.cpp ${SOURCE_DIR}/utils.cpp)
target_include_directories(test_config PRIVATE ${SOURCE_DIR})
target_link_libraries(test_config PRIVATE ${OpenCV_LIBS} OpenSSL::SSL OpenSSL::Crypto)


add_executable(test_lfcalibrate
    test_lfcalibrate.cpp
    ${SOURCE_DIR}/centers_extract.cpp
    ${SOURCE_DIR}/centers_sort.cpp
    ${SOURCE_DIR}/hexgrid_fit.cpp
    ${SOURCE_DIR}/lfcalibrate.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/config.cpp

)
target_include_directories(test_lfcalibrate PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR} ${Eigen_DIR})
target_link_directories(test_lfcalibrate PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_lfcalibrate PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    OpenMP::OpenMP_CXX
)

add_executable(test_json test_json.cpp)
target_include_directories(test_lfcalibrate PRIVATE ${SOURCE_DIR})

add_executable(test_isp
    test_isp.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/lfcalibrate.cpp
    ${SOURCE_DIR}/centers_extract.cpp
    ${SOURCE_DIR}/centers_sort.cpp
    ${SOURCE_DIR}/hexgrid_fit.cpp
)
target_include_directories(test_isp PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR} ${Eigen_DIR})
target_link_directories(test_isp PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_isp PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
    OpenMP::OpenMP_CXX
)

add_executable(test_lfresample
    test_lfresample.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/lfcalibrate.cpp
    ${SOURCE_DIR}/centers_extract.cpp
    ${SOURCE_DIR}/centers_sort.cpp
    ${SOURCE_DIR}/hexgrid_fit.cpp
)
target_include_directories(test_lfresample PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR} ${Eigen_DIR})
target_link_directories(test_lfresample PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_lfresample PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm
    OpenMP::OpenMP_CXX
)


add_executable(test_trtcli test_trtcli.cpp)
target_include_directories(test_trtcli PRIVATE
    "${TENSORRT_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS} # 必须！否则 NvInfer.h 里找不到 cuda_runtime_api.h
)
target_link_libraries(test_trtcli PRIVATE
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    "${TENSORRT_DIR}/lib/nvonnxparser_10.lib"
    CUDA::cudart
)


add_executable(test_distgssr_onnx test_distgssr_onnx.cpp ${SOURCE_DIR}/utils.cpp)
target_include_directories(test_distgssr_onnx PRIVATE ${ORT_DIR}/include)
target_link_directories(test_distgssr_onnx PRIVATE ${ORT_DIR}/lib)
target_link_libraries(test_distgssr_onnx PRIVATE
    onnxruntime.lib
    ${OpenCV_LIBS}
)
if(WIN32)
    set(ORT_LIB_DIR "${ORT_DIR}/lib")

    add_custom_command(TARGET test_distgssr_onnx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:test_distgssr_onnx>"

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB_DIR}/onnxruntime_providers_shared.dll" # <--- 关键文件 1
        "$<TARGET_FILE_DIR:test_distgssr_onnx>"

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB_DIR}/onnxruntime_providers_cuda.dll" # <--- 关键文件 2 (CUDA支持)
        "$<TARGET_FILE_DIR:test_distgssr_onnx>"

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB_DIR}/onnxruntime_providers_tensorrt.dll" # <--- 关键文件 2 (CUDA支持)
        "$<TARGET_FILE_DIR:test_distgssr_onnx>"
    )
endif()

add_executable(test_distgssr_trt
    test_distgssr_trt.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/distgssr.cpp
    ${SOURCE_DIR}/trtwrapper.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfmbase.cpp
    ${SOURCE_DIR}/lfsr_base.cpp
)
target_include_directories(test_distgssr_trt PRIVATE
    "${TENSORRT_DIR}/include"
    "${OpenCV_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_distgssr_trt PRIVATE
    "${OpenCV_LIBS}"
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    CUDA::cudart
    OpenSSL::SSL OpenSSL::Crypto
    OpenMP::OpenMP_CXX
)

add_executable(test_epit
    test_epit.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/epit.cpp
    ${SOURCE_DIR}/trtwrapper.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfmbase.cpp
    ${SOURCE_DIR}/lfsr_base.cpp
)
target_include_directories(test_epit PRIVATE
    "${TENSORRT_DIR}/include"
    "${OpenCV_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_epit PRIVATE
    "${OpenCV_LIBS}"
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    CUDA::cudart
    OpenSSL::SSL OpenSSL::Crypto
    OpenMP::OpenMP_CXX
)

add_executable(test_distgdisp
    test_distgdisp.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/distgdisp.cpp
    ${SOURCE_DIR}/trtwrapper.cpp
    ${SOURCE_DIR}/lfmbase.cpp
    ${SOURCE_DIR}/lfdisp_base.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/lfio.cpp
)
target_include_directories(test_distgdisp PRIVATE
    "${TENSORRT_DIR}/include"
    "${OpenCV_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_distgdisp PRIVATE
    "${OpenCV_LIBS}"
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    CUDA::cudart
    OpenMP::OpenMP_CXX
    OpenSSL::SSL OpenSSL::Crypto
)

add_executable(test_lfdepth
    test_lfdepth.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/distgdisp.cpp
    ${SOURCE_DIR}/trtwrapper.cpp
    ${SOURCE_DIR}/lfmbase.cpp
    ${SOURCE_DIR}/lfdisp_base.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/lfio.cpp
    ${SOURCE_DIR}/lfdepth.cpp
)
target_include_directories(test_lfdepth PRIVATE
    "${TENSORRT_DIR}/include"
    "${OpenCV_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_lfdepth PRIVATE
    "${OpenCV_LIBS}"
    "${TENSORRT_DIR}/lib/nvinfer_10.lib"
    CUDA::cudart
    OpenMP::OpenMP_CXX
    OpenSSL::SSL OpenSSL::Crypto
)


