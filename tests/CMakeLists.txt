cmake_minimum_required(VERSION 3.10)
project(test)

# test
add_executable(test test.cpp)

# test_ui
add_executable(test_ui
    test_ui.cpp
    ${UI_DIR}/ui.cpp
    ${UI_DIR}/ui.h
)
target_include_directories(test_ui PRIVATE ${CMAKE_SOURCE_DIR}/ui)
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

# test_load
add_executable(test_lfload
    test_lfload.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/lfload.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/utils.cpp
)
target_include_directories(test_lfload PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_directories(test_lfload PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_lfload PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
)

if(WIN32)
    # ===== Copy OpenSSL DLLs to output directory =====
    set(OPENSSL_DLLS
        "${OPENSSL_ROOT_DIR}/bin/libssl-3-x64.dll"
        "${OPENSSL_ROOT_DIR}/bin/libcrypto-3-x64.dll"
    )

    add_custom_command(TARGET test_lfload POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENSSL_DLLS}
        "$<TARGET_FILE_DIR:test_lfload>/"
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
        # --- Copy to Main Target Directory (test_lfload) ---
        add_custom_command(TARGET test_lfload POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS_TO_COPY}
            "$<TARGET_FILE_DIR:test_lfload>/"
            COMMENT "Copying OpenCV DLLs to test_lfload output directory..."
            VERBATIM
        )
    endif()

endif()


# test_lfrefocus
add_executable(test_lfrefocus
    test_lfrefocus.cpp
    ${SOURCE_DIR}/lfrefocus.cpp
    ${SOURCE_DIR}/lfload.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/utils.cpp
)
target_include_directories(test_lfrefocus PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_libraries(test_lfrefocus PRIVATE ${OpenCV_LIBS} OpenSSL::SSL OpenSSL::Crypto)

# add_executable(test_future test_future.cpp)

add_executable(test_readraw test_readraw.cpp)
target_link_libraries(test_readraw PRIVATE ${OpenCV_LIBS})
# message(STATUS "OpenCV LIBS: ${OpenCV_LIBS}")


add_executable(test_openssl
    test_openssl.cpp
    ${SOURCE_DIR}/lfload.cpp
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
)
target_include_directories(test_lfcalibrate PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR} ${Eigen_DIR})
target_link_directories(test_lfcalibrate PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_lfcalibrate PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
)

add_executable(test_json test_json.cpp)
target_include_directories(test_lfcalibrate PRIVATE ${SOURCE_DIR})

add_executable(test_isp
    test_isp.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/config.cpp
    ${SOURCE_DIR}/raw_decode.cpp
    ${SOURCE_DIR}/lfload.cpp
)
target_include_directories(test_isp PRIVATE ${SOURCE_DIR} ${OPENSSL_INCLUDE_DIR})
target_link_directories(test_isp PRIVATE "${OPENSSL_ROOT_DIR}/lib")
target_link_libraries(test_isp PRIVATE
    OpenSSL::SSL OpenSSL::Crypto
    ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudaarithm opencv_cudawarping
)
