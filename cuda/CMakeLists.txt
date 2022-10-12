find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_EXECUTABLE(cuda_test
        test.cu
    )
    target_link_libraries(cuda_test cpu_lib)
    include_directories(cuda_test ../cpu)

    CUDA_ADD_EXECUTABLE(cuda_profile
        profile.cu
    )
endif()

add_test(NAME cuda_test COMMAND cuda_test)