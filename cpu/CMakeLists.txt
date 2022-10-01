add_executable(test_cpu
    test.cpp
)


add_executable(profile_cpu
    profile.cpp
)

# add_executable(search_cpu
#     search.cpp
# )

add_library(cpu_lib STATIC
    matmul.hpp
    matmul.cpp
)

set_target_properties(
    profile_cpu
PROPERTIES
    COMPILE_FLAGS "-S"
)

set_target_properties(
    cpu_lib
    profile_cpu
PROPERTIES
    COMPILE_FLAGS "-ftree-vectorize"
)

set_target_properties(
    test_cpu
    cpu_lib
    profile_cpu
PROPERTIES
    COMPILE_FLAGS "-DENABLE_AVX"
    COMPILE_FLAGS "-march=native"
)