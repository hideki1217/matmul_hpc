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