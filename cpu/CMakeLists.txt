add_executable(test_cpu
    test.cpp
)

add_executable(profile_cpu
    profile.cpp
)
set_target_properties(profile_cpu PROPERTIES COMPILE_FLAGS "-save-temps") 

# add_executable(search_cpu
#     search.cpp
# )

add_library(cpu_lib STATIC
    matmul.hpp
    matmul.cpp
)