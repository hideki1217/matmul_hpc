cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(matmul_hpc VERSION 1.0)

enable_testing()

include_directories(include)

add_subdirectory(cpu)
add_subdirectory(mpi)