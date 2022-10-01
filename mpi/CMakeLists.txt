find_package(MPI REQUIRED)

# SET(CMAKE_C_COMPILER ${MPI_C_COMPILER})
# SET(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

add_executable(mpi_sample sample.cpp)
target_link_libraries(mpi_sample PRIVATE MPI::MPI_CXX)
# target_include_directories(mpi_sample PRIVATE ${MPI_INCLUDE_PATH})