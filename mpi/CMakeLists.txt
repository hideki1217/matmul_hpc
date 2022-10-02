find_package(MPI REQUIRED)

add_executable(sample_mpi 
    sample.cpp
    )
target_link_libraries(sample_mpi PRIVATE MPI::MPI_CXX)

add_executable(test_mpi 
    test.cpp
    )
target_link_libraries(test_mpi PRIVATE MPI::MPI_CXX)

add_executable(profile_mpi 
    profile.cpp
    )
target_link_libraries(profile_mpi PRIVATE MPI::MPI_CXX)

enable_testing()

add_test(NAME test_mpi COMMAND mpirun -n 4 test_mpi)