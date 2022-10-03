#pragma once

#include <mpi.h>

#include <stdexcept>

namespace my_mpi {
int world_proc_i;
int world_proc_num;

inline bool is_(int i){ return world_proc_i == i; }

template <typename T>
MPI_Datatype get_dtype() {
  throw std::runtime_error();
}
#define define_get_dtype(type_, dtype_) \
  template <>                           \
  MPI_Datatype get_dtype<type_>() {     \
    return dtype_;                      \
  }

define_get_dtype(double, MPI_DOUBLE);
define_get_dtype(float, MPI_FLOAT);
define_get_dtype(int32_t, MPI_INT);
define_get_dtype(int16_t, MPI_SHORT);

#undef define_get_dtype

#define printf_once(fmt, ...) if ( my_mpi::world_proc_i == 0 ) std::printf(fmt, ##__VA_ARGS__)

class MPI_Core {
public:
  MPI_Core(int *argc, char ***argv) {
    MPI_Init(argc, argv);
  }
  void get_ready() {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_proc_i);
    MPI_Comm_size(MPI_COMM_WORLD, &world_proc_num);
  }
  ~MPI_Core() {
    MPI_Finalize();
  }
};

}  // namespace my_mpi