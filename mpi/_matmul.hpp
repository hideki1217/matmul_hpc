#pragma once

#include <mpi.h>

#include <cstring>
#include <vector>

#include "./_mpi_core.hpp"
#include "matmul_core.h"

namespace matmul_mpi_v1 {

template <typename T>
void _mmmul(const usize M, const usize N, const usize K, const T *a, const T *b,
            T *c) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T reg(0);
      for (int k = 0; k < K; k++) {
        reg += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = reg;
    }
  }
}

template <typename T>
void mmmul(const usize M, const usize N, const usize K, const T *a, const T *b,
           T *c) {
  if (M % my_mpi::world_proc_num == 0) {
    const usize sM = M / my_mpi::world_proc_num;

    std::vector<T> sub_a(sM * K);
    std::vector<T> sub_c(sM * N, 0);

    MPI_Scatter(a, my_mpi::is_(0) ? sub_a.size() : 0, my_mpi::get_dtype<T>(),
                &sub_a[0], sub_a.size(), my_mpi::get_dtype<T>(), 0,
                MPI_COMM_WORLD);

    _mmmul(sM, N, K, &sub_a[0], b, &sub_c[0]);

    MPI_Gather(&sub_c[0], sub_c.size(), my_mpi::get_dtype<T>(), c,
               my_mpi::is_(0) ? sub_c.size() : 0, my_mpi::get_dtype<T>(), 0,
               MPI_COMM_WORLD);
  } else {
    const usize sM = (M + my_mpi::world_proc_num - 1) / my_mpi::world_proc_num;

    std::vector<T> sub_a(sM * K);
    std::vector<T> sub_c(sM * N, 0);

    std::vector<T> c_(my_mpi::is_(0) ? my_mpi::world_proc_num * sub_c.size()
                                     : 0);

    MPI_Scatter(a, my_mpi::is_(0) ? sub_a.size() : 0, my_mpi::get_dtype<T>(),
                &sub_a[0], sub_a.size(), my_mpi::get_dtype<T>(), 0,
                MPI_COMM_WORLD);

    _mmmul(sM, N, K, &sub_a[0], b, &sub_c[0]);

    MPI_Gather(&sub_c[0], sub_c.size(), my_mpi::get_dtype<T>(), &c_[0],
               my_mpi::is_(0) ? sub_c.size() : 0, my_mpi::get_dtype<T>(), 0,
               MPI_COMM_WORLD);

    std::memcpy(c, &c_[0], sizeof(T) * M * N);
  }
}

}  // namespace matmul_mpi_v1