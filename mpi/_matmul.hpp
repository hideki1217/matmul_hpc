#pragma once

#include <mpi.h>

#include <cstring>
#include <vector>

#include "../cpu/matmul.hpp"
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
                sub_a.data(), sub_a.size(), my_mpi::get_dtype<T>(), 0,
                MPI_COMM_WORLD);

    _mmmul(sM, N, K, sub_a.data(), b, sub_c.data());

    MPI_Gather(sub_c.data(), sub_c.size(), my_mpi::get_dtype<T>(), c,
               my_mpi::is_(0) ? sub_c.size() : 0, my_mpi::get_dtype<T>(), 0,
               MPI_COMM_WORLD);
  } else {
    static constexpr int MAX_PROC = 32;
    int sM;
    int each_sM[MAX_PROC];
    int each_displs[MAX_PROC];
    int each_size[MAX_PROC];
    if (my_mpi::is_(0)) {
      const int x = M / my_mpi::world_proc_num;
      const int m = M % my_mpi::world_proc_num;
      for (int i = 0; i < my_mpi::world_proc_num; i++) {
        const int row_n = x + (i < m);
        each_sM[i] = row_n;
      }
    }
    MPI_Scatter(each_sM, my_mpi::is_(0) ? 1 : 0, MPI_INT, &sM, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    std::vector<T> sub_a(sM * K);
    std::vector<T> sub_c(sM * N, 0);

    {
      int c = 0;
      for (int i = 0; i < my_mpi::world_proc_num; i++) {
        each_size[i] = each_sM[i] * K;
        each_displs[i] = c;

        c += each_size[i];
      }
    }
    MPI_Scatterv(a, each_size, each_displs, my_mpi::get_dtype<T>(),
                 sub_a.data(), sub_a.size(), my_mpi::get_dtype<T>(), 0,
                 MPI_COMM_WORLD);

    _mmmul(sM, N, K, sub_a.data(), b, sub_c.data());

    {
      int c = 0;
      for (int i = 0; i < my_mpi::world_proc_num; i++) {
        each_size[i] = each_sM[i] * N;
        each_displs[i] = c;

        c += each_size[i];
      }
    }
    MPI_Gatherv(sub_c.data(), sub_c.size(), my_mpi::get_dtype<T>(), c,
                each_size, each_displs, my_mpi::get_dtype<T>(), 0,
                MPI_COMM_WORLD);
  }
}

}  // namespace matmul_mpi_v1

namespace matmul_mpi_v2 {

template <typename T>
void mmmul(const usize M, const usize N, const usize K, const T *a, const T *b,
           T *c) {
  if (M % my_mpi::world_proc_num == 0) {
    const usize sM = M / my_mpi::world_proc_num;

    std::vector<T> sub_a(sM * K);
    std::vector<T> sub_c(sM * N, 0);

    MPI_Scatter(a, my_mpi::is_(0) ? sub_a.size() : 0, my_mpi::get_dtype<T>(),
                sub_a.data(), sub_a.size(), my_mpi::get_dtype<T>(), 0,
                MPI_COMM_WORLD);

    matmul_cpu::mmmul(sM, N, K, sub_a.data(), b, sub_c.data());

    MPI_Gather(sub_c.data(), sub_c.size(), my_mpi::get_dtype<T>(), c,
               my_mpi::is_(0) ? sub_c.size() : 0, my_mpi::get_dtype<T>(), 0,
               MPI_COMM_WORLD);
  } else {
    static constexpr int MAX_PROC = 32;
    int sM;
    int each_sM[MAX_PROC];
    int each_displs[MAX_PROC];
    int each_size[MAX_PROC];
    if (my_mpi::is_(0)) {
      const int x = M / my_mpi::world_proc_num;
      const int m = M % my_mpi::world_proc_num;
      for (int i = 0; i < my_mpi::world_proc_num; i++) {
        const int row_n = x + (i < m);
        each_sM[i] = row_n;
      }
    }
    MPI_Scatter(each_sM, my_mpi::is_(0) ? 1 : 0, MPI_INT, &sM, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    std::vector<T> sub_a(sM * K);
    std::vector<T> sub_c(sM * N, 0);

    MPI_Request requests[2];
    MPI_Status statuses[2];
    
    {
      int c = 0;
      for (int i = 0; i < my_mpi::world_proc_num; i++) {
        each_size[i] = each_sM[i] * K;
        each_displs[i] = c;

        c += each_size[i];
      }
    }
    MPI_Iscatterv(a, each_size, each_displs, my_mpi::get_dtype<T>(),
                 sub_a.data(), sub_a.size(), my_mpi::get_dtype<T>(), 0,
                 MPI_COMM_WORLD, &requests[0]);
    printf("%d: scatter recs [", my_mpi::world_proc_i);
    for(int i=0; i<sub_a.size(); i++) printf(" %d", (int)sub_a[i]);
    printf("]end\n");

    matmul_cpu::mmmul(sM, N, K, sub_a.data(), b, sub_c.data());
    printf("%d: calc [", my_mpi::world_proc_i);
    for(int i=0; i<sub_c.size(); i++) printf(" %d", (int)sub_c[i]);
    printf("]end\n");
    {
      int c = 0;
      for (int i = 0; i < my_mpi::world_proc_num; i++) {
        each_size[i] = each_sM[i] * N;
        each_displs[i] = c;

        c += each_size[i];
      }
    }
    MPI_Igatherv(sub_c.data(), sub_c.size(), my_mpi::get_dtype<T>(), c,
                each_size, each_displs, my_mpi::get_dtype<T>(), 0,
                MPI_COMM_WORLD, &requests[1]);

    MPI_Waitall(2, requests, statuses);
  }
}

}  // namespace matmul_mpi_v2