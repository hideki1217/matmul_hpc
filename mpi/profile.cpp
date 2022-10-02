#include <mpi.h>

#include <chrono>
#include <functional>
#include <memory>

#include "_matmul.hpp"
#include "_mpi_core.hpp"
#include "matmul_core.h"

using time_ms_t = int64_t;

template <typename T, typename F>
time_ms_t _mmmul_profile(const usize M, const usize N, const usize K, F mmmul) {
  auto a = std::make_unique<T[]>(M * K);
  auto b = std::make_unique<T[]>(K * N);
  auto c = std::make_unique<T[]>(M * N);

  for (int i = 0; i < M * K; i++) a[i] = i % 1024 - 512;
  for (int i = 0; i < K * N; i++) b[i] = i % 1024 - 512;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 50; i++) mmmul(M, N, K, a.get(), b.get(), c.get());
  auto time_obj = std::chrono::system_clock::now() - start;
  time_ms_t time =
      std::chrono::duration_cast<std::chrono::milliseconds>(time_obj).count();

  return time;
}

template <typename T, typename F>
time_ms_t mmmul_profile(const usize M, const usize N, const usize K, F f) {
  return _mmmul_profile<T>(M, N, K, std::function<mmmul_t<T>>(f));
}

#define timeit(exp)                          \
  {                                          \
    time_ms_t time = (exp);                  \
    print_once("%ld(ms): %s\n", time, #exp); \
  }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  my_mpi::init();
  {
    const usize N = 256;

#define do_short(target) \
  timeit(mmmul_profile<short>(N, N, N, target::mmmul<short>))
#define do_int(target) timeit(mmmul_profile<int>(N, N, N, target::mmmul<int>))
#define do_float(target) \
  timeit(mmmul_profile<float>(N, N, N, target::mmmul<float>))

    print_once("--- mmmul( Square matrix ) N=%d\n", N);

    do_short(matmul_mpi_v1);

    do_int(matmul_mpi_v1);

    do_float(matmul_mpi_v1);

#undef do_short
#undef do_int
#undef do_float
  }
  MPI_Finalize();
}