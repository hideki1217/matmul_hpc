#include <chrono>
#include <functional>
#include <iostream>
#include <memory>

#include "./_matmul.hpp"

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
time_ms_t _mvmul_profile(const usize M, const usize K, F mvmul) {
  auto a = std::make_unique<T[]>(M * K);
  auto b = std::make_unique<T[]>(K);
  auto c = std::make_unique<T[]>(M);

  for (int i = 0; i < M * K; i++) a[i] = i % 1024 - 512;
  for (int i = 0; i < K; i++) b[i] = i % 1024 - 512;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 50; i++) mvmul(M, K, a.get(), b.get(), c.get());
  auto time_obj = std::chrono::system_clock::now() - start;
  time_ms_t time =
      std::chrono::duration_cast<std::chrono::milliseconds>(time_obj).count();

  return time;
}

template <typename T, typename F>
time_ms_t mmmul_profile(const usize M, const usize N, const usize K, F f) {
  return _mmmul_profile<T>(M, N, K, std::function<mmmul_t<T>>(f));
}
template <typename T, typename F>
time_ms_t mvmul_profile(const usize M, const usize K, F f) {
  return _mvmul_profile<T>(M, K, std::function<mvmul_t<T>>(f));
}

#define timeit(exp)                                     \
  {                                                     \
    time_ms_t time = (exp);                             \
    std::cout << time << "(ms): " << #exp << std::endl; \
  }

int main() {
  // square matrix
  {
    const usize N = 256;
#define do_short(target) \
  timeit(mmmul_profile<short>(N, N, N, target::mmmul<short>))
#define do_int(target) timeit(mmmul_profile<int>(N, N, N, target::mmmul<int>))
#define do_float(target) \
  timeit(mmmul_profile<float>(N, N, N, target::mmmul<float>))

    std::cout << "--- mmmul( Square matrix ) N=" << N << std::endl;

    do_short(matmul_cpu_v1);
    do_short(matmul_cpu_v2);
    do_short(matmul_cpu_v3);
    do_short(matmul_cpu_v4);

    do_int(matmul_cpu_v1);
    do_int(matmul_cpu_v2);
    do_int(matmul_cpu_v3);
    do_int(matmul_cpu_v4);


    do_float(matmul_cpu_v1);
    do_float(matmul_cpu_v2);
    do_float(matmul_cpu_v3);
    do_float(matmul_cpu_v4);

#undef do_short
#undef do_int
#undef do_float
  }
  {
    const usize N = 1024;
#define do_short(target) \
  timeit(mvmul_profile<short>(N, N, target::mvmul<short>))
#define do_int(target) timeit(mvmul_profile<int>(N, N, target::mvmul<int>))
#define do_float(target) \
  timeit(mvmul_profile<float>(N, N, target::mvmul<float>))

    std::cout << "--- mvmul( Square matrix ) N=" << N << std::endl;

    do_short(matmul_cpu_v1);
    do_short(matmul_cpu_v2);
    do_short(matmul_cpu_v3);
    do_short(matmul_cpu_v4);

    do_int(matmul_cpu_v1);
    do_int(matmul_cpu_v2);
    do_int(matmul_cpu_v3);
    do_int(matmul_cpu_v4);


    do_float(matmul_cpu_v1);
    do_float(matmul_cpu_v2);
    do_float(matmul_cpu_v3);
    do_float(matmul_cpu_v4);

#undef do_short
#undef do_int
#undef do_float
  }
}