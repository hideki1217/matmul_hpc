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
  for (int i = 0; i < 10; i++) mmmul(M, N, K, a.get(), b.get(), c.get());
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
  for (int i = 0; i < 10; i++) mvmul(M, K, a.get(), b.get(), c.get());
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
    timeit(mmmul_profile<int>(N, N, N, matmul_cpu_v1::mmmul<int>));

    timeit(mmmul_profile<float>(N, N, N, matmul_cpu_v1::mmmul<float>));
  }
  {
    const usize N = 2048;
    timeit(mvmul_profile<int>(N, N, matmul_cpu_v1::mvmul<int>));

    timeit(mvmul_profile<float>(N, N, matmul_cpu_v1::mvmul<float>));
  }
}