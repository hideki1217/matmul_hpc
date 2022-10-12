#include <chrono>
#include <functional>
#include <iostream>
#include <memory>

#include "_matmul.cuh"

using time_ms_t = int64_t;

template <typename T, typename F>
double mmmul_profile(const usize M, const usize N, const usize K, F mmmul, int m=50) {
  auto a = std::make_unique<T[]>(M * K);
  auto b = std::make_unique<T[]>(K * N);
  auto c = std::make_unique<T[]>(M * N);

  for (int i = 0; i < M * K; i++) a[i] = i % 1024 - 512;
  for (int i = 0; i < K * N; i++) b[i] = i % 1024 - 512;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < m; i++) mmmul.mmmul(M, N, K, a.get(), b.get(), c.get());
  auto time_obj = std::chrono::system_clock::now() - start;
  auto time =
      std::chrono::duration_cast<std::chrono::milliseconds>(time_obj).count();

  const double op_n = M * N * K * m;
  const double gflops = op_n / (time / 1000.0) / 1e9;

  return gflops;
}

#define gflopit(exp) \
 { \
    auto gflops = (exp); \
    std::cout << gflops << "(GFlops): " << #exp << std::endl; \
 } 

int main() {
  // square matrix
  {
    const usize N = 1024;
#define do_short(target) \
  gflopit(mmmul_profile<short>(N, N, N, target::MMmull<int16_t>(N, N, N)))
#define do_int(target) gflopit(mmmul_profile<int>(N, N, N, target::MMmull<int32_t>(N, N, N)))
#define do_float(target) \
  gflopit(mmmul_profile<float>(N, N, N, target::MMmull<float>(N, N, N), 1))

    std::cout << "--- mmmul( Square matrix ) N=" << N << std::endl;

    do_short(matmul_cuda_v1);
    do_short(matmul_cuda_v2);

    do_int(matmul_cuda_v1);
    do_int(matmul_cuda_v2);

    do_float(matmul_cuda_v1);
    do_float(matmul_cuda_v2);

#undef do_short
#undef do_int
#undef do_float
  }
}