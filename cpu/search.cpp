#include <chrono>
#include <iostream>
#include <memory>

#include "./_matmul.hpp"

template <typename T>
void v4_search() {
  const char* name = "mmull_by_submatrix";
  std::cout << name << ": start" << std::endl;

  int64_t time_min = 100000;
  const char* func_name;

#define time_op(func, sM, sN, sK)                                            \
  {                                                                          \
    auto start = std::chrono::system_clock::now();                           \
    for (int i = 0; i < 50; i++)                                             \
      func<sM, sN, sK>(M, N, K, a.get(), b.get(), c.get());                  \
    auto time = std::chrono::system_clock::now() - start;                    \
    auto time_ms =                                                           \
        std::chrono::duration_cast<std::chrono::milliseconds>(time).count(); \
    if (time_min > time_ms) {                                                \
      func_name = #func "<" #sM "," #sN "," #sK ">";                         \
      time_min = time_ms;                                                    \
    }                                                                        \
  }
#define apply(sM, sN, sK) \
  time_op(matmul_cpu_v4::mmmul_by_submatrix, sM, sN, sK);
  {
    const usize M = 256;
    const usize N = 256;
    const usize K = 256;
    auto a = std::make_unique<T[]>(M * K);
    auto b = std::make_unique<T[]>(K * N);
    auto c = std::make_unique<T[]>(M * N);

    for (int i = 0; i < M * K; i++) a[i] = i % 1024 - 512;
    for (int i = 0; i < K * N; i++) b[i] = i % 1024 - 512;

    apply(4, 4, 4);
    apply(4, 4, 8);
    apply(4, 4, 16);
    apply(4, 4, 32);
    apply(4, 8, 4);
    apply(4, 8, 8);
    apply(4, 8, 16);
    apply(4, 8, 32);
    apply(8, 16, 4);
    apply(8, 16, 8);
    apply(8, 16, 16);
    apply(8, 16, 32);
    apply(8, 32, 4);
    apply(8, 32, 8);
    apply(8, 32, 16);
    apply(8, 32, 32);
    apply(16, 4, 4);
    apply(16, 4, 8);
    apply(16, 4, 16);
    apply(16, 4, 32);
    apply(16, 8, 4);
    apply(16, 8, 8);
    apply(16, 8, 16);
    apply(16, 8, 32);
    apply(16, 16, 4);
    apply(16, 16, 8);
    apply(16, 16, 16);
    apply(16, 16, 32);
    apply(16, 32, 4);
    apply(16, 32, 8);
    apply(16, 32, 16);
    apply(16, 32, 32);
    apply(32, 4, 4);
    apply(32, 4, 8);
    apply(32, 4, 16);
    apply(32, 4, 32);
    apply(32, 8, 4);
    apply(32, 8, 8);
    apply(32, 8, 16);
    apply(32, 8, 32);
    apply(32, 16, 4);
    apply(32, 16, 8);
    apply(32, 16, 16);
    apply(32, 16, 32);
    apply(32, 32, 4);
    apply(32, 32, 8);
    apply(32, 32, 16);
    apply(32, 32, 32);
    std::cout << "\t" << M << ", " << N << ", " << K << ": " << func_name
              << std::endl;
  }
  {
    const usize M = 128;
    const usize N = 512;
    const usize K = 256;
    auto a = std::make_unique<T[]>(M * K);
    auto b = std::make_unique<T[]>(K * N);
    auto c = std::make_unique<T[]>(M * N);

    for (int i = 0; i < M * K; i++) a[i] = i % 1024 - 512;
    for (int i = 0; i < K * N; i++) b[i] = i % 1024 - 512;

    apply(4, 4, 4);
    apply(4, 4, 8);
    apply(4, 4, 16);
    apply(4, 4, 32);
    apply(4, 8, 4);
    apply(4, 8, 8);
    apply(4, 8, 16);
    apply(4, 8, 32);
    apply(8, 16, 4);
    apply(8, 16, 8);
    apply(8, 16, 16);
    apply(8, 16, 32);
    apply(8, 32, 4);
    apply(8, 32, 8);
    apply(8, 32, 16);
    apply(8, 32, 32);
    apply(16, 4, 4);
    apply(16, 4, 8);
    apply(16, 4, 16);
    apply(16, 4, 32);
    apply(16, 8, 4);
    apply(16, 8, 8);
    apply(16, 8, 16);
    apply(16, 8, 32);
    apply(16, 16, 4);
    apply(16, 16, 8);
    apply(16, 16, 16);
    apply(16, 16, 32);
    apply(16, 32, 4);
    apply(16, 32, 8);
    apply(16, 32, 16);
    apply(16, 32, 32);
    apply(32, 4, 4);
    apply(32, 4, 8);
    apply(32, 4, 16);
    apply(32, 4, 32);
    apply(32, 8, 4);
    apply(32, 8, 8);
    apply(32, 8, 16);
    apply(32, 8, 32);
    apply(32, 16, 4);
    apply(32, 16, 8);
    apply(32, 16, 16);
    apply(32, 16, 32);
    apply(32, 32, 4);
    apply(32, 32, 8);
    apply(32, 32, 16);
    apply(32, 32, 32);
    std::cout << "\t" << M << ", " << N << ", " << K << ": " << func_name
              << std::endl;
  }

  std::cout << name << ": end" << std::endl;
}

int main() {
  v4_search<short>();
  v4_search<int>();
  v4_search<long>();
  v4_search<float>();
  v4_search<double>();
}