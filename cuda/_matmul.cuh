#include <cuda_runtime_api.h>

#include <iostream>

#include "matmul_core.h"

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
static void checkLast(const char *const file, const int line) {
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    // We don't exit when we encounter CUDA errors in this example.
    // std::exit(EXIT_FAILURE);
  }
}
#define div_up(a, b) (((a) + (b) - 1)/(b))

namespace matmul_cuda_v1 {
template <typename T>
__global__ void mmmull_kernel(const usize M, const usize N, const usize K,
                              const T *a, const T *b, T *c) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  T reg(0);
  for (int k = 0; k < K; k++) {
    reg += a[i * K + k] * b[k * N + j];
  }
  if (i < M && j < N) c[i * N + j] = reg;
}
template <typename T>
class MMmull {
 public:
  MMmull(const usize maxM, const usize maxN, const usize maxK)
      : max_a_byte(sizeof(T) * maxM * maxK),
        max_b_byte(sizeof(T) * maxK * maxN),
        max_c_byte(sizeof(T) * maxM * maxN) {
    cudaMalloc(&a_d, max_a_byte);
    cudaMalloc(&b_d, max_b_byte);
    cudaMalloc(&c_d, max_c_byte);
    CHECK_LAST_CUDA_ERROR();
  }
  ~MMmull() {
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    CHECK_LAST_CUDA_ERROR();
  }
  void mmmul(const usize M, const usize N, const usize K, const T *a,
             const T *b, T *c) {
    const usize a_byte = M * K * sizeof(T);
    const usize b_byte = K * N * sizeof(T);
    const usize c_byte = M * N * sizeof(T);

    cudaMemcpyAsync(a_d, a, a_byte, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(b_d, b, b_byte, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(div_up(N, block.x), div_up(M, block.y));
    mmmull_kernel<T><<<grid, block>>>(M, N, K, a_d, b_d, c_d);

    cudaMemcpyAsync(c, c_d, c_byte, cudaMemcpyDeviceToHost);
  }
  const usize max_a_byte;
  const usize max_b_byte;
  const usize max_c_byte;

 private:
  T *a_d;
  T *b_d;
  T *c_d;
};
}  // namespace matmul_cuda_v1