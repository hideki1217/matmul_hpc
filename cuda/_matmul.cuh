#include <cuda_runtime_api.h>

#include <cassert>
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
#define div_up(a, b) (((a) + (b)-1) / (b))

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

namespace matmul_cuda_v2 {
template <typename T>
__device__ void mmmull_in_warp(const usize M, const usize N, const usize K,
                               const T *a, const int lda, const T *b,
                               const int ldb, T *c, const int ldc) {
  const int lane = threadIdx.x % warpSize;

  for (int i = 0; i < M; i++) {
    T c_i_l(0);

    for (int k = 0; k < K; k++) {
      const T a_i_k = a[i * lda + k];
      const T b_k_l = b[k * ldb + lane];

      c_i_l += a_i_k * b_k_l;
    }

    c[i * ldc + lane] += c_i_l;
  }
}

template <typename T, usize WarpSize>
__global__ void mmmull_kernel(const usize M, const usize N, const usize K,
                              const T *a, const T *b, T *c) {
  assert(WarpSize == blockDim.x);
  assert(warpSize == WarpSize);

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int warp_n = blockDim.y;
  const int j0 = warpSize * blockIdx.x;
  const int i0 = warpSize * (warp_n * blockIdx.y + warp);

  __shared__ T B[WarpSize][WarpSize];
  T *c_ = c + (i0 * N + j0);
  const usize M_ = min(warpSize, M - i0);
  const usize N_ = min(warpSize, N - j0);

  if (lane < N_) {
    for (int i = 0; i < M_; i++) {
      c_[i * N + lane] = 0;
    }
  }

  for (int bk = 0; bk < div_up(K, warpSize); bk++) {
    const int k0 = bk * warpSize;

    const T *a_ = a + (i0 * K + k0);
    const T *b_ = b + (k0 * N + j0);
    const usize K_ = min(warpSize, K - k0);

    if (lane < N_) {
      B[warp][lane] = b_[warp * N + lane];
    }

    // below is needless because run in warp
    __syncwarp();

    mmmull_in_warp(M_, N_, K_, a_, K, &B[0][0], warpSize, c_, N);
  }
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

    dim3 block(32, 1);
    dim3 grid(div_up(N, block.x), div_up(M, block.x * block.y));
    mmmull_kernel<T, 32><<<grid, block>>>(M, N, K, a_d, b_d, c_d);

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
}  // namespace matmul_cuda_v2

namespace matmul_cuda_v3 {
template <typename T, usize BN, usize BK>
__global__ void mmmull_kernel(const usize M, const usize N, const usize K,
                              const T *a, const T *b, T *c) {
  static_assert(BN == BK);

  const int j = threadIdx.x;
  const int i = threadIdx.y;
  const int x0 = blockDim.x * blockIdx.x;
  const int y0 = blockDim.y * blockIdx.y;

  __shared__ T sA[BN][BK];
  __shared__ T sB[BK][BN];

  T reg(0);
  for (int bk = 0; bk < div_up(K, BK); bk++) {
    sA[i][j] = a[(y0 + i) * K + (bk * BK + j)];
    sB[i][j] = b[(bk * BK + i) * N + (x0 + j)];

    __syncthreads();

    for (int k = 0; k < min(BK, K - bk * BK); k++) {
      reg += sA[i][k] * sB[k][j];
    }

    __syncthreads();
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

    static constexpr usize BN = 32;
    dim3 block(BN, BN);
    dim3 grid(div_up(N, BN), div_up(M, BN));
    mmmull_kernel<T, BN, BN><<<grid, block>>>(M, N, K, a_d, b_d, c_d);

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
}  // namespace matmul_cuda_v3

namespace matmul_cuda_v4 {
template <typename T, usize BN, usize BK, usize RPT>
__global__ void mmmull_kernel(const usize M, const usize N, const usize K,
                              const T *a, const T *b, T *c) {
  static_assert(BN == BK);
  static_assert(BN % RPT == 0);

  const int j = threadIdx.x;
  const int i = threadIdx.y;
  const int x0 = blockDim.x * blockIdx.x;
  const int y0 = blockDim.y * RPT * blockIdx.y;

  __shared__ T sA[BN][BK];
  __shared__ T sB[BK][BN];

  T reg[RPT];
  for(int i=0; i<RPT; i++) reg[i] = 0;

  for (int bk = 0; bk < div_up(K, BK); bk++) {
    for(int r=0; r<RPT; r++){
      sA[i * RPT + r][j] = a[(y0 + i * RPT + r) * K + (bk * BK + j)];
      sB[i * RPT + r][j] = b[(bk * BK + i * RPT + r) * N + (x0 + j)];
    }
    __syncthreads();

    for (int k = 0; k < min(BK, K - bk * BK); k++) {
      for(int r=0; r<RPT; r++){
        reg[r] += sA[i * RPT + r][k] * sB[k][j];
      }
    }

    __syncthreads();
  }

  for(int r=0; r<RPT; r++){
    const int i_r = i * RPT + r;
    if (i_r < M && j < N) c[i_r * N + j] = reg[r];
  }
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

    static constexpr usize BN = 32;
    static constexpr usize RPT = 8;
    dim3 block(BN, BN/RPT);
    dim3 grid(div_up(N, BN), div_up(M, BN));
    mmmull_kernel<T, BN, BN, RPT><<<grid, block>>>(M, N, K, a_d, b_d, c_d);

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
}  // namespace matmul_cuda_v4