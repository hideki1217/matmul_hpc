#include <cstring>

#include "matmul_core.h"

namespace matmul_cpu_v1 {

template <typename T>
void mmmul(usize M, usize N, usize K, const T *a, const T *b, T *c) {
  std::memset(c, 0, sizeof(T) * M * N);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        c[i * N + j] += a[i * K + k] * b[k * N + j];
      }
    }
  }
}

template <typename T>
void mvmul(usize M, usize K, const T *a, const T *b, T *c) {
  std::memset(c, 0, sizeof(T) * M);

  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      c[i] += a[i * K + k] * b[k];
    }
  }
}
}  // namespace matmul_cpu_v1
