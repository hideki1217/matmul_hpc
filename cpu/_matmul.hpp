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

namespace matmul_cpu_v2 {

template <typename T>
void mmmul(usize M, usize N, usize K, const T *a, const T *b, T *c) {
  std::memset(c, 0, sizeof(T) * M * N);

  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        c[i * N + j] += a[i * K + k] * b[k * N + j];
      }
    }
  }
}

template <typename T>
void mvmul(usize M, usize K, const T *a, const T *b, T *c) {
  std::memset(c, 0, sizeof(T) * M);

  for (int i = 0; i < M; i++) {
    int k;
    for (k = 0; k < K - 4; k += 4) {
      T reg0 = a[i * K + k] * b[k];
      T reg1 = a[i * K + k + 1] * b[k + 1];
      T reg2 = a[i * K + k + 2] * b[k + 2];
      T reg3 = a[i * K + k + 3] * b[k + 3];

      c[i] += reg0 + reg1 + reg2 + reg3;
    }
    switch (K - k) {
      case 3: {
        T reg0 = a[i * K + k + 2] * b[k + 2];
        T reg1 = a[i * K + k + 1] * b[k + 1];
        T reg2 = a[i * K + k] * b[k];

        c[i] += reg0 + reg1 + reg2;
        break;
      }
      case 2: {
        T reg0 = a[i * K + k + 1] * b[k + 1];
        T reg1 = a[i * K + k] * b[k];

        c[i] += reg0 + reg1;
        break;
      }
      case 1:
        c[i] += a[i * K + k] * b[k];
        break;
    }
  }
}
}  // namespace matmul_cpu_v2

namespace matmul_cpu_v3 {

template <typename T>
void mmmul(usize M, usize N, usize K, const T *a, const T *b, T *c) {
  std::memset(c, 0, sizeof(T) * M * N);

  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      int j;
      const T a_ik = a[i * K + k];

      for (j = 0; j < N - 4; j += 4) {
        c[i * N + j] += a_ik * b[k * N + j];
        c[i * N + j + 1] += a_ik * b[k * N + j + 1];
        c[i * N + j + 2] += a_ik * b[k * N + j + 2];
        c[i * N + j + 3] += a_ik * b[k * N + j + 3];
      }

      switch (N - j) {
        case 3:
          c[i * N + j] += a_ik * b[k * N + j];
          c[i * N + j + 1] += a_ik * b[k * N + j + 1];
          c[i * N + j + 2] += a_ik * b[k * N + j + 2];
          break;
        case 2:
          c[i * N + j] += a_ik * b[k * N + j];
          c[i * N + j + 1] += a_ik * b[k * N + j + 1];
          break;
        case 1:
          c[i * N + j] += a_ik * b[k * N + j];
          break;
      }
    }
  }
}
template <typename T>
void mvmul(usize M, usize K, const T *a, const T *b, T *c) {
  std::memset(c, 0, sizeof(T) * M);

  for (int i = 0; i < M; i++) {
    int k;

    for (k = 0; k < K - 4; k += 4) {
      T vreg[4];

      vreg[0] = a[i * K + k] * b[k];
      vreg[1] = a[i * K + k + 1] * b[k + 1];
      vreg[2] = a[i * K + k + 2] * b[k + 2];
      vreg[3] = a[i * K + k + 3] * b[k + 3];

      c[i] += vreg[0] + vreg[1] + vreg[2] + vreg[3];
    }
    switch (K - k) {
      case 3: {
        T reg0 = a[i * K + k + 2] * b[k + 2];
        T reg1 = a[i * K + k + 1] * b[k + 1];
        T reg2 = a[i * K + k] * b[k];

        c[i] += reg0 + reg1 + reg2;
        break;
      }
      case 2: {
        T reg0 = a[i * K + k + 1] * b[k + 1];
        T reg1 = a[i * K + k] * b[k];

        c[i] += reg0 + reg1;
        break;
      }
      case 1:
        c[i] += a[i * K + k] * b[k];
        break;
    }
  }
}
}  // namespace matmul_cpu_v3