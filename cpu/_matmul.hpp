#include <algorithm>
#include <cstring>

#include "matmul_core.h"

#define to_str(x_) #x_
#define asm_log(message) asm volatile ("#" asm_log_NAME ": " #message)
#define asm_log_start() asm_log(start)
#define asm_log_end() asm_log(end)


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
  for (int i = 0; i < M; i++) {
    int k;
    T reg(0);
    const T *a_row = a + (i * K);

    for (k = 0; k <= K - 4; k += 4) {
      T reg0 = a_row[k] * b[k];
      T reg1 = a_row[k + 1] * b[k + 1];
      T reg2 = a_row[k + 2] * b[k + 2];
      T reg3 = a_row[k + 3] * b[k + 3];

      reg += reg0 + reg1 + reg2 + reg3;
    }
    switch (K - k) {
      case 3: {
        T reg0 = a_row[k + 2] * b[k + 2];
        T reg1 = a_row[k + 1] * b[k + 1];
        T reg2 = a_row[k] * b[k];

        reg += reg0 + reg1 + reg2;
        break;
      }
      case 2: {
        T reg0 = a_row[k + 1] * b[k + 1];
        T reg1 = a_row[k] * b[k];

        reg += reg0 + reg1;
        break;
      }
      case 1:
        reg += a_row[k] * b[k];
        break;
    }

    c[i] = reg;
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

      for (j = 0; j <= N - 4; j += 4) {
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
  for (int i = 0; i < M; i++) {
    int k;
    T reg(0);
    const T *a_row = a + i * K;

    for (k = 0; k <= K - 4; k += 4) {
      T vreg[4];

      vreg[0] = a_row[k] * b[k];
      vreg[1] = a_row[k + 1] * b[k + 1];
      vreg[2] = a_row[k + 2] * b[k + 2];
      vreg[3] = a_row[k + 3] * b[k + 3];

      reg += vreg[0] + vreg[1] + vreg[2] + vreg[3];
    }
    switch (K - k) {
      case 3: {
        T reg0 = a_row[k + 2] * b[k + 2];
        T reg1 = a_row[k + 1] * b[k + 1];
        T reg2 = a_row[k] * b[k];

        reg += reg0 + reg1 + reg2;
        break;
      }
      case 2: {
        T reg0 = a_row[k + 1] * b[k + 1];
        T reg1 = a_row[k] * b[k];

        reg += reg0 + reg1;
        break;
      }
      case 1:
        reg += a_row[k] * b[k];
        break;
    }
    c[i] = reg;
  }
}
}  // namespace matmul_cpu_v3

namespace matmul_cpu_v4 {

template <typename T>
inline static constexpr T div_up(T a, T b) {
  return (a + b - 1) / b;
}

/**
 * @brief mmmul by submatrix multiply
 * This method has an advantage on cache efficincy
 *
 * @tparam sM A's submatrix row_n
 * @tparam sN A's submatrix col_n and B's submatrix row_n
 * @tparam sK B's submatrix row_n
 */
template <usize sM, usize sN = sM, usize sK = sM, typename T>
void mmmul_by_submatrix(usize M, usize N, usize K, const T *a, const T *b,
                        T *c) {
  std::memset(c, 0, sizeof(T) * M * N);

  T A[sM][sK];
  T B_T[sN][sK];

  const int b_i_sup = div_up(M, sM);
  const int b_j_sup = div_up(N, sN);
  const int b_k_sup = div_up(K, sK);
  for (int b_i = 0; b_i < b_i_sup; b_i++) {
    for (int b_j = 0; b_j < b_j_sup; b_j++) {
      for (int b_k = 0; b_k < b_k_sup; b_k++) {
        const int c_0 = b_i * N * sM + b_j * sN;
        const int a_0 = b_i * K * sM + b_k * sK;
        const int b_0 = b_k * N * sK + b_j * sN;

        const int i_sup = std::min(sM, M - b_i * sM);
        const int k_sup = std::min(sK, K - b_k * sK);
        const int j_sup = std::min(sN, N - b_j * sN);

        for (int i = 0; i < i_sup; i++) {
          for (int k = 0; k < k_sup; k++) {
            A[i][k] = a[a_0 + (i * K + k)];
          }
        }
        for (int k = 0; k < k_sup; k++) {
          for (int j = 0; j < j_sup; j++) {
            B_T[j][k] = b[b_0 + (k * N + j)];
          }
        }
        {
          for (int i = 0; i < i_sup; i++) {
            for (int j = 0; j < j_sup; j++) {
              T reg(0);
              for (int k = 0; k < k_sup; k++) {
                reg += A[i][k] * B_T[j][k];
              }
              c[c_0 + (i * N + j)] += reg;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void mmmul(usize M, usize N, usize K, const T *a, const T *b, T *c) {
  mmmul_by_submatrix<16>(M, N, K, a, b, c);
}

#define mmull_define(T_, sM_, sN_, sK_)                                        \
  template <>                                                                  \
  void mmmul<T_>(usize M, usize N, usize K, const T_ *a, const T_ *b, T_ *c) { \
    mmmul_by_submatrix<sM_, sN_, sK_>(M, N, K, a, b, c);                       \
  }

mmull_define(int16_t, 32, 32, 32);
mmull_define(int32_t, 16, 16, 16);
mmull_define(int64_t, 16, 16, 16);
mmull_define(float, 16, 16, 16);

template <typename T>
void mvmul(usize M, usize K, const T *a, const T *b, T *c) {
  for (int i = 0; i < M; i++) {
    const T *a_row = a + i * K;

    T reg(0);
    for (int k = 0; k < K; k++) {
      reg += a_row[k] * b[k];
    }
    c[i] = reg;
  }
}
}  // namespace matmul_cpu_v4

namespace matmul_cpu_v5 {

template <typename T>
inline static constexpr T div_up(T a, T b) {
  return (a + b - 1) / b;
}

/**
 * @brief mmmul by submatrix multiply
 * This method has an advantage on cache efficincy
 *
 * @tparam sM A's submatrix row_n
 * @tparam sN A's submatrix col_n and B's submatrix row_n
 * @tparam sK B's submatrix row_n
 * @param M must be multiple of sM
 * @param N must be multiple of sN
 * @param K must be multiple of sK
 */
template <usize sM, usize sN = sM, usize sK = sM, typename T>
void mmmul_by_submatrix_unsafe(usize M, usize N, usize K, const T *a,
                               const T *b, T *c) {
  std::memset(c, 0, sizeof(T) * M * N);

  T A[sM][sK];
  T B_T[sN][sK];

  const int b_i_sup = M / sM;
  const int b_j_sup = N / sN;
  const int b_k_sup = K / sK;
  for (int b_i = 0; b_i < b_i_sup; b_i++) {
    for (int b_j = 0; b_j < b_j_sup; b_j++) {
      for (int b_k = 0; b_k < b_k_sup; b_k++) {
        const int c_0 = b_i * N * sM + b_j * sN;
        const int a_0 = b_i * K * sM + b_k * sK;
        const int b_0 = b_k * N * sK + b_j * sN;

#pragma unroll
        for (int i = 0; i < sM; i++) {
          T *A_ = A[i];
          const T *a_ = a + (a_0 + i * K);
#pragma unroll
          for (int k = 0; k < sK; k++) {
            A_[k] = a_[k];
          }
        }
#pragma unroll
        for (int k = 0; k < sK; k++) {
#pragma unroll
          for (int j = 0; j < sN; j++) {
            B_T[j][k] = b[b_0 + (k * N + j)];
          }
        }
#pragma unroll
        for (int i = 0; i < sM; i++) {
          const T *A_ = A[i];

#pragma unroll
          for (int j = 0; j < sN; j++) {
            const T *B_T_ = B_T[j];

            T reg(0);
#pragma unroll
            for (int k = 0; k < sK; k++) {
              reg += A_[k] * B_T_[k];
            }
            c[c_0 + (i * N + j)] += reg;
          }
        }
      }
    }
  }
}

template <usize sM, usize sN = sM, usize sK = sM, typename T>
void mmmul_by_submatrix(usize M, usize N, usize K, const T *a, const T *b,
                        T *c) {
  if (M % sM || N % sN || K % sK)
    matmul_cpu_v4::mmmul_by_submatrix<sM, sN, sK>(M, N, K, a, b, c);
  else
    mmmul_by_submatrix_unsafe<sM, sN, sK>(M, N, K, a, b, c);
}

template <typename T>
void mmmul(usize M, usize N, usize K, const T *a, const T *b, T *c) {
  mmmul_by_submatrix<16>(M, N, K, a, b, c);
}

#define mmull_define(T_, sM_, sN_, sK_)                                        \
  template <>                                                                  \
  void mmmul<T_>(usize M, usize N, usize K, const T_ *a, const T_ *b, T_ *c) { \
    mmmul_by_submatrix<sM_, sN_, sK_>(M, N, K, a, b, c);                       \
  }

mmull_define(int16_t, 64, 64, 64);
mmull_define(int32_t, 32, 32, 32);
mmull_define(float, 8, 8, 8);

template <typename T>
void mvmul(usize M, usize K, const T *a, const T *b, T *c) {
  matmul_cpu_v4::mvmul(M, K, a, b, c);
}
}  // namespace matmul_cpu_v5