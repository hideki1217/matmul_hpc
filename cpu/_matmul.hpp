#include <algorithm>
#include <cassert>
#include <cstring>

#include "matmul_core.h"

#define to_str(x_) #x_
#define asm_log(message) asm volatile("#" asm_log_NAME ": " #message)
#define asm_log_start() asm_log(start)
#define asm_log_end() asm_log(end)

#define is_aligned(POINTER, BYTE_COUNT) \
  (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

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

#undef mmull_define

template <typename T>
void mvmul(usize M, usize K, const T *a, const T *b, T *c) {
  matmul_cpu_v4::mvmul(M, K, a, b, c);
}
}  // namespace matmul_cpu_v5

#include <immintrin.h>
namespace matmul_cpu_v6 {

template <typename T>
inline static constexpr T div_up(T a, T b) {
  return (a + b - 1) / b;
}

// #if defined(ENABLE_AVX)
// static constexpr int ALIGN = alignof(__m256);
// #else
// static constexpr int ALIGN = 8;
// #endif
static constexpr int ALIGN = alignof(__m256);

template <bool ALIGNED>
inline __m256 _mm256_load_f32(const float *p) {
  return ALIGNED ? _mm256_load_ps(p) : _mm256_loadu_ps(p);
}
template <bool ALIGNED>
inline __m256i _mm256_load_i32(const int32_t *p) {
  return ALIGNED ? _mm256_load_si256((const __m256i *)p)
                 : _mm256_loadu_si256((const __m256i_u *)p);
}
template <bool ALIGNED>
inline __m256i _mm256_load_i16(const int16_t *p) {
  return ALIGNED ? _mm256_load_si256((const __m256i *)p)
                 : _mm256_loadu_si256((const __m256i_u *)p);
}

template <bool ALIGNED>
inline void _mm256_store_f32(float *p, __m256 r) {
  return ALIGNED ? _mm256_store_ps(p, r) : _mm256_storeu_ps(p, r);
}
template <bool ALIGNED>
inline void _mm256_store_i32(int32_t *p, __m256i r) {
  return ALIGNED ? _mm256_store_si256((__m256i *)p, r)
                 : _mm256_storeu_si256((__m256i_u *)p, r);
}
template <bool ALIGNED>
inline void _mm256_store_i16(int16_t *p, __m256i r) {
  return ALIGNED ? _mm256_store_si256((__m256i *)p, r)
                 : _mm256_storeu_si256((__m256i_u *)p, r);
}

/**
 * @brief
 *
 * @tparam A_ALIGNED
 * @tparam B_ALIGNED
 * @tparam C_ALIGNED
 * @param a
 * @param lda must be multiply of 8
 * @param b
 * @param ldb must be multiply of 8
 * @param c
 * @param ldc must be multiply of 8
 */
template <bool A_ALIGNED, bool B_ALIGNED, bool C_ALIGNED>
inline void mmmulladd_avx_float_8x8(const float *a, const int lda,
                                    const float *b, const int ldb, float *c,
                                    const int ldc) {
  /**
   * a_ik  b_kj = c_ij
   * c_0j = a_0k b_kj
   */

  static constexpr size_t N = sizeof(__m256) / sizeof(float);
  static_assert(N == 8);
  // assert(lda % 8 == 0);
  assert(ldb % N == 0);
  assert(ldc % N == 0);

  __m256 b_row0 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 0));
  __m256 b_row1 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 1));
  __m256 b_row2 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 2));
  __m256 b_row3 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 3));
  __m256 b_row4 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 4));
  __m256 b_row5 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 5));
  __m256 b_row6 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 6));
  __m256 b_row7 = _mm256_load_f32<B_ALIGNED>(b + (ldb * 7));

  for (int i = 0; i < N; i++) {
    const float *a_row_i0 = a + (lda * i);

    __m256 c_row_i = _mm256_load_f32<C_ALIGNED>(c + (ldc * i));

    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 0), b_row0, c_row_i);
    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 1), b_row1, c_row_i);
    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 2), b_row2, c_row_i);
    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 3), b_row3, c_row_i);
    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 4), b_row4, c_row_i);
    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 5), b_row5, c_row_i);
    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 6), b_row6, c_row_i);
    c_row_i =
        _mm256_fmadd_ps(_mm256_broadcast_ss(a_row_i0 + 7), b_row7, c_row_i);

    _mm256_store_f32<C_ALIGNED>(c + (ldc * i), c_row_i);
  }
}
template <bool A_ALIGNED, bool B_ALIGNED, bool C_ALIGNED>
inline void mmmulladd_avx_i32_8x8(const int32_t *a, const int lda,
                                  const int32_t *b, const int ldb, int32_t *c,
                                  const int ldc) {
  static constexpr size_t N = sizeof(__m256i) / sizeof(int32_t);
  static_assert(N == 8);
  // assert(lda % N == 0);
  assert(ldb % N == 0);
  assert(ldc % N == 0);

  __m256i b_row0 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 0));
  __m256i b_row1 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 1));
  __m256i b_row2 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 2));
  __m256i b_row3 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 3));
  __m256i b_row4 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 4));
  __m256i b_row5 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 5));
  __m256i b_row6 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 6));
  __m256i b_row7 = _mm256_load_i32<B_ALIGNED>(b + (ldb * 7));

  for (int i = 0; i < N; i++) {
    const int32_t *a_row_i0 = a + (lda * i);

    __m256i c_row_i = _mm256_load_i32<C_ALIGNED>(c + (ldc * i));

    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[0]), b_row0), c_row_i);
    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[1]), b_row1), c_row_i);
    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[2]), b_row2), c_row_i);
    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[3]), b_row3), c_row_i);
    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[4]), b_row4), c_row_i);
    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[5]), b_row5), c_row_i);
    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[6]), b_row6), c_row_i);
    c_row_i = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(a_row_i0[7]), b_row7), c_row_i);

    _mm256_store_i32<C_ALIGNED>(c + (ldc * i), c_row_i);
  }
}

template <bool A_ALIGNED, bool B_ALIGNED, bool C_ALIGNED>
inline void mmmulladd_avx_i16_16x16(const int16_t *a, const int lda,
                                    const int16_t *b, const int ldb, int16_t *c,
                                    const int ldc) {
  static constexpr size_t N = sizeof(__m256i) / sizeof(int16_t);
  static_assert(N == 16);
  // assert(lda % 8 == 0);
  assert(ldb % N == 0);
  assert(ldc % N == 0);

  __m256i b_row0 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 0));
  __m256i b_row1 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 1));
  __m256i b_row2 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 2));
  __m256i b_row3 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 3));
  __m256i b_row4 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 4));
  __m256i b_row5 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 5));
  __m256i b_row6 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 6));
  __m256i b_row7 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 7));

  for (int i = 0; i < N; i++) {
    const int16_t *a_row_i0 = a + (lda * i);

    __m256i c_row_i = _mm256_load_i16<C_ALIGNED>(c + (ldc * i));

    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[0]), b_row0), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[1]), b_row1), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[2]), b_row2), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[3]), b_row3), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[4]), b_row4), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[5]), b_row5), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[6]), b_row6), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[7]), b_row7), c_row_i);

    _mm256_store_i16<C_ALIGNED>(c + (ldc * i), c_row_i);
  }

  b_row0 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 8));
  b_row1 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 9));
  b_row2 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 10));
  b_row3 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 11));
  b_row4 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 12));
  b_row5 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 13));
  b_row6 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 14));
  b_row7 = _mm256_load_i16<B_ALIGNED>(b + (ldb * 15));

  for (int i = 0; i < N; i++) {
    const int16_t *a_row_i0 = a + (lda * i);

    __m256i c_row_i = _mm256_load_i16<C_ALIGNED>(c + (ldc * i));

    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[8]), b_row0), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[9]), b_row1), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[10]), b_row2), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[11]), b_row3), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[12]), b_row4), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[13]), b_row5), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[14]), b_row6), c_row_i);
    c_row_i = _mm256_add_epi16(
        _mm256_mullo_epi16(_mm256_set1_epi16(a_row_i0[15]), b_row7), c_row_i);

    _mm256_store_i16<C_ALIGNED>(c + (ldc * i), c_row_i);
  }
}

#define mmmulladd_avx_core_define(type_, func_)                            \
  template <bool A_ALIGNED, bool B_ALIGNED, bool C_ALIGNED>                \
  inline void mmmulladd_avx_core(const type_ *a, const int lda,            \
                                 const type_ *b, const int ldb, type_ *c,  \
                                 const int ldc) {                          \
    return func_<A_ALIGNED, B_ALIGNED, C_ALIGNED>(a, lda, b, ldb, c, ldc); \
  }
mmmulladd_avx_core_define(float, mmmulladd_avx_float_8x8);
mmmulladd_avx_core_define(int32_t, mmmulladd_avx_i32_8x8);
mmmulladd_avx_core_define(int16_t, mmmulladd_avx_i16_16x16);

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

template <bool A_ALIGNED, bool B_ALIGNED, bool C_ALIGNED, typename T>
void mmmul_avx(usize M, usize N, usize K, const T *a, const T *b, T *c) {
  static constexpr size_t INTERVAL = sizeof(__m256) / sizeof(T);
  std::memset(c, 0, sizeof(T) * M * N);

  const int b_i_sup = M / INTERVAL;
  const int b_j_sup = N / INTERVAL;
  const int b_k_sup = K / INTERVAL;
  for (int b_i = 0; b_i < b_i_sup; b_i++) {
    for (int b_j = 0; b_j < b_j_sup; b_j++) {
      for (int b_k = 0; b_k < b_k_sup; b_k++) {
        const int c_0 = b_i * N * INTERVAL + b_j * INTERVAL;
        const int a_0 = b_i * K * INTERVAL + b_k * INTERVAL;
        const int b_0 = b_k * N * INTERVAL + b_j * INTERVAL;

        mmmulladd_avx_core<true, B_ALIGNED, C_ALIGNED>(a + a_0, K, b + b_0, N,
                                                       c + c_0, N);
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

template <>
void mmmul<float>(usize M, usize N, usize K, const float *a, const float *b,
                  float *c) {
  if (M % 8 || N % 8 || K % 8)
    mmmul_by_submatrix<8>(M, N, K, a, b, c);
  else {
    if (is_aligned(c, ALIGN))
      if (is_aligned(b, ALIGN))
        mmmul_avx<true, true, true>(M, N, K, a, b, c);
      else
        mmmul_avx<true, false, true>(M, N, K, a, b, c);
    else if (is_aligned(b, ALIGN))
      mmmul_avx<true, true, false>(M, N, K, a, b, c);
    else
      mmmul_avx<true, false, false>(M, N, K, a, b, c);
  }
}

template <>
void mmmul<int32_t>(usize M, usize N, usize K, const int32_t *a,
                    const int32_t *b, int32_t *c) {
  if (M % 8 || N % 8 || K % 8)
    mmmul_by_submatrix<32>(M, N, K, a, b, c);
  else {
    if (is_aligned(c, ALIGN))
      if (is_aligned(b, ALIGN))
        mmmul_avx<true, true, true>(M, N, K, a, b, c);
      else
        mmmul_avx<true, false, true>(M, N, K, a, b, c);
    else if (is_aligned(b, ALIGN))
      mmmul_avx<true, true, false>(M, N, K, a, b, c);
    else
      mmmul_avx<true, false, false>(M, N, K, a, b, c);
  }
}

template <>
void mmmul<int16_t>(usize M, usize N, usize K, const int16_t *a,
                    const int16_t *b, int16_t *c) {
  if (M % 16 || N % 16 || K % 16)
    mmmul_by_submatrix<64>(M, N, K, a, b, c);
  else {
    if (is_aligned(c, ALIGN))
      if (is_aligned(b, ALIGN))
        mmmul_avx<true, true, true>(M, N, K, a, b, c);
      else
        mmmul_avx<true, false, true>(M, N, K, a, b, c);
    else if (is_aligned(b, ALIGN))
      mmmul_avx<true, true, false>(M, N, K, a, b, c);
    else
      mmmul_avx<true, false, false>(M, N, K, a, b, c);
  }
}

template <typename T>
void mvmul(usize M, usize K, const T *a, const T *b, T *c) {
  matmul_cpu_v4::mvmul(M, K, a, b, c);
}
}  // namespace matmul_cpu_v6