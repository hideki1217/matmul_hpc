#include <cstdint>

#include "matmul_core.h"

void mmmul_i16(usize M, usize N, usize K, const std::int16_t *a,
               const std::int16_t *b, std::int16_t *c);
void mmmul_i32(usize M, usize N, usize K, const std::int32_t *a,
               const std::int32_t *b, std::int32_t *c);
void mmmul_i64(usize M, usize N, usize K, const std::int64_t *a,
               const std::int64_t *b, std::int64_t *c);

void mmmul_f32(usize M, usize N, usize K, const float *a, const float *b,
               float *c);
void mmmul_f64(usize M, usize N, usize K, const double *a, const double *b,
               double *c);

inline void mmmul(usize M, usize N, usize K, const std::int16_t *a,
                  const std::int16_t *b, std::int16_t *c) {
  mmmul_i16(M, N, K, a, b, c);
};
inline void mmmul(usize M, usize N, usize K, const std::int32_t *a,
                  const std::int32_t *b, std::int32_t *c) {
  mmmul_i32(M, N, K, a, b, c);
};
inline void mmmul(usize M, usize N, usize K, const std::int64_t *a,
                  const std::int64_t *b, std::int64_t *c) {
  mmmul_i64(M, N, K, a, b, c);
};
inline void mmmul(usize M, usize N, usize K, const float *a, const float *b,
                  float *c) {
  mmmul_f32(M, N, K, a, b, c);
};
inline void mmmul(usize M, usize N, usize K, const double *a, const double *b,
                  double *c) {
  mmmul_f64(M, N, K, a, b, c);
};