#include "./matmul.hpp"

#include <cstdint>

#include "./_matmul.hpp"

namespace matmul_cpu {
void mmmul_i16(usize M, usize N, usize K, const std::int16_t *a,
               const std::int16_t *b, std::int16_t *c) {
  return matmul_cpu_v6::mmmul(M, N, K, a, b, c);
}
void mmmul_i32(usize M, usize N, usize K, const std::int32_t *a,
               const std::int32_t *b, std::int32_t *c) {
  return matmul_cpu_v6::mmmul(M, N, K, a, b, c);
}
void mmmul_i64(usize M, usize N, usize K, const std::int64_t *a,
               const std::int64_t *b, std::int64_t *c) {
  return matmul_cpu_v4::mmmul(M, N, K, a, b, c);
}
void mmmul_f32(usize M, usize N, usize K, const float *a, const float *b,
               float *c) {
  return matmul_cpu_v6::mmmul(M, N, K, a, b, c);
}
void mmmul_f64(usize M, usize N, usize K, const double *a, const double *b,
               double *c) {
  return matmul_cpu_v4::mmmul(M, N, K, a, b, c);
}
}  // namespace matmul_cpu