// #include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "_matmul.cuh"
#include "matmul.hpp"

template <typename T>
bool is_same(const usize N, const T* lhs, const T* rhs) {
  for (int i = 0; i < N; i++) {
    if (lhs[i] != rhs[i]) {
      printf("    FAILED: lhs[%d] = %d, rhs[%d] = %d\n", i, lhs[i], i, rhs[i]);
      return false;
    }
  }
  return true;
}

template <>
bool is_same<float>(const usize N, const float* lhs, const float* rhs) {
  for (int i = 0; i < N; i++) {
    if (std::abs(lhs[i] - rhs[i]) >= 1e-6) {
      printf("    FAILED: lhs[%d] = %f, rhs[%d] = %f\n", i, lhs[i], i, rhs[i]);
      return false;
    }
  }
  return true;
}

template <>
bool is_same<double>(const usize N, const double* lhs, const double* rhs) {
  for (int i = 0; i < N; i++) {
    if (std::abs(lhs[i] - rhs[i]) >= 1e-6) {
      printf("    FAILED: lhs[%d] = %lf, rhs[%d] = %lf\n", i, lhs[i], i,
             rhs[i]);
      return false;
    }
  }
  return true;
}

#define PRINT_TEST_FAILED() \
  printf("failed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)
#define PRINT_TEST_PASSED() \
  printf("passed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)

#define TEST_FAILED()    \
  {                      \
    PRINT_TEST_FAILED(); \
    return false;        \
  }
#define TEST_PASSED()        \
  { /*PRINT_TEST_PASSED();*/ \
    return true;             \
  }

template <typename T>
bool mm_test() {
  const usize M = 512;
  const usize N = 256;
  const usize K = 128;

  std::vector<T> a(M * K);
  std::vector<T> b(K * N);
  std::vector<T> c_true(M * N);
  std::vector<T> c_act(M * N);

  for (int i = 0; i < M * K; i++) a[i] = i % 512 - 256;
  for (int i = 0; i < K * N; i++) b[i] = i % 512 - 256;

  matmul_cpu::mmmul(M, N, K, a.data(), b.data(), c_true.data());

#define do_test(target)                                     \
{ \
    target::MMmull<T> st(M, N, K);\
    st.mmmul(M, N, K, a.data(), b.data(), c_act.data());\
    if (!::is_same(M * N, c_true.data(), c_act.data())) TEST_FAILED();\
}

  do_test(matmul_cuda_v1);
  do_test(matmul_cuda_v2);
#undef do_test
  TEST_PASSED();
}

int main() {
    mm_test<int16_t>();
    mm_test<int32_t>();
    mm_test<float>();
}