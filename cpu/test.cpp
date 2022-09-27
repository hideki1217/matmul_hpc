#include <cmath>
#include <iostream>
#include <vector>

#include "./_matmul.hpp"

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

#define PRINT_TEST_FAILED() printf("failed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)
#define PRINT_TEST_PASSED() printf("passed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)

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
bool mm_basic_test() {
  const usize M = 3;
  const usize N = 3;
  const usize K = 3;

  std::vector<T> a = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<T> b = {1, 2, 3, 5, 4, 6, 3, 12, 9};
  std::vector<T> c(M * N);

#define do_test(target)                                     \
  target::mmmul(M, N, K, a.data(), b.data(), c.data());     \
  if (!::is_same(M * N, c.data(), b.data())) TEST_FAILED(); \
  target::mmmul(M, N, K, b.data(), a.data(), c.data());     \
  if (!::is_same(M * N, c.data(), b.data())) TEST_FAILED();

  do_test(matmul_cpu_v1);
#undef do_test
  TEST_PASSED();
}

template <typename T>
bool mv_basic_test() {
  const usize M = 4;
  const usize K = 3;

  std::vector<T> a = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  std::vector<T> b = {1, 2, 3, 0};
  std::vector<T> c(M);

#define do_test(target)                              \
  target::mvmul(M, K, a.data(), b.data(), c.data()); \
  if (!::is_same(M, c.data(), b.data())) TEST_FAILED();

  do_test(matmul_cpu_v1);

#undef do_test

  TEST_PASSED();
}

template <typename T>
bool mm_test() {
  const usize M = 128;
  const usize N = 64;
  const usize K = 128;

  std::vector<T> a(M*K);
  std::vector<T> b(K*N);
  std::vector<T> c_true(M*N);
  std::vector<T> c_act(M*N);

  for(int i=0; i<M*K; i++) a[i] = i % 512 - 256;
  for(int i=0; i<K*N; i++) b[i] = i % 512 - 256;

  matmul_cpu_v1::mmmul(M, N, K, a.data(), b.data(), c_true.data());

#define do_test(target)                                     \
  target::mmmul(M, N, K, a.data(), b.data(), c_act.data());     \
  if (!::is_same(M * N, c_true.data(), c_act.data())) TEST_FAILED();

  do_test(matmul_cpu_v2);
  do_test(matmul_cpu_v3);
  do_test(matmul_cpu_v4);
  do_test(matmul_cpu_v5);
#undef do_test
  TEST_PASSED();
}

template <typename T>
bool mv_test() {
  const usize M = 1024;
  const usize K = 512;

  std::vector<T> a(M*K);
  std::vector<T> b(K);
  std::vector<T> c_true(M);
  std::vector<T> c_act(M);

  for(int i=0; i<M*K; i++) a[i] = i % 512 - 256;
  for(int i=0; i<K; i++) b[i] = i % 512 - 256;

  matmul_cpu_v1::mvmul(M, K, a.data(), b.data(), c_true.data());

#define do_test(target)                              \
  target::mvmul(M, K, a.data(), b.data(), c_act.data()); \
  if (!::is_same(M, c_act.data(), c_true.data())) TEST_FAILED();

  do_test(matmul_cpu_v2);
  do_test(matmul_cpu_v3);
  do_test(matmul_cpu_v4);
  do_test(matmul_cpu_v5);

#undef do_test

  TEST_PASSED();
}

int main() {
  bool res = true;

  res &= mm_basic_test<int>();
  res &= mm_basic_test<short>();
  res &= mm_basic_test<float>();

  res &= mv_basic_test<int>();
  res &= mv_basic_test<short>();
  res &= mv_basic_test<float>();

  res &= mm_test<int>();
  res &= mm_test<short>();
  res &= mm_test<float>();

  res &= mv_test<int>();
  res &= mv_test<short>();
  res &= mv_test<float>();

  return !res;
}