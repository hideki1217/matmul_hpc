#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "_matmul.hpp"
#include "_mpi_core.hpp"
#include "matmul_core.h"

template <typename T>
bool is_same(const size_t N, const T* lhs, const T* rhs) {
  for (int i = 0; i < N; i++) {
    if (lhs[i] != rhs[i]) {
      printf_once("    FAILED: lhs[%d] = %d, rhs[%d] = %d\n", i, lhs[i], i,
                  rhs[i]);
      return false;
    }
  }
  return true;
}

template <>
bool is_same<float>(const size_t N, const float* lhs, const float* rhs) {
  for (int i = 0; i < N; i++) {
    if (std::abs(lhs[i] - rhs[i]) >= 1e-6) {
      printf_once("    FAILED: lhs[%d] = %f, rhs[%d] = %f\n", i, lhs[i], i,
                  rhs[i]);
      return false;
    }
  }
  return true;
}

template <>
bool is_same<double>(const size_t N, const double* lhs, const double* rhs) {
  for (int i = 0; i < N; i++) {
    if (std::abs(lhs[i] - rhs[i]) >= 1e-6) {
      printf_once("    FAILED: lhs[%d] = %lf, rhs[%d] = %lf\n", i, lhs[i], i,
                  rhs[i]);
      return false;
    }
  }
  return true;
}

#define PRINT_TEST_FAILED() \
  printf_once("failed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)
#define PRINT_TEST_PASSED() \
  printf_once("passed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)

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
bool mmmull_test() {
#define do_test(target)                                     \
  target::mmmul(M, N, K, a.data(), b.data(), c_act.data()); \
  if (!::is_same(M * N, c_act.data(), c_true.data())) TEST_FAILED();
#define do_all_target()   \
  do_test(matmul_mpi_v1); \
  // do_test(matmul_mpi_v2);

  {
    const usize M = 256, N = 256, K = 256;
    std::vector<T> a(M * K);
    std::vector<T> b(K * N);
    std::vector<T> c_act(M * N);
    std::vector<T> c_true(M * N);
    for (int i = 0; i < a.size(); i++) a[i] = i % 256 - 128;
    for (int i = 0; i < b.size(); i++) b[i] = -i % 256 + 128;

    matmul_mpi_v1::_mmmul(M, N, K, a.data(), b.data(), c_true.data());

    do_all_target();
  }
  {
    const usize M = 10, N = 23, K = 15;
    std::vector<T> a(M * K);
    std::vector<T> b(K * N);
    std::vector<T> c_act(M * N);
    std::vector<T> c_true(M * N);
    for (int i = 0; i < a.size(); i++) a[i] = i % 256 - 128;
    for (int i = 0; i < b.size(); i++) b[i] = -i % 256 + 128;

    matmul_mpi_v1::_mmmul(M, N, K, a.data(), b.data(), c_true.data());

    do_all_target();
  }
  {
    const usize M = 100, N = 101, K = 102;
    std::vector<T> a(M * K);
    std::vector<T> b(K * N);
    std::vector<T> c_act(M * N);
    std::vector<T> c_true(M * N);
    for (int i = 0; i < a.size(); i++) a[i] = i % 256 - 128;
    for (int i = 0; i < b.size(); i++) b[i] = -i % 256 + 128;

    matmul_mpi_v1::_mmmul(M, N, K, a.data(), b.data(), c_true.data());

    do_all_target();
  }

  TEST_PASSED();
#undef do_test
}

int main(int argc, char** argv) {
  my_mpi::MPI_Core core(&argc, &argv);
  core.get_ready();

  bool no_error = true;

  no_error &= mmmull_test<int16_t>();
  no_error &= mmmull_test<int32_t>();
  no_error &= mmmull_test<float>();
}