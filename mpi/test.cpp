#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "_mpi_core.hpp"
#include "matmul_core.h"
#include "_matmul.hpp"

template <typename T>
bool is_same(const size_t N, const T* lhs, const T* rhs) {
  for (int i = 0; i < N; i++) {
    if (lhs[i] != rhs[i]) {
      print_once("    FAILED: lhs[%d] = %d, rhs[%d] = %d\n", i, lhs[i], i,
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
      print_once("    FAILED: lhs[%d] = %f, rhs[%d] = %f\n", i, lhs[i], i,
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
      print_once("    FAILED: lhs[%d] = %lf, rhs[%d] = %lf\n", i, lhs[i], i,
                 rhs[i]);
      return false;
    }
  }
  return true;
}

#define PRINT_TEST_FAILED() \
  print_once("failed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)
#define PRINT_TEST_PASSED() \
  print_once("passed: %s: %s:%d\n\n", __PRETTY_FUNCTION__, __FILE__, __LINE__)

#define TEST_FAILED()    \
  {                      \
    PRINT_TEST_FAILED(); \
    return false;        \
  }
#define TEST_PASSED()        \
  { /*PRINT_TEST_PASSED();*/ \
    return true;             \
  }

template<typename T>
bool mmmull_test() {
  const usize M = 128, N = 256, K=128; 
  std::vector<T> a(M*K);
  std::vector<T> b(K*N);
  std::vector<T> c_act(M*N);
  std::vector<T> c_true(M*N);
  for(int i=0; i<a.size(); i++) a[i] = i % 256 - 128;
  for(int i=0; i<b.size(); i++) b[i] = - i % 256 + 128;

  matmul_mpi_v1::_mmmul(M, N, K, a.data(), b.data(), c_true.data());

#define do_test(target)                                     \
  target::mmmul(M, N, K, a.data(), b.data(), c_act.data());     \
  if (!::is_same(M * N, c_act.data(), c_true.data())) TEST_FAILED(); \

  do_test(matmul_mpi_v1);
#undef do_test

  TEST_PASSED();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  my_mpi::init();

  bool no_error = true;

  no_error &= mmmull_test<int16_t>();
  no_error &= mmmull_test<int32_t>();
  no_error &= mmmull_test<float>();

  MPI_Finalize();
}