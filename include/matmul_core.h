using usize = unsigned int;

template <typename T>
using mmmul_t = void(usize M, usize N, usize K, const T *a, const T *b, T *c);

template <typename T>
using mvmul_t = void(usize M, usize K, const T *a, const T *b, T *c);