#pragma once

#include <cuda_runtime.h>
#include <cstdio>

namespace tinysplat {
namespace cuda {
namespace detail {

inline bool cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
    return false;
  }
  return true;
}

}  // namespace detail
}  // namespace cuda
}  // namespace tinysplat

#define TINYSPLAT_CUDA_CHECK(call) \
  ::tinysplat::cuda::detail::cuda_check((call), __FILE__, __LINE__)
