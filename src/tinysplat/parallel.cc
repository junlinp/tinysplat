#include "tinysplat/parallel.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tinysplat {

void parallel_for(int64_t begin, int64_t end,
                  const std::function<void(int64_t, int64_t)>& fn) {
  if (end <= begin) {
    return;
  }
#ifdef _OPENMP
#pragma omp parallel
  {
    const int num_threads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t chunk = (end - begin + num_threads - 1) / num_threads;
    const int64_t chunk_begin = begin + tid * chunk;
    const int64_t chunk_end = std::min(end, chunk_begin + chunk);
    if (chunk_begin < chunk_end) {
      fn(chunk_begin, chunk_end);
    }
  }
#else
  fn(begin, end);
#endif
}

int num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // namespace tinysplat
