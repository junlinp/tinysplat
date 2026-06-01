#pragma once

#include <cstdint>
#include <functional>

namespace tinysplat {

/// Parallel loop over [begin, end) with grain size 1 when OpenMP is enabled.
void parallel_for(int64_t begin, int64_t end, const std::function<void(int64_t, int64_t)>& fn);

int num_threads();

}  // namespace tinysplat
