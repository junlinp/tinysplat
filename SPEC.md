# TinySplat × Halide — SPEC

## 1. Current Architecture

```
gaussian_splat_2d.py
├── GaussianSplat2DFunction (torch.autograd.Function)
│   ├── forward()  → routes to backend.forward()
│   └── backward() → routes to backend.backward()
│
backends/
├── BackendOps { name, forward, backward, is_compiled }
├── cpu.py   → C++ extension (tinysplat_cpp) or PyTorch fallback
├── cuda.py  → CUDA extension (tinysplat_cuda) or PyTorch fallback
├── mps.py   → MPS kernel or PyTorch fallback
└── python.py → pure PyTorch chunked fallback
```

- **Forward**: alpha-composite N Gaussians onto (H, W, C) canvas.
  `T_src = Σᵢ αᵢcᵢ / Σᵢ αᵢ`, with normalization.
- **Backward**: analytical gradient replay or PyTorch autograd.
- **Extension model**: PyBind11 in `cpp/`, `mps/`.

---

## 2. Goal

Replace each backend's compiled kernel (C++/CUDA/Metal) with **a single Halide algorithm description** compiled to native code for each target from one source of truth.

**Target backends:**
| Target   | Halide target    | Schedule strategy                     |
|----------|------------------|---------------------------------------|
| CPU      | host             | vectorized × threaded (TBB/OpenMP)    |
| CUDA     | CUDA             | shared-memory tiled, stream-parallel   |
| MPS/Metal| Metal            | GPU tiling, threadgroup同步            |

**Constraint:** Must remain differentiable (PyTorch autograd compatible).

---

## 3. Halide Algorithm Design

### 3.1 Forward Pass (single algorithm, scheduled per target)

```
Input:
  means[N,2]       — (x, y) centers
  covariances[N,2,2] — 2×2 positive-definite matrices
  colors[N,C]       — per-Gaussian color (C=3 or 4)
  opacities[N]      — α in [0,1]
  height, width     — canvas resolution

Algorithm:
  for y in 0..height-1:
    for x in 0..width-1:
      accum_color  = 0   (C channels)
      accum_weight = 0
      for i in 0..N-1:
        d = [x, y] - means[i]                          (2,)
        S_inv = inverse(covariances[i])                (2,2)
        mahalanobis = d · S_inv · d                     (scalar)
        weight = opacities[i] * det_normalization(i) * exp(-0.5 * mahalanobis)
        accum_color  += weight * colors[i]
        accum_weight += weight
      output[y,x,:] = accum_color / max(accum_weight, ε)
```

**Output**: `(height, width, C)` rendered image.

### 3.2 Backward Pass (analytical, scheduled per target)

For input gradient `∂L/∂output[y,x,c]` we need `∂L/∂means`, `∂L/∂covariances`, `∂L/∂colors`, `∂L/∂opacities`.

Gradient for mean `i` (chain rule through weight and normalization):
```
∂weight/∂mean[i] = -weight_i * S_inv[i] · (pixel - mean[i])
∂L/∂mean[i] = Σ_{y,x} ∂L/∂output[y,x] · (∂accum_color/∂mean[i] - output * ∂accum_weight/∂mean[i]) / accum_weight
```
Similar analytical forms for covariance, color, and opacity gradients.

**Key**: backward does NOT replay forward. It computes exact gradients analytically, same O(NHW) structure as forward.

### 3.3 Halide-JIT Compilation Strategy

Instead of ahead-of-time compilation, use **Halide's JIT compilation** via `Halide::JITModule`:
- On first call, Halide compiles the pipeline for the detected target.
- Compilation is cached in memory for the session.
- No binary distribution of `.h` files needed.

Alternatively, ahead-of-time with a build step (see Phase 3).

---

## 4. File Layout

```
tinysplat/
├── halide/
│   ├── __init__.py
│   ├── algorithm.h          # Forward + backward Func definitions (Halide C++ headers)
│   ├── schedule_cpu.h       # CPU schedule definitions
│   ├── schedule_cuda.h      # CUDA schedule definitions
│   ├── schedule_metal.h     # Metal/MPS schedule definitions
│   ├── pipeline_forward.cpp # JIT pipeline construction
│   ├── pipeline_backward.cpp
│   ├── bindings.h           # PyBind11/JITPython Callable wrappers
│   └── CMakeLists.txt       # Build halide pipeline to .so
│
├── halide_backend.py        # New backend (halide::Func ↔ torch.Tensor)
│
├── gaussian_splat_2d.py      # No change needed (already dispatches to backends)
└── backends/
    ├── __init__.py          # Add halide to registry
    └── halide.py            # ← new entry point
```

---

## 5. Autodiff Strategy

**Option: analytical gradients (chosen)**

Halide generates both forward AND backward functions. Both are scheduled separately. The `GaussianSplat2DFunction.backward()` calls the Halide-generated backward kernel directly — no PyTorch graph replay needed.

**Why analytical over autograd replay:**
- Replay doubles the forward cost on every backward call.
- Analytical gradients are the same quality and O(1) memory.
- Halide's reverse-diff generator can produce these from the forward definition.

**If Halide's auto-diff is too immature**: fall back to hand-derived gradients in the `schedule_backward.*` files.

**Gradient checkpointing**: Not needed — forward is O(NHW) with constant memory per pixel, no large intermediates stored.

---

## 6. Build / Dependencies

### 6.1 Halide Installation

```bash
# Option A: prebuilt release
wget https://github.com/halide/Halide/releases/download/v16.0.0/Halide-16.0.0-x86-64-linux.tar.gz
tar xzf Halide-16.0.0-x86-64-linux.tar.gz
export HL_PATH=/path/to/Halide-16.0.0
export PATH=$HL_PATH/bin:$PATH

# Option B: conda
conda install -c conda-forge halide
```

### 6.2 CMake Build

```bash
mkdir halide_build && cd halide_build
cmake -DHalide_DIR=$HL_PATH/lib/cmake/Halide ..
make -j$(nproc)
```

### 6.3 Python Binding

- Load JIT-compiled `.so` via `ctypes` or `PyBind11`.
- Wrap `Halide::Buffer<float>` ↔ `torch::from_blob`.
- Register as `HALIDE_BACKEND` in `backends/__init__.py`.

---

## 7. Implementation Phases

### Phase 1 — Halide Forward Kernel (week 1)
- [ ] Install Halide, verify `HL_TARGET` env.
- [ ] Implement `algorithm.h` forward `Func`.
- [ ] Write `schedule_cpu.h` — vectorized × threaded CPU schedule.
- [ ] Write Python bindings + smoke test (`torch` ↔ `Halide::Buffer`).
- [ ] Benchmark vs. existing C++ backend on CPU.

### Phase 2 — Halide Backward Kernel (week 2)
- [ ] Implement reverse-mode gradient Funcs in `algorithm.h`.
- [ ] Write `schedule_cpu.h` for backward.
- [ ] Verify gradient correctness vs. PyTorch finite differences.
- [ ] Hook into `GaussianSplat2DFunction.backward()`.

### Phase 3 — CUDA + Metal Targets (week 3)
- [ ] Add `schedule_cuda.h` — shared-memory tiled schedule.
- [ ] Add `schedule_metal.h` — GPU schedule.
- [ ] Add target selection logic in `halide_backend.py`.
- [ ] Benchmark vs. existing `tinysplat_cuda` backend.

### Phase 4 — Integration + Optimization (week 4)
- [ ] Register `HALIDE_BACKEND` in `backends/__init__.py`, toggle via env `TINYSPLAT_BACKEND=halide`.
- [ ] Profile-guided schedule tuning (Playwright-style search over tile sizes).
- [ ] Multi-GPU support via Halide `Pipeline::produce_multiple`.
- [ ] Update README, deprecate old C++/CUDA/MPS extensions.

### Phase 5 — 3D Splatting (future)
- Same pattern: one `gaussian_splat_3d.h` algorithm, three schedules.
- Currently 3D is in `backends_3d/` — same interface applies.

---

## 8. Open Questions

1. **Halide auto-diff maturity** — worth using `Halide::generate_adjoints()` or hand-write gradients?
2. **JIT vs AOT** — JIT for dev, AOT for distribution? Preference: JIT-first.
3. **Metal on Linux** — `schedule_metal.h` targets macOS Metal. For Linux MPS, schedule needs to map to MPS-specific schedule API. Verify Halide MPS support for your target.
4. **CUDA version** — Halide supports `cuda` and `cuda_capability_86` (Ampere). Confirm your CUDA arch.
5. **Old extensions** — deprecate `cpp/` and `mps/` kernels once Halide backend is validated, or keep as fallbacks?

---

*Owner: junlin.pan | Status: Draft*
