# Legacy Python / PyTorch implementation

This directory contains the original TinySplat codebase (Python API, PyTorch autograd, JIT-compiled C++/CUDA extensions, training scripts).

The active project is now the **C++ / Bazel** build at the repository root. Use `pip install -e legacy` only if you need the old Python workflow.

The FastGS, Metal backend, and spherical-harmonic helper implementations live
in `legacy/tinysplat/`. The corresponding checkout-level Python modules are
relative symlinks so existing scripts can import the same source files.
