#!/usr/bin/env bash
# Build shared library for Python ctypes Metal backend.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/metal/build"
OUT="$ROOT/metal/build/libtinysplat_metal_py.dylib"
clang++ -std=c++17 -O3 -fobjc-arc -shared -fPIC \
  -I "$ROOT/src/tinysplat/include" -I "$ROOT/metal/include" -DTINYSPLAT_METAL \
  "$ROOT/metal/gaussian_3d_metal.mm" \
  "$ROOT/metal/python_api.cc" \
  -framework Metal -framework Foundation \
  -o "$OUT"
echo "Wrote $OUT"
