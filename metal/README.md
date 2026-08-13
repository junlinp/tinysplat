# Metal 3DGS rasterizer (FastGS-compatible)

Tiled alpha compositing on Apple Metal, with compact-box footprints and
per-splat projected backward. Used as the default MPS backend when the
Python dylib is built.

## Build Python dylib

```bash
./metal/build_python_dylib.sh
```

Produces `metal/build/libtinysplat_metal_py.dylib`. Override path with
`TINYSPLAT_METAL_LIB`.

## Bazel (optional)

```bash
bazel build --config=metal //metal:...
```

Requires `--define=metal=1` (see `.bazelrc` `build:metal`).

## Train with FastGS on MPS

```bash
export PYTHONPATH=$PWD/legacy
export TINYSPLAT_METAL_LIB=$PWD/metal/build/libtinysplat_metal_py.dylib
# Run from a non-repo cwd if needed so ./tinysplat does not shadow legacy.
python train_3d_gaussians_json.py data/tandt/truck/dataset.json \
  --device mps --fastgs --sh-degree 3 --no-viser --iterations 30000
```
