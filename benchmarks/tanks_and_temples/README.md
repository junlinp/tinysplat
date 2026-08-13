# Tanks & Temples quality benchmark

Novel-view quality eval on the standard 3DGS **train** / **truck** scenes.

| Item | Value |
|------|--------|
| Data | Hugging Face [`alexmkwizu/gaussian_training_datasets`](https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets) (`tandt/train`, `tandt/truck`) |
| Split | Every 8th image held out (`--eval-hold 8`, same as official 3DGS / LLFF) |
| Metrics | PSNR ↑, SSIM ↑, LPIPS ↓ on held-out views |
| Train | [`train_3d_gaussians_json.py`](../../train_3d_gaussians_json.py) with **`--fastgs`** (VCD/VCP) + Metal tiled raster on MPS |

## Setup

```bash
# Complete Python package lives under legacy/
pip install -e legacy
pip install huggingface_hub lpips pytorch-msssim
# Metal FastGS rasterizer (macOS)
./metal/build_python_dylib.sh
```

## Run

```bash
# Full FastGS-protocol run on Apple Silicon (~30k iters / scene)
python benchmarks/tanks_and_temples/run_benchmark.py --device mps --iterations 30000

# Smoke test
python benchmarks/tanks_and_temples/run_benchmark.py --device mps --iterations 200 --skip-download --skip-prepare

# Re-eval only
python benchmarks/tanks_and_temples/run_benchmark.py --skip-download --skip-prepare --skip-train
```

Outputs land in `outputs/tandt_benchmark/<scene>/` (`gaussians.pt`, `metrics.json`) plus `summary.json`.

## Steps separately

```bash
export PYTHONPATH=$PWD/legacy
export TINYSPLAT_METAL_LIB=$PWD/metal/build/libtinysplat_metal_py.dylib
python benchmarks/tanks_and_temples/download_hf.py --data-dir data
python benchmarks/tanks_and_temples/prepare_scenes.py --data-dir data
python train_3d_gaussians_json.py data/tandt/truck/dataset.json \
  --eval-hold 8 --iterations 30000 --no-viser --num-downscales 0 \
  --fastgs --sh-degree 3 --device mps \
  --output-dir outputs/tandt_benchmark/truck
python benchmarks/tanks_and_temples/eval_heldout.py \
  data/tandt/truck/dataset.json outputs/tandt_benchmark/truck/gaussians.pt
```
