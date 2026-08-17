#!/usr/bin/env bash
# Train the Tanks & Temples "truck" benchmark on a CUDA box (e.g. vast.ai).
#
# Run this ON the rented instance, not locally:
#   bash vastai_truck_cuda.sh
#
# NOTE ON --fastgs: it is deliberately NOT passed. FastGS's VCD/VCP scores come
# from count_session_hits(), which is implemented only in the Metal backend. On
# CUDA it returns None, the trainer substitutes all-zero hit counts, and no
# Gaussian ever clears the densify threshold -- training silently never
# densifies and lands around ~9 PSNR. Without --fastgs the trainer uses
# the built-in densify_and_prune path instead.
# For the same reason benchmarks/.../run_benchmark.py is NOT used here: it
# hardcodes --fastgs with no way to disable it.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/junlinp/tinysplat.git}"
WORK="${WORK:-/workspace}"
REPO="$WORK/tinysplat"
ITERS="${ITERS:-30000}"
SMOKE_ITERS="${SMOKE_ITERS:-200}"
SCENE="${SCENE:-truck}"

echo "=============== 1. environment ==============="
nvidia-smi || { echo "FATAL: no NVIDIA GPU visible"; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
  echo "FATAL: nvcc not found. The legacy CUDA extension JIT-compiles at first"
  echo "       use, so a runtime-only image will not work. Pick a vast.ai"
  echo "       template with the full CUDA toolkit (pytorch/pytorch *-devel)."
  exit 1
fi
nvcc --version | tail -2
# cv2 needs these in the pytorch devel image
apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y libxcb1 libgl1 libglib2.0-0 >/dev/null 2>&1 || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "=============== 2. repo ==============="
mkdir -p "$WORK"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch --quiet origin && git -C "$REPO" checkout --quiet main && git -C "$REPO" pull --ff-only --quiet origin main
else
  git clone --quiet "$REPO_URL" "$REPO"
fi
cd "$REPO"
echo "at commit: $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"

echo "=============== 3. dependencies ==============="
python -m pip install --quiet --upgrade pip
python -m pip install --quiet huggingface_hub lpips pytorch-msssim plyfile opencv-python-headless tqdm
# legacy/ is the complete `tinysplat` package: gaussian_splat_3d, backends_3d,
# cpp/ (the CUDA extension). The root package only carries fastgs/metal/sh.
python -m pip install --quiet -e legacy
python -c "import lpips, plyfile; print('deps ok')"

echo "=============== 4. dataset ==============="
export PYTHONPATH="$REPO/legacy:${PYTHONPATH:-}"
python benchmarks/tanks_and_temples/download_hf.py --data-dir data
python benchmarks/tanks_and_temples/prepare_scenes.py --data-dir data
DATASET="data/tandt/$SCENE/dataset.json"
test -f "$DATASET" || { echo "FATAL: $DATASET missing"; exit 1; }

train () {  # $1 = iterations, $2 = output dir
  # -u so densify logs stream instead of sitting in an 8KB stdout buffer.
  python -u train_3d_gaussians_json.py "$DATASET" \
    --iterations "$1" --output-dir "$2" \
    --eval-hold 8 --device cuda --num-downscales 0 \
    --sh-degree 3 --no-viser --cache-images \
    --densify-every 500 --densify-from 500 --densify-until 15000
}

echo "=============== 5. smoke test ($SMOKE_ITERS iters) ==============="
# Cheap guard: proves the CUDA extension compiles and that densification is
# actually happening before paying for a full run.
train "$SMOKE_ITERS" "$WORK/out_smoke" 2>&1 | tee "$WORK/smoke.log"
python - <<'PY'
import re, sys, pathlib
log = pathlib.Path("/workspace/smoke.log").read_text(errors="ignore")
ns = [int(m) for m in re.findall(r"N=(\d+)", log)]
if not ns:
    sys.exit("SMOKE FAIL: no Gaussian counts in log")
print(f"smoke N: {ns[0]} -> {ns[-1]}")
if ns[-1] <= ns[0]:
    sys.exit(f"SMOKE FAIL: N did not grow ({ns[0]} -> {ns[-1]}). "
             "Densification is dead -- do NOT start the 30k run.")
print("SMOKE OK: densification is live")
PY

echo "=============== 6. full run ($ITERS iters) ==============="
time train "$ITERS" "$WORK/out_$SCENE" 2>&1 | tee "$WORK/train.log"

echo "=============== 7. held-out eval ==============="
python benchmarks/tanks_and_temples/eval_heldout.py "$DATASET" "$WORK/out_$SCENE/gaussians.pt" \
  | tee "$WORK/eval.log"

echo
echo "=============== done ==============="
echo "checkpoint : $WORK/out_$SCENE/gaussians.pt"
echo "metrics    : $WORK/out_$SCENE/metrics.json"
echo "M1 baseline for comparison: PSNR 25.08 / SSIM 0.8697 / LPIPS 0.1255 (with FastGS)"
echo "Retrieve with:  vastai copy <INSTANCE_ID>:$WORK/out_$SCENE/gaussians.pt ."

# MEASURED 2026-08-17 on an RTX 4090: ~37-46 s/iter, i.e. ~308 h for 30k.
# The CUDA 3D rasterizer (splat_3d_per_pixel_kernel) has no tile binning and is
# O(N*H*W), making it ~74x slower than the Metal path on an M1. Until that kernel
# is tiled, the full benchmark is not practical here -- use --iterations for
# smoke tests only.
