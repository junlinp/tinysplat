#!/usr/bin/env bash
# Train the Tanks & Temples "truck" benchmark on a CUDA box (e.g. vast.ai).
#
# Run this ON the rented instance, not locally:
#   bash vastai_truck_cuda.sh
#
# --fastgs IS passed. FastGS's VCD/VCP statistics used to be Metal-only, so on
# CUDA they returned None, the trainer substituted all-zero hit counts, and
# training silently never densified. That was fixed in #20 (CUDA
# footprint_hit_count + AbsGS accumulator) and #21 (the EWA Jacobian clamp that
# CUDA was missing, without which gradients were dominated by outliers and
# training did not converge). Reaching the published quality needs both.
#
# benchmarks/.../run_benchmark.py hardcodes --fastgs, which is now correct on
# CUDA; this script exists mainly to drive a single scene with explicit flags.

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
    --sh-degree 3 --no-viser --cache-images --fastgs \
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

# MEASURED on an RTX 4090 at commit 901a7db: ~20 it/s, i.e. ~25 min for 30k.
#
# For context on how that number moved: it was 37-46 s/iter before #18 (a 12x
# buffer overflow, aliased tile bins, a density-normalisation factor applied
# under alpha compositing, and a per-pixel linear scan of the tile bin in the
# backward) and #19 (a per-pixel backward replacing the per-Gaussian one, which
# also made the gradient exact rather than dropping transmittance). Do not trust
# older timing notes -- and if you microbenchmark the backward, use realistic
# projected covariances: synthetic 1-4px ones understate its cost by ~20x.
