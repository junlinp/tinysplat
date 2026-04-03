#!/usr/bin/env python3
"""
Benchmark: Halide vs PyTorch 2D Gaussian Splatting on Lena image.
Runs each backend in a clean subprocess for accurate timing.
"""

import os, sys, json, time, subprocess, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HALIDE_LIB = os.path.join(SCRIPT_DIR, "tinysplat/halide/build/libtinysplat_halide_pipeline.so")
LENA_PATH = os.path.join(SCRIPT_DIR, "tests/data/lena.png")


def run_backend(backend: str, n_runs: int = 10, warmup: int = 2) -> dict:
    """Run benchmark in clean subprocess, return parsed results."""
    env = {k: v for k, v in os.environ.items()}
    if backend == "halide":
        env.update({
            "TINYSPLAT_BACKEND": "halide",
            "TINYSPLAT_HALIDE_LIB": HALIDE_LIB,
        })
    else:
        env["TINYSPLAT_BACKEND"] = "python"
        env.pop("TINYSPLAT_HALIDE_LIB", None)

    code = f"""
import os, sys, time, json, cv2, torch, numpy as np
sys.path.insert(0, {repr(SCRIPT_DIR)})
from tinysplat.gaussian_splat_2d import gaussian_splat_2d
from sklearn.cluster import MiniBatchKMeans

img = cv2.imread({repr(str(LENA_PATH))})
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
H, W, C = img.shape
pixels = img.reshape(-1, 3)
coords = np.mgrid[0:H, 0:W].reshape(2, -1).T.astype(np.float32)
features = np.concatenate([pixels, coords / max(H, W)], axis=1)
kmeans = MiniBatchKMeans(n_clusters=500, random_state=42, batch_size=1000)
labels = kmeans.fit_predict(features)
ml, cl = [], []
for k in range(500):
    m = labels == k
    ml.append([coords[m][:,1].mean(), coords[m][:,0].mean()])
    cl.append(pixels[m].mean(axis=0))
means = np.array(ml, dtype=np.float32)
colors = np.array(cl, dtype=np.float32)
covs = np.zeros((500, 2, 2), dtype=np.float32)
for k in range(500):
    m = labels == k
    c = coords[m]
    covs[k] = (np.cov(c[:,0], c[:,1]) + np.eye(2) * 4.0) if len(c) > 1 else np.eye(2) * 16.0
opacities = np.ones(500, dtype=np.float32) * 0.7
means_t = torch.from_numpy(means)
cov_t = torch.from_numpy(covs)
colors_t = torch.from_numpy(colors)
opacities_t = torch.from_numpy(opacities)

for _ in range({warmup}):
    _ = gaussian_splat_2d(means_t, cov_t, colors_t, opacities_t, H, W)

times = []
for _ in range({n_runs}):
    t0 = time.perf_counter()
    out = gaussian_splat_2d(means_t, cov_t, colors_t, opacities_t, H, W)
    t1 = time.perf_counter()
    times.append(t1 - t0)

out_np = out.detach().numpy()
print(json.dumps({{
    "backend": {repr(backend)},
    "times": times,
    "out_min": float(out_np.min()),
    "out_max": float(out_np.max()),
    "canvas": [H, W, C],
    "n_gaussians": 500,
}}))
"""

    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=SCRIPT_DIR,
        timeout=120,
    )
    if r.returncode != 0:
        print(f"  [{backend}] ERROR:")
        print(r.stderr[-1000:] if len(r.stderr) > 1000 else r.stderr)
        return None
    try:
        return json.loads(r.stdout.strip())
    except Exception as e:
        print(f"  [{backend}] Failed to parse: {e}\n{r.stdout[:500]}")
        return None


def save_image(backend: str):
    """Save one output image for the given backend."""
    env = {k: v for k, v in os.environ.items()}
    if backend == "halide":
        env.update({
            "TINYSPLAT_BACKEND": "halide",
            "TINYSPLAT_HALIDE_LIB": HALIDE_LIB,
        })
    else:
        env["TINYSPLAT_BACKEND"] = "python"
        env.pop("TINYSPLAT_HALIDE_LIB", None)

    out_path = os.path.join(SCRIPT_DIR, f"output_lena_{backend}.png")
    code = f"""
import os, sys, cv2, torch, numpy as np
sys.path.insert(0, {repr(SCRIPT_DIR)})
from tinysplat.gaussian_splat_2d import gaussian_splat_2d
from sklearn.cluster import MiniBatchKMeans
img = cv2.imread({repr(str(LENA_PATH))})
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
H, W, C = img.shape
pixels = img.reshape(-1, 3)
coords = np.mgrid[0:H, 0:W].reshape(2, -1).T.astype(np.float32)
features = np.concatenate([pixels, coords / max(H, W)], axis=1)
kmeans = MiniBatchKMeans(n_clusters=500, random_state=42, batch_size=1000)
labels = kmeans.fit_predict(features)
ml, cl = [], []
for k in range(500):
    m = labels == k
    ml.append([coords[m][:,1].mean(), coords[m][:,0].mean()])
    cl.append(pixels[m].mean(axis=0))
means = np.array(ml, dtype=np.float32)
colors = np.array(cl, dtype=np.float32)
covs = np.zeros((500, 2, 2), dtype=np.float32)
for k in range(500):
    m = labels == k
    c = coords[m]
    covs[k] = (np.cov(c[:,0], c[:,1]) + np.eye(2) * 4.0) if len(c) > 1 else np.eye(2) * 16.0
opacities = np.ones(500, dtype=np.float32) * 0.7
means_t = torch.from_numpy(means)
cov_t = torch.from_numpy(covs)
colors_t = torch.from_numpy(colors)
opacities_t = torch.from_numpy(opacities)
out = gaussian_splat_2d(means_t, cov_t, colors_t, opacities_t, H, W)
out_np = (out.detach().numpy().clip(0,1) * 255).astype(np.uint8)
cv2.imwrite({repr(out_path)}, cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                      text=True, env=env, cwd=SCRIPT_DIR, timeout=60)
    if r.returncode == 0:
        return out_path
    return None


def main():
    if not os.path.exists(LENA_PATH):
        sys.exit(f"Lena not found: {LENA_PATH}")
    if not os.path.exists(HALIDE_LIB):
        sys.exit(f"Halide lib not found: {HALIDE_LIB}")

    n_runs, warmup = 10, 2
    print("=" * 60)
    print("Benchmark: Halide vs PyTorch 2D Gaussian Splatting (Lena)")
    print("=" * 60)
    print(f"Canvas: 500x500x3  |  Gaussians: 500  |  {n_runs} runs")
    print()

    print(f"[1/4] Halide backend ({warmup} warmup + {n_runs} runs)...")
    res_h = run_backend("halide", n_runs=n_runs, warmup=warmup)

    print(f"[2/4] PyTorch backend ({warmup} warmup + {n_runs} runs)...")
    res_p = run_backend("python", n_runs=n_runs, warmup=warmup)

    if res_h is None or res_p is None:
        sys.exit("Benchmark failed.")

    t_h = np.array(res_h["times"])
    t_p = np.array(res_p["times"])

    print(f"[3/4] Saving output images...")
    path_h = save_image("halide")
    path_p = save_image("python")

    print(f"[4/4] Results:")
    print()
    print(f"{'':22s} {'Halide':>12s}  {'PyTorch':>12s}  {'Ratio':>8s}")
    print(f"{'mean (s)':22s} {t_h.mean():12.4f}  {t_p.mean():12.4f}  {t_p.mean()/t_h.mean():8.2f}x")
    print(f"{'std (s)':22s} {t_h.std():12.4f}  {t_p.std():12.4f}")
    print(f"{'min (s)':22s} {t_h.min():12.4f}  {t_p.min():12.4f}")
    print(f"{'max (s)':22s} {t_h.max():12.4f}  {t_p.max():12.4f}")
    print()
    print(f"Halide range:  [{res_h['out_min']:.4f}, {res_h['out_max']:.4f}]")
    print(f"PyTorch range: [{res_p['out_min']:.4f}, {res_p['out_max']:.4f}]")
    print()
    print(f"Output images:")
    if path_h: print(f"  Halide:  {path_h}")
    if path_p: print(f"  PyTorch: {path_p}")


if __name__ == "__main__":
    main()
