"""Native CUDA/Metal 3DGUT and a PyTorch reference.

The reference evaluates 3D Gaussian responses along calibrated rays directly;
it is not an accelerated unscented-transform rasterizer.
"""
import torch
import torch.nn.functional as F

from .gaussian_splat_3d_core import validate_camera_to_world


def quaternion_matrix(quats):
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).reshape(-1,3,3)


def render_3dgut(means, quats, scales, opacities, colors, camera, camera_to_world,
                  *, backend="auto", near_plane=.01, pixel_chunk=256,
                  reference_pair_limit=2_000_000):
    """Return HWC RGB and HW1 alpha. Calibration is fixed, pose is optical c2w.

    `native` uses TinySplat's CUDA/Metal UT and eval3D forward/backward kernels.
    `reference` supports CPU/CUDA/Apple MPS for small scenes. No implicit
    undistortion, approximate pinhole conversion, or device fallback occurs.
    """
    if means.device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Supported devices are CPU, CUDA and Apple MPS")
    validate_camera_to_world(camera_to_world)
    if backend == "auto":
        backend = "native" if means.device.type in ("cuda", "mps") else "reference"
    if backend == "native":
        from .native_3dgut import render_native
        return render_native(means,quats,scales,opacities,colors,camera,camera_to_world,near_plane)
    if backend != "reference":
        raise ValueError(f"Unknown 3DGUT backend: {backend}")
    if pixel_chunk <= 0:
        raise ValueError("pixel_chunk must be positive")
    if len(means)*camera.width*camera.height > reference_pair_limit:
        raise ValueError("Reference scene exceeds pair limit; use the native CUDA/Metal backend or a smaller scene")
    rays, valid = camera.rays(device=means.device, dtype=means.dtype)
    # Express rays in each Gaussian's local, scale-normalized space.
    rays = rays.reshape(-1,3) @ camera_to_world[:3,:3].T
    rotation = quaternion_matrix(quats)
    origin = camera_to_world[:3,3]-means
    local_origin = torch.einsum("nj,njk->nk", origin, rotation)/scales
    z = ((means-camera_to_world[:3,3]) @ camera_to_world[:3,:3])[:,2]
    order = z.argsort()
    images, alphas = [], []
    for ray_chunk in rays.split(pixel_chunk):
        direction = torch.einsum("pj,njk->pnk", ray_chunk, rotation)/scales[None]
        t = -(direction*local_origin[None]).sum(-1)/direction.square().sum(-1).clamp_min(1e-12)
        distance2 = (local_origin[None]+t[...,None]*direction).square().sum(-1)
        alpha = (opacities.reshape(1,-1)*(-.5*distance2).exp()).clamp(0,.999)
        alpha = torch.where((z[None]>near_plane)&(t>0)&(distance2<9),alpha,0)[:,order]
        trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),1-alpha],-1),-1)[:,:-1]
        weight = alpha*trans
        images.append(weight @ colors[order]); alphas.append(weight.sum(-1,keepdim=True))
    mask = valid[...,None]
    return (torch.cat(images).reshape(camera.height,camera.width,-1)*mask,
            torch.cat(alphas).reshape(camera.height,camera.width,1)*mask)
