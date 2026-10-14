"""Autograd bridge for TinySplat-owned CUDA and Metal UT/eval3D kernels."""
from functools import lru_cache
from pathlib import Path
import ctypes
import torch


@lru_cache(None)
def cuda_extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).parent/'cpp/native_3dgut.cu'
    return load(name='tinysplat_native_3dgut',sources=[str(source)],extra_cuda_cflags=['-O3'],verbose=False)


@lru_cache(None)
def metal_library():
    path=Path(__file__).resolve().parents[2]/'metal/build/libtinysplat_3dgut.dylib'
    if not path.exists():
        raise RuntimeError('Build Metal 3DGUT with python metal/build_native_3dgut.py')
    lib=ctypes.CDLL(str(path));lib.tinysplat_gut_available.restype=ctypes.c_int
    if not lib.tinysplat_gut_available():raise RuntimeError('Native Metal 3DGUT kernels unavailable')
    lib.tinysplat_gut_run.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*8+[ctypes.c_int]+[ctypes.c_void_p]*7
    lib.tinysplat_gut_run.restype=ctypes.c_int
    return lib


def metal_run(mode,m,cov,q,c,o,k,s,ids,h,w,image=None,grad=None):
    cpu=lambda t:None if t is None else t.detach().cpu().contiguous()
    m,cov,q,c,o,k,s,ids,image,grad=map(cpu,(m,cov,q,c,o,k,s,ids,image,grad))
    out=torch.zeros((len(m),4) if mode==0 else (h,w,4),dtype=torch.float32)
    grads=[torch.zeros_like(t) for t in (m,q,c,o)] if mode==2 else [None]*4
    ptr=lambda t:None if t is None else t.data_ptr()
    code=metal_library().tinysplat_gut_run(mode,len(m),h,w,*map(ptr,(m,cov,q,c,o,k,s,ids)),len(ids) if ids is not None else 0,*map(ptr,(image,grad,out,*grads)))
    if code:raise RuntimeError(f'Native Metal 3DGUT failed: {code}')
    return grads if mode==2 else out


def tile_bins(bounds,h,w,max_intersections=8_000_000):
    """Build sorted (tile, depth-order Gaussian) pairs with a hard memory bound."""
    valid=(bounds[:,2]>=bounds[:,0])&(bounds[:,3]>=bounds[:,1])
    lo=torch.floor(bounds[:,:2]/16).long();hi=torch.floor(bounds[:,2:]/16).long()
    size=(hi-lo+1).clamp_min(0);counts=size[:,0]*size[:,1]*valid
    total=int(counts.sum())
    if total>max_intersections:raise RuntimeError('3DGUT tile intersections exceed memory safety limit')
    offsets=counts.cumsum(0)-counts
    gauss=torch.repeat_interleave(torch.arange(len(bounds),device=bounds.device),counts)
    local=torch.arange(total,device=bounds.device)-offsets[gauss]
    sx=size[gauss,0].clamp_min(1)
    tx=lo[gauss,0]+local%sx;ty=lo[gauss,1]+local//sx
    tile=ty*((w+15)//16)+tx
    order=(tile*max(1,len(bounds))+gauss).argsort()
    tile=tile[order];gauss=gauss[order]
    nt=((h+15)//16)*((w+15)//16)
    starts=torch.cat([torch.zeros(1,device=bounds.device,dtype=torch.int64),torch.bincount(tile,minlength=nt).cumsum(0)]).int()
    return starts.contiguous(),gauss.int().contiguous()


class NativeRaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx,m,cov,q,c,o,k,h,w,max_intersections):
        metal=m.device.type=='mps'
        if metal:
            bounds=metal_run(0,m,cov,q,c,o,k,None,None,h,w)
        else:bounds=cuda_extension().project(m,cov,k,h,w)
        starts,ids=tile_bins(bounds,h,w,max_intersections)
        if metal:image=metal_run(1,m,cov,q,c,o,k,starts,ids,h,w).to(m.device)
        else:image=cuda_extension().forward(m,q,c,o,k,starts,ids,h,w)
        ctx.save_for_backward(m,q,c,o,k,starts,ids,image);ctx.h=h;ctx.w=w;ctx.metal=metal
        return image

    @staticmethod
    def backward(ctx,grad):
        m,q,c,o,k,starts,ids,image=ctx.saved_tensors
        if ctx.metal:grads=[t.to(m.device)for t in metal_run(2,m,None,q,c,o,k,starts,ids,ctx.h,ctx.w,image,grad)]
        else:grads=cuda_extension().backward(m,q,c,o,k,starts,ids,image,grad.contiguous(),ctx.h,ctx.w)
        return grads[0],None,grads[1],grads[2],grads[3],None,None,None,None


def render_native(means,quats,scales,opacities,colors,camera,pose,near_plane,max_intersections=8_000_000):
    from .render_3dgut import quaternion_matrix
    if means.device.type not in ('cuda','mps'):raise ValueError('Native 3DGUT requires CUDA or Apple MPS')
    n=len(means)
    if any(t.dtype!=torch.float32 or t.device!=means.device for t in (means,quats,scales,opacities,colors,pose)):
        raise ValueError('Native 3DGUT requires float32 tensors on the same device')
    if means.shape!=(n,3) or quats.shape!=(n,4) or scales.shape!=(n,3) or colors.shape!=(n,3) or opacities.numel()!=n:
        raise ValueError('Invalid native 3DGUT tensor shapes')
    if not torch.isfinite(scales).all() or not (scales>0).all():raise ValueError('Scales must be positive and finite')
    if any(not torch.isfinite(t).all() for t in (means,quats,opacities,colors)) or (quats.norm(dim=-1)==0).any():
        raise ValueError('Gaussian parameters must be finite and quaternions nonzero')
    if near_plane<=0:raise ValueError('near_plane must be positive')
    # Fixed Camera metadata is copied to constants, never optimization parameters.
    k=means.new_tensor([camera.fx,camera.fy,camera.cx,camera.cy,*(camera.distortion or (0.,)*4),float(camera.model=='OPENCV_FISHEYE'),near_plane])
    m=(means-pose[:3,3])@pose[:3,:3]
    rotation=pose[:3,:3].T[None]@quaternion_matrix(quats)
    cov=(rotation*scales.square()[:,None,:])@rotation.transpose(-1,-2)
    precision=(rotation*scales.reciprocal().square()[:,None,:])@rotation.transpose(-1,-2)
    order=m[:,2].argsort()
    image=NativeRaster.apply(m[order].contiguous(),cov[order].contiguous(),precision[order].contiguous(),colors[order].contiguous(),opacities.reshape(-1)[order].contiguous(),k,camera.height,camera.width,max_intersections)
    return image[...,:3],image[...,3:]
