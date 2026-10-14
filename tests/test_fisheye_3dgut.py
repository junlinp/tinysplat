import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tinysplat.cameras import Camera
from tinysplat.render_3dgut import render_3dgut


def test_opencv_projection_and_pixel_ray_roundtrip():
    c = Camera(40, 32, 18., 19., 20., 16., "OPENCV_FISHEYE", (.02,-.003,.001,-.0001))
    xyz = np.array([[0.,0.,2.], [.7,-.5,1.], [1.8,.3,.4]], dtype=np.float64)
    expected, _ = cv2.fisheye.projectPoints(xyz[None], np.zeros(3), np.zeros(3), c.matrix(dtype=torch.float64).numpy(), np.array(c.distortion))
    np.testing.assert_allclose(c.project(torch.tensor(xyz)).numpy(), expected[0], atol=1e-10)
    rays, valid = c.rays(dtype=torch.float64)
    yy,xx = torch.meshgrid(torch.arange(c.height)+.5,torch.arange(c.width)+.5,indexing='ij')
    torch.testing.assert_close(c.project(rays)[valid],torch.stack([xx,yy],-1).double()[valid],atol=1e-7,rtol=0)
    # A zero-distortion fisheye remains equidistant, not pinhole.
    zero = Camera(40,32,18.,19.,20.,16.,"OPENCV_FISHEYE",(0.,)*4)
    assert not torch.allclose(zero.project(torch.tensor(xyz)),Camera(40,32,18.,19.,20.,16.).project(torch.tensor(xyz)))


def test_reject_ambiguous_calibration_and_reference_oversize():
    with pytest.raises(ValueError): Camera(32,32,20,20,16,16,"OPENCV_FISHEYE")
    with pytest.raises(ValueError): Camera(32,32,20,20,16,16,"PINHOLE",(.1,))
    with pytest.raises(ValueError): Camera(32,32,float('nan'),20,16,16)
    with pytest.raises(ValueError): Camera(32,32,20,20,16,16,"OPENCV_FISHEYE",(-1.,0.,0.,0.))
    with pytest.raises(ValueError,match="pair limit"):
        render_scene('cpu',limit=1)


def render_scene(device, backend="reference", limit=2_000_000):
    c = Camera(24,20,15.,15.,12.,10.,"OPENCV_FISHEYE",(.03,-.004,0.,0.))
    tensors = [torch.tensor(v,device=device,dtype=torch.float32,requires_grad=True) for v in (
        [[-.15,.1,2.],[.35,-.1,2.7]], [[1.,.1,.05,.02],[1.,.02,-.1,.04]],
        [[.18,.22,.25],[.2,.15,.3]], [.7,.6], [[.8,.2,.1],[.1,.5,.9]])]
    image, alpha = render_3dgut(*tensors,c,torch.eye(4,device=device),backend=backend,reference_pair_limit=limit)
    return image,alpha,tensors


@pytest.mark.parametrize("device",["cpu","cuda","mps"])
def test_reference_gradients_and_occlusion(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Apple unavailable')
    image,alpha,tensors=render_scene(device)
    assert image.shape==(20,24,3) and alpha.shape==(20,24,1)
    assert 0 < alpha.max() <= 1
    weights=torch.linspace(.3,1.,24,device=device)[None,:,None]
    (image*weights).sum().backward()
    for t in tensors:assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0
    # Finite difference checks the 3D response gradient, not only autograd plumbing.
    c=Camera(24,20,15.,15.,12.,10.,"OPENCV_FISHEYE",(.03,-.004,0.,0.))
    values=[]
    with torch.no_grad():
        for sign in [-1,1]:
            means=tensors[0].clone();means[0,0]+=sign*.0005
            out,_=render_3dgut(means,*tensors[1:],c,torch.eye(4,device=device),backend='reference')
            values.append(float((out*weights).sum()))
    assert float(tensors[0].grad[0,0])==pytest.approx((values[1]-values[0])/.001,rel=.02,abs=.01)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_gsplat_3dgut_forward_backward():
    pytest.importorskip('gsplat')
    image,alpha,tensors=render_scene('cuda','gsplat')
    image.sum().backward()
    assert alpha.max()>0
    for t in tensors:assert t.grad is not None and torch.isfinite(t.grad).all()
    ref,_,_=render_scene('cuda','reference')
    # UT bounds/filtering can differ; dominant calibrated structure should agree.
    assert float((image-ref).abs().mean().detach()) < .025


def test_reference_front_to_back_and_behind_camera():
    c=Camera(1,1,10.,10.,.5,.5)
    means=torch.tensor([[0.,0.,2.],[0.,0.,3.],[0.,0.,-2.]])
    quats=torch.tensor([[1.,0.,0.,0.]]*3)
    image,alpha=render_3dgut(means,quats,torch.ones(3,3)*.1,
        torch.tensor([.8,.5,1.]),torch.eye(3),c,torch.eye(4),backend='reference')
    torch.testing.assert_close(image[0,0],torch.tensor([.8,.1,0.]))
    torch.testing.assert_close(alpha[0,0],torch.tensor([.9]))


def test_pinhole_loader_rejects_raw_distorted_dataset(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('_fisheye_guard_test',repo/'train_3d_gaussians_json.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    d=dict(scene_dir=str(tmp_path),frames=[dict(camera_model='OPENCV_FISHEYE')])
    path=tmp_path/'dataset.json';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='train_3dgut.py'):module.load_dataset_frames(path,'cpu')


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_raw_fisheye_training_cli(tmp_path, device):
    if device == "cuda":
        if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
        pytest.importorskip("gsplat")
    c=Camera(24,20,15.,15.,12.,10.,"OPENCV_FISHEYE",(.03,-.004,0.,0.))
    with torch.no_grad():target,_,tensors=render_scene('cpu')
    cv2.imwrite(str(tmp_path/'raw.png'),(target.numpy()[...,::-1]*255).round().astype(np.uint8))
    frame=dict(image_id=0,file_path='raw.png',width=c.width,height=c.height,
               intrinsics=dict(fx=c.fx,fy=c.fy,cx=c.cx,cy=c.cy),camera_model=c.model,
               distortion=list(c.distortion),transform_matrix=np.eye(4).tolist())
    d=dict(scene_dir=str(tmp_path),frames=[frame],train_ids=[0],val_ids=[])
    path=tmp_path/'dataset.json';path.write_text(json.dumps(d))
    means,quats,scales,opa,colors=[t.detach() for t in tensors]
    ck=tmp_path/'initial.pt';torch.save(dict(means=means,rotations=quats,log_scales=scales.log(),opacities=torch.logit(opa),colors=colors),ck)
    repo=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env['PYTHONPATH']=str(repo/'legacy')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,str(repo/'train_3dgut.py'),'--dataset',str(path),'--checkpoint',str(ck),
                    '--output',str(tmp_path/'result'),'--device',device,'--steps','3','--eval-views','1'],env=env,check=True,cwd=tmp_path)
    assert json.loads((tmp_path/'result/dataset.json').read_text())==d
    report=json.loads((tmp_path/'result/evaluation.json').read_text())
    assert report['intrinsics_fixed'] and report['distortion_fixed'] and report['poses_fixed']
    assert (tmp_path/'result/comparison_0.png').exists()
