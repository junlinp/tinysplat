"""Native GPU kernel parity, finite differences and calibration invariance."""
import pytest
import torch
from tinysplat.cameras import Camera
from tinysplat.render_3dgut import render_3dgut
from tinysplat.native_3dgut import tile_bins


def scene(device):
    return [torch.tensor(v,device=device,requires_grad=True) for v in (
        [[-.3,.1,2.],[.3,-.2,2.8],[1.5,.5,1.5],[0.,0.,-1.]],
        [[1.,.1,.05,.02],[1.,.02,-.1,.04],[1.,.1,.1,.1],[1.,0.,0.,0.]],
        [[.18,.22,.25],[.2,.15,.3],[.15,.12,.14],[.1,.1,.1]],
        [.7,.6,.5,.8],[[.8,.2,.1],[.1,.5,.9],[.3,.8,.4],[1.,1.,1.]])]


@pytest.fixture(params=['cuda','mps'])
def device(request):
    d=request.param
    if d=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if d=='mps' and not torch.backends.mps.is_available():pytest.skip('Apple GPU unavailable')
    return d


@pytest.mark.parametrize('model',['PINHOLE','OPENCV_FISHEYE'])
def test_native_image_alpha_and_all_parameter_gradients(device,model):
    c=Camera(40,32,22.,23.,20.,16.,model,(.03,-.004,.001,0.) if model!='PINHOLE' else ())
    params=scene(device);refs=[t.detach().clone().requires_grad_()for t in params]
    pose=torch.eye(4,device=device);pose[:3,3]=pose.new_tensor([.04,-.02,.01])
    image,alpha=render_3dgut(*params,c,pose,backend='native')
    expected,ea=render_3dgut(*refs,c,pose,backend='reference')
    torch.testing.assert_close(image,expected,atol=2e-5,rtol=3e-4)
    torch.testing.assert_close(alpha,ea,atol=2e-5,rtol=3e-4)
    weights=torch.linspace(.3,1.,40,device=device)[None,:,None]
    ((image*weights).sum()+.17*alpha.sum()).backward()
    ((expected*weights).sum()+.17*ea.sum()).backward()
    for p,r in zip(params,refs):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad,r.grad,atol=8e-4,rtol=4e-3)
    assert torch.count_nonzero(params[0].grad[-1])==0
    assert c.fx==22. and c.distortion==((.03,-.004,.001,0.)if model!='PINHOLE'else())


def test_native_finite_difference_and_large_particle(device):
    c=Camera(24,20,15.,15.,12.,10.,'OPENCV_FISHEYE',(.03,-.004,0.,0.))
    p=scene(device)
    p[2]=torch.tensor([[.6,.5,.6]]*4,device=device,requires_grad=True) # full-frame UT fallback
    image,_=render_3dgut(*p,c,torch.eye(4,device=device),backend='native')
    expected,_=render_3dgut(*[v.detach()for v in p],c,torch.eye(4,device=device),backend='reference')
    torch.testing.assert_close(image,expected,atol=2e-5,rtol=3e-4)
    image.sum().backward();values=[]
    with torch.no_grad():
        for sign in [-1,1]:
            m=p[0].clone();m[0,0]+=sign*.0005
            out,_=render_3dgut(m,*p[1:],c,torch.eye(4,device=device),backend='native');values.append(float(out.sum()))
    assert float(p[0].grad[0,0])==pytest.approx((values[1]-values[0])/.001,rel=.02,abs=.03)


def test_bins_are_sorted_and_memory_bounded():
    b=torch.tensor([[0.,0.,31.,31.],[16.,16.,31.,31.],[1.,1.,0.,0.]])
    starts,ids=tile_bins(b,32,32)
    assert starts.tolist()==[0,1,2,3,5]
    assert ids.tolist()==[0,0,0,0,1]
    with pytest.raises(RuntimeError,match='memory safety'):tile_bins(b,32,32,4)


def test_native_small_fisheye_fit_keeps_calibration(device):
    c=Camera(24,20,15.,15.,12.,10.,'OPENCV_FISHEYE',(.03,-.004,0.,0.))
    p=scene(device);pose=torch.eye(4,device=device)
    with torch.no_grad():target,_=render_3dgut(*p,c,pose,backend='reference')
    with torch.no_grad():p[4].mul_(.6)
    optimizer=torch.optim.Adam(p,lr=.003);losses=[]
    for _ in range(35):
        optimizer.zero_grad();pred,_=render_3dgut(*p,c,pose,backend='native')
        loss=(pred-target).square().mean();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]*.3
    assert c==Camera(24,20,15.,15.,12.,10.,'OPENCV_FISHEYE',(.03,-.004,0.,0.))


def test_native_empty_scene_and_invalid_tensor_type(device):
    c=Camera(24,20,15.,15.,12.,10.,'OPENCV_FISHEYE',(0.,)*4)
    p=[v[:0]for v in scene(device)]
    image,alpha=render_3dgut(*p,c,torch.eye(4,device=device),backend='native')
    assert not image.any() and not alpha.any()
    p=scene(device);p[4]=torch.ones(4,3,device=device,dtype=torch.int32)
    with pytest.raises(ValueError,match='float32'):render_3dgut(*p,c,torch.eye(4,device=device),backend='native')
