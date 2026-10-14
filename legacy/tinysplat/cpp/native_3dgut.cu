#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#define DEVICE __device__ inline
#define CPTR const float*
#define WPTR float*
#define TPTR float*
#define TCPTR const float*
#define IPTR const int*
#define APTR float*
#define SQRT sqrtf
#define FMIN fminf
#define FMAX fmaxf
#define SIN sinf
#define COS cosf
#define ATAN2 atan2f
#define EXP expf
#define ABS fabsf
#define FLOOR floorf
#define CEIL ceilf
#define ADD atomicAdd
#include "native_3dgut_common.h"
__global__ void project_kernel(int n,int h,int w,const float*m,const float*c,const float*k,float*b){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)gut_project(i,h,w,m,c,k,b);}
__global__ void forward_kernel(int h,int w,const float*m,const float*q,const float*c,const float*o,const float*k,const int*s,const int*ids,float*out){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<h*w)gut_forward(p,h,w,m,q,c,o,k,s,ids,out);}
__global__ void backward_kernel(int h,int w,const float*m,const float*q,const float*c,const float*o,const float*k,const int*s,const int*ids,const float*im,const float*g,float*gm,float*gp,float*gc,float*go){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<h*w)gut_backward(p,h,w,m,q,c,o,k,s,ids,im,g,gm,gp,gc,go);}
void check(torch::Tensor t,torch::Tensor m,bool integer=false){TORCH_CHECK(t.is_cuda()&&t.device()==m.device()&&t.is_contiguous(),"Expected contiguous tensors on the same CUDA device");TORCH_CHECK(t.scalar_type()==(integer?torch::kInt32:torch::kFloat32),"Expected float32 / int32 tensors");}
torch::Tensor project(torch::Tensor m,torch::Tensor cov,torch::Tensor k,int h,int w){
  check(m,m);check(cov,m);check(k,m);c10::cuda::CUDAGuard guard(m.device());auto b=torch::empty({m.size(0),4},m.options());
  if(m.size(0))project_kernel<<<(m.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(m.size(0),h,w,m.data_ptr<float>(),cov.data_ptr<float>(),k.data_ptr<float>(),b.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();return b;
}
torch::Tensor forward(torch::Tensor m,torch::Tensor q,torch::Tensor c,torch::Tensor o,torch::Tensor k,torch::Tensor s,torch::Tensor ids,int h,int w){
  for(auto t:{m,q,c,o,k})check(t,m);check(s,m,true);check(ids,m,true);c10::cuda::CUDAGuard guard(m.device());auto out=torch::empty({h,w,4},m.options());
  forward_kernel<<<(h*w+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(h,w,m.data_ptr<float>(),q.data_ptr<float>(),c.data_ptr<float>(),o.data_ptr<float>(),k.data_ptr<float>(),s.data_ptr<int>(),ids.data_ptr<int>(),out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> backward(torch::Tensor m,torch::Tensor q,torch::Tensor c,torch::Tensor o,torch::Tensor k,torch::Tensor s,torch::Tensor ids,torch::Tensor im,torch::Tensor g,int h,int w){
  for(auto t:{m,q,c,o,k,im,g})check(t,m);check(s,m,true);check(ids,m,true);c10::cuda::CUDAGuard guard(m.device());auto gm=torch::zeros_like(m),gp=torch::zeros_like(q),gc=torch::zeros_like(c),go=torch::zeros_like(o);
  backward_kernel<<<(h*w+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(h,w,m.data_ptr<float>(),q.data_ptr<float>(),c.data_ptr<float>(),o.data_ptr<float>(),k.data_ptr<float>(),s.data_ptr<int>(),ids.data_ptr<int>(),im.data_ptr<float>(),g.data_ptr<float>(),gm.data_ptr<float>(),gp.data_ptr<float>(),gc.data_ptr<float>(),go.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();return {gm,gp,gc,go};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("project",&project);m.def("forward",&forward);m.def("backward",&backward);}
