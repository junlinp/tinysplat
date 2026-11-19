#include <metal_stdlib>
using namespace metal;
#define DEVICE inline
#define CPTR device const float*
#define WPTR device float*
#define TPTR thread float*
#define TCPTR thread const float*
#define IPTR device const int*
#define APTR device atomic_uint*
#define SQRT sqrt
#define FMIN min
#define FMAX max
#define SIN sin
#define COS cos
#define ATAN2 atan2
#define EXP exp
#define ABS abs
#define FLOOR floor
#define CEIL ceil
inline void gut_add(device atomic_uint* ptr,float value){
  uint old=atomic_load_explicit(ptr,memory_order_relaxed);
  while(!atomic_compare_exchange_weak_explicit(ptr,&old,as_type<uint>(as_type<float>(old)+value),memory_order_relaxed,memory_order_relaxed)){}
}
#define ADD gut_add
// The shared math header is concatenated here by build_native_3dgut.py.
// COMMON_SOURCE
#define GUT_ARGS device const int*params [[buffer(0)]],device const float*m [[buffer(1)]],device const float*cov [[buffer(2)]],device const float*q [[buffer(3)]],device const float*c [[buffer(4)]],device const float*o [[buffer(5)]],device const float*k [[buffer(6)]],device const int*s [[buffer(7)]],device const int*ids [[buffer(8)]],device const float*im [[buffer(9)]],device const float*g [[buffer(10)]],device float*out [[buffer(11)]],device atomic_uint*gm [[buffer(12)]],device atomic_uint*gp [[buffer(13)]],device atomic_uint*gc [[buffer(14)]],device atomic_uint*go [[buffer(15)]],uint tid [[thread_position_in_grid]]
kernel void gut_projection(GUT_ARGS){if(tid<uint(params[0]))gut_project(tid,params[1],params[2],m,cov,k,out);}
kernel void gut_raster(GUT_ARGS){if(tid<uint(params[1]*params[2]))gut_forward(tid,params[1],params[2],m,q,c,o,k,s,ids,out);}
kernel void gut_vjp(GUT_ARGS){if(tid<uint(params[1]*params[2]))gut_backward(tid,params[1],params[2],m,q,c,o,k,s,ids,im,g,gm,gp,gc,go);}
