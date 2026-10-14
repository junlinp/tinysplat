#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <mutex>
#include "build/native_3dgut_source.h"
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipelines[3];
static std::once_flag once;
static void initialize(){
  device=MTLCreateSystemDefaultDevice();if(!device)return;
  queue=[device newCommandQueue];NSError*error=nil;
  MTLCompileOptions*options=[MTLCompileOptions new];options.fastMathEnabled=NO;
  id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:gut_source] options:options error:&error];
  if(!lib){NSLog(@"TinySplat 3DGUT shader error: %@",error);return;}
  const char*names[]={"gut_projection","gut_raster","gut_vjp"};
  for(int i=0;i<3;i++)pipelines[i]=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:names[i]]] error:&error];
}
extern "C" int tinysplat_gut_available(){std::call_once(once,initialize);return pipelines[0]&&pipelines[1]&&pipelines[2]?1:0;}
extern "C" int tinysplat_gut_run(int mode,int n,int h,int w,const float*m,const float*cov,const float*q,const float*c,const float*o,const float*k,const int*s,const int*ids,int numids,const float*im,const float*g,float*out,float*gm,float*gp,float*gc,float*go){
  @autoreleasepool {
    if(mode<0||mode>2||!tinysplat_gut_available())return -1;
    int params[]={n,h,w,numids};size_t pixels=size_t(h)*w,tiles=size_t((h+15)/16)*((w+15)/16);
    const void*ptrs[]={params,m,cov,q,c,o,k,s,ids,im,g,nullptr,nullptr,nullptr,nullptr,nullptr};
    size_t sizes[]={sizeof(params),size_t(n)*3*4,size_t(n)*9*4,size_t(n)*9*4,size_t(n)*3*4,size_t(n)*4,10*4,(tiles+1)*4,size_t(numids)*4,pixels*4*4,pixels*4*4,mode==0?size_t(n)*4*4:pixels*4*4,size_t(n)*3*4,size_t(n)*9*4,size_t(n)*3*4,size_t(n)*4};
    id<MTLBuffer> buffers[16];
    for(int i=0;i<16;i++){
      size_t bytes=sizes[i]>0?sizes[i]:4;
      buffers[i]=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];if(!buffers[i])return -2;
      if(ptrs[i]&&sizes[i])std::memcpy(buffers[i].contents,ptrs[i],sizes[i]);else std::memset(buffers[i].contents,0,bytes);
    }
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipelines[mode]];for(int i=0;i<16;i++)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
    size_t count=mode==0?size_t(n):pixels;
    if(count)[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status==MTLCommandBufferStatusError){NSLog(@"TinySplat 3DGUT command error: %@",command.error);return -3;}
    if(mode<2&&out&&sizes[11])std::memcpy(out,buffers[11].contents,sizes[11]);
    float*grads[]={gm,gp,gc,go};if(mode==2)for(int i=0;i<4;i++)if(grads[i]&&sizes[i+12])std::memcpy(grads[i],buffers[i+12].contents,sizes[i+12]);
    return 0;
  }
}
