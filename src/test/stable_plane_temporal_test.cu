/* SPDX-License-Identifier: Apache-2.0 */
#include <cuda_runtime.h>
#include <cstdio>
#include "kernel/device/cuda/compat.h"
#define PASS_UNUSED (-1)
#include "kernel/film/temporal_history.h"

__global__ void check_planes(float *out) {
  KernelTemporalPixel ring[2] = {};
  KernelTemporalFrame frames[2] = {};
  for (int i=0;i<2;++i) {
    frames[i].width=frames[i].height=frames[i].samples=1;
    frames[i].stride=4; frames[i].combined=0; frames[i].planes=0;
    frames[i].worldtocamera=make_transform(1,0,0,0,0,1,0,0,0,0,1,0);
    frames[i].worldtoraster={make_float4(1,0,0,0),make_float4(0,1,0,0),
                            make_float4(0,0,1,0),make_float4(0,0,0,1)};
    ring[i].weight=1; ring[i].color=make_float4(10,10,10,0);
  }
  for(int k=0;k<2;++k) {
    KernelTemporalPlane p={}; p.branch=k+5; p.key=k+123; p.valid=true;
    p.previous_position=p.previous_virtual_position=p.position=p.virtual_position=make_float3(0,0,1);
    p.normal=p.view=make_float3(0,0,1); p.roughness=1;
    p.color=make_float3(k?4:2); p.throughput=make_float3(k?.25f:.5f);
    ring[1].planes[k]=p; p.color=make_float3(k?8:6);
    ring[0].planes[1-k]=p; // plane slot order is not an identity
  }
  float film[4]={10,10,10,0}; uint count=0;
  temporal_reproject(film,ring,frames,&count,1,2,1,0);
  out[0]=fabsf(film[0]-11.5f)<1e-6f && (count & 0xffffu)==1;
  for(int k=0;k<2;++k) ring[0].planes[k].key+=1;
  film[0]=film[1]=film[2]=10;
  temporal_reproject(film,ring,frames,&count,1,2,1,0);
  out[1]=film[0]==10 && (count & 0xffffu)==0; // optical branch/material changes reject
  for(int k=0;k<2;++k) {ring[0].planes[k].key-=1;ring[0].planes[k].position.x=10;}
  temporal_reproject(film,ring,frames,&count,1,2,1,0);
  out[2]=film[0]==10 && (count & 0xffffu)==0; // disocclusion rejects, despite virtual match
  temporal_reproject(film,ring,frames,&count,1,2,0,0);
  out[3]=film[0]==10 && (count & 0xffffu)==0; // no recursive/expired history
  // Actual motion: current target moved, but previous target/virtual positions
  // point to the raw sample in the previous frame.
  for (int k=0;k<2;++k) {
    ring[0].planes[k].position=make_float3(0,0,1);
    ring[1].planes[k].position=make_float3(10,0,1);
    ring[1].planes[k].virtual_position=make_float3(10,0,1);
  }
  temporal_reproject(film,ring,frames,&count,1,2,1,0);
  out[4]=fabsf(film[0]-11.5f)<1e-6f && (count & 0xffffu)==1;
  float records[STABLE_PLANE_COUNT*STABLE_PLANE_STRIDE]={};
  records[0]=__uint_as_float(1); records[22]=1;
  records[STABLE_PLANE_STRIDE]=__uint_as_float(5); // reflected child
  records[STABLE_PLANE_STRIDE+22]=1;
  records[2*STABLE_PLANE_STRIDE]=__uint_as_float(26); // transmitted grandchild
  records[2*STABLE_PLANE_STRIDE+22]=1;
  out[5]=stable_plane_find(records,5)==1 && stable_plane_find(records,26)==2 &&
         stable_plane_find(records,25)==0 && stable_plane_find(records,0x8000001au)==2;
  out[6]=stable_plane_emission_covered(records,1) &&
         stable_plane_emission_covered(records,6) &&
         stable_plane_emission_covered(records,26) &&
         !stable_plane_emission_covered(records,25) &&
         !stable_plane_emission_covered(records,0x8000001au);
  records[2*STABLE_PLANE_STRIDE+22]=0;
  out[7]=!stable_plane_emission_covered(records,26) &&
         stable_plane_emission_covered(records,6); // unsupported terminal keeps its emission

}
int main() {
  float *device=nullptr, result[8]={};
  if(cudaMalloc(&device,sizeof(result))!=cudaSuccess) return 2;
  check_planes<<<1,1>>>(device);
  const auto error=cudaMemcpy(result,device,sizeof(result),cudaMemcpyDeviceToHost);
  cudaFree(device);
  if(error!=cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(error));return 2;}
  for(int i=0;i<8;++i) if(result[i]!=1) {std::fprintf(stderr,"case %d failed\n",i);return 1;}
  std::puts("8 GPU plane decomposition and reprojection checks passed");
}
