/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/projection.h"

CCL_NAMESPACE_BEGIN

struct KernelTemporalPixel {
  float4 color;
  float3 position, normal;
  float roughness, weight, object_id;
};
struct KernelTemporalFrame {
  ProjectionTransform worldtoraster;
  Transform worldtocamera;
  float3 camera_position;
  float2 lookup_shift;
  int width, height, stride, combined, position, normal, roughness, object_id, count, samples;
};

#ifdef __KERNEL_GPU__
ccl_device void temporal_snapshot(ccl_global const float *film,
                                  ccl_global KernelTemporalPixel *ring,
                                  ccl_global const KernelTemporalFrame *frames,
                                  const int slot, const int i)
{
  const KernelTemporalFrame f = frames[slot];
  ccl_global const float *src = film + size_t(i) * f.stride;
  const uint n = f.count == PASS_UNUSED ? uint(f.samples) : __float_as_uint(src[f.count]);
  const float inv = n ? 1.0f / float(n) : 0.0f;
  KernelTemporalPixel p;
  p.color = make_float4(src[f.combined], src[f.combined+1], src[f.combined+2],
                        src[f.combined+3]) * inv;
  p.position = make_float3(src[f.position], src[f.position+1], src[f.position+2]);
  p.normal = make_float3(src[f.normal], src[f.normal+1], src[f.normal+2]) * inv;
  p.roughness = src[f.roughness] * inv;
  p.object_id = src[f.object_id];
  p.weight = float(n);
  ring[size_t(slot) * f.width * f.height + i] = p;
}

ccl_device void temporal_reproject(ccl_global float *film,
                                   ccl_global const KernelTemporalPixel *ring,
                                   ccl_global const KernelTemporalFrame *frames,
                                   ccl_global uint *counts,
                                   const int slot, const int capacity, const int history,
                                   const int i)
{
  const KernelTemporalFrame f = frames[slot];
  const size_t size = size_t(f.width) * f.height;
  const KernelTemporalPixel p = ring[size_t(slot) * size + i];
  counts[i] = 0;
  if (p.weight <= 0 || !isfinite(p.weight) || !isfinite(p.roughness) ||
      !isfinite(p.object_id) || !isfinite(len_squared(p.position)) ||
      len_squared(p.normal) < 0.5f ||
      !isfinite(len_squared(p.normal)) || !isfinite(p.color.x) || !isfinite(p.color.y) ||
      !isfinite(p.color.z) || !isfinite(p.color.w) || fabsf(p.color.w) > 0.001f) return;
  double sum[4] = {double(p.color.x)*p.weight, double(p.color.y)*p.weight,
                   double(p.color.z)*p.weight, double(p.color.w)*p.weight};
  double weight = p.weight;
  uint accepted = 0;
  for (int age = history; age > 0; --age) {
    const int old_slot = (slot + capacity - age) % capacity;
    const KernelTemporalFrame old = frames[old_slot];
    const float3 cam = transform_point(&old.worldtocamera, p.position);
    if (!isfinite(cam.z) || cam.z <= 0) continue;
    const float3 raster = transform_perspective(&old.worldtoraster, p.position);
    const float rx = raster.x + old.lookup_shift.x, ry = raster.y + old.lookup_shift.y;
    if (!isfinite(rx) || !isfinite(ry) || rx < 0 || ry < 0 ||
        rx >= f.width || ry >= f.height) continue;
    const KernelTemporalPixel q = ring[size_t(old_slot)*size + size_t(int(ry))*f.width + int(rx)];
    if (q.weight <= 0 || !isfinite(q.weight) || !isfinite(q.roughness) ||
        q.object_id != p.object_id || !isfinite(len_squared(q.position)) ||
        fabsf(q.color.w) > 0.001f ||
        !isfinite(len_squared(q.normal)) || len_squared(q.normal) < 0.5f ||
        dot(normalize(p.normal), normalize(q.normal)) < 0.98f) continue;
    const float3 old_cam = transform_point(&old.worldtocamera, q.position);
    const float footprint = max(cam.z, 1e-3f)*2.0f / float(max(f.width, f.height));
    if (fabsf(cam.z-old_cam.z) > max(footprint*0.5f, 1e-4f) ||
        len_squared(p.position-q.position) > footprint*footprint) continue;
    const float3 view = safe_normalize(f.camera_position-p.position);
    const float3 old_view = safe_normalize(old.camera_position-p.position);
    const float rough = clamp(min(p.roughness, q.roughness), 0.0f, 1.0f);
    if (dot(view, old_view) < 1.0f-max(1e-7f, rough*rough*0.002f)) continue;
    if (!isfinite(q.color.x) || !isfinite(q.color.y) || !isfinite(q.color.z) ||
        !isfinite(q.color.w)) continue;
    sum[0] += double(q.color.x)*q.weight; sum[1] += double(q.color.y)*q.weight;
    sum[2] += double(q.color.z)*q.weight; sum[3] += double(q.color.w)*q.weight;
    weight += q.weight;
    ++accepted;
  }
  ccl_global float *dst = film + size_t(i)*f.stride + f.combined;
  for (int c = 0; c < 4; ++c) dst[c] = float(sum[c]/weight*p.weight);
  counts[i] = accepted;
}
#endif
CCL_NAMESPACE_END
