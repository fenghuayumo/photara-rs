/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/projection.h"
#include "kernel/film/stable_planes.h"

CCL_NAMESPACE_BEGIN

struct KernelTemporalPlane {
  float3 color, virtual_position, position, normal, view, throughput;
  float3 previous_virtual_position, previous_position;
  float roughness;
  uint branch, key;
  bool valid, sky;
};
struct KernelTemporalPixel {
  KernelTemporalPlane planes[3];
  float4 color;
  float3 position, normal, previous_position;
  uint surface_key;
  float roughness, weight, object_id;
};
struct KernelTemporalFrame {
  ProjectionTransform worldtoraster;
  Transform worldtocamera;
  float3 camera_position;
  float2 lookup_shift;
  int width, height, stride, combined, position, normal, roughness, object_id, count, samples, planes;
  int surface, frame_number;
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
  p.previous_position = f.surface == PASS_UNUSED ? p.position :
      make_float3(src[f.surface], src[f.surface+1], src[f.surface+2]);
  p.surface_key = f.surface == PASS_UNUSED ? __float_as_uint(p.object_id) :
      __float_as_uint(src[f.surface+3]);
  p.weight = float(n);
  for (int k = 0; k < 3; ++k) {
    KernelTemporalPlane plane = {};
    if (f.planes != PASS_UNUSED) {
      ccl_global const float *r = src + f.planes + k*STABLE_PLANE_STRIDE;
      plane.branch = __float_as_uint(r[0]); plane.key = __float_as_uint(r[1]);
      plane.previous_virtual_position = make_float3(r[24], r[25], r[26]);
      plane.previous_position = make_float3(r[27], r[28], r[29]);
      plane.virtual_position = make_float3(r[2], r[3], r[4]);
      plane.position = make_float3(r[5], r[6], r[7]);
      plane.normal = make_float3(r[8], r[9], r[10]);
      plane.roughness = r[11]; plane.view = make_float3(r[12], r[13], r[14]);
      plane.color = make_float3(r[15], r[16], r[17]) * inv;
      plane.throughput = make_float3(r[18], r[19], r[20]);
      plane.color = safe_divide(plane.color, plane.throughput);
      plane.sky = r[21] == 2.0f;
      plane.valid = reduce_min(plane.throughput) > 1e-6f && r[21] > 0.0f && plane.branch != 0;
    }
    p.planes[k] = plane;
  }
  ring[size_t(slot) * f.width * f.height + i] = p;
}

ccl_device void temporal_reproject(ccl_global float *film,
                                   ccl_global const KernelTemporalPixel *ring,
                                   ccl_global const KernelTemporalFrame *frames,
                                   ccl_global uint *counts,
                                   const int slot, const int capacity, const int history,
                                   const int i)
{
#ifdef __KERNEL_METAL__
  using Accumulator = float;
#else
  using Accumulator = double;
#endif
  const KernelTemporalFrame f = frames[slot];
  const size_t size = size_t(f.width) * f.height;
  const KernelTemporalPixel p = ring[size_t(slot) * size + i];
  counts[i] = 0;
  if (f.planes != PASS_UNUSED && p.weight > 0.0f && isfinite(p.weight)) {
    /* The unclassified residual is current-frame-only. Keep each history layer
     * independent; never warp a Fresnel mixture using one dominant surface. */
    float3 result = make_float3(p.color);
    uint reused = 0, active = 0, cross_frame = 0, moving = 0;
    for (int k = 0; k < 3; ++k) {
      const KernelTemporalPlane a = p.planes[k];
      if (!a.valid || !isfinite(len_squared(a.color)) ||
          !isfinite(len_squared(a.position)) || !isfinite(len_squared(a.normal)) ||
          !isfinite(a.roughness) || !isfinite(len_squared(a.view)) ||
          !isfinite(len_squared(a.throughput))) continue;
      ++active;
      Accumulator sums[3] = {Accumulator(a.color.x)*p.weight, Accumulator(a.color.y)*p.weight,
                             Accumulator(a.color.z)*p.weight};
      Accumulator weights = p.weight;
      uint accepted = 0;
      KernelTemporalPlane cursor = a;
      int cursor_frame = f.frame_number;
      for (int age = 1; age <= history; ++age) {
        const int old_slot = (slot + capacity - age) % capacity;
        const KernelTemporalFrame old = frames[old_slot];
        const bool same_frame = old.frame_number == cursor_frame;
        if (!same_frame && old.frame_number != cursor_frame - 1) break;
        const float3 proposal_position = same_frame ? cursor.virtual_position : cursor.previous_virtual_position;
        const float3 target_position = same_frame ? cursor.position : cursor.previous_position;
        const float3 proposal = a.sky ? old.camera_position +
            safe_normalize(a.virtual_position-f.camera_position)*1e6f : proposal_position;
        const float3 projected = transform_perspective(&old.worldtoraster, proposal);
        if (!isfinite(projected.x) || !isfinite(projected.y) ||
            fabsf(projected.x) > 1e6f || fabsf(projected.y) > 1e6f) break;
        const int x = int(floorf(projected.x + old.lookup_shift.x));
        const int y = int(floorf(projected.y + old.lookup_shift.y));
        const float depth = transform_point(&old.worldtocamera, proposal).z;
        if (!(depth > 0.0f) || !isfinite(depth)) break;
        const float footprint = (a.sky ? 1.0f : max(depth, 1e-3f))*2.0f / float(max(f.width, f.height));
        float best = footprint*footprint;
        int match_pixel = -1, match_plane = -1;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
          const int px = x+dx, py = y+dy;
          if (px < 0 || py < 0 || px >= f.width || py >= f.height) continue;
          const KernelTemporalPixel q = ring[size_t(old_slot)*size + size_t(py)*f.width+px];
          if (!(q.weight > 0.0f) || !isfinite(q.weight)) continue;
          for (int t = 0; t < 3; ++t) {
            const KernelTemporalPlane b = q.planes[t];
            if (!b.valid || a.sky != b.sky || !isfinite(b.roughness) ||
                !isfinite(len_squared(b.view)) || a.branch != b.branch || a.key != b.key ||
                !isfinite(len_squared(b.color)) || !isfinite(len_squared(b.normal)) ||
                dot(a.normal, b.normal) < 0.98f) continue;
            const float d = len_squared(target_position-b.position);
            const float rough = clamp(min(a.roughness, b.roughness), 0.0f, 1.0f);
            if (!isfinite(d) || d >= best ||
                dot(a.view, b.view) < 1.0f-max(1e-7f, rough*rough*0.002f)) continue;
            best = d; match_pixel = py*f.width+px; match_plane = t;
          }
        }
        if (match_pixel >= 0) {
          const KernelTemporalPixel q = ring[size_t(old_slot)*size+match_pixel];
          const KernelTemporalPlane matched = q.planes[match_plane];
          const float3 color = matched.color;
          cross_frame |= !same_frame;
          moving |= !same_frame && len_squared(cursor.position-cursor.previous_position) > 1e-10f;
          cursor = matched;
          cursor_frame = old.frame_number;
          sums[0] += Accumulator(color.x)*q.weight; sums[1] += Accumulator(color.y)*q.weight;
          sums[2] += Accumulator(color.z)*q.weight; weights += q.weight; ++accepted;
        }
        else break; /* Never jump past a disoccluded intermediate frame. */
      }
      if (weights > 0.0) result += make_float3(float(sums[0]/weights),
          float(sums[1]/weights), float(sums[2]/weights)) * a.throughput - a.color * a.throughput;
      reused = max(reused, accepted);
    }
    ccl_global float *dst = film + size_t(i)*f.stride + f.combined;
    if (isfinite(len_squared(result))) {
      dst[0] = result.x*p.weight; dst[1] = result.y*p.weight; dst[2] = result.z*p.weight;
      counts[i] = reused | (cross_frame << 8) | (moving << 9) | (active << 16);
    }
    return;
  }
  if (p.weight <= 0 || !isfinite(p.weight) || !isfinite(p.roughness) ||
      !isfinite(p.object_id) || !isfinite(len_squared(p.position)) ||
      len_squared(p.normal) < 0.5f ||
      !isfinite(len_squared(p.normal)) || !isfinite(p.color.x) || !isfinite(p.color.y) ||
      !isfinite(p.color.z) || !isfinite(p.color.w) || fabsf(p.color.w) > 0.001f) return;
  Accumulator sum[4] = {Accumulator(p.color.x)*p.weight, Accumulator(p.color.y)*p.weight,
                   Accumulator(p.color.z)*p.weight, Accumulator(p.color.w)*p.weight};
  Accumulator weight = p.weight;
  uint accepted = 0, cross_frame = 0, moving = 0;
  KernelTemporalPixel cursor = p;
  int cursor_frame = f.frame_number;
  for (int age = 1; age <= history; ++age) {
    const int old_slot = (slot + capacity - age) % capacity;
    const KernelTemporalFrame old = frames[old_slot];
    const bool same_frame = old.frame_number == cursor_frame;
    if (!same_frame && old.frame_number != cursor_frame - 1) break;
    const float3 target = same_frame ? cursor.position : cursor.previous_position;
    const float3 cam = transform_point(&old.worldtocamera, target);
    if (!isfinite(cam.z) || cam.z <= 0) break;
    const float3 raster = transform_perspective(&old.worldtoraster, target);
    const float rx = raster.x + old.lookup_shift.x, ry = raster.y + old.lookup_shift.y;
    if (!isfinite(rx) || !isfinite(ry) || rx < 0 || ry < 0 ||
        rx >= f.width || ry >= f.height) break;
    const float footprint = max(cam.z, 1e-3f)*2.0f / float(max(f.width, f.height));
    float best = footprint*footprint;
    int match = -1;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
      const int x = int(rx)+dx, y = int(ry)+dy;
      if (x < 0 || y < 0 || x >= f.width || y >= f.height) continue;
      const int pixel = y*f.width+x;
      const KernelTemporalPixel q = ring[size_t(old_slot)*size+pixel];
      if (q.weight <= 0 || !isfinite(q.weight) || !isfinite(q.roughness) ||
          q.surface_key != p.surface_key || !isfinite(len_squared(q.position)) ||
          fabsf(q.color.w) > 0.001f || !isfinite(len_squared(q.normal)) ||
          len_squared(q.normal) < 0.5f ||
          dot(normalize(cursor.normal), normalize(q.normal)) < 0.98f) continue;
      const float3 old_cam = transform_point(&old.worldtocamera, q.position);
      const float distance = len_squared(target-q.position);
      if (fabsf(cam.z-old_cam.z) > max(footprint*0.5f, 1e-4f) || distance >= best) continue;
      const float3 view = safe_normalize(f.camera_position-p.position);
      const float3 old_view = safe_normalize(old.camera_position-q.position);
      const float rough = clamp(min(p.roughness, q.roughness), 0.0f, 1.0f);
      if (dot(view, old_view) < 1.0f-max(1e-7f, rough*rough*0.002f)) continue;
      if (!isfinite(q.color.x) || !isfinite(q.color.y) || !isfinite(q.color.z) ||
          !isfinite(q.color.w)) continue;
      best = distance; match = pixel;
    }
    if (match < 0) break; /* Reject disocclusions; never skip intermediate frames. */
    const KernelTemporalPixel q = ring[size_t(old_slot)*size+match];
    sum[0] += Accumulator(q.color.x)*q.weight; sum[1] += Accumulator(q.color.y)*q.weight;
    sum[2] += Accumulator(q.color.z)*q.weight; sum[3] += Accumulator(q.color.w)*q.weight;
    weight += q.weight;
    ++accepted;
    cross_frame |= !same_frame;
    moving |= !same_frame && len_squared(cursor.position-cursor.previous_position) > 1e-10f;
    cursor = q;
    cursor_frame = old.frame_number;
  }
  ccl_global float *dst = film + size_t(i)*f.stride + f.combined;
  /* Moving shadows/lighting can change on otherwise stationary geometry.
   * Bound history by the current surface neighborhood to limit stale trails. */
  float3 lower = make_float3(p.color), upper = lower;
  for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
    const int x = i % f.width + dx, y = i / f.width + dy;
    if (x < 0 || y < 0 || x >= f.width || y >= f.height) continue;
    const KernelTemporalPixel q = ring[size_t(slot)*size + size_t(y)*f.width+x];
    if (q.surface_key != p.surface_key || !isfinite(len_squared(make_float3(q.color)))) continue;
    lower = min(lower, make_float3(q.color)); upper = max(upper, make_float3(q.color));
  }
  const float3 color = clamp(make_float3(float(sum[0]/weight), float(sum[1]/weight),
                                       float(sum[2]/weight)), lower, upper);
  dst[0] = color.x*p.weight; dst[1] = color.y*p.weight; dst[2] = color.z*p.weight;
  dst[3] = p.color.w*p.weight;
  counts[i] = accepted | (cross_frame << 8) | (moving << 9);
}
#endif
CCL_NAMESPACE_END
