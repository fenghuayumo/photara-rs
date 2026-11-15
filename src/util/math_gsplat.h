/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/color.h"
#include "util/math.h"
#include "util/transform.h"

CCL_NAMESPACE_BEGIN

ccl_static_constexpr int GSPLAT_RASTER_TILE_SIZE = 16;
ccl_static_constexpr float GSPLAT_RASTER_EXTENT = 8.0f;
ccl_static_constexpr float GSPLAT_RASTER_ALPHA_MIN = 1.0f / 255.0f;
/* Bit 7 is outside PATH_RAY_VISIBILITY_ALL. Use it to trace the mesh layer
 * without visiting the BVHs of rasterized point clouds. */
ccl_static_constexpr uint GSPLAT_RASTER_MESH_MASK = 128;
ccl_static_constexpr uint GSPLAT_RASTER_SHADOW = (1u << 14);

ccl_device_inline int2 gsplat_shadow_receiver_grid(const int width, const int height)
{
  const int x = min((width + GSPLAT_RASTER_TILE_SIZE - 1) / GSPLAT_RASTER_TILE_SIZE, 60);
  return make_int2(x, (height * x + width - 1) / width);
}

/* Weighted affine inverse-depth fit in a local image patch. Coordinates are
 * relative to the query and inverse depths relative to a nearby depth anchor.
 * The covariance test rejects lines, sparse support and non-planar layers. */
#ifndef __KERNEL_GPU__
struct GSplatShadowReceiverFit {
  float weight = 0, weight2 = 0;
  float3 sum = {}, covariance = {}, depth_covariance = {};

  /* Input is sorted far-to-near by inverse depth. Pick the nearest cluster
   * with distributed, significant support instead of averaging table and
   * occluded floor centers. A single bright floater is not sufficient. */
  static float front_layer(const float2 *depths, const size_t count)
  {
    float maximum = 0, anchor = 0;
    for (int pass = 0; pass < 2; ++pass) {
      size_t begin = 0;
      float total = 0, square = 0, moment = 0;
      for (size_t end = 0; end < count; ++end) {
        const float2 d = depths[end];
        total += d.y;
        square += d.y * d.y;
        moment += d.x * d.y;
        while (begin < end && depths[begin].x * 1.075f < d.x) {
          const float2 old = depths[begin++];
          total -= old.y;
          square -= old.y * old.y;
          moment -= old.x * old.y;
        }
        const bool supported = end - begin + 1 >= 8 && total * total / fmaxf(square, 1e-20f) >= 8;
        if (pass == 0 && supported) {
          maximum = fmaxf(maximum, total);
        }
        else if (total > 0.1f && total >= 0.1f * maximum && supported && pass == 1) {
          anchor = moment / total;
        }
      }
    }
    return anchor;
  }

  void add(const float2 p, const float z, const float w)
  {
    weight += w;
    weight2 += w * w;
    sum += w * make_float3(p.x, p.y, z);
    covariance += w * make_float3(p.x * p.x, p.x * p.y, p.y * p.y);
    depth_covariance += w * make_float3(p.x * z, p.y * z, z * z);
  }

  float4 solve(const float anchor) const
  {
    if (!(weight > 0 && anchor > 0)) {
      return zero_float4();
    }
    const float3 mean = sum / weight;
    const float3 c = covariance / weight -
                     make_float3(mean.x * mean.x, mean.x * mean.y, mean.y * mean.y);
    const float3 d = depth_covariance / weight -
                     make_float3(mean.x * mean.z, mean.y * mean.z, mean.z * mean.z);
    const float ridge = 1e-5f * (c.x + c.z) + 1e-7f;
    const float determinant = (c.x + ridge) * (c.z + ridge) - c.y * c.y;
    const float2 slope = make_float2(((c.z + ridge) * d.x - c.y * d.y) / determinant,
                                     ((c.x + ridge) * d.y - c.y * d.x) / determinant);
    const float intercept = anchor + mean.z - slope.x * mean.x - slope.y * mean.y;
    const float residual = sqrtf(fmaxf(0.0f, d.z - slope.x * d.x - slope.y * d.y));
    const float effective = weight * weight / fmaxf(weight2, 1e-20f);
    const float conditioning = fmaxf(0.0f, c.x * c.z - c.y * c.y) / fmaxf(sqr(c.x + c.z), 1e-20f);
    const float confidence = smoothstep(4.0f, 12.0f, effective) *
                             smoothstep(0.001f, 0.02f, conditioning) *
                             (1 - smoothstep(0.005f, 0.02f, residual / anchor));
    if (!(intercept > 0 && isfinite_safe(intercept) && isfinite_safe(slope.x) &&
          isfinite_safe(slope.y)))
    {
      return zero_float4();
    }
    return make_float4(intercept, slope.x, slope.y, confidence);
  }
};
#endif

/* Attenuate linear radiance, then return to the training compositing space.
 * A strength of 0.5 must halve radiance rather than halve encoded RGB. */
ccl_device_inline float3 gsplat_shadow_color(const float3 color, const float3 attenuation)
{
  if (reduce_min(attenuation) >= 1.0f) {
    return color;
  }
  return color_linear_to_srgb_v3(color_srgb_to_linear_v3(color) * attenuation);
}

/* An affine inverse-depth guide is a plane in perspective camera space.
 * Differentiate P = q / inverse_depth, with q.z = 1, to recover its normal.
 * Particle covariance axes need not agree with the captured surface. */
ccl_device_inline float3 gsplat_receiver_normal(const float3 q,
                                                const float3 dx,
                                                const float3 dy,
                                                const float3 inverse_depth)
{
  const float3 tx = dx * inverse_depth.x - q * inverse_depth.y;
  const float3 ty = dy * inverse_depth.x - q * inverse_depth.z;
  const float3 N = safe_normalize(cross(tx, ty));
  return dot(N, q) > 0 ? -N : N;
}

/* Conditional mean camera depth given the projected Gaussian position. The
 * low-pass covariance regularizes collapsed/grazing splats without dividing by
 * an estimated surface normal. Mesh occlusion and shadow queries use the same local depth. */
ccl_device_inline float2 gsplat_project_depth_gradient(const float3 center,
                                                       const float3x3 axes,
                                                       const float2 focal,
                                                       const float3 covariance)
{
  const float3 u = (focal.x / center.z) * (axes.x - (center.x / center.z) * axes.z);
  const float3 v = (focal.y / center.z) * (axes.y - (center.y / center.z) * axes.z);
  const float a = dot(axes.z, u), b = dot(axes.z, v);
  const float determinant = covariance.x * covariance.z - sqr(covariance.y);
  return make_float2((a * covariance.z - b * covariance.y) / determinant,
                     (b * covariance.x - a * covariance.y) / determinant);
}

/* Conditional depth of a projected Gaussian at the actual raster sample. */
ccl_device_inline float gsplat_projected_depth(const float4 center,
                                              const float4 conic,
                                              const float4 color,
                                              const float3 pixel)
{
  return center.z + conic.w * (pixel.x - center.x) + color.w * (pixel.y - center.y);
}

/* Material-aware artistic opacity for the Gaussian shadow overlay. */
ccl_device_inline float gsplat_material_shadow_opacity(const float transmission,
                                                       const float glass_opacity)
{
  return 1 - saturatef(transmission) * (1 - saturatef(glass_opacity));
}

/* Caustica's straight-through glass approximation: split a closed shell's
 * tint/transmission across its entry and exit, then apply interface Fresnel. */
ccl_device_inline float4 gsplat_material_shadow_transmission(const float3 color,
                                                             const float transmission,
                                                             const float ior,
                                                             const bool thin)
{
  const float amount = saturatef(transmission);
  float3 value = min(max(color * amount, zero_float3()), one_float3());
  if (!thin) {
    value = sqrt(value);
  }
  const float ratio = (ior - 1) / fmaxf(ior + 1, 1e-4f);
  value *= 1 - saturatef(ratio * ratio * amount);
  return make_float4(value, amount);
}

/* Scaled Gaussian axes are columns of axes, in camera space. */
ccl_device_inline float3 gsplat_project_covariance(const float3 center,
                                                   const float3x3 axes,
                                                   const float2 focal,
                                                   const float dilation)
{
  const float3 u = (focal.x / center.z) * (axes.x - (center.x / center.z) * axes.z);
  const float3 v = (focal.y / center.z) * (axes.y - (center.y / center.z) * axes.z);
  return make_float3(dot(u, u) + dilation, dot(u, v), dot(v, v) + dilation);
}

ccl_device_inline float gsplat_projected_alpha(const float3 conic,
                                               const float2 offset,
                                               const float opacity)
{
  const float e = conic.x * sqr(offset.x) + 2.0f * conic.y * offset.x * offset.y +
                  conic.z * sqr(offset.y);
  if (!(e >= 0.0f && e <= GSPLAT_RASTER_EXTENT)) {
    return 0.0f;
  }
  const float alpha = fminf(0.99f, opacity * expf(-0.5f * e));
  return alpha >= GSPLAT_RASTER_ALPHA_MIN ? alpha : 0.0f;
}

/* Maximum density along a ray, including the planar limit produced by raster
 * training. Do not invert a collapsed axis or subtract large squared norms. */
ccl_device_inline float gsplat_ray_alpha(const float3 origin,
                                         const float3 direction,
                                         const float3 center,
                                         const float3 scale,
                                         const Quaternion rotation,
                                         const float opacity,
                                         ccl_private float &t)
{
  const float3x3 inverse_rotation = transposed(quaternion_to_rotation(rotation));
  float3 o = inverse_rotation * (origin - center);
  float3 d = inverse_rotation * direction;
  float3 s = fabs(scale);
  const float largest = reduce_max(s);
  if (!(largest > 0)) {
    return 0;
  }
  const float threshold = largest * 1e-5f;
  if (s.x <= s.y && s.x <= s.z) {
    o = make_float3(o.y, o.z, o.x);
    d = make_float3(d.y, d.z, d.x);
    s = make_float3(s.y, s.z, s.x);
  }
  else if (s.y <= s.z) {
    o = make_float3(o.z, o.x, o.y);
    d = make_float3(d.z, d.x, d.y);
    s = make_float3(s.z, s.x, s.y);
  }
  float e;
  if (s.z <= threshold && s.x > threshold && s.y > threshold) {
    if (!(fabsf(d.z) > 1e-8f * len(d))) {
      return 0;
    }
    t = -o.z / d.z;
    const float x = (o.x + t * d.x) / s.x;
    const float y = (o.y + t * d.y) / s.y;
    e = x * x + y * y;
  }
  else {
    const float3 relative_scale = max(s / largest, make_float3(1e-5f));
    const float3 os = o / relative_scale;
    const float3 ds = d / relative_scale;
    t = -dot(os, ds) / dot(ds, ds);
    const float3 perpendicular = cross(safe_normalize(ds), os) / largest;
    e = dot(perpendicular, perpendicular);
  }
  return e >= 0 ? opacity * expf(-0.5f * e) : 0;
}

CCL_NAMESPACE_END
