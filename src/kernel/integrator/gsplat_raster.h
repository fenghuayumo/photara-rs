/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/bvh/bvh.h"
#include "kernel/film/write.h"
#include "kernel/integrator/path_state.h"
#include "kernel/light/light.h"
#include "util/color.h"
#include "util/math_gsplat.h"

CCL_NAMESPACE_BEGIN

#if defined(__KERNEL_OPTIX__) && defined(__GSPLATS__)
/* Geometric visibility with material-aware artistic shadow weights. The
 * strongest blocker wins, so entry/exit and repeated shells do not accumulate
 * opacity. Continue past glass to retain opaque blockers behind it. */
ccl_device_inline float3 gsplat_mesh_shadow_visibility(KernelGlobals kg,
                                                       Ray ray,
                                                       const float contact_radius = 0)
{
  float occlusion = 0;
  for (int hit = 0; hit < 64; ++hit) {
    Intersection isect;
    if (!scene_intersect(
            kg, &ray, PATH_RAY_VISIBILITY_SHADOW_TRANSPARENT | GSPLAT_RASTER_SHADOW, &isect))
    {
      break;
    }
    const int shader = intersection_get_shader(kg, &isect);
    float opacity = kernel_data_fetch(gsplat_shader_shadow_transmission, shader).w;
    if (contact_radius > 0) {
      opacity *= 1 - smoothstep(0.0f, contact_radius, isect.t);
    }
    occlusion = fmaxf(occlusion, opacity);
    if (occlusion >= 1) {
      break;
    }
    ray.tmin = intersection_t_offset(isect.t);
    ray.self.object = isect.object;
    ray.self.prim = isect.prim;
    if (ray.tmin >= ray.tmax) {
      break;
    }
  }
  return make_float3(1 - occlusion);
}

/* Query mesh visibility at a receiver before alpha compositing. */
ccl_device_noinline float3 integrator_gsplat_receiver_shadow(KernelGlobals kg,
                                                             IntegratorState state,
                                                             const float3 receiver_P,
                                                             const float3 N,
                                                             const float time,
                                                             const uint salt)
{
  /* Every projected particle carries the same scene-wide offset in w. Reuse
   * the packed record without changing the shared KernelData ABI. Only shadow
   * queries move; raster depth, color, alpha and mesh composition stay fixed. */
  const float bias = kernel_data_fetch(gsplat_raster_splats, 0).receiver_plane.w;
  const float3 P = receiver_P - N * bias;
  RNGState rng_state;
  path_state_rng_load(state, &rng_state);
  path_state_rng_scramble(&rng_state, 0x475300u + salt);
  float total = 0;
  float3 visible = zero_float3();
  Ray shadow;
  shadow.tmin = 0;
  shadow.time = time;
  shadow.dP = shadow.dD = 0;
  shadow.self.object = shadow.self.prim = OBJECT_NONE;
  shadow.self.light_object = shadow.self.light_prim = OBJECT_NONE;
  for (int lamp = 0; lamp < kernel_data.integrator.num_lights; ++lamp) {
    const KernelLight light = kernel_data_fetch(lights, lamp);
    if (light.type == LIGHT_BACKGROUND) {
      continue;
    }
    RNGState light_rng = rng_state;
    path_state_rng_scramble(&light_rng, lamp);
    const float2 random = path_state_rng_2D(kg, &light_rng, PRNG_LIGHT);
    LightSample ls;
    if (!light_sample<false>(kg, lamp, random, P, N, 0, 0, &ls)) {
      continue;
    }
    const KernelShader shader = kernel_data_fetch(shaders, light.shader_id & SHADER_MASK);
    const float3 emission = make_float3(shader.constant_emission[0] * light.strength[0],
                                        shader.constant_emission[1] * light.strength[1],
                                        shader.constant_emission[2] * light.strength[2]);
    const float weight = fmaxf(0.0f, average(emission)) * ls.eval_fac / ls.pdf;
    if (!(weight > 0 && isfinite_safe(weight))) {
      continue;
    }
    total += weight;
    if (!(light.shader_id & SHADER_CAST_SHADOW)) {
      visible += make_float3(weight);
      continue;
    }
    shadow.D = ls.D;
    shadow.P = P + ls.D * 1e-4f;
    shadow.tmax = ls.t == FLT_MAX ? FLT_MAX : fmaxf(0.0f, ls.t - 2e-4f);
    shadow.self.light_object = ls.object;
    shadow.self.light_prim = lamp;
    visible += weight * gsplat_mesh_shadow_visibility(kg, shadow);
  }
  float3 visibility = total > 0 ? visible / total : one_float3();
  /* Captured ambient illumination still needs local occlusion when analytic
   * lights are present. Direct light visibility alone misses glass contact. */
  if (kernel_data.gsplat_raster.contact_distance > 0 && len_squared(N) > 0.5f) {
    const float radius = kernel_data.gsplat_raster.contact_distance;
    shadow.P = ray_offset(P, N);
    float pdf;
    sample_cos_hemisphere(
        N, path_state_rng_2D(kg, &rng_state, PRNG_SURFACE_BSDF), &shadow.D, &pdf);
    shadow.tmax = radius;
    shadow.self.light_object = shadow.self.light_prim = OBJECT_NONE;
    const float3 contact = gsplat_mesh_shadow_visibility(kg, shadow, radius);
    /* Material opacity, distance falloff and the final user strength already
     * bound local attenuation. An additional fixed 0.3 multiplier suppressed
     * the contact cue a second time, especially through clear glass. */
    visibility *= contact;
  }
  return one_float3() - kernel_data.gsplat_raster.shadow_strength * (one_float3() - visibility);
}

/* Interpolate compatible receiver planes, retaining the inverse-depth slopes
 * for the surface normal. Confidence blends visibility estimates, never world
 * positions between distinct layers. */
ccl_device_inline float4 gsplat_raster_receiver(KernelGlobals kg, const float2 pixel)
{
  if (kernel_data.gsplat_raster.receiver_offset <= 0) {
    return zero_float4();
  }
  const int2 grid = gsplat_shadow_receiver_grid(int(kernel_data.cam.width),
                                                int(kernel_data.cam.height));
  const float2 step = make_float2(kernel_data.cam.width / grid.x, kernel_data.cam.height / grid.y);
  const int gx = min(int(pixel.x / step.x), grid.x - 1);
  const int gy = min(int(pixel.y / step.y), grid.y - 1);
  const float2 fraction = make_float2(pixel.x / step.x - gx, pixel.y / step.y - gy);
  float reference = 0, maximum_weight = 0;
  float4 result = zero_float4();
  for (int pass = 0; pass < 2; ++pass) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int ix = 0; ix < 2; ++ix) {
        const float4 guide = kernel_data_fetch(gsplat_raster_splats,
                                               kernel_data.gsplat_raster.receiver_offset +
                                                   (gy + iy) * (grid.x + 1) + gx + ix)
                                 .center;
        const float inverse = guide.x + guide.y * (pixel.x - (gx + ix) * step.x) +
                              guide.z * (pixel.y - (gy + iy) * step.y);
        const float weight = (ix ? fraction.x : 1 - fraction.x) *
                             (iy ? fraction.y : 1 - fraction.y) * guide.w;
        if (inverse > 0 && weight > 0) {
          if (pass == 0 && weight > maximum_weight) {
            reference = inverse;
            maximum_weight = weight;
          }
          else if (pass == 1 && reference > 0) {
            const float layer = 1 - smoothstep(0.025f, 0.075f, fabsf(inverse / reference - 1));
            result += weight * layer * make_float4(inverse, guide.y, guide.z, 1);
          }
        }
      }
    }
  }
  if (result.w > 0) {
    const float3 plane = make_float3(result) / result.w;
    return make_float4(plane, saturatef(result.w));
  }
  return zero_float4();
}
/* Refracted camera paths evaluate the default Gaussian emission rather than
 * the primary raster layer. Project the hit into the coherent guide and intersect
 * its plane with the actual incoming ray before querying the same mesh overlay.
 * Never darken custom materials or secondary hits without supported geometry. */
ccl_device_inline float3 integrator_gsplat_secondary_shadow(KernelGlobals kg,
                                                            IntegratorState state,
                                                            const Ray &ray,
                                                            const Intersection &isect)
{
  if (!kernel_data.gsplat_raster.enabled || !kernel_data.gsplat_raster.receive_shadows ||
      !kernel_data.gsplat_raster.have_shadow_geometry ||
      kernel_data.gsplat_raster.shadow_strength <= 0 || !(isect.type & PRIMITIVE_GSPLAT) ||
      kernel_data_fetch(gsplat_shader_shadow_transmission, intersection_get_shader(kg, &isect)).x != -1)
  {
    return one_float3();
  }
  const float3 pixel = transform_perspective(&kernel_data.cam.worldtoraster, ray.P + ray.D * isect.t);
  if (!(pixel.x >= 0 && pixel.x < kernel_data.cam.width &&
        pixel.y >= 0 && pixel.y < kernel_data.cam.height)) {
    return one_float3();
  }
  const float4 guide = gsplat_raster_receiver(kg, make_float2(pixel.x, pixel.y));
  if (guide.w <= 0) {
    return one_float3();
  }
  const float3 p = transform_perspective(&kernel_data.cam.rastertocamera,
                                         make_float3(pixel.x, pixel.y, 0));
  const float3 q = p / p.z;
  const float3 px = transform_perspective(&kernel_data.cam.rastertocamera,
                                          make_float3(pixel.x + 1, pixel.y, 0));
  const float3 py = transform_perspective(&kernel_data.cam.rastertocamera,
                                          make_float3(pixel.x, pixel.y + 1, 0));
  const float3 N = transform_direction(&kernel_data.cam.cameratoworld,
      gsplat_receiver_normal(q, px / px.z - q, py / py.z - q, make_float3(guide)));
  const float3 plane_P = transform_point(&kernel_data.cam.cameratoworld, q / guide.x);
  const float3 origin = ray.P;
  const float denominator = dot(ray.D, N);
  if (fabsf(denominator) < 1e-5f) {
    return one_float3();
  }
  const float t = dot(plane_P - origin, N) / denominator;
  if (!(t > 0 && isfinite_safe(t))) {
    return one_float3();
  }
  const float3 P = origin + ray.D * t;
  return mix(one_float3(), integrator_gsplat_receiver_shadow(
      kg, state, P, N, ray.time, 0x200u), guide.w);
}
#endif

/* Compute rasterization of a preprojected, depth-sorted tile list. This pass
 * consumes mesh depth; it neither traverses nor sorts a Gaussian BVH. It is
 * fused with primary intersection to reuse the exact jittered depth sample.
 * The separate layer is composited after mesh denoising, in the training RGB
 * space, by the common CPU/GPU film reader. */
ccl_device_inline void integrator_gsplat_raster(KernelGlobals kg,
                                                IntegratorState state,
                                                const ccl_private Ray &ray,
                                                const float mesh_t,
                                                ccl_global float *render_buffer)
{
#if defined(__KERNEL_OPTIX__) && defined(__GSPLATS__)
  if (!kernel_data.gsplat_raster.enabled || kernel_data.gsplat_raster.tiles_x == 0 ||
      INTEGRATOR_STATE(state, path, visibility) != PATH_RAY_VISIBILITY_CAMERA ||
      INTEGRATOR_STATE(state, path, transparent_bounce) != 0 ||
      kernel_data.film.pass_gsplat == PASS_UNUSED)
  {
    return;
  }
  const float3 origin = transform_point(&kernel_data.cam.worldtocamera, ray.P);
  const float3 direction = transform_direction(&kernel_data.cam.worldtocamera, ray.D);
  const float3 pixel = transform_perspective(&kernel_data.cam.worldtoraster, ray.P + ray.D);
  if (!(pixel.x >= 0 && pixel.x < kernel_data.cam.width && pixel.y >= 0 &&
        pixel.y < kernel_data.cam.height))
  {
    return;
  }
  const int x = int(pixel.x) / GSPLAT_RASTER_TILE_SIZE;
  const int y = int(pixel.y) / GSPLAT_RASTER_TILE_SIZE;
  const uint2 tile = kernel_data_fetch(gsplat_raster_tiles,
                                       y * kernel_data.gsplat_raster.tiles_x + x);
  if (tile.x == tile.y) {
    return;
  }
  float3 color = zero_float3();
  float transmittance = 1.0f;
  const bool shadow_enabled = kernel_data.gsplat_raster.receive_shadows &&
                              kernel_data.gsplat_raster.have_shadow_geometry &&
                              kernel_data.gsplat_raster.shadow_strength > 0;
  constexpr int shadow_samples = 4;
  float4 selected_positions[shadow_samples];
  float3 selected_colors[shadow_samples];
  float3 selected_normals[shadow_samples];
  RNGState rng_state;
  path_state_rng_load(state, &rng_state);
  path_state_rng_scramble(&rng_state, 0x475353u);
  const float random = path_state_rng_1D(kg, &rng_state, PRNG_SURFACE_BSDF);
  int selected_count = 0;
  const float min_depth = origin.z + ray.tmin * direction.z;
  const float max_depth = origin.z + mesh_t * direction.z;
  float4 guide = shadow_enabled ? gsplat_raster_receiver(kg, make_float2(pixel.x, pixel.y)) :
                                  zero_float4();
  float3 guide_shadow = one_float3();
  if (guide.w > 0) {
    const float3 p = transform_perspective(&kernel_data.cam.rastertocamera,
                                           make_float3(pixel.x, pixel.y, 0));
    const float3 q = p / p.z;
    const float3 px = transform_perspective(&kernel_data.cam.rastertocamera,
                                            make_float3(pixel.x + 1, pixel.y, 0));
    const float3 py = transform_perspective(&kernel_data.cam.rastertocamera,
                                            make_float3(pixel.x, pixel.y + 1, 0));
    const float3 N = transform_direction(
        &kernel_data.cam.cameratoworld,
        gsplat_receiver_normal(q, px / px.z - q, py / py.z - q, make_float3(guide)));
    const float3 P = transform_point(&kernel_data.cam.cameratoworld, q / guide.x);
    guide_shadow = integrator_gsplat_receiver_shadow(kg, state, P, N, ray.time, 0x100u);
  }
  for (uint index = tile.x; index < tile.y; ++index) {
    const KernelGSplat splat = kernel_data_fetch(gsplat_raster_splats,
                                                 kernel_data_fetch(gsplat_raster_indices, index));
    /* A tilted ellipse does not have constant depth. Its conditional depth
     * must be compared with the mesh at this pixel, including with shadows off.
     * Center sorting cannot terminate this loop: a later center may tilt in front. */
    const float depth = gsplat_projected_depth(splat.center, splat.conic, splat.color, pixel);
    if (!(depth >= min_depth && depth < max_depth)) {
      continue;
    }
    const float alpha = gsplat_projected_alpha(
        make_float3(splat.conic),
        make_float2(pixel.x - splat.center.x, pixel.y - splat.center.y),
        splat.center.w);
    const float weight = transmittance * alpha;
    /* Integrate the coherent guide's shaded colors exactly. Only unsupported
     * receiver confidence needs the sampled per-particle control variate. */
    const float next_coverage = 1 - transmittance * (1 - alpha);
    while (shadow_enabled && guide.w < 1 && selected_count < shadow_samples &&
           (selected_count + random) / shadow_samples < next_coverage)
    {
      /* A Gaussian crossing the near plane has no reliable local projection.
       * Keep its center query instead of emitting an invalid shadow ray. */
      selected_positions[selected_count] = make_float4(
          splat.center.x,
          splat.center.y,
          splat.center.z,
          depth > min_depth && isfinite_safe(depth) ? depth : 0);
      selected_colors[selected_count] = make_float3(splat.color);
      selected_normals[selected_count] = make_float3(splat.receiver_plane);
      ++selected_count;
    }
    const float3 source = make_float3(splat.color);
    color += weight * (guide.w > 0 ?
                           mix(source, gsplat_shadow_color(source, guide_shadow), guide.w) :
                           source);
    transmittance *= 1.0f - alpha;
    if (transmittance < 1e-4f) {
      break;
    }
  }
  if (selected_count > 0) {
    for (int i = 0; i < selected_count; ++i) {
      const float4 center = selected_positions[i];
      float3 P = transform_perspective(&kernel_data.cam.rastertocamera,
                                       center.w > 0 ? make_float3(pixel.x, pixel.y, 0) :
                                                      make_float3(center.x, center.y, 0));
      P = transform_point(&kernel_data.cam.cameratoworld,
                          P * ((center.w > 0 ? center.w : center.z) / P.z));
      const float3 shadow = integrator_gsplat_receiver_shadow(
          kg, state, P, selected_normals[i], ray.time, i * 17u);
      color += (gsplat_shadow_color(selected_colors[i], shadow) - selected_colors[i]) /
               shadow_samples * (1 - guide.w);
    }
  }
  if (kernel_data.integrator.filter_closures & FILTER_CLOSURE_EMISSION) {
    color = zero_float3();
  }
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  film_write_pass_float4(buffer + kernel_data.film.pass_gsplat,
                         make_float4(color, 1.0f - transmittance));
#endif
}

CCL_NAMESPACE_END
