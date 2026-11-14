/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/integrator/surface_shader.h"
#include "kernel/film/denoising_passes.h"

CCL_NAMESPACE_BEGIN

#ifdef __DENOISING_FEATURES__
/* Deterministic delta-tree guidance for RR. Radiance still uses the unbiased
 * integrator. In particular, never pick the guide branch from its random BSDF
 * sample: glass would change its depth and motion between reflection/refraction.
 * Three leaves and fifteen vertices bound both local storage and tracing work. */
struct StableGuideBranch {
  Ray ray;
  Transform image;
  Spectrum throughput;
  float distance;
  int depth;
  PathRayVisibility visibility;
};

ccl_device_inline bool stable_guide_delta(const ccl_private ShaderData *sd)
{
  if (sd->num_closure != 1 || !(sd->type & PRIMITIVE_TRIANGLE) ||
      (sd->shader & SHADER_SMOOTH_NORMAL) ||
      (sd->object_flag & (SD_OBJECT_MOTION | SD_OBJECT_HAS_VERTEX_MOTION)) ||
      fabsf(dot(sd->N, sd->Ng)) < 0.99999f)
    return false;
  const ccl_private ShaderClosure *sc = &sd->closure[0];
  /* Restrict to microfacet closures whose singular eval convention is 1e6.
   * Other closures need their own deterministic lobe enumeration. */
  switch (sc->type) {
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      return bsdf_get_specular_roughness_squared(sc) <= 1e-16f &&
             fabsf(dot(sc->N, sd->Ng)) > 0.99999f;
    default:
      return false;
  }
}

ccl_device_inline float3 stable_guide_unbend(const float3 v,
                                             const float3 outgoing,
                                             const float3 incoming,
                                             const float3 N,
                                             const bool transmit)
{
  if (!transmit) return v - 2.0f * dot(v, N) * N;
  /* Shortest rotation from the refracted ray back to its incident direction.
   * This is a local planar approximation, not an exact refractive Jacobian. */
  const float3 axis = cross(outgoing, incoming);
  const float cosine = dot(outgoing, incoming);
  return v + cross(axis, v) + cross(axis, cross(axis, v)) / max(1.0f + cosine, 1e-6f);
}

/* Keep the SVM interpreter outside the delta-tree loop. Inlining its large
 * switch into that loop causes pathological OptiX optimization at cold start. */
template<uint64_t node_feature_mask>
ccl_device_noinline void stable_guide_shader_eval(
    KernelGlobals kg, IntegratorState state, ccl_private ShaderData *sd,
    const PathRayVisibility visibility)
{
#ifdef __KERNEL_OPTIX__
  optixContinuationCall<void>(2, kg, state, sd, (float *)nullptr, visibility, uint32_t(0));
#else
  surface_shader_eval<node_feature_mask & ~KERNEL_FEATURE_NODE_RAYTRACE>(
      kg, state, sd, nullptr, visibility, 0);
#endif
}

template<uint64_t node_feature_mask>
ccl_device_noinline bool film_write_stable_plane_guides(
    KernelGlobals kg, IntegratorState state, const ccl_private ShaderData *primary,
    ccl_global float *ccl_restrict render_buffer)
{
#if defined(__KERNEL_GPU__) && !defined(__KERNEL_OPTIX__)
  /* OptiX scenes have no software BVH for the ordinary CUDA shading kernel. */
  return false;
#else
  if (!(kernel_data.film.denoising_pass_options_flag & DENOISING_PASS_STABLE_MIRROR_GUIDE) ||
      !(kernel_data.film.denoising_pass_options_flag & DENOISING_PASS_FOLLOW_REFLECTIONS) ||
      INTEGRATOR_STATE(state, path, bounce) != 0 ||
      INTEGRATOR_STATE(state, path, transparent_bounce) != 0 ||
      INTEGRATOR_STATE(state, path, volume_bounds_bounce) != 0 ||
      !integrator_state_volume_stack_is_empty(kg, state) || !stable_guide_delta(primary))
    return false;

  StableGuideBranch pending[3];
  integrator_state_read_ray(state, &pending[0].ray);
  const float3 camera_direction = pending[0].ray.D;
  const float3 camera_origin = pending[0].ray.P;
  pending[0].image = make_transform(1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0);
  pending[0].throughput = one_spectrum();
  pending[0].distance = 0.0f;
  pending[0].depth = 0;
  pending[0].visibility = PATH_RAY_VISIBILITY_CAMERA;
  int count = 1, leaves = 1;
  float3 normal_sum = zero_float3(), diffuse_sum = zero_float3(), specular_sum = zero_float3();
  float roughness_sum = 0.0f, weight_sum = 0.0f, dominant_weight = -1.0f;
  float dominant_depth = 0.0f;
  float3 dominant_motion = zero_float3();

  while (count) {
    StableGuideBranch branch = pending[--count];
    ShaderData sd;
    if (branch.depth == 0) sd = *primary;
    else {
      Intersection isect;
      const PathRayVisibility visibility = branch.visibility;
      if (!scene_intersect(kg, &branch.ray, visibility, &isect)) {
        const float weight = average(branch.throughput);
        if (!(weight > 0.0f) || !isfinite(weight)) return false;
        normal_sum -= camera_direction * weight;
        diffuse_sum += make_float3(0.5f, 0.5f, 0.5f) * weight;
        roughness_sum += weight;
        weight_sum += weight;
        if (weight > dominant_weight) {
          const float3 direction = transform_direction(&branch.image, branch.ray.D);
          const float4 motion = camera_motion_vector_direction(kg, camera_origin, direction);
          if (!isfinite(motion.x) || !isfinite(motion.y)) return false;
          dominant_weight = weight;
          dominant_depth = kernel_data.cam.cliplength;
          dominant_motion = make_float3(motion.x, motion.y, 0.0f);
        }
        continue;
      }
      shader_setup_from_ray(kg, &sd, &branch.ray, &isect);
      if (sd.shader_flag & (SD_HAS_LIGHT_PATH_NODE | SD_HAS_RAYTRACE)) return false;
#ifdef __SPECTRAL__
      shader_setup_wavelength(kg, &sd, state);
#endif
      stable_guide_shader_eval<node_feature_mask>(kg, state, &sd, visibility);
    }
    if ((sd.runtime_flag & (SR_CACHE_MISS | SR_HOLDOUT | SR_RAY_PORTAL |
                            SR_BSDF_HAS_DISPERSION)) || (sd.shader_flag & SD_HAS_VOLUME) ||
        (sd.object_flag & SD_OBJECT_HOLDOUT_MASK) ||
        (sd.shader_flag & (SD_HAS_LIGHT_PATH_NODE | SD_HAS_RAYTRACE))) return false;
    branch.distance += sd.ray_length;

    if (stable_guide_delta(&sd) && branch.depth < min(15, kernel_data.integrator.max_bounce)) {
      float3 directions[2];
      Spectrum weights[2];
      int labels[2], lobes = 0;
      const ccl_private MicrofacetBsdf *bsdf =
          (const ccl_private MicrofacetBsdf *)&sd.closure[0];
      const float cos_in = dot(bsdf->N, sd.wi);
      float cos_out;
      const FresnelCoeff fresnel = microfacet_fresnel(kg, bsdf, cos_in, &cos_out);
      for (int side = 0; side < 2; ++side) {
        const bool transmit = side != 0;
        const Spectrum weight = sd.closure[0].weight *
            (transmit ? fresnel.transmittance : fresnel.reflectance);
        if (!(average(weight) > 1e-6f)) continue;
        const float3 direction = transmit ?
            refract_angle(sd.wi, bsdf->N, cos_out, safe_divide(1.0f, bsdf->ior)) :
            2.0f * cos_in * bsdf->N - sd.wi;
        if (cos_in <= 0.0f || ((dot(sd.Ng, direction) < 0.0f) != transmit)) continue;
        directions[lobes] = direction;
        weights[lobes] = weight;
        labels[lobes++] = transmit ? LABEL_TRANSMIT : LABEL_REFLECT;
      }
      if (lobes > 0 && leaves + lobes - 1 <= 3) {
        leaves += lobes - 1;
        for (int i = 0; i < lobes; ++i) {
          StableGuideBranch child = branch;
          const bool transmit = labels[i] & LABEL_TRANSMIT;
          const float3 x = stable_guide_unbend(make_float3(1, 0, 0), directions[i],
                                               branch.ray.D, sd.N, transmit);
          const float3 y = stable_guide_unbend(make_float3(0, 1, 0), directions[i],
                                               branch.ray.D, sd.N, transmit);
          const float3 z = stable_guide_unbend(make_float3(0, 0, 1), directions[i],
                                               branch.ray.D, sd.N, transmit);
          const Transform local = make_transform(x.x, y.x, z.x, 0, x.y, y.y, z.y, 0,
                                                  x.z, y.z, z.z, 0);
          child.image = branch.image * local;
          child.throughput *= weights[i];
          child.depth++;
          child.visibility = transmit ? PATH_RAY_VISIBILITY_TRANSMIT : PATH_RAY_VISIBILITY_GLOSSY;
          child.ray.P = sd.P;
          child.ray.D = directions[i];
          child.ray.tmin = 0.0f;
          child.ray.tmax = FLT_MAX;
          child.ray.self.object = sd.object;
          child.ray.self.prim = sd.prim;
          child.ray.self.light_object = OBJECT_NONE;
          child.ray.self.light_prim = PRIM_NONE;
          pending[count++] = child;
        }
        continue;
      }
    }

    float3 normal = zero_float3();
    Spectrum diffuse = zero_spectrum(), specular = zero_spectrum();
    float roughness = 0.0f, material_weight = 0.0f;
    for (int i = 0; i < sd.num_closure; ++i) {
      const ccl_private ShaderClosure *sc = &sd.closure[i];
      if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) continue;
      if (CLOSURE_IS_BSDF_TRANSPARENT(sc->type)) return false;
      const Spectrum albedo = closure_albedo(kg, &sd, sc, true, true);
      const float w = average(albedo);
      const float alpha = sqrtf(bsdf_get_specular_roughness_squared(sc));
      normal += sc->N * w;
      roughness += sqrtf(alpha) * w;
      material_weight += w;
      if (CLOSURE_IS_BSDF_DIFFUSE(sc->type) || CLOSURE_IS_BSSRDF(sc->type)) diffuse += albedo;
      else specular += albedo;
    }
    if (!(material_weight > 0.0f)) return false;
    const float weight = average(branch.throughput);
    if (!(weight > 0.0f) || !isfinite(weight)) return false;
    const float3 virtual_position = camera_origin + camera_direction * branch.distance;
    float3 center, previous, next;
    primitive_motion_data_without_camera(kg, &sd, &center, &previous, &next);
    const float3 virtual_previous = virtual_position +
        transform_direction(&branch.image, previous - center);
    const float3 virtual_next = virtual_position + transform_direction(&branch.image, next - center);
    const float4 motion = camera_motion_vector(kg, virtual_position, virtual_previous, virtual_next);
    const Transform camera = kernel_data.cam.worldtocamera;
    const Transform previous_camera = kernel_data.cam.motion_pass_pre;
    const float depth = transform_point(&camera, virtual_position).z;
    const float previous_depth = transform_point(&previous_camera, virtual_previous).z;
    if (!(depth > 0.0f) || !isfinite(depth) || !isfinite(previous_depth) ||
        !isfinite(motion.x) || !isfinite(motion.y)) return false;
    normal_sum += transform_direction(&branch.image, normal / material_weight) * weight;
    diffuse_sum += spectrum_to_rgb(diffuse) * weight;
    specular_sum += spectrum_to_rgb(specular) * weight;
    roughness_sum += roughness / material_weight * weight;
    weight_sum += weight;
    if (weight > dominant_weight) {
      dominant_weight = weight;
      dominant_depth = depth;
      dominant_motion = make_float3(motion.x, motion.y, previous_depth - depth);
    }
  }
  if (!(weight_sum > 0.0f) || len_squared(normal_sum) < 1e-12f) return false;
  /* Commit together only after all leaves succeeded. No partial virtual depth
   * with primary motion if a branch hits unsupported geometry/volume/sky. */
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  const Transform camera = kernel_data.cam.worldtocamera;
  if (kernel_data.film.pass_denoising_depth != PASS_UNUSED)
    film_write_pass_float(buffer + kernel_data.film.pass_denoising_depth, dominant_depth);
  if (kernel_data.film.pass_denoising_normal != PASS_UNUSED)
    film_write_pass_float3(buffer + kernel_data.film.pass_denoising_normal,
                          transform_direction(&camera, normalize(normal_sum)));
  if (kernel_data.film.pass_denoising_albedo != PASS_UNUSED)
    film_write_pass_float3(buffer + kernel_data.film.pass_denoising_albedo, diffuse_sum / weight_sum);
  if (kernel_data.film.pass_denoising_specular_albedo != PASS_UNUSED)
    film_write_pass_float3(buffer + kernel_data.film.pass_denoising_specular_albedo,
                          specular_sum / weight_sum);
  if (kernel_data.film.pass_denoising_roughness != PASS_UNUSED)
    film_write_pass_float(buffer + kernel_data.film.pass_denoising_roughness,
                         roughness_sum / weight_sum);
  if (kernel_data.film.pass_denoising_backward_motion != PASS_UNUSED)
    film_write_pass_float3(buffer + kernel_data.film.pass_denoising_backward_motion, dominant_motion);
  if (kernel_data.film.pass_denoising_specular_motion != PASS_UNUSED) {
    film_write_pass_float(buffer + kernel_data.film.pass_denoising_specular_motion, dominant_motion.x);
    film_write_pass_float(buffer + kernel_data.film.pass_denoising_specular_motion + 1, dominant_motion.y);
  }
  INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_DENOISING_FEATURES;
  return true;
#endif
}
#endif

CCL_NAMESPACE_END
