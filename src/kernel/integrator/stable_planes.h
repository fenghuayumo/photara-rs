/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "kernel/integrator/surface_shader.h"
#include "kernel/film/denoising_passes.h"
#include "kernel/film/light_passes.h"
#include "kernel/geom/triangle.h"
#include "kernel/light/common.h"

CCL_NAMESPACE_BEGIN

#ifdef __DENOISING_FEATURES__
ccl_device_forceinline float3 integrate_surface_ray_offset(KernelGlobals kg,
    const ccl_private ShaderData *sd, const float3 ray_P, const float3 ray_D);

struct StablePlaneBranch {
  Ray ray;
  Transform image;
  Spectrum throughput;
  float distance, blocked_distance;
  float3 blocked_previous, blocked_next;
  int depth, glossy, transmission, plane;
  uint branch, key;
  bool dominant;
  PathRayVisibility visibility;
};

ccl_device_inline void stable_plane_store3(ccl_private float *r, int offset, float3 v)
{
  r[offset] = v.x; r[offset+1] = v.y; r[offset+2] = v.z;
}

ccl_device_inline bool stable_plane_delta_closure(const ccl_private ShaderClosure *sc)
{
  switch (sc->type) {
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      /* Build and Fill must classify the same closures as delta. The sampler
       * snaps almost-specular microfacets to H=N, including small nonzero alpha. */
      {
        const ccl_private MicrofacetBsdf *bsdf = (const ccl_private MicrofacetBsdf *)sc;
        return alpha_is_almost_specular(bsdf->alpha_x, bsdf->alpha_y);
      }
    default: return false;
  }
}

ccl_device_inline bool stable_plane_flat_static(KernelGlobals kg,
                                                const ccl_private ShaderData *sd)
{
  if (!(sd->type & PRIMITIVE_TRIANGLE) ||
      (sd->object_flag & (SD_OBJECT_MOTION | SD_OBJECT_HAS_VERTEX_MOTION)) ||
      fabsf(dot(sd->N, sd->Ng)) < 0.99999f) return false;
  if (sd->shader & SHADER_SMOOTH_NORMAL) {
    float3 positions[3], normals[3];
    triangle_vertices_and_normals(kg, sd, positions, normals);
    for (int i = 0; i < 3; ++i) {
      if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED))
        object_normal_transform(kg, sd, &normals[i]);
      if (fabsf(dot(safe_normalize(normals[i]), sd->Ng)) < 0.99999f) return false;
    }
  }
  return true;
}

ccl_device_inline float3 stable_plane_unbend(float3 v, float3 outgoing,
                                            float3 incoming, float3 N, bool transmit)
{
  if (!transmit) return v - 2.0f * dot(v, N) * N;
  const float3 axis = cross(outgoing, incoming);
  return v + cross(axis, v) + cross(axis, cross(axis, v)) /
      max(1.0f + dot(outgoing, incoming), 1e-6f);
}

/* A continuation callable permits nested OptiX traces from AO/bevel nodes and
 * keeps the material interpreter outside the bounded tree traversal loop. */
template<uint64_t node_feature_mask>
ccl_device_noinline void stable_planes_shader_eval(KernelGlobals kg, IntegratorState state,
    ccl_private ShaderData *sd, PathRayVisibility visibility, uint32_t flag = 0)
{
#ifdef __KERNEL_OPTIX__
  optixContinuationCall<void>(2, kg, state, sd, (float *)nullptr, visibility, flag);
#else
  surface_shader_eval<node_feature_mask & ~KERNEL_FEATURE_NODE_RAYTRACE>(
      kg, state, sd, nullptr, visibility, flag);
#endif
}

/* Aligned delta closures share the reflection direction. Refraction with
 * different IORs cannot share our two-bit lobe identity: leave that junction on
 * its base plane rather than associating a sample with the wrong surface. */
ccl_device_inline int stable_plane_delta_lobes(KernelGlobals kg,
    const ccl_private ShaderData *sd, ccl_private float3 *directions,
    ccl_private Spectrum *weights, ccl_private bool *has_non_delta)
{
  weights[0] = weights[1] = zero_spectrum();
  *has_non_delta = false;
  bool have_transmission = false;
  for (int i = 0; i < sd->num_closure; ++i) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) continue;
    if (!stable_plane_delta_closure(sc)) {
      /* Refraction near IOR=1 is labelled singular even for rough microfacets.
       * Its sampled half-vector is not our deterministic lobe; don't let it
       * alias another closure's transmission branch. */
      if (CLOSURE_IS_BSDF_MICROFACET(sc->type)) {
        const ccl_private MicrofacetBsdf *b = (const ccl_private MicrofacetBsdf *)sc;
        if (alpha_is_almost_specular(b->alpha_x, b->alpha_y) ||
            ((CLOSURE_IS_REFRACTION(sc->type) || CLOSURE_IS_GLASS(sc->type)) &&
             fabsf(b->ior - 1.0f) < 1e-4f)) return -1;
      }
      if (CLOSURE_IS_BSDF_SINGULAR(sc->type) && !CLOSURE_IS_BSDF_TRANSPARENT(sc->type))
        return -1;
      *has_non_delta = true;
      continue;
    }
    /* All singular samples with this event ID must follow the same ray.
     * A differently oriented coat cannot alias the reflected base lobe. */
    if (!isequal(sc->N, sd->N)) return -1;
    const ccl_private MicrofacetBsdf *bsdf = (const ccl_private MicrofacetBsdf *)sc;
    const float cos_in = dot(bsdf->N, sd->wi);
    if (!(cos_in > 0.0f)) continue;
    float cos_out;
    const FresnelCoeff fresnel = microfacet_fresnel(kg, bsdf, cos_in, &cos_out);
    const float3 reflect = 2.0f * cos_in * bsdf->N - sd->wi;
    if (dot(sd->Ng, reflect) > 0.0f) {
      directions[0] = reflect;
      weights[0] += sc->weight * fresnel.reflectance;
    }
    const Spectrum transmission = sc->weight * fresnel.transmittance;
    if (average(transmission) > 0.0f) {
      const float3 refract = refract_angle(sd->wi, bsdf->N, cos_out,
                                           safe_divide(1.0f, bsdf->ior));
      if (dot(sd->Ng, refract) >= 0.0f) continue;
      if (have_transmission && !isequal(refract, directions[1])) return -1;
      have_transmission = true;
      directions[1] = refract;
      weights[1] += transmission;
    }
  }
  return int(average(weights[0]) > 1e-6f) + int(average(weights[1]) > 1e-6f);
}

/* Result: 0=bypass, 1=committed, 2=texture cache miss (retry before Fill).
 * Build terminates in at most three base planes and fifteen delta vertices per
 * exploration. Plane zero stays at a glass/mixed junction; pure planar mirrors
 * can replace the primary surface. Fill is the ordinary unbiased path tracer,
 * matched against this tree, and uses the deepest base reached by its prefix. */
template<uint64_t node_feature_mask>
ccl_device_noinline int stable_planes_build(KernelGlobals kg, IntegratorState state,
    const ccl_private ShaderData *primary, ccl_global float *ccl_restrict render_buffer)
{
#if defined(__KERNEL_GPU__) && !defined(__KERNEL_OPTIX__)
  return false;
#else
  if (!(kernel_data.film.denoising_pass_options_flag & DENOISING_PASS_STABLE_PLANES) ||
      kernel_data.film.pass_stable_planes == PASS_UNUSED ||
      kernel_data.integrator.filter_closures != 0 || kernel_data.integrator.use_caustics ||
      INTEGRATOR_STATE(state, path, bounce) != 0 ||
      INTEGRATOR_STATE(state, path, transparent_bounce) != 0 ||
      INTEGRATOR_STATE(state, path, volume_bounds_bounce) != 0 ||
      !integrator_state_volume_stack_is_empty(kg, state) ||
      (kernel_data.kernel_features & (KERNEL_FEATURE_LIGHT_LINKING | KERNEL_FEATURE_SHADOW_LINKING)))
    return false;

  StablePlaneBranch pending[STABLE_PLANE_COUNT];
  StablePlaneBranch root = {};
  integrator_state_read_ray(state, &root.ray);
  const float3 camera_direction = root.ray.D, camera_origin = root.ray.P;
  root.image = make_transform(1,0,0,0, 0,1,0,0, 0,0,1,0);
  root.throughput = INTEGRATOR_STATE(state, path, throughput);
  root.branch = 1; root.key = 2166136261u; root.dominant = true;
  root.visibility = PATH_RAY_VISIBILITY_CAMERA;
  pending[0] = root;
  float records[STABLE_PLANE_COUNT][STABLE_PLANE_STRIDE] = {};
  Spectrum stable_radiance = zero_spectrum();
  int count = 1, allocated = 1;

  while (count) {
    StablePlaneBranch branch = pending[--count];
    ShaderData sd;
    bool sky = false, emission_built = false, shader_supported = true;
    Spectrum emission = zero_spectrum();
    if (!branch.depth) sd = *primary;
    else {
      Intersection isect;
      if (!scene_intersect(kg, &branch.ray, branch.visibility, &isect)) {
        sky = true;
        shader_setup_from_background(kg, &sd, branch.ray.P, branch.ray.D,
                                     branch.ray.dD, branch.ray.time);
      }
      else shader_setup_from_ray(kg, &sd, &branch.ray, &isect);
      /* These materials require the full path state or a nested trace. Keep
       * their emission in Fill and stop decomposition here, without discarding
       * the other branches already built. */
      shader_supported = !(sd.shader_flag & (SD_HAS_LIGHT_PATH_NODE | SD_HAS_RAYTRACE));
      if (shader_supported) {
#ifdef __SPECTRAL__
        shader_setup_wavelength(kg, &sd, state);
#endif
        stable_planes_shader_eval<node_feature_mask>(kg, state, &sd, branch.visibility,
            sky ? PATH_RAY_EMISSION : uint32_t(0));
      }
    }
    if (shader_supported && (sd.runtime_flag & SR_CACHE_MISS)) return 2;
    if (!sky) branch.distance += sd.ray_length;
    shader_supported &= !(sd.shader_flag & SD_HAS_LIGHT_PATH_NODE);
    if (shader_supported) {
      emission_built = true;
      if (sky) {
        const uint flags = PATH_RAY_MIS_SKIP | PATH_RAY_SINGULAR |
            ((branch.visibility & PATH_RAY_VISIBILITY_GLOSSY) ? PATH_RAY_REFLECT : 0);
        emission_built = !kernel_data.background.transparent;
        if (is_light_shader_visible_to_path(sd.shader, branch.visibility, flags) && emission_built)
          emission = surface_shader_background(&sd);
      }
      else if ((sd.object_flag & SD_OBJECT_HOLDOUT_MASK) || (sd.runtime_flag & SR_HOLDOUT))
        emission_built = false;
      else emission = surface_shader_emission(&sd);
      stable_radiance += branch.throughput * emission;
    }

    float3 directions[2]; Spectrum weights[2] = {zero_spectrum(), zero_spectrum()};
    bool has_non_delta = true;
    const bool can_split = !sky && shader_supported &&
        !(sd.runtime_flag & (SR_HOLDOUT | SR_RAY_PORTAL | SR_BSDF_HAS_DISPERSION)) &&
        !(sd.shader_flag & SD_HAS_VOLUME) && !(sd.object_flag & SD_OBJECT_HOLDOUT_MASK) &&
        branch.depth < min(15, kernel_data.integrator.max_bounce) &&
        (branch.depth == 0 || (branch.glossy < kernel_data.integrator.max_glossy_bounce &&
                              branch.transmission < kernel_data.integrator.max_transmission_bounce));
    const int lobes = can_split ? stable_plane_delta_lobes(kg, &sd, directions, weights,
                                                         &has_non_delta) : 0;
    const bool blocked = lobes > 0 && !stable_plane_flat_static(kg, &sd);
    float3 center = sky ? branch.ray.D : sd.P, previous = center, next = center;
    if (!sky) primitive_motion_data_without_camera(kg, &sd, &center, &previous, &next);
    if (blocked && branch.blocked_distance == 0.0f) {
      branch.blocked_distance = branch.distance;
      branch.blocked_previous = transform_direction(&branch.image, previous-center);
      branch.blocked_next = transform_direction(&branch.image, next-center);
    }

    int reuse = -1;
    if (lobes > 0 && !has_non_delta &&
        (branch.plane != 0 || (lobes == 1 && average(weights[1]) <= 1e-6f && !blocked)))
      reuse = average(weights[1]) > average(weights[0]) ? 1 : 0;
    int dominant = average(weights[1]) > average(weights[0]) ? 1 : 0;
    bool spawned_dominant = false;
    if (lobes > 0) {
      /* Keep stable lobe order; only the continuation reuses its own slot. */
      for (int side = 0; side < 2; ++side) {
        if (!(average(weights[side]) > 1e-6f)) continue;
        if (side != reuse && allocated == STABLE_PLANE_COUNT) continue;
        StablePlaneBranch child = branch;
        child.plane = side == reuse ? branch.plane : allocated++;
        child.depth++;
        child.glossy += side == 0; child.transmission += side == 1;
        child.branch = (branch.branch << 2) | uint(side+1);
        child.key = ((branch.key ^ uint(sd.object+1)) * 16777619u ^
                      uint(sd.shader & SHADER_MASK)) * 16777619u;
        child.throughput *= weights[side];
        child.dominant = branch.dominant && side == dominant;
        spawned_dominant |= child.dominant;
        if (branch.blocked_distance == 0.0f) {
          const float3 x = stable_plane_unbend(make_float3(1,0,0), directions[side], branch.ray.D, sd.N, side);
          const float3 y = stable_plane_unbend(make_float3(0,1,0), directions[side], branch.ray.D, sd.N, side);
          const float3 z = stable_plane_unbend(make_float3(0,0,1), directions[side], branch.ray.D, sd.N, side);
          child.image = branch.image * make_transform(x.x,y.x,z.x,0, x.y,y.y,z.y,0, x.z,y.z,z.z,0);
        }
        child.visibility = side ? PATH_RAY_VISIBILITY_TRANSMIT : PATH_RAY_VISIBILITY_GLOSSY;
        child.ray.P = integrate_surface_ray_offset(kg, &sd, sd.P, directions[side]);
        child.ray.D = directions[side];
        child.ray.tmin = 0.0f; child.ray.tmax = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
        /* Preserve the footprint already travelled before this delta bounce,
         * just as Fill does. Textured emitters behind optics must not lose
         * the camera-to-interface portion of their texture footprint. */
        child.ray.dP = differential_make_compact(sd.dP);
#endif
        child.ray.self.object = sd.object; child.ray.self.prim = sd.prim;
        child.ray.self.light_object = OBJECT_NONE; child.ray.self.light_prim = PRIM_NONE;
        pending[count++] = child;
      }
      if (reuse != -1) continue;
    }

    float *r = records[branch.plane];
    r[STABLE_PLANE_BRANCH] = __uint_as_float(branch.branch);
    r[STABLE_PLANE_KEY] = __uint_as_float(((branch.key ^ uint(sd.object+1)) * 16777619u ^
                                         uint(sd.shader & SHADER_MASK)) * 16777619u);
    const float distance = branch.blocked_distance > 0.0f ? branch.blocked_distance :
                           (sky ? 1e6f : branch.distance);
    const float3 virtual_position = camera_origin + camera_direction * distance;
    float3 virtual_previous = virtual_position, virtual_next = virtual_position;
    if (branch.blocked_distance > 0.0f) {
      virtual_previous += branch.blocked_previous; virtual_next += branch.blocked_next;
    }
    else if (!sky) {
      virtual_previous += transform_direction(&branch.image, previous-center);
      virtual_next += transform_direction(&branch.image, next-center);
    }
    const float4 motion = sky && branch.blocked_distance == 0.0f ?
        camera_motion_vector_direction(kg, camera_origin, transform_direction(&branch.image, branch.ray.D)) :
        camera_motion_vector(kg, virtual_position, virtual_previous, virtual_next);
    const float depth = sky && branch.blocked_distance == 0.0f ? kernel_data.cam.cliplength :
        transform_point(&kernel_data.cam.worldtocamera, virtual_position).z;
    if (!(depth > 0.0f) || !isfinite(depth) || !isfinite(motion.x) || !isfinite(motion.y)) return false;
    float3 normal = sky ? -branch.ray.D : sd.N;
    Spectrum diffuse = zero_spectrum(), specular = zero_spectrum();
    float roughness = 0.0f, material_weight = 0.0f;
    if (!sky && (shader_supported || branch.depth == 0)) {
      normal = zero_float3();
      for (int i = 0; i < sd.num_closure; ++i) {
        const ccl_private ShaderClosure *sc = &sd.closure[i];
        if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) continue;
        const Spectrum albedo = closure_albedo(kg, &sd, sc, true, true);
        const float w = average(albedo);
        normal += sc->N*w;
        roughness += sqrtf(sqrtf(bsdf_get_specular_roughness_squared(sc)))*w;
        material_weight += w;
        if (CLOSURE_IS_BSDF_DIFFUSE(sc->type) || CLOSURE_IS_BSSRDF(sc->type)) diffuse += albedo;
        else specular += albedo;
      }
      if (material_weight > 0.0f) roughness /= material_weight;
      normal = safe_normalize_fallback(normal, sd.N);
    }
    if (sky) roughness = 1.0f;
    if (branch.blocked_distance > 0.0f) roughness *= 0.095f;
    stable_plane_store3(r, STABLE_PLANE_VIRTUAL_POSITION, virtual_position);
    stable_plane_store3(r, STABLE_PLANE_POSITION, center);
    stable_plane_store3(r, STABLE_PLANE_NORMAL, normal);
    stable_plane_store3(r, STABLE_PLANE_VIEW, sky ? branch.ray.D : sd.wi);
    stable_plane_store3(r, STABLE_PLANE_THROUGHPUT, spectrum_to_rgb(branch.throughput));
    stable_plane_store3(r, STABLE_PLANE_PREVIOUS_VIRTUAL_POSITION, virtual_previous);
    stable_plane_store3(r, STABLE_PLANE_PREVIOUS_POSITION, previous);
    stable_plane_store3(r, STABLE_PLANE_NEXT_VIRTUAL_POSITION, virtual_next);
    stable_plane_store3(r, STABLE_PLANE_VIRTUAL_NORMAL, transform_direction(&branch.image, normal));
    stable_plane_store3(r, STABLE_PLANE_DIFFUSE, spectrum_to_rgb(diffuse));
    stable_plane_store3(r, STABLE_PLANE_SPECULAR, spectrum_to_rgb(specular));
    stable_plane_store3(r, STABLE_PLANE_MOTION, make_float3(motion.x, motion.y,
        sky ? 0.0f : transform_point(&kernel_data.cam.motion_pass_pre, virtual_previous).z-depth));
    r[STABLE_PLANE_KIND] = sky ? 2.0f : 1.0f;
    r[STABLE_PLANE_EMISSION] = emission_built ? 1.0f : 0.0f;
    r[STABLE_PLANE_ROUGHNESS] = roughness; r[STABLE_PLANE_DEPTH] = depth;
    r[STABLE_PLANE_DOMINANT] = branch.dominant && !spawned_dominant;
  }

  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  /* Transactional commit: a cache miss or unsupported camera cannot leave half
   * a tree or suppress Fill emission without its matching stable contribution. */
  for (int p = 0; p < STABLE_PLANE_COUNT; ++p)
    for (int c = 0; c < STABLE_PLANE_STRIDE; ++c)
      if (c < STABLE_PLANE_RADIANCE || c >= STABLE_PLANE_RADIANCE+3)
        film_overwrite_pass_float(buffer + kernel_data.film.pass_stable_planes +
                                  p*STABLE_PLANE_STRIDE+c, records[p][c]);
  film_write_pass_spectrum(buffer + kernel_data.film.pass_stable_radiance, stable_radiance);
  film_write_combined_pass(kg, INTEGRATOR_STATE(state, path, visibility),
      INTEGRATOR_STATE(state, path, flag), INTEGRATOR_STATE(state, path, sample), stable_radiance, buffer);

  /* RTXPT-style RR preparation: combine stable radiance with noisy Fill planes,
   * mix material guides by residual throughput plus a dominant-plane bias, and
   * select one coherent plane for depth and motion. NGX owns temporal history. */
  float plane_weights[STABLE_PLANE_COUNT] = {}, base = 1.0f, sum = 0.0f;
  for (int p = 1; p < allocated; ++p) {
    const float *r = records[p];
    const float w = saturatef((r[18]+r[19]+r[20])/3.0f);
    plane_weights[p] = w; base = max(0.0f, base-w);
  }
  plane_weights[0] = base;
  for (int p = 0; p < allocated; ++p) {
    plane_weights[p] = 0.2f*plane_weights[p] + 0.01f +
                      0.05f*records[p][STABLE_PLANE_DOMINANT];
    sum += plane_weights[p];
  }
  float3 normal = zero_float3(), diffuse = zero_float3(), specular = zero_float3();
  float roughness = 0.0f; int primary_plane = 0;
  for (int p = 0; p < allocated; ++p) {
    const float *r = records[p]; const float w = plane_weights[p]/sum;
    if (plane_weights[p] > plane_weights[primary_plane]) primary_plane = p;
    if (r[STABLE_PLANE_KIND] == 2.0f) continue;
    normal += make_float3(r[33],r[34],r[35])*w;
    diffuse += make_float3(r[36],r[37],r[38])*w;
    specular += make_float3(r[39],r[40],r[41])*w;
    roughness += r[STABLE_PLANE_ROUGHNESS]*w;
  }
  const float3 stable_rgb = max(spectrum_to_rgb(stable_radiance), zero_float3());
  const float3 stable_albedo = sqrt(stable_rgb/(1.0f+reduce_max(stable_rgb)));
  const float a = average(stable_albedo);
  diffuse = mix(diffuse, stable_albedo*0.8f+make_float3(0.1f),
                a/(average(diffuse)+sqrtf(a)+1e-7f));
  if (average(diffuse+specular) < 0.05f) diffuse += make_float3(0.05f);
  normal = transform_direction(&kernel_data.cam.worldtocamera,
                                safe_normalize_fallback(normal, -camera_direction));
  const float *r = records[primary_plane];
  if (kernel_data.film.pass_denoising_depth != PASS_UNUSED)
    film_write_pass_float(buffer+kernel_data.film.pass_denoising_depth, r[STABLE_PLANE_DEPTH]);
  if (kernel_data.film.pass_denoising_normal != PASS_UNUSED)
    film_write_pass_float3(buffer+kernel_data.film.pass_denoising_normal, normal);
  if (kernel_data.film.pass_denoising_albedo != PASS_UNUSED)
    film_write_pass_float3(buffer+kernel_data.film.pass_denoising_albedo, diffuse);
  if (kernel_data.film.pass_denoising_specular_albedo != PASS_UNUSED)
    film_write_pass_float3(buffer+kernel_data.film.pass_denoising_specular_albedo, specular);
  if (kernel_data.film.pass_denoising_roughness != PASS_UNUSED)
    film_write_pass_float(buffer+kernel_data.film.pass_denoising_roughness, roughness);
  if (kernel_data.film.pass_denoising_backward_motion != PASS_UNUSED)
    film_write_pass_float3(buffer+kernel_data.film.pass_denoising_backward_motion, make_float3(r[42],r[43],r[44]));
  if (kernel_data.film.pass_denoising_specular_motion != PASS_UNUSED) {
    film_write_pass_float(buffer+kernel_data.film.pass_denoising_specular_motion, r[42]);
    film_write_pass_float(buffer+kernel_data.film.pass_denoising_specular_motion+1, r[43]);
  }
  INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_DENOISING_FEATURES;
  return true;
#endif
}
#endif
CCL_NAMESPACE_END
