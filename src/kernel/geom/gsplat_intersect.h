/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/types.h"

#include "kernel/geom/motion_gsplat.h"
#include "kernel/geom/object.h"
#include "kernel/geom/point_intersect.h"
#include "util/math_gsplat.h"

CCL_NAMESPACE_BEGIN

/* Gaussian splats intersection functions.
 *
 * References:
 *
 *  [3dgrt2024]  Nicolas Moenne-Loccoz et. al.
 *               "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes" (2024)
 *               ACM Transactions on Graphics and SIGGRAPH Asia
 *               https://gaussiantracer.github.io/
 *
 *  [sun2025]  Xin Sun and Iliyan Georgiev and Yun Fei and Miloš Hašan
 *             "Stochastic ray tracing of transparent 3D Gaussians" (2025)
 *             EGSR conference proceedings
 *             https://iliyan.com/publications/GaussianRayTracing/
 *
 *  [Rijsdijk2026] Joris Rijsdijk et. al.
 *                 "Gaussian Point Splatting" (2026)
 *                 ACM Trans. Graph.
 *                 https://jorisar.nl/gaussian_point_splatting/
 *
 *  [chen2024pgsr]  Nicolas Moenne-Loccoz et. al.
 *                  "PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface
 *                  Reconstruction" (2024)
 *                  https://zju3dv.github.io/pgsr/
 *
 *  [ptir-gs]  Junke Zhu et. al.
 *             "Path-Traced Inverse Rendering with Global Illumination in 3D Gaussian Fields"
 *             (2026)
 *             https://junkzhu.github.io/project_pages/PTIR/ */

#ifdef __POINTCLOUD__

#  if defined(__GSPLATS__)

/* Parameters of the Gaussian splat that are needed for intersection. */
struct GSplatIsectParams {
  float3 scale;
  Quaternion rotation;
  float opacity;
};

/* Optimized access to Gaussian splat parameters for the intersection purposes.
 * Reuses steps calculation to access multiple motion attributes. */
ccl_device_forceinline GSplatIsectParams gsplat_intersect_get_params(
    KernelGlobals kg, const int object, const int prim, const float time, const int primitive_type)
{
  GSplatIsectParams params;

  if ((primitive_type & PRIMITIVE_MOTION) == 0) {
    const int scale_offset = gsplat_attr_scale_offset(kg, object);
    const int rotation_offset = gsplat_attr_rotation_offset(kg, object);
    const int radiance_base_offset = gsplat_attr_radiance_base_offset(kg, object);
    params.scale = kernel_data_fetch(attributes_float3, scale_offset + prim);
    params.rotation = kernel_data_fetch(attributes_quaternion, rotation_offset + prim);
    params.opacity = kernel_data_fetch(attributes_float4, radiance_base_offset + prim).w;
    return params;
  }

  const GSplatMotionInfo motion_info = motion_gsplat_compute_motion_info(kg, object, time);
  params.scale = motion_gsplat_scale(kg, motion_info, object, prim);
  params.rotation = motion_gsplat_rotation(kg, motion_info, object, prim);
  params.opacity = motion_gsplat_opacity(kg, motion_info, object, prim);
  return params;
}

/* Intersect ray against Gaussian splat.
 * Performs stochastic alpha test described in [sun2025]. */
ccl_device_forceinline bool gsplat_intersect_test(KernelGlobals kg,
                                                  const float3 ray_P,
                                                  const float3 ray_D,
                                                  const float ray_tmin,
                                                  const float ray_tmax,
                                                  const int object,
                                                  const int prim,
                                                  const float time,
                                                  const int primitive_type,
                                                  ccl_private float &r_t)
{
  const int position_offset = kernel_data_fetch(objects, object).position_offset;
  const float4 point = (primitive_type & PRIMITIVE_MOTION) ?
                           motion_gsplat_point(kg, object, prim, time) :
                           kernel_data_fetch(points, position_offset + prim);

  /* Intersect with the bounding sphere.
   * Equivalent to IsNegligible() check in Algorithm 1 of [sun2025].
   *
   * Disable backface culling so that we can intersect splats with the ray origin which is inside
   * of the bounding sphere. */
  float sphere_t;
  if (!point_intersect_test<false>(point, ray_P, ray_D, ray_tmin, ray_tmax, &sphere_t)) {
    return false;
  }

  const GSplatIsectParams params = gsplat_intersect_get_params(
      kg, object, prim, time, primitive_type);

  const float3 mu = make_float3(point);

  float t = 0;
  const float particle_opacity = gsplat_ray_alpha(
      ray_P, ray_D, mu, params.scale, params.rotation, params.opacity, t);
  if (!(particle_opacity > 0 && t >= ray_tmin && t <= ray_tmax)) {
    return false;
  }
  const float3 x = ray_P + t * ray_D;
  /* Include particle identity so overlapping/coplanar Gaussians do not share
   * the same transparency decision. Secondary rays still trace the 3D kernel. */
  const float random = hash_uint4_to_float(__float_as_uint(x.x),
                                           __float_as_uint(x.y),
                                           __float_as_uint(x.z),
                                           hash_uint2(object, prim));
  if (random >= particle_opacity) {
    return false;
  }

  r_t = t;

  return true;
}

ccl_device_forceinline bool gsplat_intersect(KernelGlobals kg,
                                             ccl_private Intersection *isect,
                                             const float3 ray_P,
                                             const float3 ray_D,
                                             const float ray_tmin,
                                             const float ray_tmax,
                                             const int object,
                                             const int prim,
                                             const float time,
                                             const int primitive_type)
{
  kernel_assert(primitive_type & PRIMITIVE_GSPLAT);

  if (!gsplat_intersect_test(
          kg, ray_P, ray_D, ray_tmin, ray_tmax, object, prim, time, primitive_type, isect->t))
  {
    return false;
  }

  isect->prim = prim;
  isect->object = object;
  isect->type = primitive_type;
  isect->u = 0.0f;
  isect->v = 0.0f;
  return true;
}
#  endif /* __GSPLATS__ */

ccl_device_forceinline bool point_or_gsplat_intersect(KernelGlobals kg,
                                                      ccl_private Intersection *isect,
                                                      const float3 ray_P,
                                                      const float3 ray_D,
                                                      const float ray_tmin,
                                                      const float ray_tmax,
                                                      const int object,
                                                      const int prim,
                                                      const float time,
                                                      const int primitive_type)
{
#  if defined(__GSPLATS__)
  if (primitive_type & PRIMITIVE_POINT)
#  endif
  {
    return point_intersect(
        kg, isect, ray_P, ray_D, ray_tmin, ray_tmax, object, prim, time, primitive_type);
  }

#  if defined(__GSPLATS__)
  kernel_assert(primitive_type & PRIMITIVE_GSPLAT);
  return gsplat_intersect(
      kg, isect, ray_P, ray_D, ray_tmin, ray_tmax, object, prim, time, primitive_type);
#  endif
}

#endif

CCL_NAMESPACE_END
