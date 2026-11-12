/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Isotropic texture LOD from a ray cone (Akenine-Moller et al., 2019/2021).
 * The compact ray state already propagates footprint = width + distance * spread.
 * Tangent-plane UV derivatives have determinant width^2 * UV_area / world_area,
 * so the texture-specific triangle constant can be recovered after UV mapping.
 * This is equivalent to triangle_lod + log2(width / abs(dot(D, Ng)))
 * + 0.5 * log2(texture_width * texture_height), as used by RTXPT.
 * Keep the existing derivative estimator for singular/degenerate mappings. */
ccl_device_inline bool ray_cone_texture_lod(const float2 uv_dx,
                                            const float2 uv_dy,
                                            const float texture_width,
                                            const float texture_height,
                                            const float cos_incidence,
                                            ccl_private float *lod)
{
  const float area = fabsf(uv_dx.x * uv_dy.y - uv_dx.y * uv_dy.x);
  const float texel_area = area * texture_width * texture_height;
  if (!(texel_area > 0.0f) || !isfinite_safe(texel_area) ||
      !isfinite_safe(cos_incidence))
  {
    return false;
  }
  // Clamp the grazing term, not the width: a zero-width cone stays at base LOD.
  const float incidence = max(fabsf(cos_incidence), 1e-4f);
  *lod = 0.5f * log2f(texel_area) - log2f(incidence);
  return true;
}

CCL_NAMESPACE_END
