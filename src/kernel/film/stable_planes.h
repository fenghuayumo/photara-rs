/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* One layout shared by Build, Fill and GPU reprojection. Metadata is unfiltered;
 * radiance is accumulated in RGB just like Combined. The base plane and its
 * descendants may overlap: Fill always chooses the deepest matching ancestor. */
enum StablePlaneLayout {
  STABLE_PLANE_COUNT = 3,
  STABLE_PLANE_STRIDE = 48,
  STABLE_PLANE_BRANCH = 0,
  STABLE_PLANE_KEY = 1,
  STABLE_PLANE_VIRTUAL_POSITION = 2,
  STABLE_PLANE_POSITION = 5,
  STABLE_PLANE_NORMAL = 8,
  STABLE_PLANE_ROUGHNESS = 11,
  STABLE_PLANE_VIEW = 12,
  STABLE_PLANE_RADIANCE = 15,
  STABLE_PLANE_THROUGHPUT = 18,
  STABLE_PLANE_KIND = 21,
  STABLE_PLANE_EMISSION = 22,
  STABLE_PLANE_DEPTH = 23,
  STABLE_PLANE_PREVIOUS_VIRTUAL_POSITION = 24,
  STABLE_PLANE_PREVIOUS_POSITION = 27,
  STABLE_PLANE_NEXT_VIRTUAL_POSITION = 30,
  STABLE_PLANE_VIRTUAL_NORMAL = 33,
  STABLE_PLANE_DIFFUSE = 36,
  STABLE_PLANE_SPECULAR = 39,
  STABLE_PLANE_MOTION = 42,
  STABLE_PLANE_DOMINANT = 45,
};

ccl_device_inline bool stable_plane_is_prefix(uint prefix, uint branch)
{
  if (!prefix || !branch) return false;
  while (branch > prefix) branch >>= 2;
  return prefix == branch;
}

/* Prefix hierarchy is shared by radiance routing and emission exclusion. */
ccl_device_inline int stable_plane_find(ccl_global const float *records, uint branch)
{
  branch &= 0x7fffffffu;
  uint best = 0; int result = -1;
  for (int p = 0; p < STABLE_PLANE_COUNT; ++p) {
    const uint base = __float_as_uint(records[p*STABLE_PLANE_STRIDE+STABLE_PLANE_BRANCH]);
    if (base > best && stable_plane_is_prefix(base, branch)) { best = base; result = p; }
  }
  return result;
}

ccl_device_inline bool stable_plane_emission_covered(ccl_global const float *records,
                                                    uint branch)
{
  if (!branch || (branch & 0x80000000u)) return false;
  for (int p = 0; p < STABLE_PLANE_COUNT; ++p) {
    ccl_global const float *r = records+p*STABLE_PLANE_STRIDE;
    const uint leaf = __float_as_uint(r[STABLE_PLANE_BRANCH]);
    if (stable_plane_is_prefix(branch, leaf) &&
        (branch != leaf || r[STABLE_PLANE_EMISSION] != 0.0f)) return true;
  }
  return false;
}

CCL_NAMESPACE_END
