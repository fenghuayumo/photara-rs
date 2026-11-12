/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "util/string.h"
#include "util/param.h"
#include "util/types.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN
class ShaderGraph;
class ImageTextureNode;
class Mesh;
class Progress;

/* Certified before shader graph optimization. Only a direct binary alpha test
 * mixing a white Transparent BSDF with another surface is accepted. */
struct OpacityMicromapMask {
  ImageTextureNode *image = nullptr;
  ustring uv_attribute;
  float cutoff = 0;
};
struct OpacityMicromapDesc {
  uint32_t offset;
  uint16_t level;
  uint16_t format;
};
struct OpacityMicromapCount {
  uint32_t count;
  uint16_t level;
  uint16_t format;
};
struct OpacityMicromapData {
  vector<uint8_t> data;
  vector<OpacityMicromapDesc> descriptors;
  vector<int32_t> indices;
  vector<OpacityMicromapCount> histogram;
  vector<OpacityMicromapCount> usage;
  uint64_t transparent = 0, unknown = 0;
};
OpacityMicromapMask opacity_micromap_mask(ShaderGraph *graph);
bool bake_opacity_micromap(const Mesh &mesh, Progress &progress,
                          OpacityMicromapData &data, string &reason);
CCL_NAMESPACE_END
