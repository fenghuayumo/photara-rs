/* SPDX-License-Identifier: Apache-2.0 */
#include "scene/opacity_micromap.h"
#include "scene/mesh.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "scene/image.h"
#include "util/algorithm.h"
#include "util/progress.h"
#include "util/half.h"
#include "util/math.h"
#ifdef WITH_OPTIX_OMM
#  include <omm.h>
#  include <OpenImageIO/imageio.h>
#endif

CCL_NAMESPACE_BEGIN

OpacityMicromapMask opacity_micromap_mask(ShaderGraph *graph)
{
  OpacityMicromapMask mask;
  if (!graph || graph->finalized || graph->output()->input("Volume")->link ||
      graph->output()->input("Displacement")->link) return mask;
  for (ShaderNode *node : graph->nodes) {
    if (node->special_type == SHADER_SPECIAL_TYPE_OUTPUT_AOV) return mask;
  }
  ShaderOutput *surface = graph->output()->input("Surface")->link;
  if (!surface || surface->parent->type != MixClosureNode::get_node_type()) return mask;
  ShaderNode *mix = surface->parent;
  ShaderOutput *transparent = mix->input("Closure1")->link;
  if (!transparent || transparent->parent->type != TransparentBsdfNode::get_node_type() ||
      !mix->input("Closure2")->link) return mask;
  ShaderNode *retained = mix->input("Closure2")->link->parent;
  if (retained->type == PrincipledBsdfNode::get_node_type()) {
    auto *bsdf = static_cast<PrincipledBsdfNode *>(retained);
    if (bsdf->input("Alpha")->link || bsdf->get_alpha() != 1.0f) return mask;
  }
  else if (retained->type != DiffuseBsdfNode::get_node_type()) return mask;
  auto *trans = static_cast<TransparentBsdfNode *>(transparent->parent);
  if (trans->input("Color")->link || trans->get_color() != one_float3()) return mask;
  ShaderOutput *factor = mix->input("Fac")->link;
  if (!factor || factor->parent->type != MathNode::get_node_type()) return mask;
  auto *test = static_cast<MathNode *>(factor->parent);
  if (test->get_math_type() != NODE_MATH_GREATER_THAN || test->input("Value2")->link ||
      !std::isfinite(test->get_value2()) || test->get_value2() < 0 || test->get_value2() > 1)
    return mask;
  ShaderOutput *alpha = test->input("Value1")->link;
  if (!alpha || alpha->parent->type != ImageTextureNode::get_node_type()) return mask;
  auto *image = static_cast<ImageTextureNode *>(alpha->parent);
  if (alpha != image->output("Alpha") || !image->tex_mapping.skip() ||
      image->get_projection() != NODE_IMAGE_PROJ_FLAT || image->get_animated() ||
      image->get_alpha_type() == IMAGE_ALPHA_IGNORE || !image->get_tiles().empty() ||
      (image->get_interpolation() != INTERPOLATION_LINEAR &&
       image->get_interpolation() != INTERPOLATION_CLOSEST) ||
      (image->get_extension() != EXTENSION_REPEAT && image->get_extension() != EXTENSION_EXTEND))
    return mask;
  ShaderOutput *uv = image->input("Vector")->link;
  if (uv) {
    if (uv->parent->type == UVMapNode::get_node_type()) {
      auto *map = static_cast<UVMapNode *>(uv->parent);
      if (map->get_from_dupli()) return mask;
      mask.uv_attribute = map->get_attribute();
    }
    else if (uv->parent->type != TextureCoordinateNode::get_node_type() ||
             uv != uv->parent->output("UV")) return mask;
  }
  mask.image = image;
  mask.cutoff = test->get_value2();
  return mask;
}

bool bake_opacity_micromap(const Mesh &mesh, Progress &progress,
                          OpacityMicromapData &out, string &reason)
{
#ifndef WITH_OPTIX_OMM
  reason = "OMM SDK not configured";
  return false;
#else
  const auto version = ommGetLibraryDesc();
  if (version.versionMajor != OMM_VERSION_MAJOR || version.versionMinor != OMM_VERSION_MINOR) {
    reason = "OMM header/runtime version mismatch";
    return false;
  }
  if (mesh.get_used_shaders().size() != 1 || mesh.num_triangles() > 131072 ||
      mesh.has_volume || mesh.has_true_displacement() || mesh.get_subdivision_type() != Mesh::SUBDIVISION_NONE) { reason = "mesh topology/material layout unsupported"; return false; }
  for (int index : mesh.get_shader()) if (index != 0) return false;
  auto *shader = static_cast<Shader *>(mesh.get_used_shaders()[0]);
  const auto &mask = shader->opacity_micromap;
  if (!mask.image || !shader->get_use_transparent_shadow() || shader->has_surface_bssrdf ||
      shader->has_surface_raytrace || shader->has_light_path_node || shader->has_aov_output_node ||
      reduce_max(shader->emission_estimate) > 0) { reason = "shader is not a certified binary alpha test"; return false; }
  // Never dereference a node that graph optimization has removed.
  bool found = false;
  for (ShaderNode *node : shader->graph->nodes) found |= node == mask.image;
  if (!found) { reason = "alpha image removed by graph optimization"; return false; }
  auto *uv = mask.uv_attribute.empty() ? mesh.attributes.find(ATTR_STD_UV) :
                                        mesh.attributes.find(mask.uv_attribute);
  if (!uv || uv->type != TypeFloat2 ||
      (uv->element != ATTR_ELEMENT_CORNER && uv->element != ATTR_ELEMENT_VERTEX)) { reason = "missing/nonlinear UV attribute"; return false; }
  vector<float2> coordinates(mesh.num_triangles() * 3);
  const float2 *uvs = uv->data<float2>();
  const auto &triangles = mesh.get_triangles();
  for (size_t i = 0; i < coordinates.size(); ++i) {
    const float2 p = uvs[uv->element == ATTR_ELEMENT_CORNER ? i : triangles[i]];
    // Bound SDK work and reject nonfinite/degenerate UVs before fast-math code.
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || fabsf(p.x) > 1024 || fabsf(p.y) > 1024)
      return false;
    coordinates[i] = p;
  }
  for (size_t i = 0; i < coordinates.size(); i += 3) {
    const float2 a = coordinates[i+1]-coordinates[i], b = coordinates[i+2]-coordinates[i];
    const float area = fabsf(a.x*b.y-a.y*b.x);
    if (!(area > 1e-12f) || area > 64.0f) return false;
  }
  if (mask.image->handle.empty()) return false;
  const ImageMetaData metadata = mask.image->handle.metadata(progress);
  if (progress.get_cancel() || metadata.ignore_alpha || !metadata.is_rgba()) return false;
  set<const ImageSingle *> images;
  mask.image->handle.add_to_set(images);
  if (images.size() != 1) return false;
  const ImageSingle *image = *images.begin();
  const string filepath = image->loader->resolved_filepath();
  if (filepath.empty()) return false;
  auto input = OIIO::ImageInput::open(filepath);
  if (!input) { reason = "cannot read resolved alpha texture"; return false; }
  vector<vector<float>> pixels;
  vector<ommCpuTextureMipDesc> mips;
  for (int level = 0; input->seek_subimage(0, level); ++level) {
    const auto spec = input->spec();
    const int w = spec.width, h = spec.height;
    const int channel = spec.alpha_channel >= 0 ? spec.alpha_channel : (spec.nchannels == 4 ? 3 : -1);
    if (w <= 0 || h <= 0 || channel < 0 || spec.nchannels > 4 || spec.depth != 1 ||
        size_t(w)*h > 16u*1024u*1024u) return false;
    vector<float> alpha(size_t(w)*h);
    if (!input->read_image(0, level, channel, channel+1, OIIO::TypeFloat, alpha.data())) return false;
    vector<float> flipped(alpha.size());
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
      float a = alpha[size_t(h-1-y)*w+x];
      if (!std::isfinite(a) || a < 0 || a > 1) return false;
      // Match the GPU's storage conversion before filtering.
      if (metadata.is_half()) a = half_to_float(float_to_half(a));
      flipped[size_t(y)*w+x] = a;
    }
    pixels.push_back(std::move(flipped));
    auto mip = ommCpuTextureMipDescDefault();
    mip.width = w; mip.height = h; mip.rowPitch = w; // SDK stride is in texels.
    mip.textureData = pixels.back().data();
    mips.push_back(mip);
    // Non-tiled runtime textures sample level zero only.
    if (!metadata.has_tiles_and_mipmaps || (w == 1 && h == 1)) break;
  }
  if (mips.empty()) return false;
  struct Resources {
    ommBaker baker = nullptr;
    ommCpuTexture texture = nullptr;
    ommCpuBakeResult result = nullptr;
    ~Resources() {
      if (result) ommCpuDestroyBakeResult(result);
      if (texture) ommCpuDestroyTexture(baker, texture);
      if (baker) ommDestroyBaker(baker);
    }
  } resources;
  auto create = ommBakerCreationDescDefault();
  create.type = ommBakerType_CPU;
  create.messageInterface.userArg = &reason;
  create.messageInterface.messageCallback = [](ommMessageSeverity severity,
                                              const char *message, void *arg) {
    if (severity >= ommMessageSeverity_Error)
      *static_cast<string *>(arg) = message ? message : "OMM SDK error";
  };
  if (ommCreateBaker(&create, &resources.baker) != ommResult_SUCCESS) return false;
  auto texture = ommCpuTextureDescDefault();
  texture.format = ommCpuTextureFormat_FP32;
  texture.mips = mips.data(); texture.mipCount = mips.size();
  texture.alphaCutoff = mask.cutoff;
  if (ommCpuCreateTexture(resources.baker, &texture, &resources.texture) != ommResult_SUCCESS)
    return false;
  vector<uint32_t> indices(coordinates.size());
  for (size_t i = 0; i < indices.size(); ++i) indices[i] = uint32_t(i);
  auto bake = ommCpuBakeInputDescDefault();
  bake.bakeFlags = ommCpuBakeFlags(ommCpuBakeFlags_Force32BitIndices |
                                  ommCpuBakeFlags_EnableValidation);
  bake.texture = resources.texture;
  bake.runtimeSamplerDesc = ommSamplerDescDefault();
  bake.runtimeSamplerDesc.filter = mask.image->get_interpolation() == INTERPOLATION_CLOSEST ?
                                     ommTextureFilterMode_Nearest : ommTextureFilterMode_Linear;
  bake.runtimeSamplerDesc.addressingMode = mask.image->get_extension() == EXTENSION_REPEAT ?
                                             ommTextureAddressMode_Wrap : ommTextureAddressMode_Clamp;
  bake.alphaMode = ommAlphaMode_Test;
  bake.alphaCutoff = mask.cutoff;
  bake.texCoordFormat = ommTexCoordFormat_UV32_FLOAT;
  bake.texCoords = coordinates.data();
  bake.indexFormat = ommIndexFormat_UINT_32;
  bake.indexBuffer = indices.data(); bake.indexCount = indices.size();
  bake.maxSubdivisionLevel = 5;
  bake.dynamicSubdivisionScale = 2;
  bake.maxArrayDataSize = 32u*1024u*1024u;
  bake.maxWorkloadSize = 128ull*1024ull*1024ull;
  if (ommCpuBake(resources.baker, &bake, &resources.result) != ommResult_SUCCESS) {
    if (reason.empty()) reason = "SDK bake rejected input or work budget"; return false;
  }
  const ommCpuBakeResultDesc *result = nullptr;
  if (ommCpuGetBakeResultDesc(resources.result, &result) != ommResult_SUCCESS ||
      result->indexFormat != ommIndexFormat_UINT_32 || result->indexCount != mesh.num_triangles())
    return false;
  const auto *data = static_cast<const uint8_t *>(result->arrayData);
  if (result->arrayDataSize) out.data.assign(data, data+result->arrayDataSize);
  const auto *mapping = static_cast<const int32_t *>(result->indexBuffer);
  out.indices.assign(mapping, mapping+result->indexCount);
  for (uint32_t i = 0; i < result->descArrayCount; ++i) {
    const auto &d = result->descArray[i];
    out.descriptors.push_back({d.offset, d.subdivisionLevel, uint16_t(d.format)});
  }
  for (uint32_t i = 0; i < result->descArrayHistogramCount; ++i) {
    const auto &h = result->descArrayHistogram[i];
    out.histogram.push_back({h.count, h.subdivisionLevel, uint16_t(h.format)});
  }
  for (uint32_t i = 0; i < result->indexHistogramCount; ++i) {
    const auto &h = result->indexHistogram[i];
    out.usage.push_back({h.count, h.subdivisionLevel, uint16_t(h.format)});
  }
  // Count states for diagnostics, without relying on micro-triangle ordering.
  for (const int32_t index : out.indices) {
    if (index == -1) ++out.transparent;
    else if (index >= 0 && size_t(index) < out.descriptors.size()) {
      const auto &d = out.descriptors[index];
      if (d.format != 2) return false;
      for (uint32_t i = 0; i < (1u << (2*d.level)); ++i) {
        const uint32_t state = (out.data[d.offset+i/4] >> ((i%4)*2)) & 3;
        out.transparent += state == 0; out.unknown += state >= 2;
      }
    }
  }
  if (!out.transparent) { reason = "no fully transparent states across mip chain"; return false; } // No hole rejection benefit.
  if (out.descriptors.empty()) {
    // OptiX requires an array even if all references are predefined indices.
    out.data.assign(1, 3); out.descriptors.push_back({0,0,2});
    out.histogram.push_back({1,0,2});
  }
  return true;
#endif
}
CCL_NAMESPACE_END
