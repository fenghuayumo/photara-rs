/* SPDX-License-Identifier: Apache-2.0 */
#ifdef WITH_OPTIX
#include "device/optix/device_impl.h"
#include "bvh/optix.h"
#include "scene/mesh.h"
#include "scene/opacity_micromap.h"
#include "util/log.h"
#include "util/progress.h"

CCL_NAMESPACE_BEGIN
bool OptiXDevice::build_opacity_micromap(BVHOptiX *bvh, const Mesh &mesh, Progress &progress)
{
#ifndef WITH_OPTIX_OMM
  return false;
#else
  OpacityMicromapData baked;
  string reason;
  if (!bake_opacity_micromap(mesh, progress, baked, reason)) {
    if (!reason.empty()) LOG_INFO << "OMM fallback: " << reason;
    return false;
  }
  if (progress.get_cancel()) return false;
  const CUDAContextScope scope(this);
  device_vector<uint8_t> data(this, "OMM bake data", MEM_READ_ONLY);
  device_vector<OptixOpacityMicromapDesc> descriptors(this, "OMM descriptors", MEM_READ_ONLY);
  data.alloc(baked.data.size());
  std::copy(baked.data.begin(), baked.data.end(), data.data());
  descriptors.alloc(baked.descriptors.size());
  for (size_t i = 0; i < baked.descriptors.size(); ++i) {
    const auto &d = baked.descriptors[i];
    descriptors[i] = {d.offset, d.level, d.format};
  }
  data.copy_to_device(); descriptors.copy_to_device();
  vector<OptixOpacityMicromapHistogramEntry> histogram;
  for (const auto &h : baked.histogram) {
    histogram.push_back({h.count, h.level, OptixOpacityMicromapFormat(h.format)});
  }
  OptixOpacityMicromapArrayBuildInput input = {};
  input.flags = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE;
  input.inputBuffer = data.device_pointer;
  input.perMicromapDescBuffer = descriptors.device_pointer;
  input.numMicromapHistogramEntries = histogram.size();
  input.micromapHistogramEntries = histogram.data();
  OptixMicromapBufferSizes sizes = {};
  OptixResult result = optixOpacityMicromapArrayComputeMemoryUsage(context, &input, &sizes);
  if (result != OPTIX_SUCCESS) {
    LOG_WARNING << "OMM unavailable for mesh: " << optixGetErrorName(result);
    return false;
  }
  auto output = make_unique<device_only_memory<char>>(this, "OptiX OMM array", false);
  auto indices = make_unique<device_only_memory<char>>(this, "OptiX OMM indices", false);
  device_only_memory<char> temp(this, "OMM temporary build", true);
  output->alloc_to_device(sizes.outputSizeInBytes);
  temp.alloc_to_device(sizes.tempSizeInBytes);
  indices->alloc_to_device(baked.indices.size()*sizeof(int32_t));
  if (!output->device_pointer || !indices->device_pointer ||
      (sizes.tempSizeInBytes && !temp.device_pointer)) return false;
  OptixMicromapBuffers buffers = {};
  buffers.output = output->device_pointer;
  buffers.outputSizeInBytes = sizes.outputSizeInBytes;
  buffers.temp = temp.device_pointer;
  buffers.tempSizeInBytes = sizes.tempSizeInBytes;
  result = optixOpacityMicromapArrayBuild(context, nullptr, &input, &buffers);
  if (result != OPTIX_SUCCESS) {
    LOG_WARNING << "OMM array build fallback: " << optixGetErrorName(result);
    return false;
  }
  cuda_assert(cuMemcpyHtoD(indices->device_pointer, baked.indices.data(),
                           baked.indices.size()*sizeof(int32_t)));
  cuda_assert(cuCtxSynchronize()); // Inputs/temporary storage may be released after completion.
  if (have_error()) return false;
  bvh->omm_array_data = std::move(output);
  bvh->omm_index_data = std::move(indices);
  bvh->omm_usage = std::move(baked.usage);
  bvh->omm_transparent = baked.transparent;
  bvh->omm_unknown = baked.unknown;
  LOG_INFO << "OMM attached: triangles=" << mesh.num_triangles()
           << " transparent_states=" << bvh->omm_transparent
           << " unknown_states=" << bvh->omm_unknown
           << " array_bytes=" << sizes.outputSizeInBytes;
  return true;
#endif
}
CCL_NAMESPACE_END
#endif
