/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "device/device.h"
#include "device/queue.h"
#include "kernel/film/temporal_history.h"
#include "scene/camera.h"
#include "session/buffers.h"
#include "util/projection.h"

namespace photara::realtime {

/* GPU history stores raw frames, never reconstructed frames, so
 * removing an entry really removes all of its influence. Denoising is a later stage. */
class TemporalHistory {
 public:
  std::unique_ptr<ccl::device_only_memory<ccl::uchar>> gpu_ring;
  std::unique_ptr<ccl::device_only_memory<ccl::uint>> gpu_counts;
  std::unique_ptr<ccl::device_vector<ccl::uchar>> gpu_frames;
  std::unique_ptr<ccl::device_vector<ccl::uint4>> gpu_stats;
  std::unique_ptr<ccl::DeviceQueue> gpu_queue;
  int gpu_width = 0, gpu_height = 0, gpu_capacity = 0, gpu_count = 0, gpu_next = 0;
  bool stats_valid = false;
  uint64_t gpu_updates = 0;
  int limit = 0;
  uint64_t frame_index = 0, reused_pixels = 0, reused_frames = 0, active_planes = 0;
  uint64_t cross_frame_pixels = 0, moving_pixels = 0;
  double mean_history_frames = 0;

  void clear()
  {
    gpu_count = gpu_next = 0;
    stats_valid = false;
    reused_pixels = reused_frames = active_planes = 0;
    cross_frame_pixels = moving_pixels = 0;
    mean_history_frames = 0;
  }

  size_t memory_bytes() const
  {
    return gpu_ring ? gpu_ring->memory_size() : 0;
  }

  int stored_frames() const { return gpu_count; }

  void free_gpu()
  {
    gpu_queue.reset(); gpu_stats.reset(); gpu_frames.reset();
    gpu_counts.reset(); gpu_ring.reset();
    gpu_width = gpu_height = gpu_capacity = 0;
    clear();
  }

  void sync_stats()
  {
    if (!stats_valid) return;
    gpu_stats->copy_from_device();
    uint64_t sum = 0;
    reused_pixels = reused_frames = active_planes = 0;
    cross_frame_pixels = moving_pixels = 0;
    for (size_t i = 0; i < gpu_stats->size(); ++i) {
      const auto v = gpu_stats->data()[i];
      sum += v.x; reused_pixels += v.y; active_planes += v.w & 1023u;
      cross_frame_pixels += (v.w >> 10) & 1023u;
      moving_pixels += v.w >> 20;
      reused_frames = std::max(reused_frames, uint64_t(v.z));
    }
    mean_history_frames = double(sum) / (double(gpu_width)*gpu_height);
  }

  void reconstruct(ccl::RenderBuffers *buffers, int samples, const ccl::Camera &camera,
                   ccl::float2 pixel_jitter, int frame_number = INT_MIN)
  {
    using namespace ccl;
    const auto &p = buffers->params;
    Device *device = buffers->buffer.device;
    if (device->info.type != DEVICE_OPTIX && device->info.type != DEVICE_CUDA &&
        device->info.type != DEVICE_METAL)
      throw std::runtime_error("GPU temporal reconstruction requires OptiX, CUDA or Metal");
    if (p.width != p.full_width || p.height != p.full_height || p.full_x || p.full_y ||
        p.window_x || p.window_y)
      throw std::runtime_error("history_samples requires an untiled full-resolution frame");
    const int capacity = limit+1, size = p.width*p.height;
    if (!gpu_ring || gpu_width != p.width || gpu_height != p.height || gpu_capacity != capacity) {
      free_gpu();
      gpu_width = p.width; gpu_height = p.height; gpu_capacity = capacity;
      gpu_ring = std::make_unique<device_only_memory<uchar>>(device, "temporal_raw_ring");
      gpu_ring->alloc_to_device(size_t(capacity)*size*sizeof(KernelTemporalPixel));
      gpu_counts = std::make_unique<device_only_memory<uint>>(device, "temporal_counts");
      gpu_counts->alloc_to_device(size);
      gpu_frames = std::make_unique<device_vector<uchar>>(device, "temporal_frames", MEM_READ_ONLY);
      gpu_frames->alloc(size_t(capacity)*sizeof(KernelTemporalFrame));
      std::memset(gpu_frames->data(), 0, gpu_frames->size());
      gpu_stats = std::make_unique<device_vector<uint4>>(device, "temporal_stats", MEM_READ_WRITE);
      gpu_stats->alloc((size+255)/256); gpu_stats->zero_to_device();
      gpu_queue = device->gpu_queue_create();
    }
    KernelTemporalFrame f;
    f.worldtoraster = camera.worldtoraster; f.worldtocamera = camera.worldtocamera;
    f.camera_position = transform_get_column(&camera.get_matrix(), 3);
    f.lookup_shift = pixel_jitter.x == FLT_MAX ? make_float2(0) : pixel_jitter+make_float2(0.5f);
    f.width = p.width; f.height = p.height; f.stride = p.pass_stride; f.samples = samples;
    f.combined = p.get_pass_offset(PASS_COMBINED); f.position = p.get_pass_offset(PASS_POSITION);
    f.normal = p.get_pass_offset(PASS_NORMAL); f.roughness = p.get_pass_offset(PASS_ROUGHNESS);
    f.object_id = p.get_pass_offset(PASS_OBJECT_ID);
    f.count = p.get_pass_offset(PASS_SAMPLE_COUNT);
    f.planes = p.get_pass_offset(PASS_STABLE_PLANES);
    f.surface = p.get_pass_offset(PASS_TEMPORAL_SURFACE);
    f.frame_number = frame_number == INT_MIN ? int(gpu_updates) : frame_number;
    if (f.combined == PASS_UNUSED || f.position == PASS_UNUSED || f.normal == PASS_UNUSED ||
        f.roughness == PASS_UNUSED || f.object_id == PASS_UNUSED)
      throw std::runtime_error("Temporal guiding passes are missing");
    std::memcpy(gpu_frames->data()+size_t(gpu_next)*sizeof(f), &f, sizeof(f));
    gpu_frames->copy_to_device();
    gpu_queue->init_execution();
    const int blocks = (size+255)/256;
    if (!gpu_queue->enqueue(DEVICE_KERNEL_TEMPORAL_SNAPSHOT, size,
        DeviceKernelArguments(&buffers->buffer.device_pointer, &gpu_ring->device_pointer,
                             &gpu_frames->device_pointer, &gpu_next, &size)) ||
        !gpu_queue->enqueue(DEVICE_KERNEL_TEMPORAL_REPROJECT, size,
        DeviceKernelArguments(&buffers->buffer.device_pointer, &gpu_ring->device_pointer,
          &gpu_frames->device_pointer, &gpu_counts->device_pointer, &gpu_next, &capacity,
          &gpu_count, &size)) ||
        !gpu_queue->enqueue(DEVICE_KERNEL_TEMPORAL_STATS, blocks,
        DeviceKernelArguments(&gpu_counts->device_pointer, &gpu_stats->device_pointer, &size)) ||
        !gpu_queue->synchronize())
      throw std::runtime_error("GPU temporal reconstruction failed");
    gpu_next = (gpu_next+1)%capacity;
    gpu_count = std::min(gpu_count+1, limit);
    stats_valid = true;
    ++gpu_updates;
  }

};

}  // namespace photara::realtime
