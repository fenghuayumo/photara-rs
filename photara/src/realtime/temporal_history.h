/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

#include "device/device.h"
#include "device/queue.h"
#include "kernel/film/temporal_history.h"
#include "scene/camera.h"
#include "session/buffers.h"
#include "util/projection.h"

namespace photara::realtime {

/* GPU history with an explicit CPU reference. Store raw frames, never reconstructed frames, so
 * removing an entry really removes all of its influence. Denoising is a later stage. */
class TemporalHistory {
 public:
  struct Pixel {
    std::array<float, 4> color;
    ccl::float3 position, normal;
    float roughness = 0, weight = 0, object_id = 0;
  };
  struct Frame {
    int width, height;
    ccl::ProjectionTransform worldtoraster;
    ccl::Transform worldtocamera;
    ccl::float3 camera_position;
    ccl::float2 lookup_shift;
    std::vector<Pixel> pixels;
  };

  bool use_gpu = true;
  std::unique_ptr<ccl::device_only_memory<ccl::uchar>> gpu_ring;
  std::unique_ptr<ccl::device_only_memory<ccl::uint>> gpu_counts;
  std::unique_ptr<ccl::device_vector<ccl::uchar>> gpu_frames;
  std::unique_ptr<ccl::device_vector<ccl::uint4>> gpu_stats;
  std::unique_ptr<ccl::DeviceQueue> gpu_queue;
  int gpu_width = 0, gpu_height = 0, gpu_capacity = 0, gpu_count = 0, gpu_next = 0;
  bool stats_valid = false;
  uint64_t gpu_updates = 0, pixel_readback_bytes = 0;
  int limit = 0;
  uint64_t frame_index = 0, reused_pixels = 0, reused_frames = 0, active_planes = 0;
  double mean_history_frames = 0;
  std::deque<Frame> frames;

  void clear()
  {
    frames.clear();
    gpu_count = gpu_next = 0;
    stats_valid = false;
    reused_pixels = reused_frames = active_planes = 0;
    mean_history_frames = 0;
  }

  size_t memory_bytes() const
  {
    if (use_gpu && gpu_ring) return gpu_ring->memory_size();
    size_t bytes = 0;
    for (const auto &frame : frames) bytes += frame.pixels.size() * sizeof(Pixel);
    return bytes;
  }

  int stored_frames() const { return use_gpu ? gpu_count : int(frames.size()); }

  void free_gpu()
  {
    gpu_queue.reset(); gpu_stats.reset(); gpu_frames.reset();
    gpu_counts.reset(); gpu_ring.reset();
    gpu_width = gpu_height = gpu_capacity = 0;
    clear();
  }

  void sync_stats()
  {
    if (!use_gpu || !stats_valid) return;
    gpu_stats->copy_from_device();
    uint64_t sum = 0;
    reused_pixels = reused_frames = active_planes = 0;
    for (size_t i = 0; i < gpu_stats->size(); ++i) {
      const auto v = gpu_stats->data()[i];
      sum += v.x; reused_pixels += v.y; active_planes += v.w;
      reused_frames = std::max(reused_frames, uint64_t(v.z));
    }
    mean_history_frames = double(sum) / (double(gpu_width)*gpu_height);
  }

  void reconstruct_gpu(ccl::RenderBuffers *buffers, int samples, const ccl::Camera &camera,
                       ccl::float2 pixel_jitter)
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

  void reconstruct(ccl::RenderBuffers *buffers, const int samples, const ccl::Camera &camera,
                   const ccl::float2 pixel_jitter)
  {
    if (use_gpu) { reconstruct_gpu(buffers, samples, camera, pixel_jitter); return; }
    pixel_readback_bytes += buffers->buffer.memory_size();
    using namespace ccl;
    const auto &params = buffers->params;
    if (params.width != params.full_width || params.height != params.full_height ||
        params.full_x || params.full_y || params.window_x || params.window_y) {
      throw std::runtime_error("history_samples requires an untiled full-resolution frame");
    }
    const int combined = params.get_pass_offset(PASS_COMBINED);
    const int position = params.get_pass_offset(PASS_POSITION);
    const int normal = params.get_pass_offset(PASS_NORMAL);
    const int roughness = params.get_pass_offset(PASS_ROUGHNESS);
    const int object_id = params.get_pass_offset(PASS_OBJECT_ID);
    const int count = params.get_pass_offset(PASS_SAMPLE_COUNT);
    if (combined == PASS_UNUSED || position == PASS_UNUSED || normal == PASS_UNUSED ||
        roughness == PASS_UNUSED || object_id == PASS_UNUSED) {
      throw std::runtime_error("Temporal reconstruction guiding passes are missing");
    }
    if (!frames.empty() && (frames.back().width != params.width ||
                            frames.back().height != params.height)) clear();
    Frame current;
    current.width = params.width;
    current.height = params.height;
    current.worldtoraster = camera.worldtoraster;
    current.worldtocamera = camera.worldtocamera;
    current.camera_position = transform_get_column(&camera.get_matrix(), 3);
    // Jittered camera rays sample raster (x,y)-jitter, while ordinary first-hit
    // guides sample pixel centers. Map projections to the nearest old guide.
    current.lookup_shift = pixel_jitter.x == FLT_MAX ? make_float2(0.0f) :
                                                    pixel_jitter + make_float2(0.5f);
    current.pixels.resize(size_t(params.width) * params.height);
    float *data = buffers->buffer.data();
    for (size_t i = 0; i < current.pixels.size(); ++i) {
      const float *src = data + i * params.pass_stride;
      auto &pixel = current.pixels[i];
      uint32_t n = std::max(samples, 0);
      if (count != PASS_UNUSED) std::memcpy(&n, src + count, sizeof(n));
      pixel.weight = float(n);
      const float inv = n ? 1.0f / float(n) : 0.0f;
      for (int c = 0; c < 4; ++c) pixel.color[c] = src[combined + c] * inv;
      pixel.position = make_float3(src[position], src[position + 1], src[position + 2]);
      pixel.normal = make_float3(src[normal], src[normal + 1], src[normal + 2]) * inv;
      pixel.roughness = src[roughness] * inv;
      pixel.object_id = src[object_id];
    }

    reused_pixels = reused_frames = active_planes = 0;
    uint64_t accepted_total = 0;
    for (size_t i = 0; i < current.pixels.size(); ++i) {
      const auto &pixel = current.pixels[i];
      // The internal Combined fourth channel is transparency, not display alpha.
      if (pixel.weight <= 0 || !std::isfinite(pixel.weight) ||
          !std::isfinite(pixel.roughness) || !std::isfinite(pixel.object_id) ||
          !std::isfinite(len_squared(pixel.position)) || len_squared(pixel.normal) < 0.5f ||
          !std::isfinite(pixel.color[0]) || !std::isfinite(pixel.color[1]) ||
          !std::isfinite(pixel.color[2]) || !std::isfinite(pixel.color[3]) ||
          !std::isfinite(len_squared(pixel.normal)) ||
          fabsf(pixel.color[3]) > 0.001f) continue;
      std::array<double, 4> sum;
      for (int c = 0; c < 4; ++c) sum[c] = double(pixel.color[c]) * pixel.weight;
      double weight = pixel.weight;
      uint64_t accepted = 0;
      for (const auto &old : frames) {
        const float3 pcam = transform_point(&old.worldtocamera, pixel.position);
        if (!std::isfinite(pcam.z) || pcam.z <= 0) continue;
        const float3 raster = transform_perspective(&old.worldtoraster, pixel.position);
        const float rx = raster.x + old.lookup_shift.x;
        const float ry = raster.y + old.lookup_shift.y;
        if (!std::isfinite(rx) || !std::isfinite(ry) ||
            rx < 0 || ry < 0 || rx >= old.width || ry >= old.height)
          continue;
        const int x = int(rx), y = int(ry);
        const auto &previous = old.pixels[size_t(y) * old.width + x];
        if (previous.weight <= 0 || !std::isfinite(previous.weight) ||
            !std::isfinite(previous.roughness) || previous.object_id != pixel.object_id ||
            !std::isfinite(len_squared(previous.position)) || fabsf(previous.color[3]) > 0.001f ||
            !std::isfinite(len_squared(previous.normal)) ||
            len_squared(previous.normal) < 0.5f ||
            dot(normalize(pixel.normal), normalize(previous.normal)) < 0.98f) continue;
        const float3 previous_cam = transform_point(&old.worldtocamera, previous.position);
        /* A footprint-sized position check rejects occlusion changes and nearby surfaces. */
        const float footprint = std::max(pcam.z, 1e-3f) * 2.0f /
                                float(std::max(old.width, old.height));
        if (fabsf(pcam.z - previous_cam.z) > std::max(footprint * 0.5f, 1e-4f) ||
            len_squared(pixel.position - previous.position) > footprint * footprint) continue;
        const float3 view = safe_normalize(current.camera_position - pixel.position);
        const float3 old_view = safe_normalize(old.camera_position - pixel.position);
        const float rough = std::clamp(std::min(pixel.roughness, previous.roughness), 0.0f, 1.0f);
        /* Conservative rejection of view-dependent combined radiance. This baseline
         * does not reproject reflected/refracted secondary surfaces. */
        if (dot(view, old_view) < 1.0f - std::max(1e-7f, rough * rough * 0.002f)) continue;
        bool finite = true;
        for (int c = 0; c < 4; ++c) finite &= std::isfinite(previous.color[c]);
        if (!finite) continue;
        for (int c = 0; c < 4; ++c) sum[c] += double(previous.color[c]) * previous.weight;
        weight += previous.weight;
        ++accepted;
      }
      float *dst = data + i * params.pass_stride + combined;
      for (int c = 0; c < 4; ++c) dst[c] = float(sum[c] / weight * pixel.weight);
      accepted_total += accepted;
      reused_pixels += accepted != 0;
      reused_frames = std::max(reused_frames, accepted);
    }
    mean_history_frames = double(accepted_total) / double(current.pixels.size());
    frames.push_back(std::move(current));
    while (frames.size() > size_t(limit)) frames.pop_front();
  }
};

}  // namespace photara::realtime
