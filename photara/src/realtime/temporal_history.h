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

#include "scene/camera.h"
#include "session/buffers.h"
#include "util/projection.h"

namespace photara::realtime {

/* Experimental CPU baseline. Store raw frames, never reconstructed frames, so
 * removing an entry really removes all of its influence. Denoising is a later stage. */
class TemporalHistory {
 public:
  struct Pixel {
    std::array<float, 4> color;
    ccl::float3 position, normal;
    float roughness = 0, weight = 0;
  };
  struct Frame {
    int width, height;
    ccl::ProjectionTransform worldtoraster;
    ccl::Transform worldtocamera;
    ccl::float3 camera_position;
    ccl::float2 lookup_shift;
    std::vector<Pixel> pixels;
  };

  int limit = 0;
  uint64_t frame_index = 0, reused_pixels = 0, reused_frames = 0;
  double mean_history_frames = 0;
  std::deque<Frame> frames;

  void clear()
  {
    frames.clear();
    reused_pixels = reused_frames = 0;
    mean_history_frames = 0;
  }

  size_t memory_bytes() const
  {
    size_t bytes = 0;
    for (const auto &frame : frames) bytes += frame.pixels.size() * sizeof(Pixel);
    return bytes;
  }

  void reconstruct(ccl::RenderBuffers *buffers, const int samples, const ccl::Camera &camera,
                   const ccl::float2 pixel_jitter)
  {
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
    const int count = params.get_pass_offset(PASS_SAMPLE_COUNT);
    if (combined == PASS_UNUSED || position == PASS_UNUSED || normal == PASS_UNUSED ||
        roughness == PASS_UNUSED) {
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
    }

    reused_pixels = reused_frames = 0;
    uint64_t accepted_total = 0;
    for (size_t i = 0; i < current.pixels.size(); ++i) {
      const auto &pixel = current.pixels[i];
      // The internal Combined fourth channel is transparency, not display alpha.
      if (pixel.weight <= 0 || len_squared(pixel.normal) < 0.5f ||
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
        if (previous.weight <= 0 || fabsf(previous.color[3]) > 0.001f ||
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
        const float rough = std::min(pixel.roughness, previous.roughness);
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
