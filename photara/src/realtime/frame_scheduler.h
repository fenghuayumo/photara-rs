/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <stdexcept>

namespace photara::realtime {

/* Independent frame scheduling policy. No ownership of scene or OptiX state.
 * Each frame starts a fresh film, but consumes a different sample subset.
 * Future resampling/cache passes belong to this mode, not the reference tracer. */
class FrameScheduler {
 public:
  static constexpr int sample_period = 1 << 24;

  void configure(const int samples_per_frame)
  {
    if (samples_per_frame < 1 || samples_per_frame > 64) {
      throw std::invalid_argument("RealTime samples_per_frame must be in [1, 64]");
    }
    samples_ = samples_per_frame;
    reset();
  }

  void reset() { offset_ = 0; }
  int samples() const { return samples_; }
  int offset() const { return offset_; }
  void advance()
  {
    offset_ += samples_;
    if (offset_ > sample_period - samples_) {
      offset_ = 0;
    }
  }

 private:
  int samples_ = 1;
  int offset_ = 0;
};

}  // namespace photara::realtime
