/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace photara::realtime {
/* Integrate held keys at input events and frame boundaries. Event frequency
 * does not determine speed, and releases retain motion accrued during a frame. */
class Navigation {
 public:
  std::array<bool, 6> held = {}; // A, D, W, S, Q, E
  std::array<float, 3> advance(double now, bool enabled)
  {
    const double dt = initialized_ ? std::clamp(now - time_, 0.0, 0.1) : 0.0;
    time_ = now;
    initialized_ = true;
    std::array<float, 3> direction = {
        float(held[1]) - float(held[0]), float(held[4]) - float(held[5]),
        float(held[2]) - float(held[3])};
    const float length = std::sqrt(direction[0]*direction[0] + direction[1]*direction[1] +
                                   direction[2]*direction[2]);
    const float scale = enabled && length > 0 ? float(dt) * 2.0f / length : 0.0f;
    for (float &component : direction) component *= scale;
    return direction;
  }
  bool active() const
  {
    for (bool key : held) if (key) return true;
    return false;
  }
 private:
  double time_ = 0;
  bool initialized_ = false;
};
} // namespace photara::realtime
