/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "util/math.h"

CCL_NAMESPACE_BEGIN

/* Camera-independent local surface estimates. The samples, neighborhood size,
 * and cell anchors depend only on world geometry, never on projection or image
 * resolution. Unsupported edges and thick/mixed layers retain the raster guide. */
class GSplatWorldReceiver {
  std::vector<float4> points_;
  float cell_size_ = 0;

  void build(const size_t begin, const size_t end, const int axis)
  {
    if (begin >= end) {
      return;
    }
    const size_t mid = (begin + end) / 2;
    std::nth_element(points_.begin() + begin,
                     points_.begin() + mid,
                     points_.begin() + end,
                     [axis](const float4 a, const float4 b) {
                       if (a[axis] != b[axis]) {
                         return a[axis] < b[axis];
                       }
                       for (int i = 0; i < 4; ++i) {
                         if (a[i] != b[i]) {
                           return a[i] < b[i];
                         }
                       }
                       return false;
                     });
    build(begin, mid, (axis + 1) % 3);
    build(mid + 1, end, (axis + 1) % 3);
  }

  using Neighbors = std::priority_queue<std::pair<float, size_t>>;

  void nearest(const float3 query,
               const size_t begin,
               const size_t end,
               const int axis,
               Neighbors &result) const
  {
    if (begin >= end) {
      return;
    }
    const size_t mid = (begin + end) / 2;
    const float3 delta = make_float3(points_[mid]) - query;
    const float distance = len_squared(delta);
    result.emplace(distance, mid);
    if (result.size() > 64) {
      result.pop();
    }
    const bool left = delta[axis] > 0;
    nearest(query, left ? begin : mid + 1, left ? mid : end, (axis + 1) % 3, result);
    if (result.size() < 64 || sqr(delta[axis]) <= result.top().first) {
      nearest(query, left ? mid + 1 : begin, left ? end : mid, (axis + 1) % 3, result);
    }
  }

 public:
  explicit GSplatWorldReceiver(std::vector<float4> points) : points_(std::move(points))
  {
    build(0, points_.size(), 0);
    if (points_.size() < 32) {
      return;
    }
    std::vector<float> radii;
    for (size_t i = 0; i < points_.size(); i += std::max(size_t(1), points_.size() / 128)) {
      Neighbors neighbors;
      nearest(make_float3(points_[i]), 0, points_.size(), 0, neighbors);
      if (neighbors.top().first > 0) {
        radii.push_back(sqrtf(neighbors.top().first));
      }
    }
    if (!radii.empty()) {
      std::sort(radii.begin(), radii.end());
      cell_size_ = radii[radii.size() / 2] * 0.5f;
    }
  }

  /* Return N.xyz and plane distance in w. Confidence is separate, so zero
   * distance planes and either normal orientation are represented exactly. */
  float4 plane(const float3 query, float &confidence) const
  {
    confidence = 0;
    if (!(cell_size_ > 0)) {
      return zero_float4();
    }
    const float3 anchor = (floor(query / cell_size_) + make_float3(0.5f)) * cell_size_;
    Neighbors neighbors;
    nearest(anchor, 0, points_.size(), 0, neighbors);
    const float radius = sqrtf(neighbors.top().first);
    if (!(radius > 0)) {
      return zero_float4();
    }
    std::vector<float4> local;
    while (!neighbors.empty()) {
      const float4 p = points_[neighbors.top().second];
      local.push_back(make_float4((make_float3(p) - anchor) / radius, p.w));
      neighbors.pop();
    }
    float3 normal = zero_float3(), mean = zero_float3();
    float eigenvalues[3] = {};
    for (int iteration = 0; iteration < 3; ++iteration) {
      float total = 0;
      const float3 previous_mean = mean;
      mean = zero_float3();
      float weights[64];
      for (size_t i = 0; i < local.size(); ++i) {
        const float residual = dot(normal, make_float3(local[i]) - previous_mean);
        /* Downweight outliers relative to the fixed neighborhood radius.
         * This robust scale does not use camera depth. */
        const float robust = iteration == 0 ? 1 : 1 / (1 + sqr(residual / 0.03f));
        weights[i] = local[i].w * robust;
        total += weights[i];
        mean += weights[i] * make_float3(local[i]);
      }
      if (!(total > 0)) {
        return zero_float4();
      }
      mean /= total;
      float covariance[3][3] = {}, vectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      for (size_t i = 0; i < local.size(); ++i) {
        const float3 delta = make_float3(local[i]) - mean;
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            covariance[row][col] += weights[i] * delta[row] * delta[col] / total;
          }
        }
      }
      /* Symmetric Jacobi eigensolver in normalized local coordinates. */
      for (int sweep = 0; sweep < 12; ++sweep) {
        for (int p = 0; p < 2; ++p) {
          for (int q = p + 1; q < 3; ++q) {
            const float off = covariance[p][q];
            if (fabsf(off) < 1e-10f) {
              continue;
            }
            const float angle = 0.5f * atan2f(2 * off, covariance[q][q] - covariance[p][p]);
            const float c = cosf(angle), s = sinf(angle);
            for (int k = 0; k < 3; ++k) {
              const float a = covariance[k][p], b = covariance[k][q];
              covariance[k][p] = c * a - s * b;
              covariance[k][q] = s * a + c * b;
              const float u = vectors[k][p], v = vectors[k][q];
              vectors[k][p] = c * u - s * v;
              vectors[k][q] = s * u + c * v;
            }
            for (int k = 0; k < 3; ++k) {
              const float a = covariance[p][k], b = covariance[q][k];
              covariance[p][k] = c * a - s * b;
              covariance[q][k] = s * a + c * b;
            }
          }
        }
      }
      int smallest = 0;
      for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = fmaxf(0, covariance[i][i]);
        if (eigenvalues[i] < eigenvalues[smallest]) {
          smallest = i;
        }
      }
      normal = make_float3(vectors[0][smallest], vectors[1][smallest], vectors[2][smallest]);
    }
    std::sort(eigenvalues, eigenvalues + 3);
    confidence = (1 - smoothstep(0.02f, 0.12f, sqrtf(eigenvalues[0]))) *
                 smoothstep(0.01f, 0.05f, eigenvalues[1] / fmaxf(eigenvalues[2], 1e-20f));
    const float3 point = anchor + mean * radius;
    const float distance = dot(normal, point);
    confidence *= 1 - smoothstep(0.2f, 0.6f, fabsf(dot(normal, query) - distance) / radius);
    return make_float4(normal, distance);
  }
};

CCL_NAMESPACE_END
