/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "util/neeat_types.h"

CCL_NAMESPACE_BEGIN

constexpr int NEEAT_MAX_LIGHTS = 2 * 1024 * 1024;
constexpr int NEEAT_MAX_CLUSTERS = 4096;

/* Leaves arrive in spatial light-tree traversal order. Select a power-of-two
 * light LOD so learning stays bounded even for millions of emissive triangles.
 * The hierarchy is sampled exactly; no radiance or geometry is approximated. */
inline int neeat_build_clusters(std::vector<KernelNEEATLight> &lights,
                                const std::vector<double> &energy,
                                std::vector<KernelNEEATCluster> &clusters,
                                std::vector<KernelNEEATLookup> &lookup)
{
  const int count = int(lights.size());
  if (count <= 0 || count > NEEAT_MAX_LIGHTS || energy.size() != lights.size()) {
    throw std::invalid_argument("Invalid NEE-AT light count");
  }
  int stride = 1;
  while ((count + stride - 1) / stride > NEEAT_MAX_CLUSTERS) stride *= 2;
  clusters.clear();
  lookup.resize(count);
  std::vector<double> masses;
  for (int first = 0; first < count; first += stride) {
    const int size = std::min(stride, count - first);
    double total = 0;
    for (int i = first; i < first + size; ++i) {
      if (std::isfinite(energy[i]) && energy[i] > 0) total += energy[i];
    }
    double prefix = 0;
    float previous = 0;
    for (int j = 0; j < size; ++j) {
      const int i = first + j;
      const double e = std::isfinite(energy[i]) ? std::max(0.0, energy[i]) : 0.0;
      prefix += total > 0 ? 0.95 * e / total + 0.05 / size : 1.0 / size;
      const float endpoint = j == size - 1 ? 1.0f : float(prefix);
      lights[i].cdf = endpoint;
      lights[i].pdf = endpoint - previous;
      previous = endpoint;
      lookup[i] = {lights[i].object, lights[i].emitter, i, 0};
    }
    clusters.push_back({first, size, 0, 0});
    masses.push_back(total);
  }
  const double total = std::accumulate(masses.begin(), masses.end(), 0.0);
  for (size_t i = 0; i < clusters.size(); ++i) {
    clusters[i].prior = float(total > 0 ? 0.95 * masses[i] / total + 0.05 / clusters.size() :
                                        1.0 / clusters.size());
  }
  std::sort(lookup.begin(), lookup.end(), [](const auto &a, const auto &b) {
    return a.object < b.object || (a.object == b.object && a.emitter < b.emitter);
  });
  return stride;
}

CCL_NAMESPACE_END
