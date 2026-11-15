/* SPDX-License-Identifier: Apache-2.0 */
/* Standalone test of the production cluster builder and update policy. */
#include <cassert>
#include <iostream>
#include <limits>

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#define ccl_device inline
#define ccl_device_inline inline
#define ccl_global
#include "util/neeat.h"

namespace ccl {
using uint = unsigned int;
struct float2 { float x, y; };
inline float2 make_float2(float x, float y) { return {x, y}; }
inline bool isfinite_safe(float x) { return std::isfinite(x); }
struct float3 { float x, y, z; };
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
using KernelGlobals = int;
struct TestEmitter { struct { int id = 0; } light; };
struct {
  struct {
    int neeat_stride = 0, neeat_tile_size = 0, neeat_tiles_x = 0, neeat_local_count = 64;
    int neeat_count = 0, neeat_cluster_count = 0, neeat_cluster_size = 0;
    float neeat_tree_fraction = .25f;
  } integrator;
} kernel_data;
std::vector<TestEmitter> light_tree_emitters(128);
std::vector<KernelNEEATLookup> neeat_lookup;
std::vector<KernelNEEATLight> neeat_lights;
std::vector<float2> neeat_distribution;
}
#define kernel_data_fetch(name, index) name[index]
#include "kernel/light/neeat.h"
#include "kernel/light/neeat_update.h"

int main()
{
  using namespace ccl;
  for (int count : {1, 32, 4096, 4097, 24860, NEEAT_MAX_LIGHTS}) {
    std::vector<KernelNEEATLight> lights(count);
    std::vector<double> energy(count);
    for (int i = 0; i < count; ++i) {
      // Reuse emitter IDs across separate instances; lookups must not alias.
      lights[i] = {i % 128, i / 128, 0, 0};
      energy[i] = i % 17 == 0 ? 1e20 : i % 5;
    }
    std::vector<KernelNEEATCluster> clusters;
    std::vector<KernelNEEATLookup> lookup;
    const int stride = neeat_build_clusters(lights, energy, clusters, lookup);
    assert(clusters.size() <= NEEAT_MAX_CLUSTERS);
    double prior_sum = 0;
    for (const auto &cluster : clusters) {
      double sum = 0;
      float previous = 0;
      for (int i = cluster.first; i < cluster.first + cluster.size; ++i) {
        assert(lights[i].pdf > 0);
        assert(lights[i].pdf == lights[i].cdf - previous);
        previous = lights[i].cdf;
        sum += lights[i].pdf;
      }
      assert(std::abs(sum - 1.0) < 1e-7);
      prior_sum += cluster.prior;
    }
    assert(std::abs(prior_sum - 1.0) < 1e-6);
    for (const auto &key : lookup) {
      assert(lights[key.index].emitter == key.emitter);
      assert(lights[key.index].object == key.object);
    }
    const int size = int(clusters.size());
    std::vector<float2> distribution((size + 1) * 65);
    std::vector<float> feedback(distribution.size());
    for (int iteration = 0; iteration < 3; ++iteration) {
      feedback[0] = 1000;
      if (size > 1) feedback[1] = std::numeric_limits<float>::infinity();
      for (int table = 0; table < 65; ++table) {
        neeat_update_table(distribution.data(), feedback.data(), clusters.data(),
                           size, table, iteration == 0);
        double sum = 0;
        for (int i = 0; i < size; ++i) {
          const int base = table * (size + 1);
          assert(distribution[base + i].y > 0);
          assert(distribution[base + i].y == distribution[base + i].x -
                     (i ? distribution[base + i - 1].x : 0.0f));
          sum += distribution[base + i].y;
          assert(feedback[base + i] == 0);
        }
        assert(std::abs(sum - 1.0) < 1e-7);
      }
    }
    const size_t bytes = size_t(count) * (sizeof(KernelNEEATLight) + sizeof(KernelNEEATLookup)) +
        size_t(size) * sizeof(KernelNEEATCluster) + distribution.size() * (sizeof(float2) + sizeof(float));
    assert(bytes < 72 * 1024 * 1024);
    neeat_lookup = lookup;
    neeat_lights = lights;
    neeat_distribution = distribution;
    kernel_data.integrator.neeat_count = count;
    kernel_data.integrator.neeat_cluster_count = size;
    kernel_data.integrator.neeat_cluster_size = stride;
    for (float fraction : {0.0f, .25f, 1.0f}) {
      kernel_data.integrator.neeat_tree_fraction = fraction;
      for (int table : {0, 1}) {
        const auto w = neeat_mixture_weights(0, table);
        assert(w.x == fraction);
        assert(w.x + w.y + w.z == 1.0f);
        assert((1.0f - w.z) - w.x == w.y);
      }
      for (int i : {0, count / 2, count - 1}) {
        assert(neeat_light_index(0, lights[i].emitter, lights[i].object) == i);
        assert(neeat_cluster(0, lights[i].emitter, lights[i].object) == i / stride);
        assert(neeat_pdf(0, i) == distribution[i / stride].y * lights[i].pdf);
        const float expected = fraction * .1f + (1.0f - fraction) * neeat_pdf(0, i);
        assert(neeat_mix_pdf(0, lights[i].emitter, lights[i].object, .1f, 0) == expected);
      }
    }
    std::cout << "lights=" << count << " clusters=" << size << " lod_stride=" << stride
              << " bytes=" << bytes << '\n';
  }
  std::cout << "NEEAT_CLUSTER_TEST_PASS\n";
}
