/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* Table zero is global; subsequent tables correspond to screen tiles. Each
 * table has a bounded number of light LOD clusters followed by mixture metadata.
 * Probabilities remain frozen during each render work batch. */
ccl_device_inline int neeat_table(KernelGlobals kg, const uint pixel)
{
  const int stride = kernel_data.integrator.neeat_stride;
  const int size = kernel_data.integrator.neeat_tile_size;
  if (!stride || !size) {
    return 0;
  }
  const int x = (pixel % stride) / size;
  const int y = (pixel / stride) / size;
  const int table = 1 + x + y * kernel_data.integrator.neeat_tiles_x;
  return table <= kernel_data.integrator.neeat_local_count ? table : 0;
}

/* Map a terminal triangle and its actual instance to a proposal leaf. Analytic
 * lights use object=-1 because the light-tree branch has not sampled them yet. */
ccl_device_inline int neeat_light_index(KernelGlobals kg, const int emitter, int object)
{
  const int count = kernel_data.integrator.neeat_count;
  if (!count || emitter < 0) return -1;
  if (kernel_data_fetch(light_tree_emitters, emitter).light.id < 0) object = -1;
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const KernelNEEATLookup key = kernel_data_fetch(neeat_lookup, mid);
    if (key.object < object || (key.object == object && key.emitter < emitter)) lo = mid + 1;
    else hi = mid;
  }
  if (lo == count) return -1;
  const KernelNEEATLookup key = kernel_data_fetch(neeat_lookup, lo);
  return key.object == object && key.emitter == emitter ? key.index : -1;
}

ccl_device_inline int neeat_cluster(KernelGlobals kg, const int emitter, const int object)
{
  const int index = neeat_light_index(kg, emitter, object);
  return index < 0 ? -1 : index / kernel_data.integrator.neeat_cluster_size;
}

ccl_device_inline float neeat_pdf(KernelGlobals kg, const int index, const int table = 0)
{
  if (index < 0) return 0.0f;
  const int count = kernel_data.integrator.neeat_cluster_count;
  const int cluster = index / kernel_data.integrator.neeat_cluster_size;
  return kernel_data_fetch(neeat_distribution, table * (count + 1) + cluster).y *
         kernel_data_fetch(neeat_lights, index).pdf;
}

ccl_device_inline float3 neeat_mixture_weights(KernelGlobals kg, const int table)
{
  const float tree = kernel_data.integrator.neeat_tree_fraction;
  const float mix = 1.0f - tree;
  // Use the representable selection interval widths for MIS too, including
  // tiny feedback shares that round away at the tree/global boundary.
  const float global_end = table ? 1.0f - (1.0f - (tree + mix / 3.0f)) : 1.0f;
  return make_float3(tree, global_end - tree, 1.0f - global_end);
}

ccl_device_inline float neeat_mix_pdf(KernelGlobals kg,
                                      const int emitter,
                                      const int object,
                                      const float tree_pdf,
                                      const uint pixel)
{
  if (!kernel_data.integrator.neeat_count) {
    return tree_pdf;
  }
  const int table = neeat_table(kg, pixel);
  const float3 weights = neeat_mixture_weights(kg, table);
  const int index = neeat_light_index(kg, emitter, object);
  return weights.x * tree_pdf + weights.y * neeat_pdf(kg, index) +
         (table ? weights.z * neeat_pdf(kg, index, table) : 0.0f);
}

CCL_NAMESPACE_END
