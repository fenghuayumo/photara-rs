/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* Table zero is global; subsequent tables correspond to screen tiles. Each
 * table has count emitter entries followed by (analytic share, feedback mix).
 * All probabilities remain frozen until the next finite realtime frame job. */
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

ccl_device_inline float neeat_pdf(KernelGlobals kg, const int emitter, const int table = 0)
{
  const int count = kernel_data.integrator.neeat_count;
  return emitter >= 0 && emitter < count ?
             kernel_data_fetch(neeat_distribution, table * (count + 1) + emitter).y : 0.0f;
}

ccl_device_inline float3 neeat_mixture_weights(KernelGlobals kg, const int table)
{
  const int count = kernel_data.integrator.neeat_count;
  const float mix = kernel_data_fetch(neeat_distribution, table * (count + 1) + count).y;
  const float tree = 1.0f - mix;
  // Use the representable selection interval widths for MIS too, including
  // tiny feedback shares that round away at the tree/global boundary.
  const float global_end = table ? tree + mix / 3.0f : 1.0f;
  return make_float3(tree, global_end - tree, 1.0f - global_end);
}

ccl_device_inline float neeat_mix_pdf(KernelGlobals kg,
                                      const int emitter,
                                      const float tree_pdf,
                                      const uint pixel)
{
  if (!kernel_data.integrator.neeat_count) {
    return tree_pdf;
  }
  const int table = neeat_table(kg, pixel);
  const float3 weights = neeat_mixture_weights(kg, table);
  return weights.x * tree_pdf + weights.y * neeat_pdf(kg, emitter) +
         (table ? weights.z * neeat_pdf(kg, emitter, table) : 0.0f);
}

CCL_NAMESPACE_END
