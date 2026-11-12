/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* Table zero is global; subsequent tables correspond to screen tiles. All
 * probabilities remain frozen until the next finite realtime frame job. */
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
             kernel_data_fetch(neeat_distribution, table * count + emitter).y : 0.0f;
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
  return 0.25f * tree_pdf + (table ? 0.25f : 0.75f) * neeat_pdf(kg, emitter) +
         (table ? 0.5f * neeat_pdf(kg, emitter, table) : 0.0f);
}

CCL_NAMESPACE_END
