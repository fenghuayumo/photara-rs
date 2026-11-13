/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* One independent thread per bounded table. Global is dispatched before local
 * tables so empty tiles see the newly updated global proposal. No pixel data
 * or feedback leaves the device. Double sums match the host reference policy. */
ccl_device void neeat_update_table(ccl_global float2 *distribution,
                                  ccl_global float *feedback,
                                  ccl_global const KernelLightTreeEmitter *emitters,
                                  const int count, const int analytic_count,
                                  const int table, const bool fresh)
{
  const int base = table * (count + 1);
  double total = 0.0, mesh_total = 0.0;
  for (int i = 0; i < count; ++i) {
    const float f = fresh ? 0.0f : feedback[base + i];
    if (isfinite(f) && f > 0.0f) {
      if (emitters[i].light.id < 0) total += f;
      else mesh_total += f;
    }
  }
  /* Mixed scenes start conservatively with the tree. Visible, PDF-corrected
   * contributions learn how much budget analytic-only proposals deserve.
   * The tree always retains at least 25% support, including for mesh lights. */
  const float initial_share = analytic_count == count ? 1.0f : 0.0f;
  const float fallback_share = table ? distribution[count].x : initial_share;
  const float target_share = total + mesh_total > 0.0 ?
      float(total / (total + mesh_total)) : fallback_share;
  const float share = fresh ? initial_share :
      0.8f * distribution[base + count].x + 0.2f * target_share;
  distribution[base + count] = make_float2(share, 0.75f * share);
  double cdf = 0.0;
  for (int i = 0; i < count; ++i) {
    float pdf = 0.0f;
    if (emitters[i].light.id < 0) {
      const float uniform = 1.0f / analytic_count;
      const float f = fresh ? 0.0f : feedback[base + i];
      const float fallback = table ? distribution[i].y : uniform;
      const float target = total > 0.0 && isfinite(f) ? float(fmaxf(0.0f, f) / total) : fallback;
      const float learned = fresh ? uniform : 0.8f * distribution[base + i].y + 0.2f * target;
      pdf = 0.95f * learned + 0.05f * uniform;
    }
    cdf += pdf;
    distribution[base + i] = make_float2(float(cdf), pdf);
    feedback[base + i] = 0.0f;
  }
  double prefix = 0.0;
  float previous = 0.0f;
  int remaining = analytic_count;
  for (int i = 0; i < count; ++i) {
    prefix += distribution[base + i].y;
    float endpoint = previous;
    if (distribution[base + i].y > 0.0f) {
      --remaining;
      endpoint = remaining == 0 ? 1.0f : float(prefix / cdf);
    }
    distribution[base + i] = make_float2(endpoint, endpoint - previous);
    previous = endpoint;
  }
}

CCL_NAMESPACE_END
