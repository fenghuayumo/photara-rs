/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* One thread per bounded cluster table, dispatched global before local.
 * Shared with the CPU fallback to keep normalization and rounding identical. */
ccl_device void neeat_update_table(ccl_global float2 *distribution,
                                  ccl_global float *feedback,
                                  ccl_global const KernelNEEATCluster *clusters,
                                  const int count, const int table, const bool fresh)
{
  const int base = table * (count + 1);
  /* Metal has no FP64 support. Tables are bounded, so FP32 accumulation is
   * sufficient there; retain FP64 normalization on CPU and CUDA. */
#ifdef __KERNEL_METAL__
  using Accumulator = float;
#else
  using Accumulator = double;
#endif
  Accumulator total = 0.0f;
  for (int i = 0; i < count; ++i) {
    const float f = fresh ? 0.0f : feedback[base + i];
    if (isfinite_safe(f) && f > 0.0f) total += f;
  }
  Accumulator cdf = 0.0f;
  for (int i = 0; i < count; ++i) {
    const float prior = clusters[i].prior;
    const float f = fresh ? 0.0f : feedback[base + i];
    const float fallback = table ? distribution[i].y : prior;
    const float target = total > 0.0f && isfinite_safe(f) ?
                             float(fmaxf(0.0f, f) / total) : fallback;
    const float learned = fresh ? prior : 0.8f * distribution[base + i].y + 0.2f * target;
    const float pdf = 0.95f * learned + 0.05f * prior;
    cdf += pdf;
    distribution[base + i] = make_float2(float(cdf), pdf);
    feedback[base + i] = 0.0f;
  }
  Accumulator prefix = 0.0f;
  float previous = 0.0f;
  for (int i = 0; i < count; ++i) {
    prefix += distribution[base + i].y;
    const float endpoint = i == count - 1 ? 1.0f : float(prefix / cdf);
    distribution[base + i] = make_float2(endpoint, endpoint - previous);
    previous = endpoint;
  }
  /* Keep at least 25% of the ordinary Light Tree proposal at every point. */
  distribution[base + count] = make_float2(1.0f, 0.75f);
}

CCL_NAMESPACE_END
