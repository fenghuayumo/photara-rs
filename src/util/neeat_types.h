/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

CCL_NAMESPACE_BEGIN

/* Exact leaf sampling, never a replacement triangle or averaged material. */
struct KernelNEEATLight {
  int emitter;
  int object;
  float cdf;
  float pdf;
};

struct KernelNEEATCluster {
  int first;
  int size;
  float prior;
  float pad;
};

/* Sorted by (object, emitter), preserving separate mesh instances. */
struct KernelNEEATLookup {
  int object;
  int emitter;
  int index;
  int pad;
};

CCL_NAMESPACE_END
