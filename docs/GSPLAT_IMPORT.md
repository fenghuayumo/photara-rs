# Cycles Gaussian splatting source import

Source: local Blender `intern/cycles`, Blender commit
`47d5372d4dd92ccde4025ce9132c606c7407c2fd` (working-tree source snapshot).
Per-file SHA-256 hashes and import modes are recorded in
`cycles_gsplat_sources.json`. Original SPDX and copyright notices are retained.

The import includes Gaussian intersection and SH evaluation, OptiX masks,
tiled independent raster projection/compositing, mesh and glass shadow weights,
world-space shadow receivers, the GSplat film pass, denoising/compositing
integration, scene/integrator settings, device arrays and upstream math tests.
The rendering core and CMake lists are integrated into the existing build.
Unrelated upstream temporal denoising changes and disabled backend CMake
configuration were not used to overwrite Photara's existing code.

`src/blender/` contains the original Blender-native adapters and panel source.
They require Blender's internal C++ APIs and are retained as reference; the
standalone Photara build does not compile them. `photara_rs_engine` is still the
active Blender Python adapter. The active Python adapter now exports Gaussian point clouds, and the public
Scene/Session APIs provide Gaussian creation and standard file loading.
HIPRT, oneAPI and Metal source counterparts are included but those backends
remain disabled in the supported Windows OptiX build.

Upstream algorithm documentation: `src/doc/gsplat_raster.md`.
Upstream regression tests: `src/test/util_math_gsplat_test.cpp` (19 cases).

Validation completed for this import:

- 60 source-manifest entries verified against the local reference tree.
- CUDA sm_120 cubin, OptiX PTX and the Python extension built successfully.
- All 19 upstream GSplatRaster tests passed against the imported headers.
- Ordinary Blender mesh render, DLSS final image/two-frame animation, and
  native GPU viewport motion regressions passed with the rebuilt artifacts.

Kernel data layouts changed. Rebuild all C++ objects as well as GPU kernels,
then run `cmake --install build` before staging the Python package. Do not mix
old kernel blobs or stale C++ objects with this version.
End-to-end support and validation are described in `GSPLAT_USAGE.md`, including
the user-provided 2,957,438-Gaussian trained PLY rendered through both Python
and the installed Blender add-on.
