# OptiX opacity micromaps

OMM accelerates binary alpha-tested triangle meshes in both Reference and
RealTime. On Ada/Blackwell RTX hardware (OptiX RT-core version >= 30), eligible
meshes use hardware opacity micromaps automatically when the optional NVIDIA
OMM CPU baker is configured. Other devices/builds retain normal shader tracing.
`PHOTARA_DISABLE_OMM=1` disables the accelerator when creating a device, for
comparisons. It does not turn off alpha clipping.

## Build

The NVIDIA SDK is optional and is not vendored or downloaded during configure.
Use a matching OMM SDK header, library and shared runtime. Local validation used
SDK 1.8.0 already present with RTXPT and OptiX SDK 9.1.

```powershell
cmake -S . -B build -DPHOTARA_WITH_OMM=ON `
  -DPHOTARA_OMM_ROOT=C:/path/to/OMM `
  -DPHOTARA_OMM_LIBRARY=C:/path/to/omm-lib.lib `
  -DPHOTARA_OMM_RUNTIME=C:/path/to/omm-lib.dll
```

For an SDK installation, `PHOTARA_OMM_ROOT` can point at its `include/` and `lib/`
parent. Set `PHOTARA_OMM_RUNTIME` for shared builds so development builds/install
stage the DLL. Static builds do not need it. `PHOTARA_WITH_OMM=OFF` builds without
the SDK; a missing SDK also leaves acceleration disabled. The SDK retains its
own NVIDIA license; no SDK source or binaries are committed here.

Rebuild Python/viewer/standalone and their matching GPU kernels. The binary
cutout fallback uses the same clipping semantics with or without OMM, so
alpha holes do not consume transparency bounce counts. Genuine transparent
surfaces continue to use those counts. Shadow traversal reserves capacity for
cutout triangle crossings, with bounded packed hit counts for continuation.

## Use

```python
leaf.base_color_texture = {"path": "leaf.png", "colorspace": "sRGB"}
leaf.configure_alpha_clip(0.5)  # image alpha > threshold is retained
leaf.configure_alpha_clip(None)  # disable clipping
print(scene.omm_stats())  # supported, enabled, meshes, transparent_states
```

Setting Base Color alone preserves its original opaque behavior. Alpha clipping
is explicit. `configure_alpha_clip()` requires an assigned image and an idle
scene; changing image/cutoff/material/UVs invalidates the associated OMM. Ordinary
material edits rebuild the certified graph. Transparent blending and glass do
not get converted to binary alpha tests.

XML graphs are automatically recognized when their Surface output is a
Mix Closure whose first input is a constant white Transparent BSDF and whose
factor is `image alpha > constant cutoff`. The second input is a Diffuse BSDF
or a Principled BSDF with constant alpha 1. A direct default UV, UV Map or
Texture Coordinate UV must feed the image.

## Conservative scope

The CPU baker uses **4-state** micromaps and the entire actual runtime texture
MIP chain, including the generated `photara_tx/` file. This matters with Ray
Cones: a region is skipped only when the baked texture/filter/addressing proves
it transparent for every allowed mip. Unknown regions retain the original
shader evaluation. Opaque regions still execute required visibility/self-hit
filters because traversal keeps `OPTIX_RAY_FLAG_ENFORCE_ANYHIT`.

Initially supported: one material per ordinary triangle mesh, linear/nearest
alpha filtering, repeat/clamp addressing, static UVs and a direct alpha test.
Deforming geometry remains eligible when its UVs/material stay unchanged;
position edits currently rebuild and rebake rather than reuse the old array. UV/topology/material/texture changes force an OMM rebuild. Transforms
share the geometry's OMM through instances. Arrays and index buffers live with
the BLAS and use the existing delayed-free lifecycle.

Unsupported or unhelpful inputs use normal shading: cubic/smart filtering,
procedural or transformed alpha, UDIM, animated texture nodes, subdivision,
displacement, BSSRDF/ray-tracing/AOV/light-path/volume/emissive shaders, singular
UVs, or a bake with no guaranteed transparent states. Baking is bounded by
131072 triangles, subdivision level 5, 32 MiB raw micromap data and a workload
budget. A large or frequently edited asset can cost more to bake than it saves.
This implementation uses the CPU baker, not GPU baking or a persistent disk OMM
cache. GPU allocation failures follow the renderer's existing error handling.

## Validation

`photara/python/tests/test_omm.py` exercises hardware off/on, Reference and
RealTime, alpha/UV/material/texture changes and asynchronous mutation guards.
`scripts/test_omm.py` compares linear-image error and warm render time on six
layered alpha cards. Benchmark timing includes CPU AOV capture and excludes
first-use cache/bake costs from the warm median. Inspect `scene.omm_stats()` to
ensure acceleration actually attached; hardware support alone is not evidence.

Local validation on RTX 5090 D v2 (2026-11-12): all 20 OMM/Ray Cones/texture/
NEE-AT/RealTime regressions passed, including 17 cutout layers with direct-light
shadows and no consumption of the glass transparency-bounce budget. SDK-disabled
scene/device source branches also compiled. Both OMM-on and OMM-off Viewer runs
completed the 32-frame mode/camera/resize cycle with Vulkan validation enabled,
without reported validation or image-load errors and with zero CPU pixel readback.

The six-card 512x512, 64-spp Reference benchmark (five warm renders) measured
79.27 ms without OMM and 32.04 ms with OMM, including CPU AOV capture: about 2.47x.
All six meshes had attached OMMs (11208 referenced transparent microstates).
Linear-image RMSE was 8.40e-6; mean RGB was 0.31368873 vs 0.31368876.
This deliberately alpha-heavy fixture demonstrates a benefit, not a general
frame-rate guarantee. Results are saved under `test_outputs/omm/metrics.json`.

## Static SDK dependencies

Windows uses `.lib` for both import libraries and static archives. Set
`PHOTARA_OMM_STATIC=ON` explicitly for a static SDK. Photara then links LZ4 and
xxHash transitively into all clients and fails configuration if either is missing.
Use `PHOTARA_OMM_DEPENDENCY_ROOT` for the SDK's dependency install/build tree,
or set `PHOTARA_OMM_LZ4_LIBRARY` and `PHOTARA_OMM_XXHASH_LIBRARY` to exact libraries.
Dependencies must match the SDK architecture, configuration and CRT.

```powershell
cmake -S . -B build -DPHOTARA_OMM_STATIC=ON `
  -DPHOTARA_OMM_LIBRARY=C:/OMM/build/libraries/omm-lib/omm-lib.lib `
  -DPHOTARA_OMM_DEPENDENCY_ROOT=C:/OMM/build `
  -DPHOTARA_OMM_RUNTIME= -DPHOTARA_OMM_STATIC_OPENMP=OFF
```

Set `PHOTARA_OMM_STATIC_OPENMP=ON` when the SDK archive was built with OpenMP;
this requires and links `OpenMP::OpenMP_CXX`. Match this option to the SDK build,
not to whether your scene uses internal baking threads. Vendor-specific extra
libraries can be supplied as a CMake list in `PHOTARA_OMM_EXTRA_LIBRARIES`.
Static mode does not stage or install the OMM DLL, even if a previous shared
configuration left `PHOTARA_OMM_RUNTIME` cached. When returning to a DLL import
library, set `PHOTARA_OMM_STATIC=OFF`.

Static-link validation (2026-11-12): SDK 1.8.0 built without OpenMP, linked with
its LZ4 and xxHash static libraries. Python, Viewer and standalone linked
successfully; PE dependency inspection confirmed none imports `omm-lib.dll`.
All four OMM GPU regression tests passed with that statically linked extension.

Shared SDK configuration validates the runtime before enabling OMM. If
`PHOTARA_OMM_RUNTIME` is empty, CMake searches SDK `bin/`, `bin/Release/`, the
library directory and neighboring `bin/` directories. An explicit invalid path,
a directory, or an empty/truncated file fails configuration. Windows also
requires a `.dll` with an MZ header, rejecting accidental `.lib` inputs.
This file check does not prove binary compatibility or validate all dependent
DLLs; the SDK header/runtime version check still runs before baking. Static
mode bypasses runtime validation and staging.
