# Photara Renderer (`photara_rs`)

Photara is an OptiX-first physically based renderer. This repository contains
the renderer core, a C ABI, and the `photara_rs` Python package.

> **Upstream attribution.** The core is derived from the Cycles renderer by the
> Blender Foundation.
> - Upstream: <https://github.com/blender/cycles>
> - Base commit: `83a06e5e79613e82f2e6d3fc7c3fdb98192c6786` (2026-11-04)
> - License: Apache-2.0, see `LICENSE`. Original copyright and
>   SPDX headers are preserved. Modifications made for Photara are also
>   distributed under Apache-2.0 unless a file states otherwise.
>
> The name Cycles appears only as upstream provenance; the product and all
> user-facing interfaces are Photara / `photara_rs`.

## Status

Working, OptiX-first MVP:

- Standalone `photara_rs.exe` builds and renders with OptiX.
- A recoverable integrator-state allocation retry was added in
  `src/integrator/path_trace_work_gpu.cpp`.
- Stable C ABI in `photara/include/photara/photara.h` and
  `photara/src/photara.cpp`.
- Python package (`photara-rs`, import `photara_rs`) with triangle meshes,
  per-vertex normals/UVs, transforms, Principled materials, area and sun
  lights, constant-color world, perspective camera, OptiX/OIDN denoising, and
  `combined`/`depth`/`normal`/`albedo`/`object_id` passes.
- Mutable batch API: meshes, instances and lights return handles; objects can
  be transformed, hidden, re-materialized and removed; materials and lights can
  be updated in place; scenes can be cleared and rebuilt; seed and samples can
  be changed per frame. The session uses dynamic BVH so new instances and
  transform changes are picked up on the next render.
- A self-contained local wheel bundles the native module, runtime DLLs and
  OptiX/CUDA kernels. It has been installed outside the repository and
  verified with OptiX rendering.

The device backends are now OptiX-only from the user's perspective. HIP, HIPRT,
Metal, oneAPI and multi-device backends were removed, and CPU/CUDA are no longer
selectable. CUDA host code and non-ray-tracing kernel binaries remain because
OptiX requires them, and a small internal CPU host device remains for denoising
and buffer transfers. Internal implementation names (the `ccl` namespace and
`cycles_*` CMake targets) remain for upstream compatibility and are not part of
the product surface.

Blender integration is **paused by user decision**. The user's Blender source
tree and build are not modified.

Not implemented yet: textures, volumes, hair/curves, subdivision, OSL, USD, and
Blender integration.

## Goals

1. OptiX correctness and Cycles-level visual fidelity.
2. A stable C ABI for the Python renderer and future integrations.
3. A fast path for interactive/realtime rendering later (simplified materials,
   low sample counts, OptiX denoising), without forking the renderer.

## Build on this machine

Windows x64, Visual Studio 2022, CUDA 12.8, OptiX 9.1 and the prebuilt Blender
libraries are required.

```bat
scripts\configure_windows.bat
scripts\build_windows.bat
cmake --install build
```

The standalone renderer is installed as `install\photara_rs.exe`.

Python package:

```bat
scripts\build_python_windows.bat
python scripts\stage_python_package.py
python -m pip wheel --no-build-isolation --no-deps -w dist photara\python
```

The wheel is `dist\photara_rs-*-cp311-cp311-win_amd64.whl`. It targets CPython
3.11 and this machine's `sm_120`; adjust the CUDA architecture list and rebuild
for other GPUs, and build a matching module for other Python versions.

Run the Python example:

```bat
set PYTHONPATH=%CD%\photara\python
python photara\python\examples\render_test.py
```

Outputs are written to `test_outputs/`.

## Architecture

```text
src/                     renderer core
photara/include/photara  public C ABI
photara/src              C ABI facade (scene building + render orchestration)
photara/python           pybind11 module and `photara_rs` package
scripts/                 Windows build and packaging helpers
```

The C ABI facade keeps the renderer session alive, owns the scene, captures
full-frame passes through a custom `OutputDriver`, and copies results into the
caller's buffers. The Python wrapper exposes a small numpy-friendly API and
converts bottom-up render rows to top-down images.

## Known issues

- **Do not assign `scene->background->shader` directly.** Configure
  `scene->default_background` and update the existing `BackgroundNode` for
  color/strength changes. Replacing the graph or setting the socket explicitly
  crashed during the first render in testing.
- A transient ~9.8 GiB per-process CUDA allocation ceiling was observed on this
  machine. The integrator-state allocation retry halves the number of states
  until allocation succeeds; an `out of GPU memory` log followed by a successful
  render can be expected.
- Mesh triangle winding must be counter-clockwise when viewed from the outside.
  Supplied vertex normals do not rescue an inverted mesh.
- Depth passes contain `1e10` for rays that miss geometry. Clamp or mask them in
  sim2real pipelines.
- Colour is linear; use `photara_rs.linear_to_srgb` / `save_png` for display.
- Materials and light emission shaders are append-only at the Cycles scene
  level; removing a light clears its nodes but not the shader graph. For long
  batch runs, reuse material slots/light handles or recreate the session
  periodically instead of adding new materials every frame.
- Mesh vertex positions and topology cannot be edited in place yet. Remove and
  re-add the object, or animate it with transforms/instances.
- `photara/python/photara_rs/{bin,lib}` and the built `.pyd` are generated
  artifacts.

## Next steps

1. Optional further trim: remove unused CPU path-tracing and CPU-kernel code.
   The small CPU host device must remain for denoising/buffer transfers, and
   CUDA host/kernel code must remain for OptiX.
2. USD/glTF/OBJ ingestion for sim2real scene loading.
3. In-place mesh deformation/topology updates for cloth, soft bodies and other
   simulation outputs.
4. Point and spot lights, plus image textures and environment maps.
5. Python 3.13 / additional wheel targets.
6. A dedicated realtime path (low samples + OptiX denoise + simplified
   materials), including async render/progress APIs.
7. Blender integration remains paused.
