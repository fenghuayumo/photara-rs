# Native Blender synchronization

Photara 0.2 uses the complete vendored Cycles Blender integration when the
matching Blender build provides `_photara_native`. The original files and
their hashes are recorded in `src/blender/upstream_manifest.json`.

This module is a separately compiled Photara renderer, with its own C++
namespace, device, scene, shader graphs and session. It does not redirect
Photara rendering to `_cycles`. Blender's APIs provide evaluated geometry,
images (including packed/generated images), node graphs, attributes and
dependency-graph changes. The host SDK loader is shared for DLSS; denoiser
instances and histories are independent.

The old Python material conversion is retained only for Blender builds without
the module. It supports a limited subset and is not equivalent to native sync.
`PHOTARA_LEGACY_BLENDER_SYNC=1` selects it explicitly for regression testing.

## Build

Use a matching Blender checkout and its already configured Release build:

```powershell
python scripts/prepare_blender_native.py --blender-source D:/ProgramCode/C++/blender --blender-build D:/ProgramCode/C++/build_windows
cmake -S D:/ProgramCode/C++/blender -B D:/ProgramCode/C++/build_windows -DPHOTARA_NATIVE_SOURCE=D:/ProgramCode/C++/photara-rs/build/blender_native_source/embedded
cmake --build D:/ProgramCode/C++/build_windows --target blender --parallel 16
```

Run compilation in the VS developer environment. The helper generates a
separate source tree under `build/blender_native_source`, adapting namespaces,
target names and host interfaces without rewriting the vendored synchronization
files. It also prepares the small host-module registration changes and opt-in
Cycles benchmark diagnostics. The original Cycles renderer remains available.
Regenerate the tree after modifying Photara's C++ source.

Install the matching Photara wheel and `photara_blender_addon-0.2.0.zip` using
`scripts/install_blender.py`. Native Python adapters are generated from the
vendored addon files and packaged with the addon. Scene render controls use
`Scene.photara_native`; the user's `Scene.cycles` controls are preserved.
Material, object, world and view-layer properties retain Blender's full schema.

## Resolution and measurement

Native and legacy Photara disable progressive resolution reduction and force
pixel size 1. Only explicitly selected DLSS upscaling changes internal trace
resolution. The output always matches the viewport.

`scripts/benchmark_viewport_compare.py` opens the input file in a disposable
process, drives four timed translation sweeps and tracks an emissive probe in
the rendered framebuffer. It counts changed camera poses, not redraw callbacks
or accumulated samples. Its framebuffer stripe read adds measurable overhead;
the JSON reports that overhead separately. Input-to-visible-pose latency excludes
OS/monitor presentation and has about four milliseconds of pixel quantization
at 1280 pixels and the default sweep speed. This is scripted navigation rather
than a physical mouse-to-monitor test.

For Cycles comparison the helper adds the opt-in environment switch
`CYCLES_VIEWPORT_FULL_RESOLUTION=1`. The benchmark sets it, requests pixel size
1 and checks actual traced dimensions for both renderers. Normal Cycles use
without that environment variable is unchanged.

Use original materials for the reported comparison. Neutral materials are an
explicit diagnostic option and must not be presented as full scene fidelity.
The source blend is never saved or overwritten.

## Validation

`scripts/test_blender_native_sync.py` compares Photara and Cycles images before
and after edits using packed image textures, diffuse/glossy mixtures, node
groups, procedural color, bump and multiple material slots. It checks scene
diagnostics and full traced resolution. Unsupported backend features such as
OSL are not claimed to be covered by the SVM build.

## Sponza measurement, 2026-11-12

RTX 5090 D v2, Blender 5.3 Beta, original Sponza materials, 384 objects,
149,798 source polygons, 1280 x 800 viewport, 4096 sample limit, adaptive
sampling disabled, 8 bounces, four 3-second translation sweeps after warmup.
Both renderers reported 547 synchronized shader nodes, 68 image texture nodes,
pixel size 1 and actual 1280 x 800 tracing throughout the motion phase.

| Denoiser | Renderer | Motion FPS | Pose latency median | Pose latency P95 |
| --- | --- | ---: | ---: | ---: |
| OptiX | Cycles | 27.2 | 43.0 ms | 47.1 ms |
| OptiX | Photara native | 33.2 | 35.7 ms | 42.0 ms |
| DLSS, native pixels | Cycles | 26.6 | 40.2 ms | 50.5 ms |
| DLSS, native pixels | Photara native | 26.9 | 38.5 ms | 46.0 ms |

OptiX values average two sequential runs; DLSS values are one sequential run
per backend. These are software measurements with the probe/readback overhead
described above, not hardware input-to-display measurements. DLSS differences
are small enough that no general speed advantage is claimed. The original
scene has full materials in both backends; the neutral-material and reduced
resolution experiments from development are excluded.

Raw results live in `test_outputs/sponza_*fullres_*.json`, with the selected
valid runs listed in `test_outputs/sponza_full_resolution_summary.json`.

The native settings request display work at 1/60 second rather than Cycles'
increasing display intervals. This reduces queued sampling latency while
keeping full pixels. GPU tracing and denoising now dominate this scene; the
Python callback takes substantially less than one millisecond. Further large
gains at fixed quality require reducing rendering/denoising work, rather than
increasing Python timer frequency.
