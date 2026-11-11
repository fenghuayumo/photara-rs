# Photara Renderer (`photara_rs`)

Photara is an OptiX-first physically based renderer written in C++ and CUDA,
with a small C ABI and a Python package named `photara_rs`. It targets
production-quality path tracing and Python workflows such as sim2real synthetic
data generation.

> **Upstream attribution.** The renderer core is derived from the Cycles
> renderer by the Blender Foundation, licensed under Apache-2.0. Original
> copyright, license, and SPDX notices are preserved (see `LICENSE`).
> The name Cycles appears only as upstream provenance; the product
> and all user-facing interfaces in this repository are Photara /
> `photara_rs`.

## Status

- OptiX path tracing on the renderer core, with Cycles-level sampling, BSDFs,
  light sampling, denoising, and AOVs.
- Standalone executable: `photara_rs`.
- Stable C ABI: `photara/include/photara/photara.h`.
- Python package and wheel: distribution `photara-rs`, import `photara_rs`.
- Blender-style Python scene graph: `Scene`, `Object`, `Mesh`, `Material`,
  `Light`, `Camera`, `World` and `RenderResult`, with data-block collections.
  Objects can be instanced, transformed, hidden, re-materialized and removed;
  meshes can be deformed or rebuilt in place; materials and lights update in
  place; base-color image textures and equirectangular HDR environments are
  supported; static OBJ/MTL and glTF/GLB assets can be imported into a scene;
  renders can run asynchronously with progress/cancel and a scene can render
  many frames without being rebuilt. A low-level handle-based `Session` remains
  available.
- Supported device: **OptiX**. The HIP, HIPRT, Metal, oneAPI and multi-device
  backends and the user-facing CPU/CUDA selection have been removed. CUDA host
  code and non-ray-tracing kernel binaries remain as OptiX dependencies, and a
  small internal CPU host device remains for denoising and buffer transfers.
- Blender add-on: a `PHOTARA_RS` render engine driven by the `photara_rs`
  package, with F12 final renders (Combined + Depth passes) and a rendered
  viewport via asynchronous jobs. See `photara_rs_engine/README.md`.

## Requirements

- Windows 10/11 x64
- NVIDIA RTX GPU with a recent driver (OptiX runtime)
- Visual Studio 2022 with the C++ toolchain
- CUDA Toolkit 12.8
- OptiX SDK 9.1
- Prebuilt dependencies in `lib/windows_x64`

## Build the standalone renderer

```bat
scripts\configure_windows.bat
scripts\build_windows.bat
cmake --install build
```

The executable is installed as:

```text
install\photara_rs.exe
```

Check the build:

```bat
install\photara_rs.exe --version
install\photara_rs.exe --list-devices
```

## Render a scene

```bat
install\photara_rs.exe --background --device OPTIX --samples 64 ^
  --output image.png examples\scene_monkey.xml
```

`--help` lists all command-line options. Render passes and scene format details
are documented in `PHOTARA.md`.

## Python package

The package is built from the same core:

```bat
scripts\build_python_windows.bat
python scripts\stage_python_package.py
python -m pip wheel --no-build-isolation --no-deps -w dist photara\python
```

Install the resulting wheel and use the Scene/Object API:

```bat
pip install dist\photara_rs-0.1.0-cp311-cp311-win_amd64.whl
```

```python
import numpy as np
import photara_rs as pr

scene = pr.Scene(width=1280, height=720, samples=64, device="OPTIX")

material = scene.materials.new("prop", base_color=(0.7, 0.2, 0.1), roughness=0.3)
mesh = scene.meshes.new("prop", vertices, triangles, normals=normals, material=material)
prop = scene.objects.new("prop", mesh, location=(0.0, 1.0, 0.0), object_id=1)
instance = scene.objects.new("prop.001", mesh, location=(2.0, 1.0, 0.0), object_id=2)

scene.camera.location = (3.0, 2.0, -4.0)
scene.camera.target = (0.0, 0.0, 0.0)
scene.lights.new_area("key", position=(2.0, 4.0, -1.0), target=(0.0, 0.0, 0.0),
                      size=(2.0, 2.0), strength=400.0)
scene.lights.new_sun("sun", direction=(-0.4, -1.0, 0.2))
scene.world.color = (0.05, 0.08, 0.12)

frame = scene.render(seed=0)
frame.beauty            # (H, W, 4), linear RGBA, top-down
frame.depth             # (H, W); use frame.depth_valid as the miss mask
frame.object_id         # (H, W), int32
frame.save_npz("frame_0000.npz")

# Reuse the scene for later frames.
prop.location = (0.0, 1.2, 0.0)
material.base_color = (0.2, 0.5, 0.8)
frame = scene.render(seed=1)
```

`Scene.objects` contains mesh objects and lights. `Scene.meshes` can be shared
by several objects, and `Scene.materials`, `Scene.meshes`, `Scene.objects` and
`Scene.lights` are name-addressable collections. The scene uses dynamic BVH, so
object transforms, instances, material/light edits and removals apply on the
next `render()` without rebuilding the scene. The low-level handle-based
`pr.Session` remains available for custom scene builders and render farms.

The wheels in `dist/` target **Windows x64, CPython 3.11 or 3.13 and RTX
50-series (sm_120)**. Install the wheel matching your interpreter (`cp311` or
`cp313`). Other Python versions or GPU generations need a matching rebuild.
Wheel creation checks the extension ABI and rejects mismatched native modules.

See `photara/python/README.md`, `photara/python/examples/render_test.py` and
`photara/python/examples/batch_sim2real.py` for more details.

## Blender integration

Install into the specified Blender build and enable the engine automatically:

```bat
python scripts\install_blender.py --blender "D:\ProgramCode\C++\build_windows\bin"
```

The installer detects Blender's Python version, selects its matching wheel,
installs the native package, packages and installs the add-on, checks OptiX,
and saves the enabled add-on in Blender preferences. Close Blender before
upgrading native binaries, then reopen it and select **Photara** as the render
engine. No Blender source modifications or rebuild are needed.

The local Blender **5.3.0 Beta / Python 3.13.13 / RTX 5090 D v2** has been
verified with final Combined/Depth passes, multilayer EXR output, collection
instances, linked lights, incremental edits and rendered viewport updates.
The integration checks can be repeated with:

```bat
"D:\ProgramCode\C++\build_windows\bin\blender.exe" --background --python-exit-code 1 --python scripts\test_blender.py
"D:\ProgramCode\C++\build_windows\bin\blender.exe" --factory-startup --python-exit-code 1 --python scripts\test_blender.py -- --viewport
```

Results, logs and `photara_demo.blend` are in `test_outputs/`.

For manual installation:

```bat
python scripts\package_blender_addon.py
```

1. Install the `photara_rs` wheel matching Blender's bundled Python: `cp311`
   for the Python 3.11 build, `cp313` for the local Python 3.13 build.
2. Install `dist\photara_blender_addon-<version>.zip` through Blender's
   add-on preferences and enable **Render: Photara Renderer**.
3. Select the **Photara** engine in Render Properties.

The add-on syncs meshes (with modifiers and instances), Principled BSDF
materials, area/point/sun/spot lights, world backgrounds and HDR
environments, and the camera incrementally via `depsgraph.updates` — the
same strategy as Cycles' own `BlenderSync`. Current limitations are
   documented in `photara_rs_engine/README.md`.

## Repository layout

```text
src/                     renderer core
photara/include/photara  public C ABI
photara/src              C ABI facade
photara/python           pybind11 module and `photara_rs` package
scripts/                 Windows build and packaging helpers
examples/                standalone scene examples
```

Internal implementation identifiers (the `ccl` C++ namespace and `cycles_*`
CMake library targets) are retained as compatibility details so that upstream
fixes remain mergeable. They are not part of the product surface.

## License

Apache-2.0. See `LICENSE`. Original upstream copyright and SPDX headers are
preserved.
