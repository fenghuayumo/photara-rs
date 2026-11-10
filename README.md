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
- Mutable scene handles for batch rendering: objects can be instanced,
  transformed, hidden, re-materialized and removed; materials and lights can be
  updated in place; a session can render many frames without being rebuilt.
- Supported device: **OptiX**. The HIP, HIPRT, Metal, oneAPI and multi-device
  backends and the user-facing CPU/CUDA selection have been removed. CUDA host
  code and non-ray-tracing kernel binaries remain as OptiX dependencies, and a
  small internal CPU host device remains for denoising and buffer transfers.
- Blender integration is paused and is not part of this build.

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

Install the resulting wheel and use the numpy-friendly API:

```bat
pip install dist\photara_rs-0.1.0-cp311-cp311-win_amd64.whl
```

```python
import numpy as np
import photara_rs as pr

session = pr.Session(1280, 720, samples=64, device="OPTIX")
material = session.add_material(pr.Material(base_color=(0.7, 0.2, 0.1)))
object_handle = session.add_mesh(vertices, triangles, normals=normals, material=material)
instance = session.add_instance(object_handle, transform)
session.set_camera(position=(3, 2, -4), target=(0, 0, 0))
light = session.add_area_light(position=(2, 4, -1), target=(0, 0, 0), size=(2, 2))
session.set_background(color=(0.05, 0.08, 0.12))
session.render(seed=0)

beauty = session.beauty()   # (H, W, 4), linear RGBA
depth = session.depth()     # (H, W)

# Reuse the session for later frames.
session.set_object_transform(object_handle, next_transform)
session.update_material(material, base_color=(0.2, 0.5, 0.8))
session.update_area_light(light, position=(1, 3, -1), target=(0, 0, 0),
                          size=(2, 2), color=(1, 1, 1), strength=800)
session.render(seed=1)
```

Meshes, instances and lights return integer handles. Objects can be
transformed, hidden, given a new pass id/material, removed, or cleared; the
scene uses dynamic BVH so these changes apply to the next `render()` without
rebuilding the session.

See `photara/python/README.md`, `photara/python/examples/render_test.py` and
`photara/python/examples/batch_sim2real.py` for more details.

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
