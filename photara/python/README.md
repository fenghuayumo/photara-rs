# photara-rs

OptiX-first path tracing for Python and sim-to-real pipelines. The renderer core
is derived from the standalone Cycles renderer (Apache-2.0, see `LICENSE` and
the upstream AUTHORS file) and exposes a small scene API with beauty, depth,
normal, albedo and object-id passes.

```python
import numpy as np
import photara_rs as pr

session = pr.Session(1280, 720, samples=64, device="OPTIX")
material = session.add_material(pr.Material(base_color=(0.7, 0.2, 0.1)))
session.add_mesh(vertices, triangles, normals=normals, material=material)
session.set_camera(position=(3, 2, -4), target=(0, 0, 0))
session.add_area_light(position=(2, 4, -1), target=(0, 0, 0), size=(2, 2))
session.set_background(color=(0.05, 0.08, 0.12))
session.render()

beauty = session.beauty()   # (H, W, 4), linear RGBA
depth = session.depth()     # (H, W), distances in meters
```

Passes are returned as linear `float32` arrays with a top-down row order.
`photara_rs.save_png()` applies an sRGB transform for previews.

## Requirements

This wheel is built for Windows x64, CPython 3.11, NVIDIA RTX (sm_120), and
requires a sufficiently recent NVIDIA driver with OptiX support. It bundles the
native module, the Cycles/OptiX kernels and the runtime DLLs; no CUDA toolkit
installation is required at runtime.
