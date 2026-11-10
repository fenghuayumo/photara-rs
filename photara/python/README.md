# photara-rs

OptiX-first path tracing for Python and sim-to-real pipelines. The renderer core
is derived from the standalone Cycles renderer (Apache-2.0, see `LICENSE`)
and exposes a small scene API with beauty, depth,
normal, albedo and object-id passes.

```python
import numpy as np
import photara_rs as pr

session = pr.Session(1280, 720, samples=64, device="OPTIX")
material = session.add_material(pr.Material(base_color=(0.7, 0.2, 0.1)))
object_handle = session.add_mesh(vertices, triangles, normals=normals,
                                 material=material)
instance = session.add_instance(object_handle, transform)
session.set_camera(position=(3, 2, -4), target=(0, 0, 0))
light = session.add_area_light(position=(2, 4, -1), target=(0, 0, 0), size=(2, 2))
session.set_background(color=(0.05, 0.08, 0.12))
session.render(seed=0)

beauty = session.beauty()   # (H, W, 4), linear RGBA
depth = session.depth()     # (H, W), distances in meters

# Later frames reuse the same session and scene.
session.set_object_transform(object_handle, next_transform)
session.update_material(material, base_color=(0.2, 0.5, 0.8))
session.update_area_light(light, position=(1, 3, -1), target=(0, 0, 0),
                          size=(2, 2), color=(1, 1, 1), strength=800)
session.render(seed=1)
```

Meshes, instances and lights return integer handles. Objects can be
transformed, hidden, assigned a new object id or material, removed, and cleared
(`remove_object`, `clear_objects`, `clear_lights`, `clear_scene`). Materials and
lights support in-place updates. Per-frame `seed`/`set_samples` controls make
the same session usable for batch rendering.

Passes are returned as linear `float32` arrays with a top-down row order.
`photara_rs.save_png()` applies an sRGB transform for previews. See
`examples/batch_sim2real.py` for a complete multi-frame pipeline that writes
`beauty`, `depth`, `depth_valid`, `normal`, `albedo` and `object_id` to npz.

## Requirements

This wheel is built for Windows x64, CPython 3.11, NVIDIA RTX (sm_120), and
requires a sufficiently recent NVIDIA driver with OptiX support. It bundles the
native module, the Cycles/OptiX kernels and the runtime DLLs; no CUDA toolkit
installation is required at runtime.
