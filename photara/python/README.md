# photara-rs

OptiX-first path tracing for Python and sim-to-real pipelines. The renderer core
is derived from the standalone Cycles renderer (Apache-2.0, see `LICENSE`).

The public API is a small Blender-style scene graph: a `Scene` owns
name-addressable collections of materials, meshes, objects and lights, plus a
camera and a world. Meshes can be shared by multiple objects; materials and
lights can be edited in place; the same scene renders repeatedly for batch data
generation.

```python
import numpy as np
import photara_rs as pr

scene = pr.Scene(width=1280, height=720, samples=64, device="OPTIX")

material = scene.materials.new("prop", base_color=(0.7, 0.2, 0.1), roughness=0.3)
mesh = scene.meshes.new("prop", vertices, triangles, normals=normals,
                        material=material)
prop = scene.objects.new("prop", mesh, location=(0.0, 1.0, 0.0), object_id=1)
instance = scene.objects.new("prop.001", mesh, location=(2.0, 1.0, 0.0),
                             object_id=2)

scene.camera.location = (3.0, 2.0, -4.0)
scene.camera.target = (0.0, 0.0, 0.0)
scene.camera.fov = 45.0
scene.lights.new_area("key", position=(2.0, 4.0, -1.0), target=(0.0, 0.0, 0.0),
                      size=(2.0, 2.0), strength=400.0)
scene.lights.new_sun("sun", direction=(-0.4, -1.0, 0.2), strength=2.0)
scene.world.color = (0.05, 0.08, 0.12)

frame = scene.render(seed=0)
frame.beauty        # (H, W, 4), linear RGBA, top-down
frame.depth         # (H, W), 1e10 for rays that miss geometry
frame.depth_valid   # (H, W) boolean mask
frame.normal        # (H, W, 3)
frame.albedo        # (H, W, 3)
frame.object_id     # (H, W), int32
frame.save_npz("frame_0000.npz")
frame.save_png("frame_0000.png")   # requires Pillow
```

## Scene graph

- `scene.materials.new(name, **parameters)` creates a Principled BSDF material.
  Changing `material.base_color`, `.roughness`, `.metallic`, etc. updates the
  renderer immediately. `material.update(**kwargs)` sets several fields with one
  update. `material.base_color_texture` attaches a file-backed Base Color image.
- `scene.meshes.new(name, vertices, indices, normals=None, uvs=None,
  material=None)` creates a mesh data-block. `mesh.update_positions(...)`
  deforms vertices/normals/UVs without changing topology;
  `mesh.update_topology(...)` replaces triangles/material in place. All objects
  sharing the mesh update together.
- `scene.objects.new(name, mesh, location=..., rotation_euler=..., scale=...,
  object_id=...)` creates an object. Several objects can share one mesh
  (instanced geometry); `mesh.users` reports the number of users.
- `scene.lights.new_area(...)`, `new_sun(...)`, `new_point(...)` and
  `new_spot(...)` create lights; `scene.lights.new(name, type=..., ...)`
  dispatches. Lights are also visible through `scene.objects`.
- `scene.camera` is the active perspective camera. It supports `location`,
  `rotation_euler`, `matrix_world`, `target`, `fov`, `near_clip`, `far_clip`.
- `scene.world` controls the background: constant `color`/`strength` or an
  equirectangular `environment_texture`.
- Collections support name lookup (`scene.objects["prop"]`), iteration,
  `len()`, `.get()` and `.remove()`.
- Objects support `.location`, `.rotation_euler` (XYZ Euler in radians),
  `.scale`, `.matrix_world`, `.visible`, `.object_id`, `.material`,
  `.remove()`.
- Lights support transform properties, `.color`, `.strength`, `.size` (area),
  `.radius` (point/spot), `.spot_angle` and `.smooth` (spot), `.angle` (sun),
  `.target` (area), `.direction` (sun/spot), and `.visible`.
- `scene.samples` and `scene.seed` can change between renders. Resolution,
  maximum bounces, denoising and adaptive sampling are fixed for the lifetime
  of the scene.
- Removing a scene, mesh, object or light and then using it raises
  `PhotaraRemovedError`; using a closed scene raises `PhotaraClosedError`.

## Textures and HDR environments

```python
# Base-color texture; base_color remains a multiplier (use white for the raw texture).
material.base_color_texture = "C:/textures/albedo.png"   # sRGB, linear filtering, repeat
material.base_color_texture = {
    "path": "C:/textures/albedo.png",
    "colorspace": "sRGB",       # auto | sRGB | scene_linear | data
    "interpolation": "linear", # linear | closest | cubic | smart
    "extension": "repeat",     # repeat | extend | clip | mirror
}
material.base_color_texture = None

# Equirectangular environment map (PNG/JPG/EXR/HDR through OpenImageIO).
scene.world.environment_texture = "C:/hdri/studio.hdr"  # scene_linear by default
scene.world.environment_texture = {"path": "...", "colorspace": "sRGB"}
scene.world.environment_texture = None   # restore the constant background colour
```

Textures are loaded by OpenImageIO and cached by path/parameters. Files are
checked at assignment time. Reloading after an on-disk change is not exposed
yet; assign the texture again to force a fresh handle. HDR environment
lighting uses the background shader (background-light MIS is not implemented
yet), so low-sample environment renders can be noisier than geometry lights.

## Importing assets

Static OBJ/MTL assets can be loaded directly into a Scene:

```python
result = scene.import_obj(
    "assets/robot.obj",
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),   # XYZ radians
    scale=1.0,
    object_id_start=10,         # 0 keeps the default id
)
result.objects        # created Object handles
result.meshes         # created Mesh data-blocks
result.materials      # created Material data-blocks
result.warnings       # non-fatal MTL/material issues
```

OBJ support includes groups/objects, negative indices, quad/ngon fan
triangulation, per-corner UV/normal seams (vertices are duplicated), generated
smooth normals when absent, and MTL `Kd`/`Ke`/`Ns`/`map_Kd`.

glTF 2.0 `.gltf` and `.glb` files are supported for static triangle meshes:

```python
result = scene.import_gltf("assets/robot.glb", object_id_start=100)
```

Supported: node hierarchy/TRS and matrix transforms, multiple primitives and
materials per mesh, POSITION/NORMAL/TEXCOORD_0, indices (including normalized
and strided accessors), `baseColorFactor`, metallic/roughness/emissive factors,
`baseColorTexture` (external, data URI, or embedded GLB bufferView), GLB
containers, and sampler wrap/filter mapping. Cameras, punctual lights,
animations, skins, morph targets, Draco/meshopt, alpha modes, sparse accessors
and additional PBR texture maps are not imported.

## Batch rendering

The same scene can be rendered repeatedly with different transforms, camera
poses, material parameters, lights and per-frame seeds:

```python
for frame_index in range(1000):
    prop.location = simulated_position(frame_index)
    material.update(base_color=randomized_colour(frame_index))
    frame = scene.render(seed=1000 + frame_index)
    frame.save_npz(f"data/frame_{frame_index:04d}.npz")
```

See `examples/batch_sim2real.py` for a complete example with domain
randomization, instances and multi-pass npz output:

```bat
set PYTHONPATH=%CD%\photara\python
python photara\python\examples\batch_sim2real.py --frames 4
```

## Async rendering

`Scene.render_async()` returns a `RenderJob` while the render runs on the
renderer's session thread. The GIL is released, so the calling thread can
poll progress or do other work. Scene mutation is rejected while a job is
running.

```python
job = scene.render_async(seed=1000)
while job.state == "running":
    print(job.progress, job.current_sample, job.status)
    time.sleep(0.05)

frame = job.wait()          # RenderResult; raises on cancel/failure
# frame = job.result()      # non-blocking access once done
# job.cancel()              # stop and wait for the renderer
```

The low-level `Session` exposes the same `render_async()` API. A cancelled
job raises `PhotaraCancelledError`; failures raise `PhotaraRenderError`.

## Low-level Session API

`pr.Session` exposes the same renderer through integer handles. It is useful
for custom scene builders, render farms and tests. It is also the layer on
which `Scene` is built:

```python
with pr.Session(640, 480, samples=32) as session:
    material = session.add_material(pr.Material(base_color=(0.7, 0.2, 0.1)))
    obj = session.add_mesh(vertices, triangles, material=material)
    instance = session.add_instance(obj, transform)
    session.set_camera(position=(3, 2, -4), target=(0, 0, 0))
    session.render(seed=1)
    beauty = session.beauty()
```

Passes are linear `float32`. `pr.linear_to_srgb()` and `pr.save_png()` apply an
sRGB transform for display. Alpha is preserved for RGBA input.

## Installation requirements

Available wheels target **Windows x64, CPython 3.11 or 3.13, NVIDIA RTX 50-series (sm_120)**
and requires a recent NVIDIA driver with OptiX support. The native module,
OptiX/CUDA kernels and runtime DLLs are bundled; no CUDA toolkit is needed at
runtime. Other Python versions or GPU generations require a matching rebuild.

Use the wheel whose `cp311`/`cp313` tag matches your interpreter. The local
Blender 5.3 build uses Python 3.13.13 and requires the `cp313` wheel. The
repository's `scripts/install_blender.py` selects and installs it automatically.
This is a standalone native renderer: importing `photara_rs` does not require
Blender, `bpy`, the source checkout, or a CUDA Toolkit installation.

`save_png()` requires Pillow, available through the optional extra:

```bat
pip install "photara-rs[png]"
```

## Current limitations

- No full PBR texture set (metallic/roughness/normal/emissive maps), volumes,
  hair/curves or subdivision.
- OBJ and glTF/GLB static-mesh import are available; USD import and
  animation/skinning are not implemented yet. Geometry can always be supplied
  directly as NumPy arrays.
- One perspective camera per scene; no orthographic, panoramic, DOF or motion
  blur controls.
- Fixed passes only: combined, depth, normal, albedo and object-id.
- Materials cannot be deleted from the renderer; `materials.remove()` only
  detaches an unused material from the Python scene.

## Denoiser configuration

`Scene` and `Session` accept `denoiser="AUTO" | "OPTIX" | "OPENIMAGEDENOISE"`,
`denoise_passes="RGB" | "RGB_ALBEDO" | "RGB_ALBEDO_NORMAL"`,
`denoise_prefilter="NONE" | "FAST" | "ACCURATE"`,
`denoise_quality="HIGH" | "BALANCED" | "FAST"`, and `denoise_use_gpu=True`.
Prefilter, quality and GPU preference apply to OIDN. Settings are fixed for the
session lifetime. Use a rebuilt native wheel with this API; the local cp313
wheel includes it. The render device remains OptiX even when denoising on CPU.
