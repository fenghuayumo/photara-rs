# Rendering 3D Gaussian splats

Supported by the Windows OptiX build and Blender add-on 0.1.8.

## Python

```python
import photara_rs as pr

with pr.Scene(1200, 800, samples=16, denoise=False,
              adaptive_sampling=False) as scene:
    cloud = scene.load_gaussian_splats("splat_cleaned.ply")
    scene.camera.location = (0, 0, -5)
    scene.camera.target = (0, 0, 0)
    frame = scene.render()
    frame.save_png("render.png")
```

Set the camera for the coordinate system of your trained model. The example
camera is only suitable for a model centered near the origin.
`Scene.add_gaussian_splats(positions, scales, rotations=..., colors=...,
opacities=..., sh_rest=..., transform=..., object_id=...)` accepts arrays.
Positions/scales are N x 3; rotations are N x 4 normalized wxyz quaternions;
colors are encoded RGB (the training/compositing space, not linear RGB).
Opacity is N values in [0,1]. SH rest is N x K x 3, coefficient-major RGB,
with K = 0, 3, 8 or 15 (degrees 0 through 3). Omitted SH is zero.
The C ABI accepts 15 zero-padded SH rest RGB coefficients per Gaussian.
Clouds support `set_transform(matrix)`, `set_visible(bool)` and `remove()`.
The low-level Session method returns a native object handle with the existing
transform, visibility, instancing, ID and removal operations. Mesh-material
assignment on a Gaussian handle reports an error instead of changing its
captured radiance shader.

## Formats and Blender

The loader accepts standard trained 3DGS PLY: ASCII, binary little-endian and
binary big-endian, with x/y/z, f_dc_0..2, opacity logits, log scale_0..2,
rot_0..3 (wxyz), and optional channel-major f_rest coefficients.
It also accepts conventional 32-byte `.splat` records (float xyz/scale,
uint8 RGBA, uint8 wxyz). It does not interpret arbitrary mesh PLY as 3DGS.
`photara_rs.gsplat.save_gaussian_ply` exports binary standard PLY; exact zero
scales and alpha endpoints are bounded to tiny finite values for log encoding.

In Blender: File > Import > Gaussian Splatting (.ply/.splat), then select
Photara. The importer uses the native Gaussian point-cloud importer in the
local Blender 5.3 build. Older Blender builds without that importer are not
supported by this menu. Native Blender 3DGS PLY imports are also supported.
The sync transfers position, scale, quaternion, radiance:base and
radiance:sh_0..14, including evaluated transforms and point-data edits.
Normal point clouds without Gaussian radiance attributes are not converted.

Perspective viewport and final rendering use the independent raster layer
when its upstream eligibility rules apply; other paths use Gaussian ray
intersection. Captured illumination is preserved rather than relit as a mesh.
The layer supports mesh occlusion and the upstream approximate mesh shadows.
The independent raster layer reconstructs Combined; mesh depth/object-ID
passes do not identify its splats. Full temporal animation history, SPZ and
other compressed training formats are outside this implementation.

## Verified results

- All 19 imported upstream GSplatRaster math/compositing tests passed.
- Six Python end-to-end tests cover PLY encodings/SH layout, .splat,
  anisotropic rotations/opacity, transform/visibility/removal, shared instances,
  mesh mixing,
  OptiX/OIDN/DLSS, and DLSS upscaling's native-resolution GSplat fallback.
- Blender Gaussian import, final rendering and native DLSS viewport passed.
- User model: splat_cleaned.ply, 2,957,438 Gaussians with degree-3 SH, rendered
  in full at 1200 x 800 and 16 SPP. Standalone loading through PNG saving took
  about 5.6 seconds on the local machine (not a steady-state FPS measurement).
  The same model rendered through the installed Blender add-on.

`render_gaussian_splats.py` and `render_blender_gaussian_splats.py` under
`scripts/` reproduce the render using the saved camera JSON. The reference
camera was read from the adjacent splat_shadow.blend without modifying it.
