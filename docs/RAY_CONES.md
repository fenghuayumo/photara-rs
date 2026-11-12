# Ray cone texture LOD

Both Reference and RealTime default to `RAY_CONES`. `RAY_DIFFERENTIALS` retains
Photara's previous **compact** differential estimator; it is not the full
vector-valued Igehy implementation. Existing paths already store two scalars
(`ray.dP` footprint width and `ray.dD` spread), including compressed GPU spread.
This change does not reduce that payload size and does not claim a general speedup.

The isotropic LOD calculation follows the ray-cone texture-density and incidence
terms used in RTXPT's `TexLODHelpers.hlsli` and `TextureSampler.hlsli`, adapted to
Photara's mapped UV derivatives and tiled image cache:

```
width_at_hit = width_at_origin + distance * spread
triangle_lod = 0.5 * log2(UV_area / world_area)
lod = triangle_lod + log2(width_at_hit / abs(dot(ray_direction, geometric_normal)))
      + 0.5 * log2(texture_width * texture_height)
```

Photara recovers `width_at_hit^2 * UV_area / world_area` from the determinant of
the existing tangent-plane UV derivative matrix. This respects the actual UV
attribute, instance scale, UV transforms, UDIM mapping and per-image dimensions,
without storing a per-triangle UV constant (materials may use different UV maps).
Grazing incidence is bounded at `1e-4`. Singular UV mappings fall back to the
previous derivative LOD. Geometric derivatives remain available to bump and
procedural shaders; this is not a removal of all shader derivatives.

The new isotropic estimator applies to image lookups on static/motion triangles
with stochastic shader evaluation, including transparent-shadow and emissive
triangle evaluation through the common shader setup. Curves, point primitives,
background, displacement and importance-map queries retain their existing LOD.
The current camera initialization and symmetric NEE/BSDF roughness widening are
retained. This is not a full port of RTXPT's PDF-based scatter growth, curvature,
or refractive focusing implementation. Isotropic area-based LOD can lose detail
at grazing incidence and can underfilter highly stretched UVs. The legacy mode
is available when its result is preferable.

Python/C API, standalone and viewer clients now enable automatic tiled MIP-cache
generation for ordinary file textures using the existing OIIO pipeline. Caches
are written into `photara_tx/` beside each source texture. Existing MIP textures
are reused. If cache generation fails, the existing full-image fallback remains
available, but a texture without MIPs cannot benefit from LOD selection.
Cache generation cost occurs on first use and should be excluded from warm
render-time comparisons. `RAY_DIFFERENTIALS` also benefits from the same MIP cache.

## Controls

```python
scene.configure_texture_lod("RAY_CONES")  # default, for both render modes
scene.configure_texture_lod("RAY_DIFFERENTIALS")
print(scene.texture_lod)
```

The setter rejects mutation while an asynchronous job is running. Changing LOD
invalidates learned NEE-AT history. Mode changes retain the selected LOD method.

XML:

```xml
<integrator texture_lod_method="ray_cones" />
<!-- alternative: texture_lod_method="ray_differentials" -->
```

Viewer and standalone accept `--texture-lod RAY_CONES` or
`--texture-lod RAY_DIFFERENTIALS`, overriding XML when explicitly supplied.
The C functions are `photara_set_texture_lod` and `photara_get_texture_lod`.

## Validation

Rebuild all CUDA/OptiX kernels, Python extension and executables after changing
the integrator layout. Python/viewer build scripts stage the matching kernels.

`photara/python/tests/test_ray_cones.py` checks selection on a grazing plane using
a texture with different colors in each MIP, singular UV fallback, Reference/
RealTime switches, resize, configuration validation and the asynchronous guard.
Diagnostic MIP generation requires OpenImageIO's Python module. The kernel math
regressions are in `src/test/kernel_ray_cone_test.cpp`.

Local validation (RTX 5090 D v2, 2026-11-12): all 16 tests across ray cones,
textures, NEE-AT and realtime lifecycle passed. A standalone host harness also
passed eight analytical LOD/degeneracy checks against the kernel helper.
A 96x64, 1-spp diagnostic MIP scene (8 warm-up, 64 measured frames) measured
15.94 ms for ray cones versus 15.96 ms for compact differentials, including
Python CPU AOV capture. This is a correctness fixture with host scheduling
costs; the difference does not establish a GPU performance improvement.

Both Viewer LOD choices completed the 32-frame mode/camera/resize cycle on a
textured XML fixture with Vulkan validation enabled, no reported validation or
texture-load errors, and zero CPU pixel-readback bytes in the frame CSVs.
The standalone Reference executable also rendered the textured fixture with
`--texture-lod RAY_CONES`. Host executables and installed GPU kernels were rebuilt
together. These changes remain separate from NEE-AT commit `0b8430b`.
