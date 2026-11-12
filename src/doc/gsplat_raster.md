# Gaussian splat camera rasterization

`Render > Film > Rasterize Gaussian Splats` is enabled by default. With an
OptiX device, static pinhole perspective cameras draw default-material Gaussian
point clouds using a projected Gaussian layer. Mesh reflection, refraction,
shadow and indirect rays retain stochastic 3D intersection.

The camera layer preserves captured illumination. It is not relit by scene
lights. `Receive Mesh Shadows` (enabled by default) overlays approximate mesh
shadows without changing unoccluded captured brightness. `Shadow Strength`
controls attenuation; `Contact Distance` controls local occlusion of captured
illumination, including when analytic lights contribute. Assigning a custom material
opts that cloud into normal material tracing.
For comparisons with a raster training renderer, use the same camera, background,
resolution and color management (typically Standard, neutral look, exposure 0 in
Blender's view settings). AgX changes the displayed appearance even when the
underlying Gaussian radiance matches.

## Implementation

- Project object-space covariance with the camera Jacobian. Add 0.3 square final
  resolution pixels of low-pass dilation. Collapsed scales remain valid. Cull
  particle centers against a 20% dilated frustum, matching Caustica's default.
- Evaluate SH from the camera to each particle center in training/object space.
- Sort particles across objects by camera depth, with deterministic ties, and bin
  them into 16 by 16 pixel tiles once per scene/camera update. Tile lists have no
  fixed layer limit.
- Use a reserved OptiX instance mask for mesh primary visibility, skipping the
  Gaussian BVHs entirely. Other ray types use the ordinary visibility masks.
- Fuse the compute raster evaluation with primary intersection, using its sampled
  pixel position and mesh depth. Write display-referred premultiplied RGB and
  coverage to an internal four-channel pass. Do not modify mesh path throughput.
- Denoise the mesh layer. At display/export, blend the Gaussian layer and mesh
  background in the training RGB space, then decode the completed color to linear
  Rec.709. Apply film exposure and preserve premultiplied film alpha. The common
  CPU/GPU film conversion handles both viewport and final output without changing
  or repeatedly compositing the accumulated render buffers.

## Current boundaries

Unsupported configurations fall back to normal 3D tracing: CPU and other GPU
backends, multi-device rendering, DOF, moving/stereo/non-perspective cameras,
volumes, shadow catchers, non-Rec.709 working spaces, baking, and scenes containing
transparent surface shaders. Moving clouds, holdouts, light-linked/light-group
clouds and custom-material clouds remain on the normal tracing path individually.

The projected layer currently contributes to Combined and its alpha. Depth,
normals, Cryptomatte, emission and material AOVs continue to describe the traced
scene. Mesh-edge visibility is
sampled, then the independently accumulated layers are composited; this is not
an exact per-sample treatment of correlated partial mesh coverage. Near-camera
projection, raster extent/clipping, SH quantization and raster sort conventions
can also differ between training renderers. Pixel-identical agreement with every
3DGS renderer is not claimed.

Projection and SH evaluation run in parallel on the host, followed by a parallel
depth sort with deterministic source-order ties. Binning runs on the host. Rendering
reuses their GPU buffers between updates. Very large models or particles spanning
many tiles can make camera updates and tile-list memory significant; these costs
should be measured separately from steady-state tracing time.

## Validation

`intern/cycles/test/util_math_gsplat_test.cpp` checks projected thin Gaussians,
alpha support, planar secondary-ray intersection, training-space color blending,
transparent-film premultiplication, and adaptive layer normalization.

Run the OptiX image regression with:

```text
blender --background --factory-startup --python tests/python/cycles_gsplat_raster.py
```

It compares every pixel of synthetic renders with an independent Gaussian
reference, including mixed colors, mesh depth, transparent film, denoising,
33 coplanar layers, camera changes and toggling rasterization. It also checks
mirror/refraction visibility of planar Gaussians. This analytic comparison is
distinct from a same-camera comparison against a running Caustica renderer.

Local validation on 2026-11-01: full Blender/CUDA/OptiX build succeeded, all six
C++ tests passed, and all thirteen OptiX image checks passed. Maximum channel error
against the analytic reference was 0.000164 (linear RGBA). Added scene lighting
left the captured Gaussian appearance unchanged. The off-screen floater check
covers a failure found when inspecting the full office model after the synthetic
tests passed.

On an NVIDIA GeForce RTX 5090 D v2, the office model with 2,957,438 Gaussians at
768 by 432, without denoising or adaptive sampling, measured:

| Camera mode | Samples | Render invocation |
| --- | ---: | ---: |
| Projected raster, first invocation | 1 | 1.196 s |
| Projected raster | 8 | 1.026 s |
| Projected raster | 64 | 1.169 s |
| Stochastic 3D tracing | 8 | 0.621 s |

These are single render-call measurements including scene synchronization and PNG
writing, excluding application startup and loading the blend file. They are not
interactive viewport frame rates or a performance comparison with Caustica. The
source blend file was read without saving changes.


## Host preparation performance update

Projection and SH evaluation now use TBB worker-local buffers. Their results are
merged with one reserved allocation and sorted with TBB. Every source particle
gets a unique order key before culling, so equal-depth particles retain the same
order regardless of worker completion or object boundaries. Debug-level Cycles
logging reports projection, sorting, binning and upload times separately.

On the office scene, instrumented serial preparation measured roughly 100�C135 ms
for projection and 62�C102 ms for sorting per update. Parallel preparation measured
roughly 30�C67 ms for projection and 15�C28 ms for sorting in representative samples;
projection also had slower outliers under concurrent rendering. The render startup
can still prepare two different camera resolutions; removing that repeated work
and moving preparation to the GPU remain future optimization opportunities.

Two six-invocation comparisons at 768 by 432 and 8 samples, translating the camera
0.001 units each invocation, excluded the first invocation and measured medians:

| Run order | Serial preparation | Parallel preparation |
| --- | ---: | ---: |
| Serial then parallel | 2.729 s | 2.254 s |
| Parallel then serial | 1.146 s | 0.765 s |

Other render processes were active and produced substantial timing outliers.
These are observational render-call timings including PNG writing, not controlled
speedup guarantees. The final office PNGs had zero differing pixel channels.
All fourteen image checks passed, including a new multiple-object coplanar-order
case with culled source particles. The GPU rendering kernel and Gaussian response
are unchanged by this performance update.


## Approximate shadow reception (experimental revision)

Mesh visibility attenuates each Gaussian's linear radiance before conversion
back to training RGB and front-to-back compositing. The overlay uses geometric
mesh visibility with material-aware artistic opacity. The receiver is
now a coherent front-surface guide, rather than each photometric particle's
center or its conditional depth. Photometric Gaussians under a desk and on the
floor can contribute to the same trained pixel; directly shadowing those depths
created fragmented and stretched shadows in the supplied robot/office scene.

At camera updates, Gaussian centers are binned spatially. Each guide patch picks
the nearest inverse-depth cluster with distributed, significant support. Robust
weighted affine fits, followed by two overlapping patch-consensus passes,
estimate a continuous receiver. Sparse support, non-planar patches, isolated
bright floaters and large depth discontinuities are rejected. The grid has at
most 60 horizontal cells and scales with resolution. It is cached in trailing
`gsplat_raster_splats` records at `receiver_offset`; those records are never
included in particle tile lists or color compositing.

Rendering reads four neighboring guide patches and excludes incompatible depth
layers before interpolation. A reliable guide supplies the visible receiver,
independently of which photometric particles contribute its RGB. Unsupported
regions retain per-particle conditional-depth queries. Intermediate confidence
blends the two visibility estimates; it does not interpolate world
positions through the occluder. Geometry is regularized, not the shadow image,
so mesh silhouettes and the light's penumbra are still evaluated by shadow rays.
No proxy plane is added to the source scene and no table height is hardcoded.

This is a single visible-surface approximation for captured backgrounds, not
exact volumetric shadowing of multiple transparent Gaussian layers. Thin or
strongly curved surfaces, very sparse clouds and depth boundaries can still
need better receiver geometry. A guide cannot recover physical geometry that
was never constrained by training. Custom-material clouds retain normal tracing.

`Render > Film > Shadow Strength` is a slider from 0 to 1. It controls the
fraction of estimated occlusion applied to each Gaussian's linear radiance.
Fully blocked particles at strength 0.5 retain half their linear radiance,
rather than half their encoded RGB. New scenes default to 0.5; saved strengths
are preserved. Disabling reception or setting strength to zero preserves the
unshadowed raster appearance. Original projected alpha, SH color, ordering,
mesh depth composition and coverage are unchanged.

Point, spot, sun and area lights use Cycles sampling functions. Constant emission,
geometry factors and sampling PDFs weight the visible/total illumination ratio.
Mesh-only rays skip Gaussian primitives to avoid shadowing captured lighting a
second time. Mesh shadow visibility and light shadow switches are respected.
Cosine-weighted hemisphere samples supply contact occlusion with distance falloff,
also when analytic lights contribute. This attenuates the captured illumination
locally as well as applying the analytic light visibility ratio. A valid guide
supplies its perspective
plane normal. Unsupported regions use particle shortest-axis normals; nearly
isotropic particles have no contact normal. Contact rays use Cycles' scale-aware
origin offset and the same material-aware opacity as direct shadow rays.

`Glass Shadow Opacity` independently controls fully transmissive blockers on
Gaussian receivers (default 0.5). Opaque blockers retain weight 1. Partial
transmission interpolates these weights. Constant, directly connected Principled,
Glass and Refraction nodes supply the material transmission fraction; unsupported
or textured graphs fall back to opaque weight. Material tint, Fresnel and
refraction are deliberately excluded from this artistic shadow overlay.

Shadow rays continue past glass to find stronger blockers. The maximum blocker
weight along the ray wins; entry/exit interfaces and repeated shells do not
multiply opacity. Contact queries additionally apply distance falloff at each hit
before taking the maximum. Queries are bounded to 64 intersections. Direct and
local visibility are multiplied, then the existing Shadow Strength is applied
once. At Glass Shadow Opacity 0 glass contributes no overlay shadow; at 1 it casts
an opaque overlay shadow. Ordinary mesh appearance, reflection and refraction
continue to use the material's normal Cycles transport.

`Receiver Offset` moves shadow queries below the estimated surface along its
normal (default 0, in scene distance units). This can recover thin mesh bottoms
missed when a fitted Gaussian receiver lies slightly above them. It affects both
guide and fallback queries without changing projected colors, alpha or camera
depth. Excessive offsets detach shadows; this is an explicit scene adjustment,
not a recovered tabletop height or an automatic geometry reconstruction.

For a supported receiver, one shared mesh-visibility sample shades every
contributing Gaussian before exact front-to-back color integration. Intermediate
confidence blends this sum with the per-particle estimate, avoiding random
switches between receiver depths. Four stratified samples of cumulative alpha
weights estimate only the unsupported fraction's shadow loss. Their exact
unshadowed RGB sum is the control variate, so these fallback samples can still
have signed corrections. Light, contact and fallback sampling use scrambled
Cycles sample sequences. Area-light penumbras and contact occlusion still need
multiple samples. The independent layer is not spatially denoised and adaptive
sampling does not track its variance.

Validation includes 14 C++ math/film tests: perspective receiver normals, affine fits, rejection of degenerate
support and mixed depths, front-layer selection with a bright outlier, resolution
scaling, linear shadow strength and the original Gaussian/film checks.
`tests/python/cycles_gsplat_receiver.py` compares against an independent known
flat receiver plane with arbitrarily tilted Gaussians, slightly noisy centers,
mixed colors, and opaque/tinted-glass blockers. Its geometric reference does not
reuse the fitted guides. A one-sample mixed-color regression checks the coherent
visibility estimator, and an off-normal blocker is compared with independent
cosine-hemisphere quadrature for contact occlusion. A narrow Gaussian filter makes pixel-center comparisons
meaningful; Cycles fixes Box filtering to one pixel regardless of width settings.
The raster integration suite checks linear strength, visibility switches, mesh
depth, reflected/refracted Gaussians and unchanged coverage.

The supplied `D:/EMbdoy/demo-downloads/liquid/payload/splat_shadow.blend` is tested
at saved frame 176 using its saved Layout viewport, without adding a sphere or
saving the blend. A temporary mesh plane and temporary forced-plane shader are
used only as diagnostic references; neither is part of the renderer or delivered
scene. The independent application has its own executable, scripts and matching
GPU kernels. The normal Blender installation is not replaced. These tests are
not a same-camera comparison against a running Caustica renderer.


## Refracted receiver and lighting correction (2026-11-03)

Default captured Gaussian emission reached through nondiffuse secondary paths
now receives the same overlay as primary camera rasterization. OptiX intersection
projects the hit into the coherent guide, intersects that plane with the incoming
ray, and attenuates the terminating emission path throughput. Unsupported guide
regions and custom materials remain unchanged. The shader buffer x channel marks
the default captured-emission shader; ordinary mesh BSDFs are not changed.
An IOR-1 refraction-sheet regression independently forces the secondary path
without changing the geometric receiver/light connection: at strength 0.5 the
measured linear-radiance ratio is 0.5, compared with 1 before this correction.

Local contact visibility no longer has an extra fixed 0.3 attenuation multiplier:
material opacity, distance falloff and the global Shadow Strength provide its
bounds. Independent hemisphere references run at 16384 samples to resolve the
stronger estimator (opaque error 0.001949; glass error 0.000634).
The host guide also repairs isolated back-layer islands only when a consistent
front plane surrounds the query in all four quadrants; silhouette/depth edges
without surrounding support are retained.

The supplied scene's only Sun light was hidden from rendering (hide_render=true).
The final saved scene explicitly enables it at energy 2, keeping its direction
and angular size, exposure 0, global Shadow Strength 0.5 and glass weight 0.5.
The user authorized overwriting splat_shadow.blend. It was saved with a matching
render camera and reloaded for a 1200x800, 256-sample verification render. The
original file is therefore no longer the earlier no-key-light baseline.
Validation artifacts are under build_windows/gsplat_validation/contact_fix_20261103.
The local launcher caps concurrent GPU states at 131072 after memory allocation
failures with the automatic state count; it does not change sampling or shading.

## Mesh overlap and world receiver anchoring (2026-11-03)

Primary mesh occlusion now compares the conditional Gaussian depth at the actual
raster sample against the mesh depth. Center depth is not constant across a
tilted ellipse. The center-sorted loop must continue past occluded centers since
a later ellipse can extend in front of the mesh. Projection, alpha, material,
and lighting settings are unchanged. This fixes the dirty horizontal Gaussian
fragments over the blue enamel tray in `splat_shadow.blend`, with shadows enabled.
Gaussian-to-Gaussian compositing still uses center sorting; this is not exact
volumetric integration or a per-pixel sort of intersecting Gaussian layers.

Supported shadow receiver patches are anchored to camera-independent world-space
geometry. A deterministic, bounded sample of each supported cloud is taken before
frustum culling. A KD tree supplies 64 neighbors around fixed world grid cells;
robust covariance fits estimate their planes. The grid scale is determined by
world sample spacing rather than image resolution or camera distance. Compatible
planes are analytically reprojected into the receiver guide. Acceptance requires
distributed planar support, proximity, and agreement with the visible depth layer
and its normal. Thick layers, curves, edges, and sparse areas retain the previous
guide; their view dependence has not been eliminated. This is a conservative
surface estimate, not a reconstructed ground-truth scene mesh.

Validation: 19 math/film/geometry tests, the raster integration suite, the coherent
receiver suite (including glass/refraction/contact references), and a new tilted
ellipse/opaque mesh regression with shadows enabled and disabled. Frame 64 of the
supplied scene is rendered at its saved Layout viewport and at +/-20 degree
azimuths, at 1574x888 and 128 samples. Scene materials, light settings, global
shadow settings, and the original blend file are not changed. Test artifacts,
matching executable/kernel, and a launcher are under
`build_windows/gsplat_validation/tray_occlusion_20261103`.

The initial world-anchor implementation replaced accepted receiver cells
independently. The supplied scene exposed rectangular transitions in the robot
shadow near the monitor base. World-plane confidence, normal agreement and
depth agreement now have smooth acceptance ramps. Compatible constraints are
spread over neighboring guide cells with compact smooth weights, retaining the
original guide as a regularizer and rejecting other depth layers. This blends
receiver geometry before tracing shadows, without filtering the shadow image.
The same frame/camera confirms removal of the rectangular transition while
retaining local ellipse/mesh occlusion. Continuity regression artifacts are in
the validation directory's `continuity` subdirectory.
