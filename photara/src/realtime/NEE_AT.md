# Experimental NEE-AT global and screen-tile feedback

This independent temporal light-selection implementation is
inspired by RTXPT's feedback-based NEE-AT architecture. It is not a port of RTXPT
and uses dense screen-tile tables rather than its light proxies or candidate
reservoir pipeline. No proprietary RTXPT source is copied or linked.

For RealTime surface NEE, the sampler mixes 25% of the existing Light Tree,
25% global feedback, and 50% feedback for the pixel's screen tile. The bounded
screen grid has at most 8x8 tiles, with a minimum tile size of 32 pixels.
Point, spot, area, distant and environment lights can receive feedback. Emissive
meshes retain tree sampling and nonzero support. Volume-segment sampling retains
the original tree distribution. MNEE and portal scenes, disabled Light Tree,
pure emissive-mesh scenes, or trees exceeding 4096 emitter entries fall back to
the original sampler. Reference always uses the original sampler.

Screen tables require a single-device, untiled, unscaled buffer at full-image
origin. Other configurations use 25% tree plus 75% global feedback. Dense tables
use at most about 3.1 MiB for distribution and feedback combined. Their update
and readback cost scales with tile count times emitter count.

## Estimator and history

The learned table is frozen for an entire finite frame job. Sampling and forward
BSDF-hit MIS both evaluate the same mixture probability. Transparency attenuates
the shadow throughput before feedback is recorded. Fully occluded connections
contribute no positive feedback. AO and volume shadow paths are excluded.

At the next frame boundary, finite positive visible contribution sums are
normalized into a target distribution. An exponential moving average uses
80% history and 20% current target, followed by 5% uniform exploration. If a
global frame has no positive feedback its target is uniform. Empty local tiles
use the current global target. Actual stored CDF interval
widths define its PDF, including floating-point rounding. The tree component
additionally preserves exploration and mesh-light support.

Scene light-table rebuilds and screen-grid/stride changes invalidate the feedback
tables. Switching modes,
changing the integrator seed, or configuring NEE-AT resets its history. Initial
analytic probabilities are uniform. Camera-only movement does not invalidate
the estimator: the old global proposal is still a valid full-support proposal,
but may be less effective until it adapts.

The local proposal is conditioned on the camera pixel, including secondary
bounces; it does not reproject previous tiles or use world-space caches. Camera
movement preserves a valid proposal but adaptation can lag. The tree and uniform
exploration keep support during that transition. Mesh-dominated scenes still do
not benefit from feedback. Evaluate equal-time error as well as frame time.

## Usage

RealTime enables this experimental proposal by default. Viewer/standalone:

```powershell
install/photara_viewer.exe --light-sampler NEE_AT examples/scene_cornell_box.xml
install/photara_viewer.exe --light-sampler LIGHT_TREE examples/scene_cornell_box.xml
```

The single emissive-mesh Cornell ceiling falls back to Light Tree. To exercise
learning use analytic lights, such as the many-light benchmark below. A fresh
standalone process renders only one frame; it does not warm temporal history.

```python
scene.configure_render_mode("REALTIME", samples_per_frame=1)
scene.configure_neeat(True)  # False selects original Light Tree.
frame = scene.render()
print(scene.neeat_stats())  # active_lights, max_pdf, local_tiles, local_max_pdf; idle only.
```

Zero active lights means a fallback/Reference path. `max_pdf` is the largest
learned analytic probability, before mixing with the tree. Statistics expose
host table values, not additional pixel transfers. `local_tiles` is zero when
the screen cache is unavailable; `local_max_pdf` reports the largest tile PDF.

## Build and validation

This changes kernel data and shadow-state layout. Rebuild and stage all CUDA
and OptiX kernels together with host code. Old installed kernels are incompatible.
The Python and viewer build scripts stage their rebuilt compressed kernels
automatically. Rebuild all executables before sharing those kernels.

```powershell
$env:PYTHONPATH='photara/python'
python scripts/test_neeat.py --size 128 --frames 128 --warmup 32
```

Use Python matching the compiled extension. The benchmark scene contains a
visible key, many hidden analytic lights, indirect illumination and an emissive
mesh. It reports learning diagnostics, frame costs and image error against a
high-sample Reference image. Timing includes Python CPU AOV capture; this is
separate from the GPU-only viewer display path. Outputs are ignored under
`test_outputs/neeat`. Regression tests cover convergence, history reset, dynamic
light rebuilds and Reference isolation.

At each boundary a small feedback array is read to the CPU and the probability
table is uploaded. This is not pixel readback, but still adds synchronization and
transfer cost. GPU-only cache reduction, mesh-light feedback and candidate reservoirs remain
subsequent work.

## Screen-tile validation result (2026-11-12)

On the occluded 32-light benchmark at 128x128, 1 spp, 32 warm-up frames and
128 measured frames, compared with a 4096-spp Reference:

| Metric | Light Tree | Global + screen-tile NEE-AT |
| --- | ---: | ---: |
| Average single-frame MSE (linear RGB) | 4.5713 | 0.6997 |
| RMSE of 128-frame arithmetic mean | 0.0861 | 0.0422 |
| Mean RGB intensity | 0.7770 | 0.7775 |
| Python render/capture median | 15.92 ms | 15.90 ms |

Reference mean RGB intensity was 0.7772. The learned largest analytic PDF was
0.7982. These results show a quality improvement in this deliberately
visibility-dominated fixture, not a general speedup or a new FPS guarantee.
The temporal mean is an offline benchmark calculation, not a temporal renderer.

The screen cache used 16 tiles. The nine NEE-AT and realtime regression tests
passed on an RTX 5090 D v2. A separate symmetric two-light scene verifies that
local light preferences differ from the global proposal, preserves energy
against Reference, and resets the cache after a screen resize. Benchmark data
is in `test_outputs/neeat_screen_tiles/metrics.json`.

GPU regressions also cover emission-only fallback, adding analytic lights back,
light edits, history reset, disable/reenable and Reference reproduction. The
Vulkan viewer's 32-frame mode/camera/resize cycle completed with validation
layers enabled and no reported Vulkan validation errors. Normal-frame CSV
records retain zero CPU pixel-readback bytes.

MSVC/Ninja builds now pin compiler output and the header-dependency scanner to
the same English include prefix, retaining configured compiler launchers. This
avoids silently retaining incompatible host objects after kernel-layout edits.
