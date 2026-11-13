# Experimental NEE-AT global and screen-tile feedback

This independent temporal light-selection implementation is
inspired by RTXPT's feedback-based NEE-AT architecture. It is not a port of RTXPT
and uses dense screen-tile tables rather than its light proxies or candidate
reservoir pipeline. No proprietary RTXPT source is copied or linked.

For RealTime surface NEE, the sampler retains at least 25% of the existing
Light Tree. The remaining budget is adaptive: the feedback mixture weight is
75% times an EMA of the visible analytic-light share of NEE contribution.
Its budget is split one third global and two thirds screen-tile feedback. The bounded
screen grid has at most 8x8 tiles, with a minimum tile size of 32 pixels.
Point, spot, area, distant and environment lights can receive feedback. Emissive
meshes retain tree sampling and nonzero support. Volume-segment sampling retains
the original tree distribution. MNEE and portal scenes, disabled Light Tree,
pure emissive-mesh scenes, or trees exceeding 4096 emitter entries fall back to
the original sampler. Reference always uses the original sampler.

Screen tables require a single-device, untiled, unscaled buffer at full-image
origin. Other configurations use the adaptive tree/global mixture. Dense tables
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

Mesh shadow connections also record their visible, PDF-corrected contribution
to estimate the analytic/mesh budget split; individual mesh lights are still
sampled only through the tree. The analytic share uses 80% history and 20%
current contribution share, separately per table. Empty tiles inherit the global
share as their target. Mixed scenes start with 100% tree, while analytic-only
scenes start with 25% tree and 75% feedback. This avoids spending 75% of NEE on
a negligible analytic light in an otherwise mesh-lit scene. Contributions and
the resulting mixture weights stay on-device on the GPU update path. Both NEE
and forward BSDF-hit MIS evaluate the same frozen adaptive weight.

Scene light-table rebuilds and screen-grid/stride changes invalidate the feedback
tables. Switching modes,
changing the integrator seed, or configuring NEE-AT resets its history. Initial
analytic probabilities are uniform. Camera-only movement does not invalidate
the estimator: the old global proposal is still a valid full-support proposal,
but may be less effective until it adapts.

The viewer also preserves proposals during ordinary camera navigation and its
scripted sweep. Sample subsets continue advancing while navigating. Film history
still resets on camera motion; NEE-AT resets on explicit camera reset/cut, mode
or sample-budget changes, and the existing scene/table invalidations. The viewer
CSV `neeat_reset` records the sampling reset requested for each submitted frame,
independently of image-history resets; `sample_subset_offset` verifies that
navigation advances sampling instead of restarting the random sequence.

The local proposal is conditioned on the camera pixel, including secondary
bounces; it does not reproject previous tiles or use world-space caches. Camera
movement preserves a valid proposal but adaptation can lag. The tree and uniform
exploration keep support during that transition. Individual mesh-light preferences
are not learned; mesh-dominated scenes mostly retain tree sampling. Evaluate
equal-time error as well as frame time.

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
python scripts/test_neeat_viewer_history.py
```

Use Python matching the compiled extension. The benchmark scene contains a
visible key, many hidden analytic lights, indirect illumination and an emissive
mesh. It reports learning diagnostics, frame costs and image error against a
high-sample Reference image. Timing includes Python CPU AOV capture; this is
separate from the GPU-only viewer display path. Outputs are ignored under
`test_outputs/neeat`. Regression tests cover convergence, history reset, dynamic
light rebuilds and Reference isolation.

The NEE-AT unit regressions also compare cold and warmed mesh-dominated frames
against Light Tree and exercise camera motion with learned proposals. The finite
viewer history test checks continuous motion, explicit cuts, mode switches and
resize while retaining zero timed CPU pixel readback.

On single OptiX/CUDA devices the frame-boundary feedback reduction, EMA, exploration,
CDF construction and feedback clearing now run on the GPU. One thread processes
each bounded table, with separate ordered global/local dispatches so empty tiles
inherit the updated global proposal. Allocation initializes/registers the global
array once; warm frames do not download feedback or upload distribution tables.
CPU launch/completion synchronization remains. This bounded baseline is not a
parallel prefix-scan implementation for arbitrarily large light counts.

Set environment variable `PHOTARA_NEEAT_CPU=1` for the original host policy and
for parity tests. Other device types retain the host fallback. Explicit idle
`neeat_stats()` queries now download a distribution snapshot; these diagnostic
copies are outside the normal frame-update path. Learning individual mesh-light
selection probabilities remains future work; their class contribution already
controls the analytic/tree budget, and mesh lights retain tree-mixture support.
The idle statistics dictionary also exposes `gpu_updates` and `cpu_updates`,
counting successful table updates, so tests can verify the selected backend.

## Multiple surface NEE candidates

`scene.configure_nee_candidates(4)` or viewer `--nee-candidates 4` enables 1–16
candidate reservoir sampling in RealTime. The default is one and Reference always
uses one. This remains one new path sample per pixel per frame; only surface NEE
candidate work increases. MNEE scenes retain the original one-candidate path;
volume-segment NEE is unchanged. No temporal/spatial reservoir reuse is added.

Each candidate uses the original light proposal and its original MIS weight.
A positive bounded proxy based on unoccluded BSDF and constant-light radiance
selects one candidate, whose shadow connection is traced. Nonconstant emission
uses a unit-radiance proxy. A small positive floor retains support where the
proxy is zero; invalid samples are zero contributions counted in the fixed
candidate budget. For weights a_i, sum W, selected index J and budget M, the
selected ordinary NEE estimator is multiplied by W/(M*a_J). Conditional on the
candidates, its expectation is their arithmetic mean. Thus forward BSDF-hit MIS
continues to evaluate the original light proposal; the reservoir's conditional
selection probability is NOT substituted into either side of MIS. This is an
average of ordinary MIS estimators with stochastic selection, not an assertion
that its marginal light PDF equals the conditional reservoir probability.

Rebuild/stage host binaries, Python extension and all CUDA/OptiX kernels together:
KernelIntegrator contains the new candidate budget. `scripts/test_nee_gpu_reservoir.py`
checks GPU/CPU learning parity, finite 1-SPP output, mean energy, candidate-count
comparisons and Reference isolation on a mixed analytic/emissive multi-light scene.
Existing NEE-AT regressions continue to cover cache reset and scene edits.

## Adaptive budget and viewer history validation (2026-11-13)

Rebuilt and staged the Python extension, viewer, standalone renderer and all
CUDA/OptiX kernels. Eleven GPU/realtime unit tests and six CPU-update NEE-AT
tests passed, including the new cold-start, mesh-dominated error and camera
history regressions. GPU/CPU learning parity, candidate budgets 1/4/8, finite
output, mean energy and Reference reproduction also passed.

On the mesh-dominated 64x64 fixture (8192-spp Reference, 32 warm-up and 512
measured 1-spp frames), single-frame MSE changed from 0.129147 with the old
fixed mixture to 0.018572 with the adaptive mixture, matching Light Tree's
0.018572. The 128x128 occluded analytic-light benchmark retained its benefit:
MSE 0.703606 versus Light Tree's 4.571258, with 128-frame mean RMSE 0.042056
versus 0.086064. These fixtures do not establish a general performance gain.

The 96-frame viewer sweep/cut test requested NEE-AT resets only at frames 0
and 85 (the explicit cut), and sample offsets advanced throughout motion. The
32-frame mode/resize test reset at frames 0/8/9/24/25, preserving proposals
and advancing samples on the camera/resize frame. Film history still reset
during motion, and both runs retained zero timed CPU pixel readback.

Artifacts: `test_outputs/neeat_review/mesh_budget_after.json`,
`test_outputs/neeat_review/analytic_after/metrics.json`, and
`test_outputs/neeat_viewer_history/summary.json`.

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
