# NEE-AT light sampling

The current implementation supports clustered feedback for analytic lights and
emissive mesh instances. See [scale, estimator, limitations and validation](NEE_AT_CLUSTERED.md).

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

## Historical adaptive-budget and viewer validation (2026-11-13)

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
