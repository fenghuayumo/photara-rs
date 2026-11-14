# Experimental RealTime mode

This is the first, frame-scheduled baseline for an independent real-time pipeline.
It shares the existing OptiX wavefront tracer, scene, materials and light tree.
Reference retains its original sampling behavior; shared kernels include the
opt-in RealTime feedback path.

Python now also exposes opt-in `history_samples` raw-frame GPU reprojection,
independent of the selected denoiser. Default zero retains per-job film behavior.
See the Python README for bounds, memory costs, supported scene types and the
distinction from the GPU-only Viewer and DLSS's opaque history.

```python
import photara_rs as pr

with pr.Scene(1280, 720, render_mode="REALTIME", samples_per_frame=1) as scene:
    # Populate the scene using the normal Scene API.
    frame = scene.render()  # A fresh film; next call uses the next sample subset.
    scene.configure_render_mode("REFERENCE")
    reference = scene.render()  # Uses the original scene.samples budget.
```

`render_async()` uses the same policy and existing cancellation/mutation guards.
Budgets are 1–64 samples per submitted frame. Existing adaptive sampling may
terminate pixels early; disable it for fixed counts. The global sample sequence
has a stable length of 2^24, wraps before exhausting that sequence, and advances
on submission (including cancelled frames). Configuring the mode restarts it.
Reference sample settings and the integrator seed are retained. Queued camera
updates retain the active job's frame budget and sample subset.

This stage does **not** implement ReSTIR PT, GI caching, material
simplification, an automatic animation loop, or Blender UI controls. It makes no
frame-rate guarantee. Films are cleared per job; there is no radiance-history
accumulation in this module. Existing denoiser behavior is retained; this module
does not yet establish temporal validity for moving objects or lights.

RealTime now enables an experimental global and screen-tile NEE-AT analytic-light
proposal. Use `scene.configure_neeat(False)` or viewer `--light-sampler LIGHT_TREE`
for the original sampler. See [scope, estimator and tests](NEE_AT.md). Mesh-only
Cornell Box uses the original tree fallback.

The local RTXPT `PathTracerNEE.hlsli` and `NEEATBaker.hlsli` were inspected as
architectural references. No RTXPT source is copied or linked here. Future work:

1. Add GPU timings and explicit scene/history revisions for camera, geometry,
   lighting and material changes, plus persistent device buffers owned by this mode.
2. Move global and screen-tile feedback reduction to GPU-only caches,
   preserving exploration and an evaluable mixture PDF.
3. Add path reservoirs, temporal/spatial reuse and validated shift mappings.
4. Add selectively queried world-space radiance caching for rough indirect paths.

Correctness checks should compare equal-time output against Reference, including
moving cameras, disocclusion, moving emissives, thin geometry and glossy paths.
Frame-time measurements must include scene updates, reconstruction and display.

RealTime marks its SessionParams with `realtime_frame`. Updating only the three
host sample-subset sockets preserves a previously clean Integrator, avoiding a
redundant device update and sampling-table rebuild. Existing device dirtiness
(e.g. seed, sequence length, bounce limits or shadow-catcher changes) is retained.
Pixel-jitter/DLSS frames keep the original update path so jitter still advances.
Reference keeps its original subset update behavior. Each submitted job still
resets its film; the viewer still presents each accepted completed frame.

The Vulkan standalone viewer now adds a separate FP32 GPU radiance history.
Each completed RealTime frame contributes its sample budget to a running mean
before tone mapping; `--spp 1` still traces just one new sample per pixel per
frame. Camera edits, sample-budget edits and mode switches reset history.
Swapchain redraws do not add samples. CSV metrics expose `history_samples` and
`present_timestamp_ms` for checking accumulation and actual presentation-submission
cadence. This is static-view progressive accumulation: camera motion discards
history; motion-vector reprojection/disocclusion rejection is not implemented.
The Python API's per-job film behavior is unchanged.

## Viewer stage timing

Add `--gpu-timings --metrics test_outputs/timing.csv` to a viewer command to
record CUDA/OptiX event spans and Vulkan display timestamps. Without an explicit
metrics path, timing mode writes `test_outputs/viewer_gpu_timings.csv`. Timing
is opt-in; disabled frames leave stage columns empty instead of reporting zero.
No per-kernel synchronization or pixel readback is added by timing mode.

The CSV adds `scene_update_host_ms` (CPU wall time), `neeat_gpu_ms`,
`path_trace_gpu_ms`, `denoise_total_gpu_ms`, `dlss_evaluate_gpu_ms`, and
`film_convert_gpu_ms`. CUDA event spans follow each stage's actual stream;
NGX is explicitly bound to the denoiser's conversion stream, with evaluation
events recorded on that same stream. These are elapsed timeline spans,
including host submission gaps, not sums of kernel busy time. DLSS evaluation
is nested inside denoise total, which also includes guides, conversion and
postprocessing: do not add these two columns. Unsupported, disabled or
not-yet-complete stages are empty. Multiple spans in a completed frame are summed.

Vulkan accumulation, tone mapping and drawing are measured together in
`display_gpu_ms`. Query results are collected at existing fence completion
points without an additional wait. `display_gpu_frame` identifies the source
rendered frame; it is normally one frame older than its CSV row. Resize redraws
retain the source frame ID. This measures GPU display work, not monitor scanout
or presentation latency. The final frame's query may not appear in the CSV.

`gpu_queue_wait_host_ms` measures explicit CUDA queue initialization/synchronization
waits on the CPU, while `display_reader_wait_host_ms` measures existing Vulkan
reader-fence waits before CUDA writes. They overlap GPU work and must not be
added to GPU spans. Existing `observed_ms`, `worker_to_gpu_display_ms`,
`present_backpressure_ms` and presentation timestamps remain separate CPU metrics.
Scene updates can use several streams and include CPU BVH/driver work, so they
are deliberately reported as host wall time rather than a misleading GPU total.

`python scripts/summarize_gpu_timings.py test_outputs/timing.csv --warmup 32`
reports sample count, mean, P50 and P95 per stage and mode/motion phase, excluding
startup and cross-phase presentation intervals. It attributes delayed display
queries by frame ID and counts a source frame once. Timed and untimed runs should
be compared to assess instrumentation overhead; these statistics are not a
general FPS guarantee. `scripts/test_gpu_timings.py` checks opt-in behavior,
image parity, disabled/CPU-fallback stages and delayed Vulkan query attribution.

For example (with the usual OCIO environment and exported Sponza XML):

```powershell
install/photara_viewer.exe --size 1080 --spp 1 --denoiser DLSS --gpu-timings --smoke-frames 160 --metrics test_outputs/sponza_timing.csv test_outputs/sponza_viewer/sponza.xml
python scripts/summarize_gpu_timings.py test_outputs/sponza_timing.csv --warmup 32 --output test_outputs/sponza_timing_summary.json
```

## Viewer DLSS Ray Reconstruction

Viewer WASD/QE navigation integrates held-key state at 2 world units/second,
normalizing diagonal movement. Keyboard and mouse deltas are applied between
completed frames without cancelling the current render or restarting sample
subsets. Focus loss releases held keys; pause/minimize time is not accumulated
into a large movement on return. Explicit mode/budget/reset controls retain
their render-restart behavior. `scripts/test_viewer_navigation.bat` checks the
navigation policy at different update rates, repeats, diagonals and pause/release.

`install/photara_viewer.exe --denoiser DLSS --spp 1 --size 1080 scene.xml`
enables the existing core DLSS RR path at native trace/output resolution. The
default remains `--denoiser NONE`. DLSS mode requires one new sample per frame
and locks the +/- sample-budget controls. Depth, diffuse/specular albedo,
normal, roughness and backward/specular motion guides are enabled. Interactive
camera state supplies the previous rendered camera to the motion-guide kernels;
continuous movement preserves DLSS history. DLSS owns reprojection, per-pixel
history rejection and reconstruction; the Vulkan running mean is bypassed by
replacing its history every frame. No custom temporal/spatial denoiser is added.

First evaluation, explicit camera reset (`R`), mode switches and scripted camera
cuts reset DLSS history. Continuous camera navigation does not. Rendering fails
explicitly if the requested DLSS evaluation does not succeed; it cannot silently
count noisy fallback images as DLSS frames. CSV adds `dlss_frames`, `dlss_resets`,
`dlss_reuses`, the requested `temporal_reset`, `camera_x` and benchmark `phase`.
These counters report successful NGX evaluations, not per-pixel rejection rates.

The viewer sets both SessionParams and SceneParams to interactive mode and seeds
the previous FOV/viewplane before the first frame. SceneParams otherwise defaults
to background rendering: previous projection state is not advanced, and an XML
camera whose FOV differs from the default can produce large motion vectors even
when stationary. In Sponza at 512x512, the broken static motion had median absolute
component 109.8 pixels and p99 214.3 pixels; the corrected motion was zero. Raw
1-SPP input variability was unchanged, while warmed DLSS output temporal MAE
fell from 2.423 to 0.055 display RGB units (0..255) across eight captured frames.
This validates useful history accumulation, beyond merely counting NGX reuse.

`scripts/test_dlss_temporal.py --ocio <config.ocio>` checks actual warmed static
output pixels from deterministic runs ending at consecutive frame counts, then
checks a camera sweep and cut. It requires the exported Sponza fixture. These
capture runs measure image stability, not application performance; the final
screenshot readback happens after the timed rendered frames.

`--camera-sweep` uses 60 static frames, a 180-frame horizontal sinusoidal sweep
(0.6 world-unit amplitude), then a static view. `--camera-cut-frame N` translates
the camera by one world unit and resets history on frame N (zero-based).
`--camera-x X` sets an initial world-X offset, also useful for reference renders.
`scripts/benchmark_sponza_dlss.py` checks finite viewer runs, successful DLSS
evaluations, reset/reuse counts and zero timed pixel readback, and saves static,
moving and cut snapshots alongside 256-SPP references. It requires the XML from
`scripts/export_sponza_xml.py` and an OCIO configuration (via `--ocio` or `OCIO`).
This benchmark covers static geometry with camera motion; moving geometry and
animated lighting are not validated by it.

## Stable Planes

OptiX, DLSS and one sample per output frame automatically select one Stable
Planes implementation. This includes the viewer, Python API and Blender sessions.
There is no mirror-only override, guide-only mode or enabling socket. The retired
`--stable-planes` / `--stable-mirror-guide` switches are no longer accepted.
Multisample reference renders retain the ordinary integrator, without writing
unfiltered plane metadata from concurrent samples. Motion blur and shadow
catchers also use the ordinary integrator.

The implementation follows RTXPT's Build / Fill decomposition inside Photara's
wavefront integrator. Build runs at the primary OptiX surface before any Fill
emission write. It deterministically explores delta lobes using physical Fresnel
weights, without Russian roulette, with at most three planes and fifteen delta
vertices. A glass or mixed junction retains its base plane; reflection and
transmission can create child planes. A pure planar mirror may replace the primary
surface. Compatible GGX/Beckmann closures are aggregated; differently oriented delta
closures or incompatible refraction directions stop decomposition at the junction.
Delta classification uses the same almost-specular threshold as the BSDF sampler. Total internal reflection creates only
the surviving reflective lobe. Plane capacity never removes a contribution:
unexplored lobes continue stochastically on the closest retained base.

Build collects emission from visited surfaces and environment branches into a
separate `stable_radiance` pass. Fill uses the normal BSDF estimator, matches the
built branch hierarchy, and deposits each noisy contribution into its deepest
retained ancestor. Contributions from a branch with no matching base remain in
the current-frame residual. A zero contribution is a valid sample. Emission is
excluded from Combined/plane writes only when Build has captured that exact
vertex; an unsupported terminal keeps its emission in Fill. Non-delta scattering,
transparency, portals and volume traversal close the stable prefix. Analytic
lamp intersections remain stochastic, since those lights are transparent in the
Photara integrator. Auxiliary light passes keep their original estimators.

The hidden `stable_planes` film pass contains three 48-float records, with a
shared layout in `kernel/film/stable_planes.h`. Records include branch identity,
object/material identity, physical throughput, noisy radiance, actual/virtual
positions, previous positions, motion and material guides. Build commits the tree,
stable radiance and guides transactionally; texture-cache misses cannot leave a
partially built tree. OptiX material interpretation uses a continuation callable,
with retained device functions and an explicit continuation-stack allowance.

Reflection accumulates mirror transforms. Transmission uses the local shortest
rotation from outgoing to incoming direction, as in RTXPT, rather than an exact
refractive Jacobian. Curved, normal-mapped and moving optical surfaces stop motion
unfolding at the interface while still allowing radiance decomposition; downstream
roughness is reduced for RR specular-motion inference. Moving target motion is
transformed through the accumulated image transform. Guides mix residual plane
throughput, equalization and dominant-plane bias, include stable-emission albedo,
and select a coherent primary plane for depth/backward motion. The ordinary DLSS
path keeps NGX temporal history across camera movement and resets on cuts.

Volume boundaries and materials that need unavailable path state stop exploration
at an existing base. Nested dielectric priorities and absorption are not evaluated
by the deterministic tree; the ordinary Fill integrator retains those transport
contributions. Light/shadow linking, MNEE and render-layer closure filtering bypass
Build to preserve their existing transport semantics. This is a conservative
integration boundary, not a claim that every RTXPT rendering feature has a Photara
equivalent.

## Bounded GPU plane history

`--stable-plane-history N` in the DLSS viewer, or Python `history_samples=N`,
requests an explicit finite raw-frame window. This changes the temporal sample
budget; it does not choose another Stable Planes implementation. It is optional:
normal DLSS uses its own temporal history. An explicit window resets NGX each
frame so NGX cannot silently bypass the requested history limit.

GPU reconstruction removes prefix throughput before averaging each plane and
restores current throughput afterwards. It searches every previous slot using
branch and optical/material identity. Previous virtual positions propose the
lookup; previous actual target positions, normals and terminal view validate it.
Multi-frame correspondence follows matched records backwards and stops at a
failed intermediate frame. Sky lookup removes camera translation and validates
environment direction. Stable radiance and unmatched residuals stay current-frame
only. The ring contains raw samples, never recursively accumulated images.

Viewer CSV reports `stable_planes`, `active_planes`, `plane_reused_pixels` and
`plane_mean_history`. Reprojection stays on GPU; explicit diagnostics read only
reduced statistics. Query `history_stats()['memory_bytes']` for the actual
allocation. Geometry/light edits clear the ring.

Validation entry points:

- `src/test/stable_plane_temporal_test.cu`: actual CUDA decomposition/prefix,
  emission exclusion, current-throughput composition, slot reordering, target
  motion, identity/disocclusion rejection and finite-window checks.
- `photara/python/tests/test_stable_plane_history.py`: automatic Build radiance,
  analytic emissive/glass signals, no duplicate emission, sample-count transition,
  explicit glass history and disabling/clearing history.
- `scripts/test_stable_mirror_guides.py`: generated mirror, glass, rough and
  multiple-reflection scenes; moving-camera DLSS against 256-SPP captures. There
  is one implementation; no legacy runtime switch is retained for comparison.
- `scripts/test_stable_plane_history.py`: actual plane population/reuse, cuts,
  explicit-window limits, zero timed pixel readback and clean viewer exit.

Cold OptiX compilation can take minutes on the first frame. The viewer paints a
loading message and finite runs report errors on stderr. Performance and image
quality must be measured on representative scenes; an explicit finite history
window is not assumed to improve on DLSS's own longer temporal history.

Validation measured on 2026-11-14 (RTX 5090 D v2, 512x512, 181-frame camera
sweep, 256-SPP reference; display-space MSE, lower is better):

| Fixture | Previous guide-only implementation | Unified Build / Fill |
| --- | ---: | ---: |
| Mirror | 116.55 | 115.30 |
| Two mirrors | 74.89 | 73.13 |
| Glass slab | 119.17 | 105.61 |
| Rough reflection | 1.61 | 1.90 |

Glass error decreased about 11.4%; the rough fixture regressed slightly in
absolute display error. These generated fixtures do not establish a universal
quality win. Current captures/timings are in
`test_outputs/stable_mirror/unified_summary.json`. Normal DLSS history remains
active throughout camera movement; no explicit raw-frame window was used in this
quality comparison.
