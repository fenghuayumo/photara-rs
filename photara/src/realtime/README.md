# Experimental RealTime mode

This is the first, frame-scheduled baseline for an independent real-time pipeline.
It shares the existing OptiX wavefront tracer, scene, materials and light tree.
Reference retains its original sampling behavior; shared kernels include the
opt-in RealTime feedback path.

Python now also exposes opt-in `history_samples` raw-frame CPU reprojection,
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

## Experimental single mirror guide plane

`--denoiser DLSS --spp 1 --stable-mirror-guide` opts into a first stable-surface
guide experiment in the standalone viewer. This is not RTXPT's multi-plane
radiance decomposition. It does not split or change the sampled radiance paths.

The eligible primary surface must be a flat-shaded, static, unbumped, pure
zero-roughness reflector. If the next hit is a fully diffuse-feature surface,
the shader reflects that hit's position and normal into the mirror's virtual
image and writes a consistent camera-space depth, normal and backward motion
(including previous-depth delta), plus the target plane's roughness and specular
albedo. Specular motion retains the existing virtual
reflection projection. This avoids combining primary-surface backward motion
with secondary-surface features for this restricted case. The option is disabled
for Reference jobs, which can accumulate multiple samples in one film.

Rough or mixed reflectors, glass/refraction, smooth/curved reflectors, moving
reflectors, multiple reflection bounces and sky hits retain the original guides.
The flat-shading check assumes a locally planar reflector; it does not prove that
the entire object is a single plane. Mirror edges still rely on DLSS rejection.
No separate per-plane history, path-branch identifier, refraction reconnection or
multi-plane compositing is implemented yet. The option remains off by default
until scene-specific A/B validation establishes a benefit.

`scripts/test_stable_mirror_guides.py` generates mirror, glass-slab and rough
reflector fixtures, compares option-off/on 1-SPP moving-camera DLSS captures
against corresponding 256-SPP reference views, and records frame times. Glass
and rough-reflector cases specifically validate fallback behavior; they must not
be reported as supported stable planes. Set `OCIO` before running the script.
