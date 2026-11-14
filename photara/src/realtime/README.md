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

## Experimental deterministic stable-plane guides

`--denoiser DLSS --spp 1 --stable-planes` enables deterministic delta-tree
reconstruction guides in the standalone viewer. `--stable-mirror-guide` remains
an alias. The internal film socket retains its previous name for compatibility.
Reference jobs disable the option.

The guide trace enumerates reflection and transmission lobes of a pure, flat,
static, unbumped, zero-roughness GGX/Beckmann mirror or dielectric. It uses physical
Fresnel lobe weights, including total internal reflection, rather than the
radiance path's randomly sampled branch. It follows consecutive eligible surfaces
up to 15 vertices and retains at most three leaves. When the leaf budget is
exhausted the current surface becomes a guide plane. Normals and target motion
are transformed back through the accumulated image transform. Reflection uses a
mirror transform; refraction uses a local rotation approximation, as in RTXPT,
rather than an exact refractive Jacobian.

The dominant leaf supplies depth and backward motion including previous-depth
delta. Material normals, roughness and albedos are blended by deterministic path
throughput. All guide passes are committed together after successful exploration;
subsequent stochastic radiance bounces cannot overwrite them. The primary hit is
scheduled through the OptiX shader-raytrace kernel because ordinary CUDA shading
has no traversable OptiX BVH. Enabling the option requests that kernel even in
scenes without ray-tracing shader nodes.

Rough, mixed, smooth/curved and moving primary reflectors retain legacy guides.
Volumes, portals, holdouts, dispersion and path-dependent/ray-tracing shaders
encountered during exploration cause an atomic fallback. Background misses use
rotation-only motion. Moving target surfaces are supported; moving optical
interfaces are not unfolded. Nested dielectric priorities, volume absorption and exact refraction reconnection
remain unsupported. Per-plane radiance/history extends this guide implementation
as described below. Radiance sampling remains Monte Carlo; enumerating guide
branches does not itself provide extra radiance samples.

The option remains off by default pending wider scene validation. Run
`scripts/test_stable_mirror_guides.py` with `OCIO` set to generate mirror, glass
slab and rough-primary fixtures and compare moving-camera 1-SPP DLSS output
against 256-SPP references. Glass must now change when enabled; rough primary
reflectors remain fallback tests. MSE and frame timings are recorded in
`test_outputs/stable_mirror/summary.json`. A changed output alone is not evidence
of improved reconstruction quality.

Validation on 2026-11-14 (RTX 5090 D v2, 512x512, camera sweep, 256-SPP reference):

| Fixture | Legacy display MSE | Stable guides MSE | Change |
| --- | ---: | ---: | ---: |
| Mirror | 115.44 | 116.55 | +1.0% |
| Two mirrors | 92.75 | 74.89 | -19.3% |
| Glass slab | 215.24 | 119.17 | -44.6% |
| Rough primary fallback | 1.61 | 1.61 | Identical output |

These are captured display-space errors in generated fixtures, not a guarantee
for arbitrary materials. Glass/two-mirror tests require at least 10% improvement;
the single-mirror case deliberately does not claim an improvement. Guide tracing
adds work: glass sweep presentation intervals increased from 4.68 to 7.19 ms in
this run. Capture runs are diagnostic, not isolated performance measurements.
The new optional OptiX module took about 126 seconds for the first cold frame;
subsequent fixture startups were about 1.85 seconds with the driver cache. Cold
compilation remains a limitation. Automated runs hide their window; interactive
startup paints a loading message, and finite tests report failures on stderr.

Material interpretation executes in an OptiX continuation callable so its raycast
nodes may trace legally. The callable is retained with `--keep-device-functions`
and its continuation stack is included in pipeline stack sizing. A no-output
preflight routes excluded primary materials back to ordinary CUDA shading before
any passes/radiance are written, preserving their original reconstruction inputs.

## Per-plane raw radiance history

`--stable-planes --stable-plane-history 4 --denoiser DLSS --spp 1` enables
bounded GPU history on the deterministic delta-tree leaves. The hidden
`stable_planes` film pass has three 24-float records. Each record stores a
branch prefix, optical-object/material key, virtual and actual positions,
world normal, terminal view direction, roughness, physical prefix throughput,
and a separate RGB radiance accumulator. Sky records store environment direction.

The ordinary unbiased integrator still samples the BSDF. Direct-light shadow
paths inherit the branch prefix; surface/volume emission and background writers
classify each clamped contribution into a disjoint leaf accumulator. A branch
not sampled in a frame contributes zero, rather than being treated as a missing
sample. This is stochastic radiance decomposition, not three extra deterministic
full-path samples per pixel. Contributions before a leaf or outside the supported
tree remain in a current-frame residual. No contribution is discarded or counted
twice when the plane histories are recomposed.

The raw GPU ring stores every leaf separately. Prefix throughput is removed
before temporal averaging and the current frame's throughput is restored during
composition, so old Fresnel ratios do not become the new frame's glass mixture.
History searches all old plane slots by branch and optical/material identity;
slot number is not an identity. Virtual coordinates propose a small neighborhood,
but actual surface position, normal and terminal view must match before reuse.
Sky correspondence is rotational and validates environment direction. The finite
ring contains raw samples, never recursively accumulated images. Cuts clear it.
DLSS resets its opaque history when the explicit ring is enabled.

Viewer CSV includes `active_planes`, `plane_reused_pixels`, and
`plane_mean_history`. These are reduced GPU diagnostics; reading them explicitly
transfers only the small statistics buffer. Normal pixel reconstruction stays on
GPU. The larger raw ring now includes up to three plane records per pixel; query
`history_stats()['memory_bytes']` rather than assuming the old 64-byte pixel size.

The Python `history_samples` path selects plane history for GPU, native-resolution,
1-SPP DLSS jobs. Transmission is accepted only in this configuration. Alpha
clipping, moving optical interfaces, curved/mixed primary optics, volumes and
path-dependent shaders still have no stable unfolded history; their unsupported
radiance stays current-frame-only. Rigid optical motion and exact refractive
reconnection are not claimed. Candidate search is approximate and conservatively
rejects failed real-surface correspondence. Geometry/light edits clear the ring.

`src/test/stable_plane_temporal_test.cu` tests the actual GPU kernel using synthetic
independent layers, slot reorder, current Fresnel weights, identity changes,
disocclusion and empty history. `scripts/test_stable_plane_history.py` exercises
mirror/glass fixtures with camera motion and cuts, and asserts actual plane
population/history reuse rather than just successful DLSS evaluation.

Validation of the plane-history path (2026-11-14): the CUDA kernel's four
synthetic checks and the ten existing bounded-history tests passed. The public
Python Principled-glass test exercises reuse, camera motion, clearing history
and disabling history. A source-line memcheck caught a missing object-motion
buffer dependency; stable guides now request motion-pass resources even when
other denoising passes did not request them. Plane-history DLSS jobs request the
full depth/normal/roughness/albedo/motion guide set. Smooth-flagged API slabs are
accepted only after all three triangle vertex/corner normals match the face.

Kernel memory checking (`--tool memcheck --report-api-errors no`) reports zero
errors for the Python glass test. The unfiltered run separately reports
CUDA texture-descriptor query errors inside `nvngx_dlssd.dll`, one per successful
NGX evaluation; these are not kernel access errors and are retained in the test
logs. Empty texture destruction and explicit/opaque history ownership transitions
were also fixed during validation.

Viewer 90-frame tests verify all three fixtures, zero reconstruction pixel
readback, explicit DLSS resets and zero accepted history at camera cut frame 40.
Maximum active/reused counts in the final installed Viewer tests were:
mirror 42849/42742, two mirrors 53274/53254, glass 126907/42849 (256x256).
Final installed-binary results are recorded in
`test_outputs/stable_plane_history/summary.json`; do not interpret these counts
as a denoising-quality score. The feature remains experimental and opt-in.

A 512x512 moving-camera comparison against the existing 256-SPP fixture captures
also checks the limitation of the explicit window. Glass display MSE was 119.17
with guide-only DLSS, 122.27 with four plane-history frames, and 121.45 with 16.
Two-mirror MSE was 74.89, 77.15 and 77.41 respectively. These windows did not beat
DLSS's own longer opaque history on these scenes. Raw plane reprojection and
strict history ownership work, but improved final denoising quality is not
claimed; the explicit plane window stays opt-in. See
`test_outputs/stable_plane_history/quality.json` for the measured captures.
