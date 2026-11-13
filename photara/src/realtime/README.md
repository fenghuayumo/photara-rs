# Experimental RealTime mode

This is the first, frame-scheduled baseline for an independent real-time pipeline.
It shares the existing OptiX wavefront tracer, scene, materials and light tree.
Reference retains its original sampling behavior; shared kernels include the
opt-in RealTime feedback path.

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
