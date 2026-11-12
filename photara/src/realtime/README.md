# Experimental RealTime mode

This is the first, frame-scheduled baseline for an independent real-time pipeline.
It shares the existing OptiX wavefront tracer, scene, materials and light tree.
The default Reference path and device kernels are unchanged.

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

This stage does **not** implement ReSTIR PT, NEE-AT, GI caching, material
simplification, an automatic animation loop, or Blender UI controls. It makes no
frame-rate guarantee. Films are cleared per job; there is no radiance-history
accumulation in this module. Existing denoiser behavior is retained; this module
does not yet establish temporal validity for moving objects or lights.

The local RTXPT `PathTracerNEE.hlsli` and `NEEATBaker.hlsli` were inspected as
architectural references. No RTXPT source is copied or linked here. Future work:

1. Add GPU timings and explicit scene/history revisions for camera, geometry,
   lighting and material changes, plus persistent device buffers owned by this mode.
2. Add feedback-based light proposals with an exploration distribution and
   evaluable mixture PDF, preserving the current light tree as the baseline.
3. Add path reservoirs, temporal/spatial reuse and validated shift mappings.
4. Add selectively queried world-space radiance caching for rough indirect paths.

Correctness checks should compare equal-time output against Reference, including
moving cameras, disocclusion, moving emissives, thin geometry and glossy paths.
Frame-time measurements must include scene updates, reconstruction and display.
