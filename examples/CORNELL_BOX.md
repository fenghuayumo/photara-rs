# Cornell Box RealTime baseline

`scene_cornell_box.xml` contains inward-facing red/green walls, two opaque
white blocks and a downward-facing emissive ceiling quad. It is a synthetic
Cornell-style fixture, not a measured Cornell dataset.

Build and install the executable with its runtime DLLs and `lib/` kernels, then:

```powershell
install/photara_rs.exe --background --device OPTIX --width 256 --height 256 --samples 1024 --output reference.png examples/scene_cornell_box.xml
install/photara_rs.exe --background --device OPTIX --width 256 --height 256 --render-mode REALTIME --samples-per-frame 1 --frame-index 0 --output realtime.png examples/scene_cornell_box.xml
```

The standalone RealTime command renders one frame. Change `--frame-index` to
select a different sample subset; it does not run a continuous viewport loop.
Reference remains the default. RealTime disables adaptive sampling for a fixed
per-frame count. At this stage it still uses the original OptiX tracer and Light
Tree, without NEE-AT, ReSTIR or GI caches.

The Python benchmark creates the same geometry and camera through the Scene API:

```powershell
$env:PYTHONPATH='photara/python'
python scripts/test_cornell_realtime.py --size 256 --frames 16 --reference-samples 1024
```

Use Python matching the compiled extension. Outputs go to
`test_outputs/cornell_realtime`: an XML export, linear NPZ, three PNG previews
and `metrics.json`. PNGs use the same fixed tone mapping for comparison.
The XML fixture uses a pure emission node for the light; the Python API uses
its emissive Principled material, so bitwise cross-loader equality is not expected.

Default timings exclude the warm-up frame and image file writes, but include
`Scene.render()` synchronization and CPU capture of all passes. They are not
GPU-only timings or a frame-rate guarantee. Reference timing includes cold
initialization and is not an equal-time comparison. The averaged preview is
an offline arithmetic mean, not a temporal reconstruction feature.

Optional `--denoiser DLSS` evaluates the existing reconstruction path separately.
No moving-scene history correctness is established by this static fixture.

Cornell Box tests indirect lighting, color bleeding, shadow visibility and
energy consistency. Its single emitter cannot demonstrate a many-light sampler's
advantage. NEE-AT evaluation also needs a separate many-light scene, warm-up and
dynamic-light tests, and matching forward/NEE selection PDFs for MIS.
