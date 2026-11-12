# Photara Viewer example (Windows)

A separate `photara_viewer` application reads XML scenes and exercises the
existing OptiX frame baseline. It does not change Reference's device kernels.
Requires the same NVIDIA GPU, driver, DLLs and kernel binaries as Photara.
It does not require Python or the optional SDL/OpenGL build. Building requires
Vulkan headers/import library from `lib/windows_x64/vulkan` and `glslc` from a
Vulkan SDK on PATH (or `$VULKAN_SDK/Bin`). Generated SPIR-V shaders are installed
beside the executable under `shaders/`. `PHOTARA_BUILD_VIEWER=OFF` disables this
example without adding a Vulkan SDK requirement to the standalone tracer.

From the repository root after configuring/building the renderer and installing
its runtime dependencies:

```powershell
scripts/build_viewer_windows.bat
install/photara_viewer.exe --size 512 examples/scene_cornell_box.xml
```

Double-clicking the installed executable uses `install/examples/scene_cornell_box.xml`.
Other XML scenes can be passed as the positional argument. Window resizing scales
the image; tracing resolution remains fixed at `--size`.

| Control | Action |
| --- | --- |
| 1 / 2 | RealTime / Reference |
| + / - | RealTime samples per frame, 1–64 |
| Space | Pause / resume |
| WASD / QE | Translate camera horizontally / vertically |
| Left drag | Pan camera |
| R | Reset camera and RealTime sampling sequence |
| F12 | Save the latest completed frame as `viewer_capture.bmp` |
| Escape | Close, cancelling any active render |

Reference renders once at `--reference-samples` (default 256); camera changes or
mode changes start a new render. RealTime continuously submits fresh films and
advances the sample sequence. Denoising is disabled to expose sampling artifacts.
RealTime enables an experimental global and screen-tile NEE-AT analytic-light feedback proposal.
Use `--light-sampler LIGHT_TREE` for the original proposal. Pure emissive-mesh
Cornell Box falls back to Light Tree; use `scripts/test_neeat.py` for a many-light
learning comparison. See [algorithm scope and validation](../photara/src/realtime/NEE_AT.md).
ReSTIR and GI caching are not enabled yet.

## Presentation and measurements

The viewer uses Vulkan 1.1 and requests three swapchain images (subject to
surface capabilities), preferring MAILBOX presentation (FIFO fallback) and one graphics submission in
flight. Acquisition is polled without waiting on vblank in the UI thread. Only
one OptiX job is active. Every accepted completed render frame receives one
successful Present submission before the next job starts. If acquisition or
presentation is busy/out of date, the same image is retained and retried;
it is not overwritten by a newer frame. Resize-only redraws are tracked
separately from rendered-frame submissions. Minimized windows stop new jobs.
MAILBOX may still replace already-submitted images before physical scanout;
submission count is not the number of images scanned out by the monitor.

OptiX's display conversion kernel writes half RGBA directly into exported,
device-local Vulkan buffer memory imported by CUDA. A fragment shader reads
that buffer and applies tone mapping and the display transfer. Normal frames
perform no CPU pixel readback or upload. Vulkan and CUDA must identify the same
GPU UUID. The existing OptiX tracer and CUDA/Vulkan interop backend are reused;
there is no new Vulkan ray-tracing backend.

The shared buffer uses exclusive ownership: the Vulkan reader's fence is waited
before CUDA writes, film conversion synchronizes its CUDA queue before the
display callback ends, and Vulkan acquires/releases external queue ownership.
The window and display callback share a mutex. This baseline uses host completion
synchronization, not an external-semaphore asynchronous pipeline. Three swapchain
images do not imply three concurrent OptiX jobs.

Only explicit screenshots (including smoke-test captures) copy the shared half
buffer into host-visible staging memory and perform CPU display conversion.
Existing scalar scheduling/status transfers inside the renderer are unchanged.
A failed interop path is reported; the viewer does not continue with CPU display.

A persistent completion waiter signals a Win32 event when the finite render
job ends. The UI waits on that event and input rather than polling every 2 ms;
only pending swapchain acquisition uses a short retry timeout. Paused and idle
windows do not periodically wake. Titles update at most four times per second
in continuous rendering, independently of rendering/presentation. Swapchain
capabilities are queried on resize/out-of-date events rather than every frame.

The title's worker time runs from render submission through the GPU display
update callback, including scene updates and cold initialization. It is not a
GPU timestamp. CSV additionally records when the UI observes completion. The
zero CPU pixel-readback byte field describes normal frames and excludes explicit
screenshot capture. CSV rows are emitted after each successful rendered-frame
Present submission, with cumulative submission/attempt/query/title counters.
`rendered_frames_presented` excludes resize-only redraws; `present_backpressure_ms`
measures completion observation through successful submission, including any
acquisition retries and command recording/submission. Metrics are buffered and
flushed every 64 frames and at exit.
No displayed-FPS or input-to-photon latency claim is made by these counters.

```powershell
install/photara_viewer.exe --size 512 --spp 1 --smoke-frames 32 --metrics timing.csv --screenshot capture.bmp examples/scene_cornell_box.xml
install/photara_viewer.exe --mode REFERENCE --reference-samples 64 --smoke-frames 1 --screenshot reference.bmp examples/scene_cornell_box.xml
```

The smoke option renders the requested number of completed frames, captures the
last image after its Present submission and exits. Reference smoke tests accept one frame. For steady-state
comparisons exclude the cold first frame and keep scene, resolution and GPU load
fixed. Screenshot and CSV paths must have existing parent directories.

The optional `--smoke-cycle` flag with at least 26 smoke frames exercises two
Reference/RealTime switches, camera movement and window resizing. For example,
add it to the 32-frame command above. Validation-layer smoke runs can be enabled
using the Vulkan SDK's standard `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.

A ready-to-run analytic-light fixture is `examples/scene_many_lights_neeat.xml`:

```powershell
install/photara_viewer.exe --light-sampler NEE_AT examples/scene_many_lights_neeat.xml
```

It complements the emissive-mesh-only Cornell Box. Use the Python many-light
benchmark for quantitative linear-image comparisons rather than judging the
noise from a single tone-mapped viewer frame.

## Additional XML regression scenes

`scripts/test_xml_realtime.py` accepts original XML paths and keeps includes and
resources relative to their source scene. For example:

```powershell
python scripts/test_xml_realtime.py --size 256 --frames 32 --reference-samples 256 D:/ProgramCode/C++/cycles/examples/scene_monkey.xml D:/ProgramCode/C++/cycles/examples/scene_osl_stripes.xml D:/ProgramCode/C++/cycles/examples/scene_sphere_bump.xml
```

It renders Reference, then exercises each supported scene with Light Tree and
NEE-AT viewer jobs. Every completed frame must be presented. Results, logs and
images are saved under `test_outputs/xml_realtime`, including `summary.json`.
Original scene files are not modified. Reference PNGs and viewer BMPs have
different display transforms; their pixel values are not used as a cross-mode
quality metric. These environment-lit fixtures primarily test compatibility;
they do not establish a many-light learning advantage.

Local 256x256 validation: Monkey and Sphere Bump each completed 32/32 frames
for both samplers with zero normal-frame CPU pixel-readback bytes. Median worker
through GPU display times (after eight warm-up frames) were 1.94/1.92 ms for
Monkey and 0.72/0.74 ms for Sphere Bump (Light Tree/NEE-AT respectively).
The OSL stripes fixture is unsupported in the current build, which has OSL
disabled. The legacy XML loader can report an unknown shader and still exit
zero, so the test checks error logs as well as process status and does not
count missing-material output as a successful render.
