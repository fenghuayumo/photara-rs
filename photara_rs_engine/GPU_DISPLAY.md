# Native Blender GPU display

The accelerated viewport directly reuses Cycles' `BlenderDisplayDriver` from
the matching Blender build. Python passes ownership of the driver to the
standalone Photara session; it does not read or upload viewport pixels.

On Vulkan, the existing Cycles CUDA interop implementation imports Blender's
exported pixel-buffer memory. The renderer writes display pixels there on the
GPU. Blender performs the buffer-to-texture copy and drawing on the GPU, using
its existing contexts, fences and display shaders. OpenGL uses the driver's
existing registered-PBO implementation. This avoids CPU pixel transfers, not
every GPU-to-GPU copy.

## Build

1. Use the same `session/display_driver.h` in both repositories and a compatible
   Release MSVC runtime/toolset. The patch helper checks the header fingerprint.
2. Apply the narrow bridge to Blender:

   ```powershell
   python scripts/patch_blender_display.py --blender-source D:/ProgramCode/C++/blender
   ```

3. Rebuild Blender from its existing configured build directory in a VS
   developer shell:

   ```powershell
   cmake --build D:/ProgramCode/C++/build_windows --target blender --parallel 8
   ```

   The local build also enables DLSS and requires a compatible NVIDIA DLSS SDK;
   its include directory must contain the CUDA API used by that Blender checkout.
4. Rebuild and package Photara's Python extension for Blender's Python, then run
   `scripts/install_blender.py`. Start a fresh Blender process.

The bridge is automatically selected when `_cycles.photara_display_create` is
present. Unpatched Blender builds retain the CPU-transfer progressive path.
The independent renderer and final-render AOV capture do not require Blender.

## Ownership and allocation

The capsule identifies the pinned DisplayDriver ABI. Attaching it transfers
ownership exactly once to Photara's session. The virtual destructor runs in
Blender, so native graphics resources and the driver are freed by their owner.

Device UUID vectors cross the module boundary. Blender and standalone Cycles
use different guarded allocators, so a native allocator callback constructs
that vector in the receiving module's allocation domain. Do not replace this
with an ordinary return from the Blender driver's device-info method.

## Verification

```powershell
blender.exe --factory-startup --python-exit-code 1 --python scripts/test_blender_gpu_display.py
```

The test requires GPU updates, zero CPU pixel-buffer mappings, disabled Photara
CPU capture, and correctly aligned pixels at three different camera poses.
Framebuffer readback is used only by the verification script. Reports are saved
under `test_outputs/gpu_display_verification.json`.

CPU denoising can itself require host pixel transfers; use the OptiX denoiser
when validating an entirely GPU-resident rendering/denoising/display route.

## DLSS denoising and upscaling

Viewport and final-render Denoise panels both offer DLSS Ray Reconstruction.
Each has an independent DLSS Upscaling selection: Native Resolution (default),
Quality (1.5x), Balanced (1/0.58x), Performance (2x), Ultra Performance (3x).
Output dimensions stay unchanged. Native Resolution performs no downscaling.
Build Photara with `WITH_DLSS=ON` and `DLSS_INCLUDE_DIR` pointing to the NVIDIA
SDK headers. An unsupported device/build reports an explicit error when DLSS is
selected. The Python API accepts `denoiser="DLSS"` and
`denoise_upscale_quality="QUALITY"` (or the other enum identifiers).

Run `PHOTARA_TEST_DLSS=QUALITY` with the GPU display verification script to test
DLSS through Blender's native display driver.
