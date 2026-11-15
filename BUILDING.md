# Building Photara

Supported render platforms are Windows x64 with NVIDIA OptiX and Apple Silicon
macOS with Metal.

## Requirements

- Visual Studio 2022 with the C++ toolchain
- CUDA Toolkit 12.8
- OptiX SDK 9.1
- Prebuilt dependencies in `lib/windows_x64` (shared with the local Blender
  installation)
- Python 3.11 with `pybind11` for the Python package
- CMake and Ninja

## Standalone renderer

```bat
scripts\configure_windows.bat
scripts\build_windows.bat
cmake --install build
```

This produces `install\photara_rs.exe` together with the runtime DLLs and the
OptiX/CUDA kernels.

## Python package

```bat
scripts\build_python_windows.bat
python scripts\stage_python_package.py
python -m pip wheel --no-build-isolation --no-deps -w dist photara\python
```

The wheel is written to `dist\photara_rs-<version>-cp311-cp311-win_amd64.whl`.
`setup.py` rejects a build when the native `.pyd`, runtime DLLs or kernel blobs
have not been staged, so run the first two commands after changing C++ code.

Run the Python API tests with the development package on `PYTHONPATH`:

```bat
set PYTHONPATH=%CD%\photara\python
python -m unittest discover -s photara\python\tests -v
```

Install the wheel into a clean environment and verify it without a repository
checkout to test the packaged resource/DLL discovery:

```bat
python -m pip install --no-deps dist\photara_rs-<version>-cp311-cp311-win_amd64.whl
python -c "import photara_rs; print(photara_rs.__version__)"
```

## Build configuration

`scripts\configure_windows.bat` enables OptiX and the CUDA toolchain needed to
compile OptiX kernels, and disables OSL, USD, Alembic, Embree, OpenVDB and
Subdivision for a short, OptiX-focused build. CUDA cubins are built for
`sm_120` only; adjust `CYCLES_CUDA_BINARIES_ARCH` to target other GPUs.

OptiX and Metal are selectable render devices. CUDA host code and non-ray-tracing
kernels remain internal OptiX dependencies; an internal CPU device handles
denoising and buffer transfers.

## macOS (Apple Silicon)

Requires macOS 13+, Xcode with its command-line tools, CMake and Ninja.
MetalRT requires a macOS 14+ SDK and runtime; the backend also supports software
BVH traversal (`CYCLES_METALRT=0`). Intel/AMD Macs are not supported by this backend.
If Xcode reports a missing Metal compiler, install it with:

```sh
xcodebuild -downloadComponent MetalToolchain
```

Use Blender-compatible precompiled libraries in `lib/macos_arm64`, or install
system dependencies:

```sh
brew install cmake ninja openimageio opencolorio pugixml tbb zstd sse2neon open-image-denoise
scripts/configure_macos.sh
scripts/build_macos.sh
install/photara_rs --list-devices
install/photara_rs --background --device METAL --samples 16 --width 128 --height 128 \
  --output metal.png examples/scene_cornell_box.xml
```

The helper disables CUDA, OptiX, OSL, USD and other optional heavyweight features.
It accepts additional CMake arguments; `PHOTARA_BUILD_DIR` overrides `build-macos`.
`CMAKE_BUILD_PARALLEL_LEVEL` sets build parallelism (default 4).
OpenImageDenoise provides automatic denoising on Metal; builds without it should
pass `denoise=False` to Python sessions.

To exercise software BVH traversal, run the same render with
`CYCLES_METALRT=0` in the environment. Both this path and MetalRT have been
smoke-tested on Apple Silicon. The Metal integration tests cover automatic
selection, scene edits, image textures, OIDN denoising, NEE-AT updates and GPU
temporal history, including checks that reconstruction performs no pixel readback.

Metal kernels compile at runtime from `install/source/`. Always run the install
step after editing kernel headers. The first render can take several minutes.
The deployment target is 13.0, but external dependencies may require newer macOS;
use dependencies built for your minimum target when distributing binaries.

For Python, use Python 3.11–3.13 and pybind11:

```sh
python3.13 -m venv .venv
. .venv/bin/activate
pip install pybind11 numpy build delocate
scripts/configure_macos.sh -DPHOTARA_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
scripts/build_macos.sh
python scripts/stage_python_macos.py
PYTHONPATH=photara/python python -m unittest discover -s photara/python/tests -p test_metal.py -v
python -m build --wheel photara/python --outdir dist
delocate-wheel --lib-sdir lib -w dist/repaired dist/photara_rs-*.whl
```

Use the repaired wheel for distribution: delocate bundles external dylibs and
rewrites load paths. The wheel contains Metal runtime sources instead of CUDA
blobs. Validate in a separate environment outside the checkout. Current-system
Homebrew libraries can raise the wheel's minimum macOS version.

The Metal device, queue, pipeline cache, BVH builder and compatibility headers
are restored from [Cycles revision 6abef8c62bf9](https://github.com/blender/cycles/tree/6abef8c62bf9d6896a81faf86bfd96121802a7be/src/device/metal),
with original Apache-2.0/SPDX notices preserved. Photara adaptations cover writable
texture descriptors, NEE-AT resource binding, temporal kernels and math constants.
The Windows Vulkan/CUDA viewer and DLSS remain platform-specific.

For development details, known issues, and the current roadmap see
`PHOTARA.md`.

## Version management

The root `VERSION` file is the source of truth for Photara's `MAJOR.MINOR.PATCH`
release version. CMake reads it to generate the native version header and the
standalone `--version` output. Python packaging embeds it in the wheel and uses
it for both distribution metadata and `photara_rs.__version__`; the native Python
module exposes its compiled version as `_photara.__version__`.

```sh
python scripts/version.py --check
python scripts/version.py 0.3.0
```

The bump command updates `VERSION` and Blender's literal `bl_info["version"]`
metadata (required for Blender's add-on discovery). If `VERSION` was edited by
hand, run `python scripts/version.py --sync`. Add-on packaging refuses mismatched
version metadata. Use a `v<version>` Git tag for the matching GitHub Release, and
rebuild the native extension, wheel and add-on after bumping. A version bump does
not publish a release or create a tag automatically.

`BUILD-INFO.json` is tracked beside `VERSION` in the repository. It records the
published build's source commit, platform, Python ABI, dependency deployment
requirements and validation results. Refresh it when rebuilding release artifacts;
keep its version consistent with `VERSION` and preserve the actual build provenance.
It is source-controlled metadata, not a GitHub Release attachment.

Validate the manager with `python -m unittest discover -s scripts -p test_version.py -v`.
