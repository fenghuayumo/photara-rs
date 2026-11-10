# Building Photara

The supported configuration is Windows x64 with an NVIDIA RTX GPU and the
OptiX runtime.

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

## Build configuration

`scripts\configure_windows.bat` enables OptiX and the CUDA toolchain needed to
compile OptiX kernels, and disables OSL, USD, Alembic, Embree, OpenVDB and
Subdivision for a short, OptiX-focused build. CUDA cubins are built for
`sm_120` only; adjust `CYCLES_CUDA_BINARIES_ARCH` to target other GPUs.

OptiX is the only selectable render device. CUDA host code and the CUDA
non-ray-tracing kernels are internal dependencies of the OptiX backend, and a
small CPU host device is retained internally for denoising and buffer
transfers.

For development details, known issues, and the current roadmap see
`PHOTARA.md`.
