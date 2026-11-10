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

OptiX is the only selectable render device. CUDA host code and the CUDA
non-ray-tracing kernels are internal dependencies of the OptiX backend, and a
small CPU host device is retained internally for denoising and buffer
transfers.

For development details, known issues, and the current roadmap see
`PHOTARA.md`.
