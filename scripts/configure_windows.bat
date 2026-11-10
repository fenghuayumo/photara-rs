@echo off
setlocal
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "D:\ProgramTool\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo Failed to initialize the Visual Studio environment.
  exit /b 1
)
cd /d "%~dp0.."

set "PHOTARA_PYTHON_EXE=D:/ProgramTool/Pyton3/python.exe"
set "PHOTARA_PYBIND11_DIR=D:/ProgramTool/Pyton3/Lib/site-packages/pybind11/share/cmake/pybind11"

cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPHOTARA_BUILD_C_API=ON ^
  -DPHOTARA_BUILD_PYTHON=ON ^
  -DPython_EXECUTABLE="%PHOTARA_PYTHON_EXE%" ^
  -Dpybind11_DIR="%PHOTARA_PYBIND11_DIR%" ^
  -DOPTIX_ROOT_DIR="D:/ProgramTool/OptixSDK9.1.0" ^
  -DWITH_CYCLES_DEVICE_OPTIX=ON ^
  -DWITH_CYCLES_DEVICE_CUDA=ON ^
  -DWITH_CYCLES_CUDA_BINARIES=ON ^
  -DCYCLES_CUDA_BINARIES_ARCH=sm_120 ^
  -DWITH_CYCLES_OSL=OFF ^
  -DWITH_CYCLES_OPENSUBDIV=OFF ^
  -DWITH_CYCLES_ALEMBIC=OFF ^
  -DWITH_CYCLES_EMBREE=OFF ^
  -DWITH_CYCLES_OPENVDB=OFF ^
  -DWITH_CYCLES_NANOVDB=OFF ^
  -DWITH_CYCLES_USD=OFF ^
  -DWITH_CYCLES_HYDRA_RENDER_DELEGATE=OFF ^
  -DWITH_CYCLES_OPENIMAGEDENOISE=ON ^
  -DWITH_CYCLES_PUGIXML=ON
