@echo off
setlocal
rem Keep the compiler language consistent with CMake showIncludes detection.
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "D:\ProgramTool\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo Failed to initialize the Visual Studio environment.
  exit /b 1
)
cd /d "%~dp0.."

cmake --build build --target photara_python --parallel
if errorlevel 1 exit /b 1
rem Stage kernels matching the host data layout for development-tree Python.
if not exist install\lib mkdir install\lib
copy /Y build\src\kernel\device\cuda\kernel_*.zst install\lib\ >nul
if errorlevel 1 exit /b 1
copy /Y build\src\kernel\device\optix\kernel_*.zst install\lib\ >nul
if errorlevel 1 exit /b 1
