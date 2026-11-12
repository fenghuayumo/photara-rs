@echo off
setlocal
rem Keep the compiler language consistent with CMake showIncludes detection.
call "D:\ProgramTool\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d "%~dp0.."
cmake --build build --target photara_viewer --parallel
if errorlevel 1 exit /b 1
rem Existing install/ must contain the normal Photara runtime DLLs and lib/ kernels.
copy /Y build\bin\photara_viewer.exe install\photara_viewer.exe >nul
if errorlevel 1 exit /b 1
if not exist install\examples mkdir install\examples
copy /Y examples\scene_cornell_box.xml install\examples\scene_cornell_box.xml >nul

if not exist install\shaders mkdir install\shaders
copy /Y build\src\app\shaders\viewer.*.spv install\shaders\ >nul

rem Kernel data/shadow layouts must match the rebuilt host executable.
if not exist install\lib mkdir install\lib
copy /Y build\src\kernel\device\cuda\kernel_*.zst install\lib\ >nul
if errorlevel 1 exit /b 1
copy /Y build\src\kernel\device\optix\kernel_*.zst install\lib\ >nul
if errorlevel 1 exit /b 1
