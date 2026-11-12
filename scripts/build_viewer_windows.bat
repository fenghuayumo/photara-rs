@echo off
setlocal
set VSLANG=1033
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
