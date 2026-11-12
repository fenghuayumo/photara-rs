@echo off
setlocal
set VSLANG=1033
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "D:\ProgramTool\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo Failed to initialize the Visual Studio environment.
  exit /b 1
)
cd /d "%~dp0.."

cmake --build build --target photara_python --parallel
