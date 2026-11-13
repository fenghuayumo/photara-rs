@echo off
setlocal
call "D:\ProgramTool\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d "%~dp0.."
if not exist test_outputs mkdir test_outputs
cl /nologo /std:c++17 /EHsc photara\tests\navigation_test.cpp /Fo:test_outputs\navigation_test.obj /Fe:test_outputs\navigation_test.exe
if errorlevel 1 exit /b 1
test_outputs\navigation_test.exe
