@echo off
rem Builds basalt for the RTX 4090 (sm_89) and writes everything to ada_build.log next to this file.
setlocal
cd /d "%~dp0"
set "LOG=%~dp0ada_build.log"
echo ==== ada build %date% %time% ==== > "%LOG%"
set "VSW=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSROOT="
for /f "usebackq delims=" %%i in (`"%VSW%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT goto novs
set "VSCMD_SKIP_SENDTELEMETRY=1"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
echo ---- tools >> "%LOG%"
nvcc --version >> "%LOG%" 2>&1
cmake --version >> "%LOG%" 2>&1
ninja --version >> "%LOG%" 2>&1
echo ---- configure >> "%LOG%"
cmake -S . -B build-ada -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 >> "%LOG%" 2>&1
if errorlevel 1 goto cfgfail
echo ---- build >> "%LOG%"
cmake --build build-ada -j -- -k 0 >> "%LOG%" 2>&1
if errorlevel 1 goto bldfail
echo BUILD OK >> "%LOG%"
goto done
:novs
echo Visual Studio with the C++ tools was not found >> "%LOG%"
goto done
:cfgfail
echo CONFIGURE FAILED >> "%LOG%"
goto done
:bldfail
echo BUILD FAILED >> "%LOG%"
:done
echo ==== finished %date% %time% ==== >> "%LOG%"
echo Finished - the result is in ada_build.log
if /i not "%~1"=="nopause" pause
