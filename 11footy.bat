@echo off
rem Opens the 11footy window (README.md, "Start"). The first time, uv installs
rem Python 3.12 and the libraries the window needs: a few minutes, once.
cd /d "%~dp0"
set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" if "%UV%" neq "uv" (
    echo uv is not installed. Install it first: see README.md, "Start".
    pause
    exit /b 1
)
rem --inexact keeps anything else a maintainer installed in the same environment
"%UV%" sync --frozen --inexact
if errorlevel 1 (
    echo The libraries could not be installed: see the lines above.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m desk
