@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 py -m venv .venv
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Try Python 3.11 and see README.md.
    pause
    exit /b 1
)
.venv\Scripts\python.exe app.py
pause
