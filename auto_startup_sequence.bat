@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Could not find .venv\Scripts\python.exe in:
    echo   %cd%
    echo.
    echo Contents of this folder:
    dir /b /a
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
start "" ".venv\Scripts\pythonw.exe" src\main.py