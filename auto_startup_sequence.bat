@echo off
setlocal


set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"
set "STAMP=%VENV%\requirements.sha256"

rem Create the venv on first run
if not exist "%PY%" (
    python -m venv "%VENV%" || goto :fail
)

rem Hash requirements.txt
for /f "skip=1 delims=" %%H in ('certutil -hashfile requirements.txt SHA256') do (
    set "NEWHASH=%%H"
    goto :gothash
)
:gothash

set "OLDHASH="
if exist "%STAMP%" set /p OLDHASH=<"%STAMP%"

rem Only install if requirements changed
if not "%NEWHASH%"=="%OLDHASH%" (
    echo [%date% %time%] Installing requirements >> install.log
    "%PY%" -m pip install --disable-pip-version-check -r requirements.txt >> install.log 2>&1 || goto :fail
    > "%STAMP%" echo %NEWHASH%
)

start "" "%PYW%" src\main.py
exit /b 0

:fail
echo Dependency install failed - see install.log
pause
exit /b 1