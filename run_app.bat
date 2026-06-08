@echo off
set PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PY%" (
    echo Python 3.11 not found. Install from https://www.python.org/downloads/
    pause
    exit /b 1
)
cd /d "%~dp0"
"%PY%" -m pip install -r requirements.txt -q
"%PY%" app.py
