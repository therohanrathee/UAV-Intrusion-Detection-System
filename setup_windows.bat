@echo off
setlocal enabledelayedexpansion
title UAV Intrusion Detection System - Setup (Windows)
cd /d "%~dp0"
color 0A

echo ===============================================================================
echo      UAV CYBER-PHYSICAL INTRUSION DETECTION SYSTEM - ENVIRONMENT SETUP
echo ===============================================================================
echo.

set "PYTHON_EXE="

:: Step 1: Detect Python
where python >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        "%%I" -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=%%I"
            goto :python_detected
        )
    )
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3"
        goto :python_detected
    )
)

set "CANDIDATES="
set "CANDIDATES=!CANDIDATES! "%LocalAppData%\Programs\Python\Python312\python.exe""
set "CANDIDATES=!CANDIDATES! "%LocalAppData%\Programs\Python\Python311\python.exe""
set "CANDIDATES=!CANDIDATES! "%LocalAppData%\Programs\Python\Python310\python.exe""
set "CANDIDATES=!CANDIDATES! "%LocalAppData%\Programs\Python\Python39\python.exe""
set "CANDIDATES=!CANDIDATES! "%ProgramFiles%\Python312\python.exe""
set "CANDIDATES=!CANDIDATES! "%ProgramFiles%\Python311\python.exe""
set "CANDIDATES=!CANDIDATES! "%ProgramFiles%\Python310\python.exe""
set "CANDIDATES=!CANDIDATES! "%ProgramFiles%\Python39\python.exe""
set "CANDIDATES=!CANDIDATES! "C:\Python312\python.exe""
set "CANDIDATES=!CANDIDATES! "C:\Python311\python.exe""
set "CANDIDATES=!CANDIDATES! "C:\Python310\python.exe""
set "CANDIDATES=!CANDIDATES! "C:\Python39\python.exe""

for %%P in (!CANDIDATES!) do (
    if exist %%P (
        %%P -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=%%~P"
            goto :python_detected
        )
    )
)

color 0C
echo [ERROR] Python 3.8+ was not found on this Windows system!
echo Please download and install Python (3.9 - 3.11) from https://www.python.org/downloads/
echo IMPORTANT: Make sure to check the box "Add python.exe to PATH" during installation.
echo.
pause
exit /b 1

:python_detected
echo [OK] Using Python: !PYTHON_EXE!
echo.

:: Step 2: Auto-bootstrap pip if missing
!PYTHON_EXE! -m pip --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Attempting offline ensurepip bootstrap...
    !PYTHON_EXE! -m ensurepip --default-pip >nul 2>nul
    !PYTHON_EXE! -m pip --version >nul 2>nul
    if !errorlevel! neq 0 (
        echo [INFO] Attempting online get-pip.py bootstrap...
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py 2>nul
        if exist get-pip.py (
            !PYTHON_EXE! get-pip.py --user >nul 2>nul
            del /f /q get-pip.py 2>nul
        )
    )
)

:: Step 3: Create venv or fallback to User-Level environment
set "TARGET_PYTHON="
set "USE_VENV=1"

echo [1/3] Configuring Python execution environment...
if not exist "venv\Scripts\python.exe" (
    !PYTHON_EXE! -m venv venv >nul 2>nul
    if !errorlevel! neq 0 (
        !PYTHON_EXE! -m venv --without-pip venv >nul 2>nul
        if not exist "venv\Scripts\python.exe" (
            echo [!] Virtual environment unavailable. Using User-Level environment fallback...
            set "USE_VENV=0"
        ) else (
            venv\Scripts\python.exe -m ensurepip --default-pip >nul 2>nul
        )
    )
)

if "!USE_VENV!"=="1" (
    if not exist "venv\Scripts\pip.exe" (
        venv\Scripts\python.exe -m ensurepip --default-pip >nul 2>nul
    )
    set "TARGET_PYTHON=venv\Scripts\python.exe"
) else (
    set "TARGET_PYTHON=!PYTHON_EXE!"
)

:: Step 4: Install dependencies
echo [2/3] Upgrading pip...
if "!USE_VENV!"=="1" (
    !TARGET_PYTHON! -m pip install --upgrade pip
) else (
    !TARGET_PYTHON! -m pip install --user --upgrade pip
)

echo [3/3] Installing dependencies from requirements.txt...
if "!USE_VENV!"=="1" (
    !TARGET_PYTHON! -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    if !errorlevel! neq 0 (
        !TARGET_PYTHON! -m pip install -r requirements.txt
    )
) else (
    !TARGET_PYTHON! -m pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    if !errorlevel! neq 0 (
        !TARGET_PYTHON! -m pip install --user -r requirements.txt
    )
)

echo.
echo ===============================================================================
echo   [SUCCESS] Environment Setup Completed!
echo   You can now launch the full suite by double-clicking 'START_TACTICAL_IDS.bat'
echo ===============================================================================
echo.
pause
