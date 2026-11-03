@echo off
title UAV Intrusion Detection System - Setup (Windows)
cd /d "%~dp0"

echo =================================================================
echo   UAV Cyber-Physical Intrusion Detection System - Windows Setup
echo =================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in system PATH!
    echo Please download and install Python (3.9, 3.10, or 3.11) from python.org
    echo IMPORTANT: Make sure to check the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Creating isolated Python virtual environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo =================================================================
echo   [SUCCESS] Environment Setup Completed!
echo   You can now launch the dashboard by double-clicking 'run_dashboard.bat'
echo =================================================================
echo.
pause
