@echo off
title Tactical UAV Cyber-Physical IDS & 3D Flight Arena
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment 'venv' not found.
    echo Running automated setup first...
    call setup_windows.bat
)

call venv\Scripts\activate.bat
set PYTHONPATH=.

echo =================================================================
echo   Starting Tactical UAV Cyber-Physical IDS & 3D Flight Arena
echo   Inference Engine: Standalone ONNX Runtime (Port 8502)
echo   Tactical UI:      Streamlit Dashboard (Port 8501)
echo =================================================================
echo.

start http://localhost:8501
streamlit run dashboard/app.py

pause
