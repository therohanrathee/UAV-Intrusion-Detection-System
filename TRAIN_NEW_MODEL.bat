@echo off
setlocal enabledelayedexpansion
title UAV IDS - Model Retraining & ONNX Export Suite
cd /d "%~dp0"

echo ===============================================================================
echo        UAV CYBER-PHYSICAL IDS: MODEL RETRAINING & DEPLOYMENT SUITE
echo ===============================================================================
echo.

set "PY_CMD="
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

if "%PY_CMD%"=="" (
    echo [ERROR] No Python interpreter found! Run setup_windows.bat first.
    pause
    exit /b 1
)

set "PYTHONPATH=."

echo [1/2] Ingesting flight recordings and rebuilding dataset splits...
%PY_CMD% sim/build_datasets.py --include-recorded
if errorlevel 1 (
    echo [ERROR] Dataset build failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Training PyTorch Multi-Task Deep Model & Exporting Edge ONNX Engine...
%PY_CMD% models/train.py --epochs 12 --export-onnx --eval-report
if errorlevel 1 (
    echo [ERROR] Model training failed.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo [SUCCESS] Model training, ONNX export, and evaluation completed!
echo  - PyTorch Checkpoint:  checkpoints/best_deep_model.pt
echo  - Production ONNX:     weights/uav_ids_model.onnx
echo  - Scaler Parameters:   checkpoints/scaler.joblib
echo  - Benchmark Figures:   evaluation/figures/
echo ===============================================================================
echo.
pause
