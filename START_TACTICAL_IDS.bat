@echo off
setlocal enabledelayedexpansion
title Tactical UAV Cyber-Physical IDS - Mission Control Launcher
cd /d "%~dp0"
color 0A

echo ===============================================================================
echo      TACTICAL UAV CYBER-PHYSICAL INTRUSION DETECTION SYSTEM (IDS)
echo                DRDO Autonomous Flight & Sensor Defense Suite
echo ===============================================================================
echo.

set "PYTHON_EXE="

:: -----------------------------------------------------------------------------
:: STEP 1: DEEP PYTHON DISCOVERY
:: -----------------------------------------------------------------------------
echo [*] Step 1/4: Discovering Python 3 runtime environment...

:: 1.1 Check 'python' in system PATH
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

:: 1.2 Check 'py' Windows Python Launcher in PATH
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3"
        goto :python_detected
    )
)

:: 1.3 Check standard Windows user & system installation locations
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

:: 1.4 If Python is missing, offer automatic installation or guidance
color 0C
echo.
echo ===============================================================================
echo   [!] CRITICAL: Python 3.8+ was not detected on this Windows workstation.
echo ===============================================================================
echo.

where winget >nul 2>nul
if %errorlevel% equ 0 (
    echo Windows Package Manager (winget) is detected!
    echo Would you like to automatically install Python 3.11 with PATH and pip configured?
    echo.
    set /p "AUTO_INSTALL=Install Python 3.11 now? (Y/N, default Y): "
    if "!AUTO_INSTALL!"=="" set "AUTO_INSTALL=Y"
    if /i "!AUTO_INSTALL!"=="Y" (
        echo.
        echo [*] Installing Python 3.11 via winget...
        winget install -e --id Python.Python.3.11 --override "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1"
        echo.
        echo [*] Installation finished. Please close this window and double-click
        echo     START_TACTICAL_IDS.bat again to launch the system.
        echo.
        pause
        exit /b 0
    )
)

echo.
echo Please follow these quick steps:
echo 1. Download Python 3.11 (or 3.10) from https://www.python.org/downloads/
echo 2. Run the installer.
echo 3. IMPORTANT: Check the box "Add python.exe to PATH" at the bottom!
echo 4. Restart this launcher once installation finishes.
echo.
set /p "OPEN_PY=Open Python official download page in your browser? (Y/N, default Y): "
if "!OPEN_PY!"=="" set "OPEN_PY=Y"
if /i "!OPEN_PY!"=="Y" (
    start https://www.python.org/downloads/
)
echo.
pause
exit /b 1

:python_detected
color 0A
echo [OK] Python runtime verified: !PYTHON_EXE!
echo.

:: -----------------------------------------------------------------------------
:: STEP 2: PIP AUTO-REPAIR & BOOTSTRAP
:: -----------------------------------------------------------------------------
echo [*] Step 2/4: Verifying pip packaging subsystem...
!PYTHON_EXE! -m pip --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] pip is missing or corrupted in the detected Python installation.
    echo [*] Attempting offline bootstrap with Python ensurepip...
    !PYTHON_EXE! -m ensurepip --default-pip >nul 2>nul
    
    !PYTHON_EXE! -m pip --version >nul 2>nul
    if !errorlevel! neq 0 (
        echo [*] Offline ensurepip restricted (likely non-admin C:\Program Files).
        echo [*] Attempting online get-pip.py bootstrap...
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py 2>nul
        if exist get-pip.py (
            !PYTHON_EXE! get-pip.py --user >nul 2>nul
            del /f /q get-pip.py 2>nul
        )
    )
    
    !PYTHON_EXE! -m pip --version >nul 2>nul
    if !errorlevel! equ 0 (
        echo [OK] pip successfully bootstrapped!
    ) else (
        echo [!] Base pip restricted. Proceeding with local project virtual environment...
    )
) else (
    echo [OK] pip packaging tool verified.
)
echo.

:: -----------------------------------------------------------------------------
:: STEP 3: VIRTUAL ENVIRONMENT & DEPENDENCY MANAGEMENT
:: -----------------------------------------------------------------------------
echo [*] Step 3/4: Configuring execution environment and dependencies...

set "RUN_PYTHON="
set "USE_VENV=1"

if not exist "venv\Scripts\python.exe" (
    echo [*] Creating isolated environment in .\venv ...
    !PYTHON_EXE! -m venv venv >nul 2>nul
    if !errorlevel! neq 0 (
        echo [!] Standard venv failed. Retrying with --without-pip fallback...
        !PYTHON_EXE! -m venv --without-pip venv >nul 2>nul
        if not exist "venv\Scripts\python.exe" (
            echo [!] Virtual environment creation unavailable on this system.
            echo [*] Switching cleanly to User-Level Python fallback (no venv)...
            set "USE_VENV=0"
        ) else (
            echo [*] Bootstrapping pip inside virtual environment...
            venv\Scripts\python.exe -m ensurepip --default-pip >nul 2>nul
        )
    )
)

if "!USE_VENV!"=="1" (
    if not exist "venv\Scripts\pip.exe" (
        echo [*] Bootstrapping pip into venv...
        venv\Scripts\python.exe -m ensurepip --default-pip >nul 2>nul
        if !errorlevel! neq 0 (
            curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py 2>nul
            if exist get-pip.py (
                venv\Scripts\python.exe get-pip.py >nul 2>nul
                del /f /q get-pip.py 2>nul
            )
        )
    )
    set "RUN_PYTHON=venv\Scripts\python.exe"
) else (
    set "RUN_PYTHON=!PYTHON_EXE!"
)

:: Verify if key dependencies are already installed
set "DEPS_OK=1"
!RUN_PYTHON! -c "import streamlit, onnxruntime, pandas, sklearn, scipy, lightgbm, xgboost, matplotlib, joblib" >nul 2>nul
if %errorlevel% neq 0 (
    set "DEPS_OK=0"
)

if "!DEPS_OK!"=="0" (
    echo.
    echo [*] Installing required tactical dependencies from requirements.txt...
    echo [*] Using lightweight CPU-optimized acceleration index for rapid setup...
    echo     (First-time installation takes ~1-3 minutes; subsequent starts are instant)
    echo.
    
    if "!USE_VENV!"=="1" (
        !RUN_PYTHON! -m pip install --upgrade pip
        !RUN_PYTHON! -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
        if !errorlevel! neq 0 (
            echo.
            echo [!] Fast-index encountered an issue; trying standard PyPI...
            !RUN_PYTHON! -m pip install -r requirements.txt
        )
    ) else (
        !RUN_PYTHON! -m pip install --user --upgrade pip
        !RUN_PYTHON! -m pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
        if !errorlevel! neq 0 (
            !RUN_PYTHON! -m pip install --user -r requirements.txt
        )
    )
    
    !RUN_PYTHON! -c "import streamlit, onnxruntime, pandas" >nul 2>nul
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Failed to install required packages.
        echo Check internet connection or consult the manual installation section in README.md.
        pause
        exit /b 1
    )
    echo.
    echo [OK] All dependencies successfully installed and verified!
) else (
    echo [OK] All tactical dependencies verified! Instant Launch Mode active.
)
echo.

:: -----------------------------------------------------------------------------
:: STEP 4: LAUNCH TACTICAL SUITE & BROWSER
:: -----------------------------------------------------------------------------
echo [*] Step 4/4: Launching Tactical UAV IDS Suite...
echo.
echo ===============================================================================
echo   TACTICAL UAV SUITE RUNNING
echo   - Streamlit Tactical Cockpit: http://localhost:8501
echo   - ONNX Edge Inference Daemon: http://localhost:8502 (auto-spawned)
echo ===============================================================================
echo   NOTE FOR PRESENTATION:
echo   1. Keep this Command Prompt window open while presenting.
echo   2. Your default web browser will open http://localhost:8501 automatically.
echo   3. Press Ctrl+C in this window when you wish to terminate the session.
echo ===============================================================================
echo.

set PYTHONPATH=.

:: Launch default browser after small delay in background
start "" http://localhost:8501

:: Start Streamlit server
!RUN_PYTHON! -m streamlit run dashboard\app.py --server.headless false

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Streamlit process stopped (exit code: %errorlevel%).
)

echo.
echo Session ended. Press any key to close this console.
pause >nul
