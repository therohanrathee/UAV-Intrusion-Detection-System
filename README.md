# Tactical UAV & Loitering Munition Cyber-Physical Intrusion Detection Suite (IDS)

[![Tests](https://img.shields.io/badge/tests-8%20passed-success?style=flat-square&logo=pytest)](file:///Users/rohan/Projects/UAV%20Intrusion%20Detection%20System/tests)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Inference Engine](https://img.shields.io/badge/inference-ONNX%20Runtime%20(0.44ms)-orange?style=flat-square&logo=onnx)](file:///Users/rohan/Projects/UAV%20Intrusion%20Detection%20System/weights/uav_ids_model.onnx)
[![Model Size](https://img.shields.io/badge/model%20size-208%20KB-brightgreen?style=flat-square)](file:///Users/rohan/Projects/UAV%20Intrusion%20Detection%20System/weights)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](file:///Users/rohan/Projects/UAV%20Intrusion%20Detection%20System/LICENSE)

A military-grade, physics-informed Cyber-Physical Intrusion Detection System (IDS) and multi-task diagnostic suite designed for autonomous unmanned aerial vehicles (UAVs) and loitering munitions operating in contested electronic warfare (EW) environments.

The system ingests high-rate (50–60 Hz), 56-channel avionics telemetry, computes analytical kinematic and aerodynamic physical invariants, and concurrently outputs:
1. **Multi-Class Attack Classification**: Identifies the hostile vector across 9 profile scenarios with **95.93% macro accuracy**.
2. **Granular Sensor Compromise Attribution**: Predicts a 10-element binary sensor health vector (`[gps, gyro1, gyro2, accel, pitot, optical, radar_alt, mag, baro, c2_link]`) with **95.74% exact match**, empowering the autopilot to disengage corrupted sensors and switch to resilient backup modes (e.g., Dead Reckoning upon GPS spoofing).

---

## 📑 Table of Contents
- [System Architecture](#-system-architecture)
- [Benchmark Results](#-benchmark-results)
- [Cross-Platform Installation (Windows, Mac, Linux)](#-cross-platform-installation)
- [Quick Start & Usage](#-quick-start--usage)
- [Interactive Tactical Dashboard & 3D Flight Arena](#-interactive-tactical-dashboard--3d-flight-arena)
- [Mathematical Formulation of Physics Residuals](#-mathematical-formulation-of-physics-residuals)
- [Comprehensive Attack Vector Profiles & Detection Mechanics](#-comprehensive-attack-vector-profiles--detection-mechanics)
- [Hardware & Drone Autopilot Deployment](#-hardware--drone-autopilot-deployment)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

## 🏛️ System Architecture

```
                                  TACTICAL FLIGHT TELEMETRY (56 CHANNELS @ 50 Hz)
   [ Primary & Secondary IMU ]    [ GNSS Position & Vel ]    [ Pitot IAS & Baro Alt ]    [ Radar Alt & Optical Flow ]
                │                            │                           │                            │
                └────────────────────────────┼───────────────────────────┴────────────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────────────────────┐
                     │     ANALYTICAL KINEMATIC & AERODYNAMIC RESIDUAL ENGINE        │
                     │  • |a_GPS| vs |a_IMU|        • |v_pitot - v_ground|           │
                     │  • Dual-Gyro Discrepancy     • Triple Altitude Invariance     │
                     │  • Geomagnetic Norm ||B||    • C2 Link Packet Loss / Jitter   │
                     └───────────────────────────────┬───────────────────────────────┘
                                                     ▼
                                        32 INVARIANT FEATURES (x 50 TIMESTEPS)
                                                     ▼
                     ┌───────────────────────────────────────────────────────────────┐
                     │             DEEP MULTI-TASK SEQUENCE DIAGNOSTICIAN            │
                     │     1D Temporal Conv (k=3) ➔ Bi-GRU (64) ➔ Multi-Head Attn   │
                     └───────────────────────┬───────────────────────────────┬───────┘
                                             │                               │
                                             ▼                               ▼
                     ┌───────────────────────────────┐ ┌───────────────────────────────────────┐
                     │ HEAD 1: ATTACK CLASSIFICATION │ │   HEAD 2: SENSOR ATTRIBUTION VECTOR   │
                     │  Softmax (9 Classes)          │ │   Sigmoid (10 Multi-Label Outputs)    │
                     │  Accuracy: 95.93%             │ │   Exact Vector Match: 95.74%          │
                     └───────────────┬───────────────┘ └───────────────────┬───────────────────┘
                                     └───────────────────────┬─────────────┘
                                                             ▼
                                             AUTOPILOT FAIL-SAFE RECONFIGURATION
                                (Decouple Compromised Sensors ➔ Dead Reckoning / Synthetic Wind)
```

---

## 📊 Benchmark Results

Evaluated across **54,000 holdout test flight samples** with domain randomization (wind gusts, airframe mass drift, Gaussian sensor white noise, and bias instability):

### Overall System Diagnostics
| Evaluation Metric | Value | Reference Baseline (LightGBM) |
| :--- | :---: | :---: |
| **Multi-Class Attack Accuracy** | **95.93%** | 94.43% |
| **Attack Macro F1-Score** | **95.70%** | 93.04% |
| **Sensor Exact Match Ratio (All 10 Sensors)** | **95.74%** | 94.08% |
| **Edge Inference Latency (ONNX CPU)** | **0.44 ms** | 0.12 ms |
| **Standalone Model Disk Footprint** | **208 KB** | 1.8 MB |

### Per-Sensor Compromise Attribution Accuracy
The model determines with sub-millisecond latency whether an individual avionics sensor is reliable:

| Sensor Channel | Attribution Accuracy | Detection Mechanism |
| :--- | :---: | :--- |
| **Barometric Altimeter (`baro`)** | **100.0%** | Static pressure vs. kinematic vertical climb rate |
| **Secondary Gyroscope (`gyro2`)** | **100.0%** | Dissimilar secondary MEMS cross-comparison |
| **3-Axis Accelerometer (`accel`)** | **100.0%** | Kinematic derivative $\vec{a} = \frac{d\vec{v}}{dt}$ vs. specific force |
| **C2 Telemetry Link (`c2_link`)** | **100.0%** | Packet loss rate, jitter variance, and RSSI degradation |
| **Primary Gyroscope (`gyro1`)** | **99.98%** | Acoustic ultrasound resonance detection |
| **Radar Altimeter (`radar_alt`)** | **99.98%** | Reflection ground-echo drop & baro divergence |
| **3-Axis Magnetometer (`mag`)** | **99.98%** | Geomagnetic invariant $\|\vec{B}\| \neq 0.52\,\text{Gauss}$ |
| **Optical Seeker (`optical`)** | **99.96%** | Entropy collapse & optical quality drop |
| **GNSS Receiver (`gps`)** | **98.97%** | Satellite count drop, HDOP spike & pseudorange drift |
| **Pitot Tube (`pitot`)** | **96.86%** | Aerodynamic dynamic pressure vs. GPS ground speed |

---

## 💻 Cross-Platform Installation

The project runs natively on **macOS** (Apple Silicon M1/M2/M3/M4 & Intel), **Linux** (Ubuntu 20.04+, Debian, Fedora, Arch), and **Windows 10/11**.

### Prerequisites
- Python **3.9, 3.10, or 3.11** installed.
- Git installed.

---

### macOS Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/uav-intrusion-detection-system.git
cd uav-intrusion-detection-system

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation by running unit tests (including raw unlabeled telemetry tests)
PYTHONPATH=. pytest tests/ -v
```

---

### Linux Setup (Ubuntu / Debian / CentOS / RHEL)
```bash
# 1. Update package index & install development tools
sudo apt update && sudo apt install -y python3-venv python3-pip git

# 2. Clone repository
git clone https://github.com/your-username/uav-intrusion-detection-system.git
cd uav-intrusion-detection-system

# 3. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify installation
PYTHONPATH=. pytest tests/ -v
```

---

### Windows Setup (PowerShell / Command Prompt)

#### 🪟 1-Click Windows Autonomous Launchers (Recommended for Demos & Presentations)

For seamless execution on any Windows 10/11 machine without manual command-line typing, the repository includes self-contained batch scripts:

| Batch Script | Role & Functionality |
| :--- | :--- |
| **`START_TACTICAL_IDS.bat`** | **Primary Master Launcher**: Auto-detects Python, auto-bootstraps `pip`, creates `venv`, installs dependencies with fast CPU wheels, starts background ONNX engine (Port 8502), boots Streamlit (Port 8501), and opens the browser. |
| **`RUN_TACTICAL_SUITE.bat`** | **Launcher Alias**: Direct shortcut that delegates to `START_TACTICAL_IDS.bat`. |
| **`setup_windows.bat`** | **Environment-Only Setup**: Scans Python, bootstraps `pip`, creates `venv`, and pre-installs dependencies without launching the servers. |
| **`run_dashboard.bat`** | **Quick Start**: Autonomous launch shortcut for the tactical dashboard and inference suite. |

##### Automated Error Recovery & Self-Healing:
- **Python Multi-Path Discovery**: Automatically searches system `PATH`, the Windows `py` launcher, `%LocalAppData%\Programs\Python\Python39-312`, and `C:\Program Files\Python39-312`. Works even if the installer's "Add Python to PATH" checkbox was left unchecked.
- **Offline `pip` Auto-Bootstrapping**: If `pip` is missing, corrupted, or stripped in a minimal installation, the launcher repairs and bootstraps it directly from Python's offline standard library using `python -m ensurepip --default-pip` (zero internet required for `pip`).
- **Winget & Python Auto-Prompt**: If no Python 3.8+ runtime is found on the machine, the script detects if Windows Package Manager (`winget`) is present and offers a 1-click automated install of Python 3.11 with PATH and pip enabled. If winget is unavailable, it prompts to open the official python.org download page in the default browser.
- **Fast Startup (< 2 seconds)**: Runs an instant import check on key packages (`streamlit`, `onnxruntime`, `pandas`, `sklearn`, etc.). If dependencies are already satisfied, pip installation is bypassed.
- **Persistent Output & Diagnostics**: If any error occurs, the Command Prompt window pauses with diagnostic guidance instead of abruptly vanishing.

##### Step-by-Step Presentation Instructions:
1. Double-click **`START_TACTICAL_IDS.bat`** (or `RUN_TACTICAL_SUITE.bat`).
2. *(First run only)*: If Windows SmartScreen displays *"Windows protected your PC"*, click **More info** $\rightarrow$ **Run anyway**.
3. Keep the Command Prompt terminal window open in the background while presenting.
4. Your default web browser will automatically open **`http://localhost:8501`**.
5. To stop the application session when finished, switch to the Command Prompt window and press `Ctrl+C`.

---

---

#### 🛠️ Manual Installation (If Batch File or Venv is Not Usable)

##### Option A: Standard Isolated Environment (Venv - Recommended)

**Using Windows Command Prompt (cmd.exe):**
```cmd
# 1. Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate.bat

# 2. Upgrade pip and install dependencies (using lightweight CPU wheel index)
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Launch Tactical Suite
set PYTHONPATH=.
streamlit run dashboard/app.py
```

**Using PowerShell:**
```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Launch Tactical Suite
$env:PYTHONPATH="."
streamlit run dashboard/app.py
```

##### Option B: Global Fallback Installation (No Venv)
If `venv` creation fails or system policies prevent virtual environment creation, install directly to your user Python environment:

```cmd
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install dependencies with --user flag (prevents permission errors in Program Files)
pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Launch Tactical Suite
set PYTHONPATH=.
streamlit run dashboard/app.py
```

---

#### 🔧 Windows Troubleshooting & Edge Case Fixes

| Problem | Cause | Quick Fix Command (Run in CMD) |
| :--- | :--- | :--- |
| **`pip` is missing, broken, or not recognized** | Python was installed without pip, or `ensurepip` failed. | `curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py`<br>*(Or offline: `python -m ensurepip --default-pip`)* |
| **`python` is not recognized as an internal command** | Python wasn't added to system PATH. | Use the Windows Python launcher `py` instead:<br>`py -3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`<br>`py -3 -m streamlit run dashboard/app.py` |
| **PowerShell script execution disabled** | Windows security policy blocks `.ps1` activation. | Run in PowerShell:<br>`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`<br>*Or simply use Command Prompt (`cmd.exe`) where execution policy restrictions do not apply.* |
| **`PermissionError: [WinError 5] Access is denied`** | Python is installed in `C:\Program Files` without Admin rights. | Add `--user` flag to pip:<br>`pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`<br>*Or right-click Command Prompt $\rightarrow$ "Run as Administrator".* |
| **PyTorch installation taking too long or timing out** | Default pip resolves the 2.5 GB CUDA build. | Force the 150 MB lightweight CPU build:<br>`pip install torch --extra-index-url https://download.pytorch.org/whl/cpu` |
| **Port 8501 or 8502 already in use** | A previous instance is still running in the background. | Terminate the occupying process:<br>`for /f "tokens=5" %a in ('netstat -aon ^| findstr :8501') do taskkill /f /pid %a 2>nul`<br>*Or specify a new port: `streamlit run dashboard/app.py --server.port 8510`* |

---

#### 🧹 Master Cleanup & Uninstall Commands

If you ever need to completely remove the installed packages and reset your machine:

- **If you used a Virtual Environment (`venv`)**:
  ```cmd
  rmdir /s /q venv 2>nul & del /f /q get-pip.py 2>nul & rmdir /s /q .pytest_cache 2>nul
  ```
  *(Deletes the `./venv` folder completely. Zero trace remains on your machine).*

- **If you installed into Global Python (No Venv)**:
  ```cmd
  pip uninstall -y -r requirements.txt altair blinker pydeck rich watchdog sympy mpmath & del /f /q get-pip.py 2>nul
  ```
  *(Uninstalls all 17 packages plus Streamlit/Torch sub-dependencies in one shot).*

---

## ⚡ Quick Start & Usage

### 1. Launch the Unified Tactical Suite
Start the local real-time inference server and launch the interactive dashboard:

```bash
# Terminal 1: Start background ONNX inference server (Port 8502)
PYTHONPATH=. python runtime/inference_server.py

# Terminal 2: Launch Streamlit Tactical Dashboard (Port 8501)
PYTHONPATH=. streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.

---

### 2. Run Minimal Edge Inference in Python (< 0.5 ms)
Deploy edge inference on companion computers (Jetson Nano, Raspberry Pi, etc.):

```python
import numpy as np
from runtime.inference import UAVInferenceEngine

# Load the lightweight ONNX model (208 KB)
engine = UAVInferenceEngine(
    onnx_model_path="weights/uav_ids_model.onnx",
    scaler_path="checkpoints/scaler.joblib",
    manifest_path="weights/model_manifest.json"
)

# Provide a 50-step sliding window of 32 physical residual features (1.0s @ 50 Hz)
sample_window = np.zeros((50, 32), dtype=np.float32)

# Execute real-time inference
result = engine.predict(sample_window)

print(f"Attack Detected: {result['attack_detected']}")
print(f"Threat Profile:  {result['attack_type']} (Confidence: {result['confidence']:.2%})")
print(f"Sensor Health:   {result['sensor_health']}")
print(f"Latency:         {result['latency_ms']:.3f} ms")
```

---

### 3. Re-Generate Datasets & Train from Scratch
```bash
# 1. Generate full synthetic dataset (310,500 flight samples across 9 classes)
python sim/build_datasets.py

# 2. Train the Deep Multi-Task Sequence Model (PyTorch / MPS / CUDA / CPU)
python models/train.py

# 3. Export trained checkpoint to optimized ONNX format
python models/export_onnx.py

# 4. Generate publication-ready figures & confusion matrices
python evaluation/report_generator.py
```

---

## 🎮 Interactive Tactical Dashboard & 3D Flight Arena

The dashboard is structured into two functional modules:

### Tab 1: Live Playable 3D Flight Arena & Electronic Warfare Simulator
- **Playable 3D Drone Flight Model**: Fly the tactical drone using keyboard controls (`W`/`S` for pitch, `A`/`D` for roll, `Q`/`E` for yaw, `Space`/`Z` for throttle) or on-screen directional touchpads.
- **Dual Graphics Engine**: Features a 3D WebGL renderer with procedural terrain and lighting, backed by a resilient **2D Tactical Vector Radar & Artificial Horizon Display** fallback.
- **3-Second Animated Attack Countdowns**: Click any attack injector button to trigger an animated 3.0s timer to observe nominal flight vs. the exact moment sensor corruption begins.
- **Genuine ML-Driven Visuals**: The UI stripcharts, threat banner, and sensor trust grid are **driven exclusively by real ONNX model outputs**, querying the background server at 6.7 Hz.
- **Full 56-Channel Flight Recorder**: Capture flight telemetry at 60 FPS and export complete CSV datasets matching training data schemas.
- **In-App 3D Flight Replayer**: Upload any recorded mission CSV to replay the 3D trajectory with pause, forward, and backward step scrubber controls.

### Tab 2: Historical Mission Telemetry & Deep Analytics
- **Whole-Mission Threat Timeline**: Scrub through missions to view classification confidence over time.
- **10-Sensor Health Matrix**: Visual heatmap displaying healthy vs. compromised sensors at each millisecond.
- **Physical Residual Stripcharts**: Real-time plots comparing primary vs. secondary gyros, pitot airspeed vs. GPS ground speed, and triple-altitude redundancy.

---

## 📐 Mathematical Formulation of Physics Residuals

Rather than feeding raw uncalibrated signals, the diagnostic pipeline extracts **32 invariant features** derived from aerospace kinematics:

### 1. Kinematic vs. Inertial Horizontal Acceleration
$$\vec{a}_{\text{kinematic}} = \frac{d}{dt}\begin{bmatrix} v_{\text{north}}^{\text{GPS}} \\ v_{\text{east}}^{\text{GPS}} \end{bmatrix}, \quad r_{\text{accel}} = \left| \|\vec{a}_{\text{kinematic}}\| - \sqrt{a_x^2 + a_y^2} \right|$$

### 2. Aerodynamic Airspeed vs. Ground Speed Invariant
Under steady flight without extreme wind shear:
$$r_{\text{airspeed}} = \left| v_{\text{pitot}} - \sqrt{\left(v_{\text{north}}^{\text{GPS}}\right)^2 + \left(v_{\text{east}}^{\text{GPS}}\right)^2} \right|$$

### 3. Triple Altitude Consistency
Cross-checks three independent physical principles (RF transit time, hydrostatic air pressure, and satellite pseudorange):
$$r_{\text{alt}}^{\text{GPS-Baro}} = |h_{\text{GPS}} - h_{\text{baro}}|, \quad r_{\text{alt}}^{\text{Baro-Radar}} = |h_{\text{baro}} - h_{\text{radar}}|, \quad r_{\text{alt}}^{\text{GPS-Radar}} = |h_{\text{GPS}} - h_{\text{radar}}|$$

### 4. Dual-IMU Angular Rate Invariant
Compares primary MEMS gyro $\vec{\omega}_1$ against a dissimilar, isolated secondary gyro $\vec{\omega}_2$:
$$r_{\text{gyro}} = \|\vec{\omega}_1 - \vec{\omega}_2\| = \sqrt{(p_1 - p_2)^2 + (q_1 - q_2)^2 + (r_1 - r_2)^2}$$

### 5. Geomagnetic Field Norm Invariant
Earth's local magnetic field magnitude remains constant ($B \approx 0.52\,\text{Gauss}$) regardless of UAV orientation:
$$r_{\text{mag}} = \left| \sqrt{B_x^2 + B_y^2 + B_z^2} - 0.52 \right|$$

---

## ⚔️ Comprehensive Attack Vector Profiles & Detection Mechanics

The suite detects, isolates, and mitigates 9 distinct cyber-physical profiles. Below is the detailed breakdown of the physical avionics target, nominal vs. corrupted telemetry signatures, invariant violation physics, training simulation parameters, and automated autopilot fail-safe responses:

---

### 1. Nominal Tactical Combat Flight (`Normal`)
* **Avionics Subsystems**: Full 10-sensor avionics suite operating under dynamic combat maneuvers and atmospheric disturbances.
* **Operational Reality**: High-dynamic tactical maneuvering (loiter banking up to $25^\circ$, pitch undulations, throttle changes between $25\%\text{--}75\%$), atmospheric turbulence wind gusts ($1.5\text{--}6.5\text{ m/s}$), engine harmonic vibration ($80\text{--}240\text{ Hz}$), and stochastic sensor bias instability.
* **Physical Dynamics & Invariants**:
  * Accelerometer specific force matches GPS kinematic acceleration: $\left| \|\vec{a}_{\text{kinematic}}\| - \sqrt{a_x^2 + a_y^2} \right| \le 1.8\text{ m/s}^2$.
  * Pitot indicated airspeed matches GPS groundspeed vector within wind boundaries: $|v_{\text{pitot}} - v_{\text{ground}}| \le 3.5\text{ m/s}$.
  * Primary and secondary MEMS gyroscopes closely agree: $\|\vec{\omega}_1 - \vec{\omega}_2\| \le 0.4\text{ rad/s}$.
  * Triple altitude readings agree across independent physical sensors: $|h_{\text{GPS}} - h_{\text{baro}}| \le 3.5\text{ m}$ and $|h_{\text{baro}} - h_{\text{radar}}| \le 2.0\text{ m}$.
  * Geomagnetic field norm tracks Earth's local field: $\|\vec{B}\| \approx 0.52\text{ Gauss}$.
  * Command datalink maintains low latency and minimal drop: `c2_loss_rate` $\le 2\%$, `c2_jitter_ms` $\le 5.0\text{ ms}$, `c2_rssi_dbm` $\approx -58\text{ dBm}$.
* **Dataset Generation**: 103,998 nominal samples across 100 combat missions in `sim/generator.py` with randomized airframe mass ($3.5\text{--}4.9\text{ kg}$), motor thrust curves, and wind vectors.
* **Autopilot Mode**: **GPS-Aided Extended Kalman Filter (EKF)** — all 10 avionics sensors fully trusted.

---

### 2. GNSS Coordinated Velocity & Position Spoofing (`GNSS_Spoofing`)
* **Target Hardware**: Multi-constellation GNSS receiver (e.g., u-blox ZED-F9P, NEO-M8N).
* **Threat Mechanism**: Attacker uses a synchronized Software-Defined Radio (SDR) to transmit counterfeit GPS satellite pseudoranges, capturing the receiver tracking loops and introducing gradual drift without sudden discontinuities.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under GNSS Spoofing |
  | :--- | :---: | :---: |
  | `gps_vn` (North Velocity) | Tracks flight path ($\approx 32.0\text{ m/s}$) | Injected drift $+2.24\text{ m/s}$ ($+6.5\text{ m/s}$ offset) |
  | `gps_ve` (East Velocity) | Tracks flight path ($\approx 0.0\text{ m/s}$) | Injected drift $+1.68\text{ m/s}$ ($+4.5\text{ m/s}$ offset) |
  | `gps_lat, lon` (Coordinates) | Truth position | Quadratic drift off waypoint path |
  | `gps_sats`, `gps_hdop` | $16\text{ sats}$, $0.95$ HDOP | $16\text{ sats}$, $0.95$ HDOP *(attacker masks quality)* |
  | `res_airspeed_vs_groundspeed` | $\le 2.5\text{ m/s}$ | **$> 5.5\text{ m/s}$** *(aerodynamic invariant breach)* |
* **Physical Invariant Breached**: 
  1. Aerodynamic invariant $|v_{\text{pitot}} - v_{\text{ground}}| > 4.0\text{ m/s}$ while throttle and pitch remain unchanged.
  2. Kinematic derivative discrepancy: Numerical gradient $\frac{d\vec{v}_{\text{GPS}}}{dt}$ contradicts accelerometer specific force $\vec{f}_{\text{IMU}}$.
* **How It Was Trained**: Simulated with smooth quadratic integral drift in `sim/attacks.py`. Multi-task network learns to correlate GPS velocity changes with Pitot airspeed and accelerometer force. Attribution accuracy: **98.97%**.
* **Autopilot Fail-Safe**: **Dead Reckoning Mode (GPS Decoupled)** — autopilot sets `EKF2_GPS_CTRL=0`, isolating GPS and navigating via strapdown IMU, Pitot airspeed, and barometric altitude.

---

### 3. GNSS Broadband RF Jamming (`GNSS_Jamming`)
* **Target Hardware**: GNSS L1/L2 patch antenna and RF front-end low-noise amplifier (LNA).
* **Threat Mechanism**: Ground-based electronic warfare jammer blankets the $1575.42\text{ MHz}$ and $1227.60\text{ MHz}$ bands with high-power Gaussian white noise or swept continuous-wave jamming, raising the receiver noise floor.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under GNSS RF Jamming |
  | :--- | :---: | :---: |
  | `gps_sats` (Locked Satellites) | $16\text{ sats}$ | **$0 \text{ to } 2\text{ sats}$** *(carrier lock lost)* |
  | `gps_hdop` (Dilution of Precision) | $0.95$ | **$16.0 \text{ to } 24.0$** *(extreme geometry collapse)* |
  | `gps_cn0` (Carrier-to-Noise Ratio) | $44.5\text{ dB-Hz}$ | **$14.0 \text{ to } 22.0\text{ dB-Hz}$** |
  | `gps_vn, ve` Velocity Variance | $\sigma \approx 0.04\text{ m/s}$ | **$\sigma \approx 4.5\text{ m/s}$** *(random walk noise)* |
* **Physical Invariant Breached**: Direct carrier-to-noise and constellation geometric degradation without airframe occlusion.
* **How It Was Trained**: Simulated in `sim/attacks.py` with randomized RF attenuation floors. The 1D-CNN temporal filters detect the sudden step drop in satellite count and HDOP surge. Attribution accuracy: **100.0%**.
* **Autopilot Fail-Safe**: **INS Standalone Inertial Hold** — GPS rejected; navigation holds attitude and heading on dual gyros and optical flow.

---

### 4. Acoustic MEMS Gyroscope Ultrasonic Resonance (`Acoustic_Gyro_Resonance`)
* **Target Hardware**: Primary MEMS silicon capacitive tuning-fork / vibrating ring gyroscope (e.g., Bosch BMI088 / InvenSense MPU-6000).
* **Threat Mechanism**: Hostile directional acoustic transducer directs high-intensity ultrasound ($> 110\text{ dB SPL}$) tuned to the mechanical resonant frequency ($9.25\text{ kHz}$) of the primary gyroscope's micromachined silicon proof-mass.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under Acoustic Resonance |
  | :--- | :---: | :---: |
  | `gyro_p` (Primary Roll Rate) | $\approx 0.0\text{ rad/s}$ | **$+3.4 \text{ to } +4.5\text{ rad/s}$** ($+200^\circ/\text{s}$ false DC bias) |
  | `gyro1_p_std_window` (Roll Variance) | $0.15\text{ rad/s}$ | **$> 3.5\text{ rad/s}$** *(high-frequency resonant chatter)* |
  | `gyro2_p` (Secondary Isolated Gyro) | $\approx 0.0\text{ rad/s}$ | $\approx 0.0\text{ rad/s}$ *(acoustically isolated & nominal)* |
  | `res_gyro_discrepancy` | $\le 0.3\text{ rad/s}$ | **$> 3.5\text{ rad/s}$** *(dual-IMU invariant breach)* |
* **Physical Invariant Breached**: Dual-IMU redundancy invariant $\|\vec{\omega}_1 - \vec{\omega}_2\| > 1.0\text{ rad/s}$. The secondary dissimilar MEMS gyroscope (damped and physically isolated) contradicts the primary gyro.
* **How It Was Trained**: Ultrasonic resonance injection with DC bias and high-frequency harmonics in `sim/attacks.py`. Multi-scale 1D-CNN extracts the resonant frequency variance while Bi-GRU confirms secondary IMU stability. Attribution accuracy: **99.98%**.
* **Autopilot Fail-Safe**: **Dissimilar Sensor Transfer** — primary gyro isolated; EKF attitude estimation instantaneously transfers to secondary MEMS gyro.

---

### 5. Optical Seeker Laser Blinding / Dazzler (`Optical_Seeker_Blinding`)
* **Target Hardware**: Forward-looking EO/IR visual camera pod and optical flow visual odometry sensor.
* **Threat Mechanism**: Ground-based continuous-wave (CW) laser dazzler or dense pyrotechnic aerosol screen saturates the optical focal plane array.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under Laser Blinding |
  | :--- | :---: | :---: |
  | `optical_entropy` (Shannon Texture) | $0.88$ | **$0.01 \text{ to } 0.04$** *(complete texture collapse)* |
  | `optical_quality` (Tracking Consensus) | $0.95$ | **$0.00 \text{ to } 0.02$** *(all keypoints lost)* |
  | `optical_flow_vx, vy` (Flow Velocity) | $32.0\text{ m/s}$ | **$0.0\text{ m/s}$** *(image frozen/saturated)* |
  | `res_optical_flow_mag` | $\approx 32.0\text{ m/s}$ | **$0.0\text{ m/s}$** |
* **Physical Invariant Breached**: Visual odometry magnitude collapses to $0.0\text{ m/s}$ while Pitot airspeed ($32\text{ m/s}$) and accelerometer integration prove high cruise speed.
* **How It Was Trained**: Modeled with focal plane array saturation in `sim/attacks.py`. Temporal self-attention focuses on keypoint consensus collapse. Attribution accuracy: **99.96%**.
* **Autopilot Fail-Safe**: **Blind Homing Fallback** — terminal visual tracking decoupled; guidance falls back to GPS coordinates and inertial dead reckoning.

---

### 6. Pitot-Static Tube Freeze / Inlet Occlusion (`Pitot_Tube_Fault`)
* **Target Hardware**: Dynamic pressure differential air data probe (e.g., Sensirion SDP3x).
* **Threat Mechanism**: In-flight icing, freezing rain, or foreign particulate / insect blockage obstructing the forward total pressure inlet.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under Pitot Freeze / Occlusion |
  | :--- | :---: | :---: |
  | `pitot_ias` (Indicated Airspeed) | $32.0\text{ m/s}$ | **$0.5\text{ m/s}$** *(or frozen static reading)* |
  | `v_ground` (GPS Ground Speed) | $32.0\text{ m/s}$ | $32.0\text{ m/s}$ *(nominal high-speed flight)* |
  | `motor_rpm` (Rotor RPM) | $7,200\text{ RPM}$ | $7,200\text{ RPM}$ *(cruising throttle)* |
  | `res_airspeed_vs_groundspeed` | $\le 2.5\text{ m/s}$ | **$> 25.0\text{ m/s}$** *(aerodynamic invariant breach)* |
* **Physical Invariant Breached**: Aerodynamic invariant $|v_{\text{pitot}} - v_{\text{ground}}| > 4.0\text{ m/s}$ without stall-induced altitude loss or engine failure.
* **How It Was Trained**: Inlet occlusion and zero-pressure collapse simulated in `sim/attacks.py`. Sequence model learns that high engine RPM and high GPS groundspeed contradict zero indicated airspeed. Attribution accuracy: **96.86%**.
* **Autopilot Fail-Safe**: **Synthetic Airspeed Estimator** — Pitot probe decoupled; airspeed estimated from GPS groundspeed vector and EKF wind triangle model to prevent stall.

---

### 7. Radar Altimeter Deception Jamming (`Radar_Altimeter_Jamming`)
* **Target Hardware**: $77\text{ GHz}$ FMCW / millimeter-wave downward radar altimeter.
* **Threat Mechanism**: Electronic warfare digital radio frequency memory (DRFM) repeater captures the transmitted radar chirps and retransmits them with an artificial delay.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under Radar Jamming |
  | :--- | :---: | :---: |
  | `radar_alt_agl` (Radar Altitude) | $180.0\text{ m}$ | **$228.0\text{ m}$** ($+48.0\text{ m}$ false offset) |
  | `radar_reflectivity` (Surface Metric) | $0.92$ | **$0.05 \text{ to } 0.10$** *(repeater phase noise)* |
  | `baro_alt` (Barometric Altitude) | $180.0\text{ m}$ | $180.0\text{ m}$ *(nominal)* |
  | `gps_alt` (GNSS Altitude) | $180.0\text{ m}$ | $180.0\text{ m}$ *(nominal)* |
  | `res_alt_baro_radar` | $\le 2.0\text{ m}$ | **$> 45.0\text{ m}$** *(3-way voting discordance)* |
* **Physical Invariant Breached**: Triple-redundancy altitude invariance $|h_{\text{radar}} - h_{\text{baro}}| > 20.0\text{ m}$ and $|h_{\text{radar}} - h_{\text{GPS}}| > 20.0\text{ m}$ while Baro and GPS altitude maintain close agreement ($|h_{\text{baro}} - h_{\text{GPS}}| \le 2.0\text{ m}$).
* **How It Was Trained**: DRFM false returns simulated in `sim/attacks.py`. 3-way altitude cross-voting layer in `pipeline/residuals.py` instantly isolates the single outlier sensor. Attribution accuracy: **99.98%**.
* **Autopilot Fail-Safe**: **Baro-Inertial Altitude Channel** — radar altimeter rejected; vertical guidance relies exclusively on fused barometric pressure and vertical accelerometer integration.

---

### 8. Geomagnetic Sensor Tampering (`Magnetometer_Interference`)
* **Target Hardware**: 3-axis tri-axial magnetometer (e.g., Honeywell HMC5883L, Asahi Kasei AK09916).
* **Threat Mechanism**: Electromagnetic tampering, nearby high-voltage transmission lines, or localized magnetic pulses distorting the ambient geomagnetic field vector.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under Magnetic Interference |
  | :--- | :---: | :---: |
  | `mag_x` (X-Axis Field) | $\approx 0.36\text{ Gauss}$ | **$\times 3.4$** ($\approx 1.25\text{ Gauss}$) |
  | `mag_y` (Y-Axis Field) | $\approx 0.00\text{ Gauss}$ | **$+1.20\text{ Gauss}$** offset |
  | `mag_norm` ($\|\vec{B}\|$) | $0.52\text{ Gauss}$ | **$1.85 \text{ to } 2.10\text{ Gauss}$** *(extreme invariant breach)* |
  | `res_mag_norm` | $\le 0.05\text{ Gauss}$ | **$> 1.30\text{ Gauss}$** |
* **Physical Invariant Breached**: Earth's geomagnetic field norm invariant $\|\vec{B}\| = \sqrt{B_x^2 + B_y^2 + B_z^2} \approx 0.52\text{ Gauss}$ (World Magnetic Model) is strictly invariant to airframe attitude. Deviation $|\|\vec{B}\| - 0.52| > 0.25\text{ Gauss}$ proves local magnetic tampering.
* **How It Was Trained**: Dipole magnetic distortion injected in `sim/attacks.py`. Normalized `res_mag_norm` feature directly separates this class. Attribution accuracy: **99.98%**.
* **Autopilot Fail-Safe**: **Gyro-Heading Integration** — magnetometer rejected; yaw angle driven by integrating rate gyro `gyro_r` corrected with GNSS course over ground.

---

### 9. C2 Telemetry Databand Buffer Flooding & DoS (`C2_Link_Jamming_DoS`)
* **Target Hardware**: Ground control datalink transceiver (MAVLink / COFDM radio modem).
* **Threat Mechanism**: Adversary launches high-rate UDP packet flooding on the datalink port or broadband RF jamming on the $915\text{ MHz} / 2.4\text{ GHz}$ ISM telemetry bands, overwhelming onboard serial transceiver buffers.
* **Telemetry Contrast Matrix**:
  | Metric Channel | Nominal Flight | Under C2 Flooding / DoS |
  | :--- | :---: | :---: |
  | `c2_loss_rate` (Packet Drop) | $\approx 0.015$ ($1.5\%$) | **$0.80 \text{ to } 0.98$** ($> 80\%$ loss) |
  | `c2_jitter_ms` (Latency Jitter) | $3.5\text{ ms}$ | **$140 \text{ to } 260\text{ ms}$** *(buffer bloat delay)* |
  | `c2_rssi_dbm` (Signal Strength) | $-58.0\text{ dBm}$ | **$-100.0 \text{ to } -110.0\text{ dBm}$** *(SNR collapse)* |
* **Physical Invariant Breached**: Datalink packet arrival statistics violate standard Poisson arrival limits while onboard physical sensors remain nominal.
* **How It Was Trained**: Simulated with stochastic network queue congestion and signal attenuation in `sim/attacks.py`. Bi-GRU captures sustained communications loss. Attribution accuracy: **100.0%**.
* **Autopilot Fail-Safe**: **Autonomous RTL Fail-Safe** — datalink decoupled; drone executes pre-programmed autonomous Return-to-Launch along protected GPS/INS waypoints.

---

## 🛸 Hardware & Drone Autopilot Deployment

### Companion Computer Architecture
To deploy on physical drones running **PX4 Autopilot** or **ArduPilot** (Pixhawk 6X, Cube Orange):

```
[ Flight Controller: Pixhawk / Cube ]
       │  (Runs 400 Hz Rate PID & EKF3)
       │  TELEMETRY STREAM via MAVLink (UART @ 921600 baud)
       ▼
[ Companion Computer: NVIDIA Jetson Orin Nano / Raspberry Pi 5 ]
       │  1. Ingests HIGHRES_IMU, GPS_RAW_INT, VFR_HUD, DISTANCE_SENSOR
       │  2. Evaluates uav_ids_model.onnx (Latency: 0.44 ms @ 20 Hz)
       │  3. If attack detected: Sends MAVLink parameter override
       ▼
[ Autopilot Fail-Safe Trigger ]
       • GPS Spoofed   ➔ Set EKF2_GPS_CTRL=0 (Switch to Optical Flow / Dead Reckoning)
       • Gyro Corrupted ➔ Switch to Dissimilar Secondary Gyro (EKF IMU Failover)
       • Pitot Frozen   ➔ Switch to Synthetic Airspeed Estimation
```

---

## 📂 Repository Structure

```
.
├── LICENSE                          # MIT License
├── README.md                        # Comprehensive system documentation & user guide
├── requirements.txt                 # Python dependencies
├── config/
│   ├── sensors_config.yaml          # Avionics specifications & noise densities
│   └── attack_config.yaml           # Attack injection profiles & parameters
├── sim/
│   ├── airframe.py                  # 6-DOF loitering munition flight dynamics
│   ├── sensors.py                   # Multi-modal avionics simulator (Dual IMU, GPS, Pitot, etc.)
│   ├── attacks.py                   # Electronic warfare attack injection modules
│   ├── generator.py                 # Mission generator with domain randomization
│   └── build_datasets.py            # Generates train, val, test Parquet splits
├── pipeline/
│   ├── residuals.py                 # Analytical kinematic & aerodynamic residual engine
│   └── windowing.py                 # Sliding temporal window segmentation
├── models/
│   ├── deep_sequence.py             # 1D-CNN + BiGRU + Temporal Self-Attention architecture
│   ├── tree_ensemble.py             # LightGBM / XGBoost baseline comparison models
│   ├── train.py                     # Multi-task training script
│   └── export_onnx.py               # Exporter for lightweight ONNX runtime model
├── runtime/
│   ├── inference.py                 # Minimal ONNX runtime inference engine
│   └── inference_server.py          # Real-time HTTP inference daemon (Port 8502)
├── dashboard/
│   ├── app.py                       # Unified Streamlit application
│   └── sim.html                     # 3D tactical flight arena & 56-channel recorder
├── static/
│   └── three.min.js                 # Bundled Three.js library for offline environments
├── checkpoints/
│   ├── best_deep_model.pt           # PyTorch checkpoint
│   ├── lgb_model.joblib             # LightGBM baseline checkpoint
│   └── scaler.joblib                # Feature normalization scaler
├── weights/
│   ├── uav_ids_model.onnx           # Production ONNX model file (208 KB)
│   └── model_manifest.json          # Deployment channel & feature metadata
├── evaluation/
│   ├── benchmark_results.json       # Evaluated metrics on 54,000 test samples
│   ├── report_generator.py          # Script generating confusion matrix & charts
│   └── figures/                     # High-resolution publication figures
├── data/
│   └── recorded_missions/           # Sample CSV mission telemetry files
└── tests/
    ├── test_features.py             # Physics residual invariant tests
    ├── test_simulation.py           # 6-DOF physics and sensor validation tests
    └── test_onnx_inference.py       # Real-time ONNX inference validation tests
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](file:///Users/rohan/Projects/UAV%20Intrusion%20Detection%20System/LICENSE) file for details.
