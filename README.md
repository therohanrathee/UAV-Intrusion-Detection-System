# Tactical UAV & Loitering Munition Cyber-Physical Intrusion Detection Suite (IDS)

[![Tests](https://img.shields.io/badge/tests-7%20passed-success?style=flat-square&logo=pytest)](file:///Users/rohan/Projects/UAV%20Intrusion%20Detection%20System/tests)
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
- [Cyber-Physical Attack Profiles](#-cyber-physical-attack-profiles)
- [Hardware & Drone Autopilot Deployment](#-hardware--drone-autopilot-deployment)
- [Repository Structure](#-repository-structure)
- [License & Citation](#-license--citation)

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

# 4. Verify installation by running unit tests
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

#### Using PowerShell:
```powershell
# 1. Clone repository
git clone https://github.com/your-username/uav-intrusion-detection-system.git
cd uav-intrusion-detection-system

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run: Set-ExecutionPolicy Unrestricted -Scope Process)

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
$env:PYTHONPATH="."
pytest tests/ -v
```

#### Using Windows Command Prompt (cmd.exe):
```cmd
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
set PYTHONPATH=.
pytest tests\ -v
```

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
$$\vec{a}_{\text{kinematic}} = \frac{d}{dt}\begin{bmatrix} v_{\text{north}}^{\text{GPS}} \\ v_{\text{east}}^{\text{GPS}} \end{bmatrix}, \quad r_{\text{accel\_horiz}} = \left| \|\vec{a}_{\text{kinematic}}\| - \sqrt{a_x^2 + a_y^2} \right|$$

### 2. Aerodynamic Airspeed vs. Ground Speed Invariant
Under steady flight without extreme wind shear:
$$r_{\text{airspeed}} = \left| v_{\text{pitot}} - \sqrt{\left(v_{\text{north}}^{\text{GPS}}\right)^2 + \left(v_{\text{east}}^{\text{GPS}}\right)^2} \right|$$

### 3. Triple Altitude Consistency
Cross-checks three independent physical principles (RF transit time, hydrostatic air pressure, and satellite pseudorange):
$$r_{\text{alt\_gps\_baro}} = |h_{\text{GPS}} - h_{\text{baro}}|, \quad r_{\text{alt\_baro\_radar}} = |h_{\text{baro}} - h_{\text{radar}}|, \quad r_{\text{alt\_gps\_radar}} = |h_{\text{GPS}} - h_{\text{radar}}|$$

### 4. Dual-IMU Angular Rate Invariant
Compares primary MEMS gyro $\vec{\omega}_1$ against a dissimilar, isolated secondary gyro $\vec{\omega}_2$:
$$r_{\text{gyro}} = \|\vec{\omega}_1 - \vec{\omega}_2\| = \sqrt{(p_1 - p_2)^2 + (q_1 - q_2)^2 + (r_1 - r_2)^2}$$

### 5. Geomagnetic Field Norm Invariant
Earth's local magnetic field magnitude remains constant ($B \approx 0.52\,\text{Gauss}$) regardless of UAV orientation:
$$r_{\text{mag}} = \left| \sqrt{B_x^2 + B_y^2 + B_z^2} - 0.52 \right|$$

---

## ⚔️ Cyber-Physical Attack Profiles

The framework models 9 operational combat conditions:

1. **Normal (Nominal Flight)**: Multi-axis tactical combat maneuvering, turbulence gusts, and Gaussian sensor noise.
2. **GNSS Spoofing**: Progressive velocity and position injection ($2.24\,\text{m/s}$ north, $1.68\,\text{m/s}$ east drift) inducing coordinate divergence without abrupt discontinuities.
3. **GNSS RF Jamming**: High-power broadband noise causing satellite lock drops ($N_{\text{sats}} \le 2$), HDOP explosion ($> 16.0$), and carrier-to-noise ratio ($C/N_0$) collapse.
4. **Acoustic Gyroscope Resonance**: Targeted ultrasound injection matching the mechanical resonant frequency ($9.25\,\text{kHz}$) of the primary MEMS gyroscope, inducing massive DC bias ($+3.4\,\text{rad/s}$) and vibration.
5. **Optical Seeker Laser Blinding**: Laser dazzler saturation causing optical flow velocity drop, entropy collapse, and image quality zeroing.
6. **Pitot-Static Tube Icing / Fault**: Freezing or total occlusion of the dynamic pressure inlet, collapsing indicated airspeed ($v_{\text{IAS}} \approx 0.5\,\text{m/s}$).
7. **Radar Altimeter Jamming**: False-target RF echo generation inducing $+48.0\,\text{m}$ altitude offsets and reflectivity collapse.
8. **Magnetometer Interference**: Electromagnetic tampering from onboard motors or high-voltage lines distorting geomagnetic norms.
9. **C2 Link Flooding / DoS**: Command-and-control datalink saturation causing packet loss ($> 80\%$), latency jitter ($> 140\,\text{ms}$), and signal attenuation.

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

## 📜 License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](file:///Users/rohan/Projects/UAV%20Intrusion%20Detection%20System/LICENSE) file for details.

If you use this work, codebase, or dataset generation pipeline in academic research, please cite:

```bibtex
@article{uav_cyber_physical_ids_2026,
  title={Physics-Informed Multi-Task Intrusion Detection and Sensor Compromise Attribution for Autonomous Loitering Munitions},
  author={UAV Cyber-Physical Systems Research Group},
  year={2026},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  note={Open-Source Avionics Telemetry Diagnostic Suite}
}
```
