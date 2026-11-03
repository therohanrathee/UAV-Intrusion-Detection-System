"""
Tactical UAV Cyber-Physical Intrusion Detection Suite - Interactive Dashboard
Unified Application with Dual Tabs:
- Tab 1: Live 3D Playable Flight Arena with Cyber-Attack Simulator, Autopilot Fail-Safe & 3D Flight Replayer
- Tab 2: Historical Mission Telemetry Replayer, Saved Mission Library, ONNX Model Diagnostics & Benchmarks
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import time

from sim.generator import MissionGenerator
from pipeline.residuals import compute_physics_residuals, FEATURE_COLUMNS
from pipeline.windowing import create_sliding_windows
from runtime.inference import UAVInferenceEngine
from runtime.inference_server import start_inference_server
from sim.attacks import ATTACK_NAMES, SENSOR_KEYS

# Initialize background edge inference server for live 3D arena
start_inference_server(port=8502)

st.set_page_config(
    page_title="Tactical UAV Cyber-Physical IDS & 3D Flight Arena",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Tactical Dark Styling
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0f17 !important;
        color: #c9d1d9 !important;
    }
    .main .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        max-width: 100vw !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
        display: block !important;
        border-radius: 8px !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        color: #8b949e;
        font-weight: 600;
        font-size: 14px;
        border-radius: 6px;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }

    /* Banners & Cards */
    .threat-banner-clean {
        background-color: #0e4429;
        color: #3fb950;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        border: 1px solid #238636;
        margin-bottom: 12px;
    }
    .threat-banner-attack {
        background-color: #490202;
        color: #ff7b72;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        border: 1px solid #da3633;
        margin-bottom: 12px;
        box-shadow: 0 0 12px rgba(218, 54, 51, 0.4);
    }
    .status-healthy {
        background-color: #238636;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        display: inline-block;
        font-size: 0.8rem;
    }
    .status-compromised {
        background-color: #da3633;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        display: inline-block;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Initializes the ONNX inference runtime."""
    return UAVInferenceEngine(
        onnx_model_path=str(PROJECT_ROOT / "weights/uav_ids_model.onnx"),
        scaler_path=str(PROJECT_ROOT / "checkpoints/scaler.joblib"),
        manifest_path=str(PROJECT_ROOT / "weights/model_manifest.json")
    )


def get_sim_html_content():
    """Reads dashboard/sim.html and inlines bundled Three.js cleanly."""
    sim_path = PROJECT_ROOT / "dashboard/sim.html"
    three_path = PROJECT_ROOT / "static/three.min.js"
    
    html = sim_path.read_text(encoding="utf-8")
    three_code = three_path.read_text(encoding="utf-8") if three_path.exists() else ""
    return html.replace("/* THREE_JS_INLINE */", three_code)


def render_tab_playable_arena():
    """Renders Tab 1: Live Playable 3D Flight Simulation, Cyber Attack Injector & 3D Replayer."""
    st.markdown("### 🎮 Tactical Flight Arena, 3D Replayer & Cyber-Attack Simulator")
    st.caption("Interact directly with the 3D drone flight model or replay previously recorded flights. Inject hostile electronic warfare attacks with 3-second animated countdown timers, observe live green/red sensor telemetry stripcharts evaluated in real-time by the ONNX machine learning model, and export full 56-channel telemetry CSVs.")

    html_content = get_sim_html_content()
    components.html(html_content, height=740, scrolling=False)

    with st.expander("ℹ️ Flight Arena Controls & Threat Injection Guide", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            **🕹️ Flight Navigation Controls**
            - **Click inside viewport** to activate keyboard flight.
            - `W` / `S`: Pitch Down (Forward Speed) / Pitch Up (Climb)
            - `A` / `D`: Bank Left / Right
            - `Q` / `E`: Yaw Left / Right
            - `Space` / `Shift`: Throttle Up (Increases Rotor RPM)
            - `Z` / `Ctrl`: Throttle Down
            - `C`: Cycle Camera (Chase, Cockpit FPV, Orbit)
            """)
        with c2:
            st.markdown("""
            **⚔️ 3-Second Attack Countdown System**
            - Clicking any cyber-attack button triggers an **animated 3.0s timer**.
            - Observe the pre-attack nominal flight vs post-attack corrupted telemetry.
            - **Red dynamic stripcharts** highlight when a sensor is compromised.
            - Click **✓ RESTORE NOMINAL FLIGHT** to return to nominal flight.
            """)
        with c3:
            st.markdown("""
            **💾 56-Channel Recorder & 3D Replay**
            - Click **⏺ RECORD** to record high-fidelity telemetry at 60 FPS.
            - Click **💾 EXPORT CSV** to download the exact 56-channel dataset format.
            - Click **📂 UPLOAD & REPLAY 3D FLIGHT** to watch the drone fly any saved CSV in 3D with full sensor replay and pause/step controls!
            """)


def render_tab_historical_analytics():
    """Renders Tab 2: Historical Telemetry Replayer, Saved Mission Library & Diagnostic Analytics."""
    st.markdown("### 📊 Historical Mission Telemetry & Deep Diagnostic Analytics")
    st.caption("Inspect saved combat flight missions, run through telemetry across time with step controls, and evaluate the edge ONNX Deep Multi-Task Intrusion Detection Model.")

    mission_lib_dir = PROJECT_ROOT / "data/recorded_missions"
    mission_lib_dir.mkdir(parents=True, exist_ok=True)

    # Sidebar: Scenario Selection
    st.sidebar.markdown("---")
    st.sidebar.header("🕹️ Mission Simulation Control")
    scenario_mode = st.sidebar.radio(
        "Mission Source",
        [
            "📁 Saved Missions Library",
            "📤 Upload New Flight CSV",
            "🚀 Generate Live Combat Mission",
            "🔬 Load Benchmark Parquet Split"
        ]
    )

    gen = MissionGenerator()

    if scenario_mode == "📁 Saved Missions Library":
        saved_files = sorted(list(mission_lib_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not saved_files:
            st.sidebar.info("No recorded missions found. Record a flight in Tab 1 or upload a CSV.")
            return
        
        file_options = {f.name: f for f in saved_files}
        selected_file_name = st.sidebar.selectbox("Select Saved Flight", list(file_options.keys()))
        selected_file_path = file_options[selected_file_name]
        
        st.sidebar.caption(f"File size: {selected_file_path.stat().st_size / 1024:.1f} KB")

        if st.sidebar.button("📂 Load Selected Flight", type="primary") or "mission_data" not in st.session_state:
            with st.spinner(f"Loading {selected_file_name}..."):
                df_raw = pd.read_csv(selected_file_path)
                df_feat = compute_physics_residuals(df_raw)
                X_win, y_att, y_sen = create_sliding_windows(df_feat, window_size=50, stride=5)
                st.session_state["mission_data"] = df_feat
                st.session_state["windows"] = X_win
                st.session_state["true_attacks"] = y_att
                st.session_state["true_sensors"] = y_sen
                st.session_state["mission_preds"] = None
                st.session_state["scrubber_idx"] = 0
                st.session_state["current_mission_title"] = selected_file_name

    elif scenario_mode == "📤 Upload New Flight CSV":
        st.sidebar.markdown("Upload telemetry CSV downloaded from **Tab 1 Flight Arena** or disk:")
        uploaded_file = st.sidebar.file_uploader("Choose Telemetry CSV", type=["csv"])
        save_to_lib = st.sidebar.checkbox("Save permanently to Missions Library", value=True)

        if uploaded_file is not None:
            if st.sidebar.button("📂 Process & Load Uploaded Flight", type="primary") or "mission_data" not in st.session_state:
                with st.spinner("Processing flight telemetry & extracting physics residuals..."):
                    df_raw = pd.read_csv(uploaded_file)
                    if len(df_raw) < 50:
                        st.sidebar.error("Flight recording too short. Please record at least 50 samples (~1.0s @ 50 Hz).")
                        return
                    if save_to_lib:
                        dest_path = mission_lib_dir / uploaded_file.name
                        df_raw.to_csv(dest_path, index=False)
                    
                    df_feat = compute_physics_residuals(df_raw)
                    X_win, y_att, y_sen = create_sliding_windows(df_feat, window_size=50, stride=5)
                    st.session_state["mission_data"] = df_feat
                    st.session_state["windows"] = X_win
                    st.session_state["true_attacks"] = y_att
                    st.session_state["true_sensors"] = y_sen
                    st.session_state["mission_preds"] = None
                    st.session_state["scrubber_idx"] = 0
                    st.session_state["current_mission_title"] = uploaded_file.name
                    st.sidebar.success(f"✓ Loaded {len(df_raw)} frames ({len(X_win)} analysis windows)!")

    elif scenario_mode == "🚀 Generate Live Combat Mission":
        attack_type_name = st.sidebar.selectbox("Inject Cyber-Physical Attack", ATTACK_NAMES)
        attack_idx = ATTACK_NAMES.index(attack_type_name)
        mission_duration = st.sidebar.slider("Mission Duration (s)", 20.0, 45.0, 30.0, 5.0)
        attack_onset = st.sidebar.slider("Attack Onset Time (s)", 5.0, 15.0, 10.0, 1.0)
        save_gen_to_lib = st.sidebar.checkbox("Save generated mission to Library", value=False)
        
        if st.sidebar.button("🚀 Run Simulation & Diagnostics", type="primary") or "mission_data" not in st.session_state:
            with st.spinner(f"Simulating 6-DOF flight with {attack_type_name} injection..."):
                df_raw = gen.generate_single_mission(
                    mission_id=999,
                    attack_type=attack_idx,
                    duration_s=mission_duration,
                    dt=0.02,
                    attack_start_s=attack_onset,
                    attack_duration_s=12.0
                )
                if save_gen_to_lib:
                    gen_fname = f"gen_mission_{attack_type_name.lower()}_{int(time.time())}.csv"
                    df_raw.to_csv(mission_lib_dir / gen_fname, index=False)

                df_feat = compute_physics_residuals(df_raw)
                X_win, y_att, y_sen = create_sliding_windows(df_feat, window_size=50, stride=5)
                st.session_state["mission_data"] = df_feat
                st.session_state["windows"] = X_win
                st.session_state["true_attacks"] = y_att
                st.session_state["true_sensors"] = y_sen
                st.session_state["mission_preds"] = None
                st.session_state["scrubber_idx"] = 0
                st.session_state["current_mission_title"] = f"Generated {attack_type_name} Mission"

    else:
        test_path = PROJECT_ROOT / "data/test/test_telemetry.parquet"
        if test_path.exists():
            test_df = pd.read_parquet(test_path)
            mission_ids = sorted(test_df["mission_id"].unique())
            selected_m_id = st.sidebar.selectbox("Select Test Mission ID", mission_ids)
            if st.sidebar.button("Load Mission") or "mission_data" not in st.session_state:
                m_df = test_df[test_df["mission_id"] == selected_m_id].copy().reset_index(drop=True)
                df_feat = compute_physics_residuals(m_df)
                X_win, y_att, y_sen = create_sliding_windows(df_feat, window_size=50, stride=5)
                st.session_state["mission_data"] = df_feat
                st.session_state["windows"] = X_win
                st.session_state["true_attacks"] = y_att
                st.session_state["true_sensors"] = y_sen
                st.session_state["mission_preds"] = None
                st.session_state["scrubber_idx"] = 0
                st.session_state["current_mission_title"] = f"Test Parquet Mission {selected_m_id}"
        else:
            st.error("Test dataset not found.")
            return

    if "mission_data" not in st.session_state or len(st.session_state.get("windows", [])) == 0:
        st.info("Select or upload a flight mission in the sidebar to view diagnostics.")
        return

    df_feat = st.session_state["mission_data"]
    X_win = st.session_state["windows"]
    engine = load_engine()
    total_windows = len(X_win)

    # Precompute full mission ONNX predictions in batch (~50 ms)
    if st.session_state.get("mission_preds") is None or len(st.session_state.get("mission_preds", [])) != total_windows:
        st.session_state["mission_preds"] = [engine.predict(w) for w in X_win]
    all_preds = st.session_state["mission_preds"]

    # 1. Mission Overview Metric Cards
    mission_title = st.session_state.get("current_mission_title", "Active Flight Mission")
    st.markdown(f"#### 🛰️ Mission File: **{mission_title}**")
    
    total_time = df_feat["time"].iloc[-1] - df_feat["time"].iloc[0]
    detected_attack_types = [p["attack_type"] for p in all_preds if p["attack_detected"]]
    primary_attack = max(set(detected_attack_types), key=detected_attack_types.count) if detected_attack_types else "Normal (Nominal Flight)"
    attack_onset_win = next((i for i, p in enumerate(all_preds) if p["attack_detected"]), None)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Duration", f"{total_time:.1f} s", f"{len(df_feat):,} samples @ 50 Hz")
    with m_col2:
        st.metric("Attack Detected", primary_attack.replace("_", " "), "Isolated by AI" if detected_attack_types else "All Nominal")
    with m_col3:
        st.metric("Total Analysis Windows", f"{total_windows} windows", "1.0s window @ 0.1s stride")
    with m_col4:
        st.metric("Model Inference Latency", f"{all_preds[0]['latency_ms']:.2f} ms", "Real-Time Edge Capable")

    # 2. Whole-Mission Threat & Sensor Health Timeline Chart
    st.markdown("#### 🗺️ Full-Mission Threat Detection & Sensor Health Matrix")
    
    time_axis = np.array([i * 0.1 + 1.0 for i in range(total_windows)])
    conf_axis = np.array([p["confidence"] * 100.0 for p in all_preds])
    attack_mask = np.array([1 if p["attack_detected"] else 0 for p in all_preds])

    # Sensor matrix: 10 sensors x total_windows (1 = healthy, 0 = corrupted)
    sensor_keys_ordered = ["gps", "gyro_primary", "gyro_secondary", "accel", "pitot", "optical", "radar_alt", "mag", "baro", "c2_link"]
    sensor_display_labels = ["GPS", "GYRO1", "GYRO2", "ACCEL", "PITOT", "OPTICAL", "RADAR", "MAG", "BARO", "C2 LINK"]
    sensor_matrix = np.ones((len(sensor_keys_ordered), total_windows))
    for win_i, p in enumerate(all_preds):
        for s_i, s_k in enumerate(sensor_keys_ordered):
            if not p["sensor_health"].get(s_k, True):
                sensor_matrix[s_i, win_i] = 0

    curr_idx = st.session_state.get("scrubber_idx", 0)
    current_time = float(curr_idx * 0.1 + 1.0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True, gridspec_kw={'height_ratios': [2, 1.8]})
    fig.patch.set_facecolor('#0d1117')
    ax_top.set_facecolor('#161b22')
    ax_bot.set_facecolor('#161b22')

    # Top: Model Confidence Line
    top_colors = np.where(attack_mask == 1, '#ff4444', '#3fb950')
    ax_top.plot(time_axis, conf_axis, color='#58a6ff', lw=1.5, label='Model Confidence (%)')
    ax_top.scatter(time_axis, conf_axis, color=top_colors, s=12, zorder=3)
    ax_top.axvline(current_time, color='#f85149', linestyle='--', lw=2.0, label=f'Current Cursor: t={current_time:.1f}s')
    ax_top.set_ylabel('Confidence (%)', color='#c9d1d9', fontsize=9, weight='bold')
    ax_top.set_ylim(40, 102)
    ax_top.tick_params(colors='#c9d1d9')
    ax_top.legend(loc='lower left', fontsize=8)
    ax_top.set_title('Whole-Mission AI Threat Confidence Trace (Green: Nominal | Red: Attack Active)', color='#f0f6fc', fontsize=10, weight='bold')
    for spine in ax_top.spines.values(): spine.set_color('#30363d')

    # Bottom: 10-Sensor Health Heatmap
    cmap = ListedColormap(['#da3633', '#238636']) # 0 = red (corrupted), 1 = green (healthy)
    t_start, t_end = time_axis[0], time_axis[-1]
    ax_bot.imshow(sensor_matrix, aspect='auto', cmap=cmap, extent=[t_start, t_end, -0.5, len(sensor_keys_ordered) - 0.5], origin='lower')
    ax_bot.axvline(current_time, color='#f85149', linestyle='--', lw=2.0)
    ax_bot.set_yticks(range(len(sensor_keys_ordered)))
    ax_bot.set_yticklabels(sensor_display_labels, color='#c9d1d9', fontsize=8, weight='bold')
    ax_bot.set_xlabel('Mission Elapsed Time (seconds)', color='#c9d1d9', fontsize=9, weight='bold')
    ax_bot.tick_params(colors='#c9d1d9')
    for spine in ax_bot.spines.values(): spine.set_color('#30363d')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 3. Interactive Playback Scrubber & Step Navigation
    st.markdown("---")
    st.markdown("#### ⏱️ Mission Timeline Step Controls & Scrubber")
    
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 1.6])
    if b1.button("⏮ Start (0.0s)"):
        st.session_state["scrubber_idx"] = 0
    if b2.button("⏪ -1.0s"):
        st.session_state["scrubber_idx"] = max(0, curr_idx - 10)
    if b3.button("+1.0s ⏩"):
        st.session_state["scrubber_idx"] = min(total_windows - 1, curr_idx + 10)
    if b4.button("⏭ End"):
        st.session_state["scrubber_idx"] = total_windows - 1
    if attack_onset_win is not None:
        if b5.button(f"⚡ Jump to Attack ({attack_onset_win * 0.1 + 1.0:.1f}s)"):
            st.session_state["scrubber_idx"] = attack_onset_win

    active_idx = st.session_state.get("scrubber_idx", 0)
    scrubber_val = st.slider("Scrub Through Mission Telemetry", 0, total_windows - 1, value=active_idx, step=1)
    st.session_state["scrubber_idx"] = scrubber_val
    current_time = float(scrubber_val * 0.1 + 1.0)
    st.caption(f"Inspection Timestamp: **t = {current_time:.2f} s** (Window #{scrubber_val} of {total_windows})")

    # 4. Instant Diagnostics at Selected Timestamp
    diag = all_preds[scrubber_val]

    if not diag["attack_detected"]:
        st.markdown('<div class="threat-banner-clean">🟢 FLIGHT STATUS: NOMINAL — All Sensors Functioning Accurately</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="threat-banner-attack">🚨 THREAT DETECTED: {diag["attack_type"].replace("_", " ").upper()} (Confidence: {diag["confidence"]*100:.1f}%)</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📡 Sensor Health Trust Vector (Autopilot Input)")
        s_cols = st.columns(5)
        sensor_display_names = {
            "gps": "GNSS / GPS", "gyro_primary": "Primary Gyro", "gyro_secondary": "Secondary Gyro",
            "accel": "Accelerometers", "pitot": "Pitot Airspeed", "optical": "EO/IR Seeker",
            "radar_alt": "Radar Altimeter", "mag": "Magnetometer", "baro": "Barometer", "c2_link": "C2 Data Link"
        }
        for idx, (s_key, s_name) in enumerate(sensor_display_names.items()):
            col_target = s_cols[idx % 5]
            is_healthy = diag["sensor_health"].get(s_key, True)
            with col_target:
                if is_healthy:
                    st.markdown(f"**{s_name}**<br><span class='status-healthy'>✓ HEALTHY</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{s_name}**<br><span class='status-compromised'>⚠ CORRUPTED</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🤖 Autopilot Downstream Navigation Recommendation")
        if not diag["attack_detected"]:
            st.success("Standard Full-Sensor EKF Navigation: GPS + Primary IMU + Airspeed + Barometer.")
        elif diag["attack_type"] in ("GNSS_Spoofing", "GNSS_Jamming"):
            st.warning("🔄 **AUTOPILOT ACTION: DISENGAGE GPS FROM EKF**<br>Seamlessly switch to **Strapdown IMU Dead Reckoning** + Pitot Airspeed + Baro/Radar Altitude Hold.", unsafe_allow_html=True)
        elif diag["attack_type"] == "Acoustic_Gyro_Resonance":
            st.warning("🔄 **AUTOPILOT ACTION: ISOLATE PRIMARY GYROSCOPE**<br>Transfer attitude control to **Secondary Isolated MEMS Gyro** + Accelerometer Tilt.", unsafe_allow_html=True)
        elif diag["attack_type"] == "Optical_Seeker_Blinding":
            st.warning("🔄 **AUTOPILOT ACTION: ABORT OPTICAL HOMING**<br>Switch terminal guidance to coordinate-based GPS/IMU waypoint lock.", unsafe_allow_html=True)
        elif diag["attack_type"] == "Pitot_Tube_Fault":
            st.warning("🔄 **AUTOPILOT ACTION: DISENGAGE PITOT-STATIC FEED**<br>Estimate airspeed via GPS ground speed minus synthetic wind estimate; lock cruise throttle.", unsafe_allow_html=True)
        elif diag["attack_type"] == "Radar_Altimeter_Jamming":
            st.warning("🔄 **AUTOPILOT ACTION: IGNORE RADAR ALTIMETER**<br>Fall back to Barometric & GPS vertical channel altitude estimation.", unsafe_allow_html=True)
        elif diag["attack_type"] == "Magnetometer_Interference":
            st.warning("🔄 **AUTOPILOT ACTION: REJECT COMPASS YAW**<br>Derive true heading strictly from GNSS ground-track course vector and gyro integration.", unsafe_allow_html=True)
        elif diag["attack_type"] == "C2_Link_Jamming_DoS":
            st.error("🚨 **AUTOPILOT ACTION: AUTONOMOUS FAIL-SAFE ENGAGED**<br>C2 link severed. Isolate telemetry port and execute pre-programmed Autonomous Return-To-Launch (RTL).", unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📊 Attack Classification Probability Distribution")
        probs_df = pd.DataFrame({
            "Attack Class": list(diag["attack_probabilities"].keys()),
            "Probability (%)": [v * 100.0 for v in diag["attack_probabilities"].values()]
        }).sort_values("Probability (%)", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        colors = ["#da3633" if c == diag["attack_type"] and c != "Normal" else ("#238636" if c == "Normal" else "#58a6ff") for c in probs_df["Attack Class"]]
        ax.barh(probs_df["Attack Class"], probs_df["Probability (%)"], color=colors, edgecolor="#30363d", alpha=0.9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)", fontsize=9, weight="bold", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values(): spine.set_color('#30363d')
        for i, v in enumerate(probs_df["Probability (%)"]):
            if v > 1.5:
                ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8, weight="bold", color="#f0f6fc")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # 5. Multi-Sensor Telemetry & Physics Residual Graphs
    st.markdown("---")
    st.subheader("📈 Multi-Sensor Telemetry & Physics Invariant Residuals")

    tab_sub1, tab_sub2, tab_sub3 = st.tabs(["Aerodynamics & Altitude", "IMU & Acoustic Vibration", "RF Link & Magnetometer"])

    with tab_sub1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
            ax.plot(df_feat["time"], df_feat["pitot_ias"], label="Pitot Indicated Airspeed", color="#ff7f0e", lw=1.5)
            ax.plot(df_feat["time"], np.sqrt(df_feat["gps_vn"]**2 + df_feat["gps_ve"]**2), label="GPS Ground Speed", color="#1f77b4", lw=1.5)
            ax.axvline(current_time, color="#ff4444", linestyle="--", label="Current Cursor")
            ax.set_ylabel("Speed (m/s)", color="#c9d1d9"); ax.set_xlabel("Time (s)", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Airspeed vs Ground Speed (Wind Triangle Consistency)", color="#c9d1d9")
            for spine in ax.spines.values(): spine.set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
            ax.plot(df_feat["time"], df_feat["gps_alt"], label="GPS Alt", color="#1f77b4", lw=1.2)
            ax.plot(df_feat["time"], df_feat["baro_alt"], label="Baro Alt", color="#2ca02c", lw=1.2)
            ax.plot(df_feat["time"], df_feat["radar_alt_agl"], label="Radar AGL", color="#d62728", lw=1.2)
            ax.axvline(current_time, color="#ff4444", linestyle="--")
            ax.set_ylabel("Altitude (m)", color="#c9d1d9"); ax.set_xlabel("Time (s)", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Triple-Redundancy Altitude Comparison", color="#c9d1d9")
            for spine in ax.spines.values(): spine.set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab_sub2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
            ax.plot(df_feat["time"], df_feat["gyro_p"], label="Primary Gyro (Roll Rate p)", color="#d62728", lw=1.2)
            ax.plot(df_feat["time"], df_feat["gyro2_p"], label="Secondary Gyro (Roll Rate p)", color="#2ca02c", lw=1.2)
            ax.axvline(current_time, color="#ff4444", linestyle="--")
            ax.set_ylabel("Rate (rad/s)", color="#c9d1d9"); ax.set_xlabel("Time (s)", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Dual Gyroscope Telemetry (Resonance Divergence)", color="#c9d1d9")
            for spine in ax.spines.values(): spine.set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
            ax.plot(df_feat["time"], df_feat["gyro1_p_std_window"], label="Primary Gyro Vibration STD", color="#d62728", lw=1.5)
            ax.plot(df_feat["time"], df_feat["gyro2_p_std_window"], label="Secondary Gyro STD", color="#2ca02c", lw=1.5)
            ax.axvline(current_time, color="#ff4444", linestyle="--")
            ax.set_ylabel("Vibration Energy", color="#c9d1d9"); ax.set_xlabel("Time (s)", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("High-Frequency Gyroscope Resonant Vibration Metric", color="#c9d1d9")
            for spine in ax.spines.values(): spine.set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab_sub3:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
            ax.plot(df_feat["time"], df_feat["mag_norm"], label="Measured Field Norm ||B||", color="#9467bd", lw=1.5)
            ax.axhline(0.52, color="gray", linestyle=":", label="Geomagnetic Baseline (0.52 G)")
            ax.axvline(current_time, color="#ff4444", linestyle="--")
            ax.set_ylabel("Gauss", color="#c9d1d9"); ax.set_xlabel("Time (s)", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Geomagnetic Field Norm (Magnetic Interference Detector)", color="#c9d1d9")
            for spine in ax.spines.values(): spine.set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
            ax.plot(df_feat["time"], df_feat["c2_jitter_ms"], label="C2 Jitter (ms)", color="#e377c2", lw=1.2)
            ax.plot(df_feat["time"], df_feat["c2_loss_rate"] * 100.0, label="Packet Loss (%)", color="#17becf", lw=1.2)
            ax.axvline(current_time, color="#ff4444", linestyle="--")
            ax.set_ylabel("Latency / Loss", color="#c9d1d9"); ax.set_xlabel("Time (s)", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("C2 Databand Quality (Jamming & Telemetry Flood DoS)", color="#c9d1d9")
            for spine in ax.spines.values(): spine.set_color('#30363d')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Model Evaluation Figures
    st.markdown("---")
    st.subheader("📑 Edge Model Benchmark & Validation Suite")
    b_col1, b_col2 = st.columns(2)
    conf_path = PROJECT_ROOT / "evaluation/figures/confusion_matrix.png"
    sens_path = PROJECT_ROOT / "evaluation/figures/sensor_health_attribution.png"
    if conf_path.exists():
        with b_col1:
            st.image(str(conf_path), caption="Multi-Class Attack Confusion Matrix (Overall Accuracy: 95.93%)")
    if sens_path.exists():
        with b_col2:
            st.image(str(sens_path), caption="Per-Sensor Health Detection Metrics (Exact Match: 95.74%)")


def main():
    st.title("🛡️ Tactical UAV Cyber-Physical Intrusion Detection Suite")
    st.markdown("**Military-grade multi-task telemetry diagnostic suite: Attack Classification & Sensor Compromise Attribution**")

    tab1, tab2 = st.tabs([
        "🎮 Live 3D Playable Flight Arena (Cyber-Attack Simulator)",
        "📊 Historical Mission Replayer & Benchmark Analytics"
    ])

    with tab1:
        render_tab_playable_arena()

    with tab2:
        render_tab_historical_analytics()


if __name__ == "__main__":
    main()
