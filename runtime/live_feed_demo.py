"""
Real-Time Live Streaming Demonstration.
Simulates a live drone companion computer receiving 50 Hz telemetry packet-by-packet,
maintaining a rolling 1.0s FIFO buffer, running real-time ONNX inference,
and alerting the autopilot the millisecond an attack occurs.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Any

from sim.airframe import LoiteringMunitionSim, AirframeParams
from sim.sensors import SensorSuite
from sim.attacks import AttackInjector, ATTACK_NAMES, SENSOR_KEYS
from runtime.inference import UAVInferenceEngine
from pipeline.residuals import FEATURE_COLUMNS


def run_live_flight_stream(attack_choice: int = 1, flight_seconds: float = 20.0, playback_speed: float = 2.0):
    """
    Simulates a live flight with streaming sensor ingestion.
    """
    print("=" * 75)
    print("🚁 LIVE DRONE TELEMETRY STREAM & REAL-TIME INTRUSION MONITOR")
    print(f"Target Attack: {ATTACK_NAMES[attack_choice]} (Scheduled at t=8.0s)")
    print(f"Companion Computer Inference Engine: weights/uav_ids_model.onnx")
    print("=" * 75)

    # 1. Initialize Simulator & Hardware Sensors
    sim = LoiteringMunitionSim(dt=0.02)
    sensors = SensorSuite()
    injector = AttackInjector(attack_id=attack_choice, start_time=8.0, duration=8.0)
    
    # 2. Initialize Standalone ONNX Inference Engine
    engine = UAVInferenceEngine()

    # 3. Live 1.0-second FIFO Buffer (50 samples @ 50 Hz)
    buffer_len = 50
    live_buffer = deque(maxlen=buffer_len)
    
    # Previous sensor state for numerical gradients
    prev_gps_vn = 0.0
    prev_gps_ve = 0.0
    prev_gps_vd = 0.0
    gyro_p_history = deque(maxlen=25)
    gyro_q_history = deque(maxlen=25)
    gyro_r_history = deque(maxlen=25)
    gyro2_p_history = deque(maxlen=25)

    dt = 0.02
    steps = int(flight_seconds / dt)
    attack_alert_triggered = False

    print(f"{'Time':<8} | {'Status':<14} | {'Threat Detected':<22} | {'Confidence':<10} | {'Compromised Sensors'}")
    print("-" * 85)

    for step in range(steps):
        t = sim.time
        # Autopilot flight maneuvers
        target_roll = 0.2 * np.sin(0.15 * t)
        target_pitch = 0.03 * np.cos(0.1 * t)
        sim.step(target_roll=target_roll, target_pitch=target_pitch, target_speed=32.0)
        
        # Real-time noisy sensor packet
        state = sim.get_state_dict()
        raw_packet = sensors.step(state, dt=dt)
        mod_packet, true_att_id, true_comp = injector.apply(raw_packet, dt=dt)

        # 4. Compute Invariant Residuals on the fly for this single packet
        a_n_gps = (mod_packet["gps_vn"] - prev_gps_vn) / dt
        a_e_gps = (mod_packet["gps_ve"] - prev_gps_ve) / dt
        a_d_gps = (mod_packet["gps_vd"] - prev_gps_vd) / dt
        prev_gps_vn = mod_packet["gps_vn"]
        prev_gps_ve = mod_packet["gps_ve"]
        prev_gps_vd = mod_packet["gps_vd"]

        a_horiz_gps = np.sqrt(a_n_gps**2 + a_e_gps**2)
        a_horiz_imu = np.sqrt(mod_packet["accel_x"]**2 + mod_packet["accel_y"]**2)
        res_kin_horiz = abs(a_horiz_gps - a_horiz_imu)
        res_kin_vert = abs(a_d_gps - (mod_packet["accel_z"] + 9.81))

        v_ground = np.sqrt(mod_packet["gps_vn"]**2 + mod_packet["gps_ve"]**2)
        res_aero = abs(mod_packet["pitot_ias"] - v_ground)

        res_alt_gb = abs(mod_packet["gps_alt"] - mod_packet["baro_alt"])
        res_alt_br = abs(mod_packet["baro_alt"] - mod_packet["radar_alt_agl"])
        res_alt_gr = abs(mod_packet["gps_alt"] - mod_packet["radar_alt_agl"])

        d_p = mod_packet["gyro_p"] - mod_packet["gyro2_p"]
        d_q = mod_packet["gyro_q"] - mod_packet["gyro2_q"]
        d_r = mod_packet["gyro_r"] - mod_packet["gyro2_r"]
        res_gyro_disc = np.sqrt(d_p**2 + d_q**2 + d_r**2)

        d_ax = mod_packet["accel_x"] - mod_packet["accel2_x"]
        d_ay = mod_packet["accel_y"] - mod_packet["accel2_y"]
        d_az = mod_packet["accel_z"] - mod_packet["accel2_z"]
        res_accel_disc = np.sqrt(d_ax**2 + d_ay**2 + d_az**2)

        gyro_p_history.append(mod_packet["gyro_p"])
        gyro_q_history.append(mod_packet["gyro_q"])
        gyro_r_history.append(mod_packet["gyro_r"])
        gyro2_p_history.append(mod_packet["gyro2_p"])

        g1_p_std = float(np.std(gyro_p_history)) if len(gyro_p_history) > 1 else 0.0
        g1_q_std = float(np.std(gyro_q_history)) if len(gyro_q_history) > 1 else 0.0
        g1_r_std = float(np.std(gyro_r_history)) if len(gyro_r_history) > 1 else 0.0
        g2_p_std = float(np.std(gyro2_p_history)) if len(gyro2_p_history) > 1 else 0.0

        res_opt_mag = np.sqrt(mod_packet["optical_flow_vx"]**2 + mod_packet["optical_flow_vy"]**2)
        res_mag_norm = abs(mod_packet["mag_norm"] - 0.52)

        # Build feature vector matching FEATURE_COLUMNS
        feat_vector = np.array([
            res_kin_horiz, res_kin_vert, res_aero,
            res_alt_gb, res_alt_br, res_alt_gr,
            res_gyro_disc, res_accel_disc,
            g1_p_std, g1_q_std, g1_r_std, g2_p_std,
            mod_packet["gps_sats"], mod_packet["gps_hdop"], mod_packet["gps_cn0"],
            mod_packet["optical_entropy"], mod_packet["optical_quality"], res_opt_mag,
            mod_packet["radar_reflectivity"], res_mag_norm,
            mod_packet["c2_jitter_ms"], mod_packet["c2_loss_rate"], mod_packet["c2_rssi_dbm"],
            mod_packet["accel_x"], mod_packet["accel_y"], mod_packet["accel_z"],
            mod_packet["gyro_p"], mod_packet["gyro_q"], mod_packet["gyro_r"],
            mod_packet["pitot_ias"], mod_packet["baro_alt"], mod_packet["radar_alt_agl"],
        ], dtype=np.float32)

        # Push to sliding buffer
        live_buffer.append(feat_vector)

        # Once buffer is filled (1.0 second elapsed), run real-time inference every 5 steps (10 Hz)
        if len(live_buffer) == buffer_len and step % 5 == 0:
            window_matrix = np.array(live_buffer)  # (50, 32)
            diag = engine.predict(window_matrix)
            
            status_str = "🟢 NOMINAL" if not diag["attack_detected"] else "🚨 ATTACK"
            bad_sensors = [s for s, ok in diag["sensor_health"].items() if not ok]
            bad_str = ", ".join(bad_sensors) if bad_sensors else "None (All Healthy)"

            # Print telemetry tick
            print(f"t={t:5.2f}s | {status_str:<14} | {diag['attack_type']:<22} | {diag['confidence']*100:5.1f}%     | {bad_str}")

            if diag["attack_detected"] and not attack_alert_triggered:
                attack_alert_triggered = True
                print(">>>" + "!" * 80)
                print(f">>> [AUTOPILOT FAIL-SAFE ENGAGED AT t={t:.2f}s]")
                print(f">>> Identified Threat : {diag['attack_type']}")
                print(f">>> Isolating Sensor  : {bad_str}")
                if "gps" in bad_sensors:
                    print(">>> Navigation Action : DECOUPLING GPS FROM EKF -> INITIATING IMU DEAD RECKONING")
                elif "gyro_primary" in bad_sensors:
                    print(">>> Attitude Action   : TRANSFERRING TO SECONDARY DISSIMILAR GYROSCOPE")
                print(">>>" + "!" * 80)

        # Optional sleep for real-time simulation visualization (accelerated by playback_speed)
        time.sleep(dt / playback_speed)


if __name__ == "__main__":
    import sys
    att = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_live_flight_stream(attack_choice=att, flight_seconds=16.0, playback_speed=15.0)
