"""
Physics-Informed Invariant Residuals & Multi-Sensor Cross-Checks.
Computes:
1. Kinematic consistency residuals: |v_dot_gps - a_ned|
2. Aerodynamic airspeed vs groundspeed residuals: |v_gps - v_pitot|
3. Triple-redundancy altitude residuals: |h_gps - h_baro|, |h_baro - h_radar|, |h_gps - h_radar|
4. Dual IMU cross-discrepancy: |gyro_1 - gyro_2|, |accel_1 - accel_2|
5. Geomagnetic field norm invariant: |B_norm - B_WMM|
6. High-frequency acoustic resonance variance features
7. Visual seeker entropy & tracking quality metrics
8. C2 RF & buffer exhaustion metrics
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


FEATURE_COLUMNS = [
    # Kinematic & Aerodynamic Invariants
    "res_kinematic_horiz",
    "res_kinematic_vert",
    "res_airspeed_vs_groundspeed",
    # Altitude Redundancy (3-Way Voting)
    "res_alt_gps_baro",
    "res_alt_baro_radar",
    "res_alt_gps_radar",
    # Dual IMU Consistency
    "res_gyro_discrepancy",
    "res_accel_discrepancy",
    # High-Frequency Acoustic Resonance Signatures
    "gyro1_p_std_window",
    "gyro1_q_std_window",
    "gyro1_r_std_window",
    "gyro2_p_std_window",
    # GNSS Signal Quality
    "gps_sats",
    "gps_hdop",
    "gps_cn0",
    # Optical Seeker Integrity
    "optical_entropy",
    "optical_quality",
    "res_optical_flow_mag",
    # Radar Altimeter Health
    "radar_reflectivity",
    # Magnetometer Invariant
    "res_mag_norm",
    # C2 Link Telemetry
    "c2_jitter_ms",
    "c2_loss_rate",
    "c2_rssi_dbm",
    # Primary telemetry signals
    "accel_x", "accel_y", "accel_z",
    "gyro_p", "gyro_q", "gyro_r",
    "pitot_ias",
    "baro_alt",
    "radar_alt_agl",
]


def compute_physics_residuals(df: pd.DataFrame, window_size: int = 25) -> pd.DataFrame:
    """
    Computes invariant physical residuals and windowed statistical features for a telemetry dataframe.
    Processes mission by mission to ensure temporal causality and prevent data leakage.
    """
    df = df.copy()
    if "mission_id" not in df.columns:
        df["mission_id"] = 1
    if "time" not in df.columns:
        df["time"] = np.arange(len(df)) * 0.02

    processed_dfs = []
    
    # Process each mission independently
    for mission_id, group in df.groupby("mission_id", sort=False):
        m_df = group.copy().sort_values("time").reset_index(drop=True)
        dt = 0.02
        
        # 1. Kinematic Acceleration from GPS velocities: a_gps = d(v_gps)/dt
        v_n = m_df["gps_vn"].values
        v_e = m_df["gps_ve"].values
        v_d = m_df["gps_vd"].values
        
        # Smoothed numerical gradient
        a_n_gps = np.gradient(v_n, dt)
        a_e_gps = np.gradient(v_e, dt)
        a_d_gps = np.gradient(v_d, dt)
        
        # Accelerometer specific force in body rotated to approximate NED
        # using roll, pitch, yaw from attitude sensors
        # For fast invariant calculation: |a_horiz_gps| vs |a_horiz_imu|
        a_horiz_gps = np.sqrt(a_n_gps**2 + a_e_gps**2)
        a_horiz_imu = np.sqrt(m_df["accel_x"].values**2 + m_df["accel_y"].values**2)
        m_df["res_kinematic_horiz"] = np.abs(a_horiz_gps - a_horiz_imu)
        m_df["res_kinematic_vert"] = np.abs(a_d_gps - (m_df["accel_z"].values + 9.81))
        
        # 2. Aerodynamic Airspeed vs GPS Ground Speed
        v_ground = np.sqrt(v_n**2 + v_e**2)
        m_df["res_airspeed_vs_groundspeed"] = np.abs(m_df["pitot_ias"].values - v_ground)
        
        # 3. Triple Altitude Redundancy
        h_gps = m_df["gps_alt"].values
        h_baro = m_df["baro_alt"].values
        h_radar = m_df["radar_alt_agl"].values
        m_df["res_alt_gps_baro"] = np.abs(h_gps - h_baro)
        m_df["res_alt_baro_radar"] = np.abs(h_baro - h_radar)
        m_df["res_alt_gps_radar"] = np.abs(h_gps - h_radar)
        
        # 4. Dual IMU Discrepancy
        d_p = m_df["gyro_p"].values - m_df["gyro2_p"].values
        d_q = m_df["gyro_q"].values - m_df["gyro2_q"].values
        d_r = m_df["gyro_r"].values - m_df["gyro2_r"].values
        m_df["res_gyro_discrepancy"] = np.sqrt(d_p**2 + d_q**2 + d_r**2)
        
        d_ax = m_df["accel_x"].values - m_df["accel2_x"].values
        d_ay = m_df["accel_y"].values - m_df["accel2_y"].values
        d_az = m_df["accel_z"].values - m_df["accel2_z"].values
        m_df["res_accel_discrepancy"] = np.sqrt(d_ax**2 + d_ay**2 + d_az**2)
        
        # 5. Acoustic Resonance Vibration: Rolling standard deviation over window (0.5s = 25 samples)
        m_df["gyro1_p_std_window"] = m_df["gyro_p"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
        m_df["gyro1_q_std_window"] = m_df["gyro_q"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
        m_df["gyro1_r_std_window"] = m_df["gyro_r"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
        m_df["gyro2_p_std_window"] = m_df["gyro2_p"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
        
        # 6. Optical Seeker Flow Magnitude
        m_df["res_optical_flow_mag"] = np.sqrt(m_df["optical_flow_vx"].values**2 + m_df["optical_flow_vy"].values**2)
        
        # 7. Magnetometer Geomagnetic Norm Residual (|B| - 0.52 Gauss)
        m_df["res_mag_norm"] = np.abs(m_df["mag_norm"].values - 0.52)
        
        processed_dfs.append(m_df)
        
    return pd.concat(processed_dfs, ignore_index=True)
