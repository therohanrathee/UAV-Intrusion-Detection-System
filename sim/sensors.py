"""
Multi-Modal Sensor Suite Simulation for Tactical Loitering Munitions.
Generates realistic noisy telemetry for:
- Primary & Secondary MEMS IMUs (with motor vibration harmonics & Allan variance bias drift)
- GNSS Receiver (coordinates, NED velocities, HDOP, C/N0, satellite counts)
- Pitot-Static Air Data Tube (Indicated Airspeed & Static Pressure)
- EO/IR Optical Seeker & Visual Odometry (Flow, Entropy, Tracking Quality)
- Radar / Laser Altimeter (AGL altitude & surface reflectivity)
- 3-Axis Magnetometer (Geomagnetic field decomposition)
- C2 Telemetry Databand Link (RSSI, Jitter, Packet Loss)
"""

import numpy as np
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class SensorSuite:
    """Simulates the comprehensive avionics sensor suite."""
    def __init__(self, config_path: Optional[str] = None):
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            # Fallback default parameters if config not loaded
            self.cfg = {
                "imu": {
                    "primary": {"accel": {"noise_density": 0.02, "bias_random_walk": 0.001},
                                "gyro": {"noise_density": 0.005, "bias_random_walk": 0.0002}},
                    "secondary": {"accel": {"noise_density": 0.035, "bias_random_walk": 0.002},
                                  "gyro": {"noise_density": 0.008, "bias_random_walk": 0.0004}}
                },
                "gnss": {"pos_std_horiz_m": 1.2, "pos_std_vert_m": 2.1, "vel_std_m_s": 0.04,
                         "nominal_sat_count": 16, "nominal_hdop": 0.95, "nominal_carrier_to_noise": 44.0},
                "pitot_static": {"airspeed_noise_std_m_s": 0.25, "static_pressure_noise_pa": 4.5},
                "optical_seeker": {"flow_noise_std_m_s": 0.04, "nominal_entropy": 0.88, "nominal_tracking_quality": 0.95},
                "radar_altimeter": {"range_noise_std_m": 0.035, "nominal_reflectivity": 0.92},
                "magnetometer": {"local_field_strength_gauss": 0.52, "noise_std_gauss": 0.008},
                "c2_link": {"nominal_jitter_ms": 3.5, "nominal_loss_rate": 0.015, "nominal_rssi_dbm": -58.0}
            }
        
        # Stochastic bias states (random-walk drift)
        self.accel1_bias = np.zeros(3)
        self.gyro1_bias = np.zeros(3)
        self.accel2_bias = np.zeros(3)
        self.gyro2_bias = np.zeros(3)
        
        # Origin reference coordinates (Lat: 34.0522° N, Lon: -118.2437° W)
        self.lat_ref = 34.0522
        self.lon_ref = -118.2437
        self.m_per_deg_lat = 111132.95
        self.m_per_deg_lon = 111132.95 * np.cos(np.radians(self.lat_ref))
        
        # Geomagnetic reference vector in NED (Gauss)
        # 0.52 Gauss total field with ~45 deg inclination
        b_norm = self.cfg["magnetometer"]["local_field_strength_gauss"]
        inc_rad = np.radians(45.0)
        self.b_ned = np.array([b_norm * np.cos(inc_rad), 0.0, b_norm * np.sin(inc_rad)])

    def step(self, state: Dict[str, Any], dt: float = 0.02) -> Dict[str, float]:
        """
        Ingests truth physical state from 6-DOF simulator and returns noisy sensor telemetry.
        """
        # 1. Update random-walk biases
        q_a1 = self.cfg["imu"]["primary"]["accel"]["bias_random_walk"] * np.sqrt(dt)
        q_g1 = self.cfg["imu"]["primary"]["gyro"]["bias_random_walk"] * np.sqrt(dt)
        self.accel1_bias += np.random.normal(0, q_a1, 3)
        self.gyro1_bias += np.random.normal(0, q_g1, 3)
        
        q_a2 = self.cfg["imu"]["secondary"]["accel"]["bias_random_walk"] * np.sqrt(dt)
        q_g2 = self.cfg["imu"]["secondary"]["gyro"]["bias_random_walk"] * np.sqrt(dt)
        self.accel2_bias += np.random.normal(0, q_a2, 3)
        self.gyro2_bias += np.random.normal(0, q_g2, 3)

        # 2. Motor vibration harmonics (80-240 Hz scaled by throttle)
        throttle = state["throttle"]
        motor_hz = 100.0 + 80.0 * throttle
        vib_phase = 2.0 * np.pi * motor_hz * state["time"]
        vib_accel = 0.4 * throttle * np.sin(vib_phase)
        vib_gyro = 0.02 * throttle * np.cos(vib_phase)
        
        # 3. Primary IMU
        sig_a1 = self.cfg["imu"]["primary"]["accel"]["noise_density"] / np.sqrt(dt)
        sig_g1 = self.cfg["imu"]["primary"]["gyro"]["noise_density"] / np.sqrt(dt)
        
        accel_x = state["accel_x_m_s2"] + self.accel1_bias[0] + np.random.normal(0, sig_a1) + vib_accel
        accel_y = state["accel_y_m_s2"] + self.accel1_bias[1] + np.random.normal(0, sig_a1)
        accel_z = state["accel_z_m_s2"] + self.accel1_bias[2] + np.random.normal(0, sig_a1) + vib_accel * 0.5
        
        gyro_p = state["gyro_p_rad_s"] + self.gyro1_bias[0] + np.random.normal(0, sig_g1) + vib_gyro
        gyro_q = state["gyro_q_rad_s"] + self.gyro1_bias[1] + np.random.normal(0, sig_g1)
        gyro_r = state["gyro_r_rad_s"] + self.gyro1_bias[2] + np.random.normal(0, sig_g1)

        # 4. Secondary IMU (dissimilar noise and isolated from prop vibration)
        sig_a2 = self.cfg["imu"]["secondary"]["accel"]["noise_density"] / np.sqrt(dt)
        sig_g2 = self.cfg["imu"]["secondary"]["gyro"]["noise_density"] / np.sqrt(dt)
        
        accel2_x = state["accel_x_m_s2"] + self.accel2_bias[0] + np.random.normal(0, sig_a2) + vib_accel * 0.2
        accel2_y = state["accel_y_m_s2"] + self.accel2_bias[1] + np.random.normal(0, sig_a2)
        accel2_z = state["accel_z_m_s2"] + self.accel2_bias[2] + np.random.normal(0, sig_a2) + vib_accel * 0.1
        
        gyro2_p = state["gyro_p_rad_s"] + self.gyro2_bias[0] + np.random.normal(0, sig_g2)
        gyro2_q = state["gyro_q_rad_s"] + self.gyro2_bias[1] + np.random.normal(0, sig_g2)
        gyro2_r = state["gyro_r_rad_s"] + self.gyro2_bias[2] + np.random.normal(0, sig_g2)

        # 5. GNSS
        cfg_gps = self.cfg["gnss"]
        gps_lat = self.lat_ref + (state["pos_n"] + np.random.normal(0, cfg_gps["pos_std_horiz_m"])) / self.m_per_deg_lat
        gps_lon = self.lon_ref + (state["pos_e"] + np.random.normal(0, cfg_gps["pos_std_horiz_m"])) / self.m_per_deg_lon
        gps_alt = state["altitude_m"] + np.random.normal(0, cfg_gps["pos_std_vert_m"])
        gps_vn = state["vel_n"] + np.random.normal(0, cfg_gps["vel_std_m_s"])
        gps_ve = state["vel_e"] + np.random.normal(0, cfg_gps["vel_std_m_s"])
        gps_vd = state["vel_d"] + np.random.normal(0, cfg_gps["vel_std_m_s"])
        gps_sats = int(np.clip(cfg_gps["nominal_sat_count"] + np.random.randint(-1, 2), 4, 24))
        gps_hdop = float(np.clip(cfg_gps["nominal_hdop"] + np.random.normal(0, 0.05), 0.7, 2.5))
        gps_cn0 = float(cfg_gps["nominal_carrier_to_noise"] + np.random.normal(0, 0.8))

        # 6. Pitot-Static Tube (Air Data)
        cfg_pitot = self.cfg["pitot_static"]
        tas = state["true_airspeed_m_s"]
        pitot_ias = float(max(tas + np.random.normal(0, cfg_pitot["airspeed_noise_std_m_s"]), 0.0))
        # Barometric standard atmosphere pressure formula: P = P0 * (1 - 2.25577e-5 * h)^5.25588
        h_baro = max(state["altitude_m"], 0.0)
        p_static = 101325.0 * ((1.0 - 2.25577e-5 * h_baro) ** 5.25588) + np.random.normal(0, cfg_pitot["static_pressure_noise_pa"])
        # Derived barometric altitude from static pressure
        baro_alt = (1.0 - (p_static / 101325.0)**(1.0 / 5.25588)) / 2.25577e-5

        # 7. EO/IR Optical Seeker / Visual Odometry
        cfg_opt = self.cfg["optical_seeker"]
        # Optical flow velocity roughly tracks body-horizontal velocity (vel_n, vel_e rotated to body)
        opt_vx = (state["vel_n"] * np.cos(np.radians(state["yaw_deg"])) + 
                  state["vel_e"] * np.sin(np.radians(state["yaw_deg"]))) + np.random.normal(0, cfg_opt["flow_noise_std_m_s"])
        opt_vy = (-state["vel_n"] * np.sin(np.radians(state["yaw_deg"])) + 
                   state["vel_e"] * np.cos(np.radians(state["yaw_deg"]))) + np.random.normal(0, cfg_opt["flow_noise_std_m_s"])
        opt_entropy = float(np.clip(cfg_opt["nominal_entropy"] + np.random.normal(0, 0.02), 0.1, 1.0))
        opt_quality = float(np.clip(cfg_opt["nominal_tracking_quality"] + np.random.normal(0, 0.015), 0.1, 1.0))

        # 8. Radar / Laser Altimeter (Above Ground Level)
        cfg_radar = self.cfg["radar_altimeter"]
        # In this terrain profile, terrain elevation is around 0m, so AGL is approx state["altitude_m"]
        radar_alt = float(max(state["altitude_m"] + np.random.normal(0, cfg_radar["range_noise_std_m"]), 0.0))
        radar_refl = float(np.clip(cfg_radar["nominal_reflectivity"] + np.random.normal(0, 0.02), 0.1, 1.0))

        # 9. Magnetometer
        cfg_mag = self.cfg["magnetometer"]
        phi = np.radians(state["roll_deg"])
        theta = np.radians(state["pitch_deg"])
        psi = np.radians(state["yaw_deg"])
        # Rotate b_ned into body frame: R_ned2b = R_b2ned.T
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_ps, s_ps = np.cos(psi), np.sin(psi)
        R_b2ned = np.array([
            [c_t * c_ps, s_p * s_t * c_ps - c_p * s_ps, c_p * s_t * c_ps + s_p * s_ps],
            [c_t * s_ps, s_p * s_t * s_ps + c_p * c_ps, c_p * s_t * s_ps - s_p * c_ps],
            [-s_t,       s_p * c_t,                     c_p * c_t]
        ])
        b_body = R_b2ned.T @ self.b_ned
        mag_x = b_body[0] + np.random.normal(0, cfg_mag["noise_std_gauss"])
        mag_y = b_body[1] + np.random.normal(0, cfg_mag["noise_std_gauss"])
        mag_z = b_body[2] + np.random.normal(0, cfg_mag["noise_std_gauss"])
        mag_norm = float(np.sqrt(mag_x**2 + mag_y**2 + mag_z**2))

        # 10. C2 Link Telemetry
        cfg_c2 = self.cfg["c2_link"]
        c2_jitter = float(max(cfg_c2["nominal_jitter_ms"] + np.random.normal(0, 0.8), 0.5))
        c2_loss = float(np.clip(cfg_c2["nominal_loss_rate"] + np.random.normal(0, 0.005), 0.0, 1.0))
        c2_rssi = float(cfg_c2["nominal_rssi_dbm"] + np.random.normal(0, 1.2))

        return {
            "time": state["time"],
            # Primary IMU
            "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
            "gyro_p": gyro_p, "gyro_q": gyro_q, "gyro_r": gyro_r,
            # Secondary IMU
            "accel2_x": accel2_x, "accel2_y": accel2_y, "accel2_z": accel2_z,
            "gyro2_p": gyro2_p, "gyro2_q": gyro2_q, "gyro2_r": gyro2_r,
            # GNSS
            "gps_lat": gps_lat, "gps_lon": gps_lon, "gps_alt": gps_alt,
            "gps_vn": gps_vn, "gps_ve": gps_ve, "gps_vd": gps_vd,
            "gps_sats": gps_sats, "gps_hdop": gps_hdop, "gps_cn0": gps_cn0,
            # Air Data & Baro
            "pitot_ias": pitot_ias, "static_pressure": p_static, "baro_alt": baro_alt,
            # Optical Seeker
            "optical_flow_vx": opt_vx, "optical_flow_vy": opt_vy,
            "optical_entropy": opt_entropy, "optical_quality": opt_quality,
            # Radar Altimeter
            "radar_alt_agl": radar_alt, "radar_reflectivity": radar_refl,
            # Magnetometer
            "mag_x": mag_x, "mag_y": mag_y, "mag_z": mag_z, "mag_norm": mag_norm,
            # C2 Link
            "c2_jitter_ms": c2_jitter, "c2_loss_rate": c2_loss, "c2_rssi_dbm": c2_rssi,
        }
