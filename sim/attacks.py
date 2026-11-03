"""
Electronic Warfare & Cyber-Physical Attack Injectors for Tactical UAV Telemetry.
Injects:
- GNSS Spoofing (progressive drift & position step)
- GNSS Broadband Jamming (satellite drop & HDOP spike)
- Acoustic MEMS Gyroscope Resonance (DC bias shift & severe vibration)
- Optical Seeker Laser Blinding (entropy collapse & tracking loss)
- Pitot-Static Tube Obstruction / Icing (airspeed freeze / collapse)
- Radar Altimeter Jamming (false return & reflectivity drop)
- Magnetometer Interference (magnetic field norm distortion)
- C2 Link Jamming / Buffer Exhaustion DoS (packet loss & jitter explosion)
"""

import numpy as np
from typing import Dict, Any, Tuple, List
import copy


SENSOR_KEYS = [
    "gps",
    "gyro_primary",
    "gyro_secondary",
    "accel",
    "pitot",
    "optical",
    "radar_alt",
    "mag",
    "baro",
    "c2_link",
]

ATTACK_NAMES = [
    "Normal",
    "GNSS_Spoofing",
    "GNSS_Jamming",
    "Acoustic_Gyro_Resonance",
    "Optical_Seeker_Blinding",
    "Pitot_Tube_Fault",
    "Radar_Altimeter_Jamming",
    "Magnetometer_Interference",
    "C2_Link_Jamming_DoS",
]


class AttackInjector:
    """Manages attack scheduling and sensor corruption injection."""
    def __init__(self, attack_id: int, start_time: float, duration: float, params: Dict[str, Any] = None):
        self.attack_id = attack_id
        self.attack_name = ATTACK_NAMES[attack_id] if 0 <= attack_id < len(ATTACK_NAMES) else "Unknown"
        self.start_time = start_time
        self.duration = duration
        self.end_time = start_time + duration
        self.params = params or {}
        
        # Internal persistent attack states (for ramps, integrals)
        self.drift_n = 0.0
        self.drift_e = 0.0
        self.drift_alt = 0.0
        self.frozen_airspeed = None

    def is_active(self, current_time: float) -> bool:
        return self.start_time <= current_time < self.end_time

    def apply(self, telemetry: Dict[str, Any], dt: float = 0.02) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
        """
        Applies attack injection if active at current time.
        Returns:
            modified_telemetry: dict of sensor telemetry
            active_attack_id: 0 if inactive/normal, else attack_id
            compromised_sensors: dict mapping sensor_name -> 0 (healthy) or 1 (compromised)
        """
        t = telemetry["time"]
        t_telemetry = copy.deepcopy(telemetry)
        
        # Default all sensors healthy
        compromised = {k: 0 for k in SENSOR_KEYS}
        
        if not self.is_active(t) or self.attack_id == 0:
            return t_telemetry, 0, compromised

        # Active attack
        elapsed = t - self.start_time
        active_id = self.attack_id

        if self.attack_id == 1:
            # 1. GNSS Spoofing (Gradual velocity drift + false position accumulation)
            drift_rate = self.params.get("drift_rate_m_s", 2.8)
            drift_dir_n = 0.8
            drift_dir_e = 0.6
            self.drift_n += drift_dir_n * drift_rate * dt
            self.drift_e += drift_dir_e * drift_rate * dt
            
            # 1 deg lat ~ 111132.95 m
            m_lat = 111132.95
            m_lon = 111132.95 * np.cos(np.radians(t_telemetry["gps_lat"]))
            t_telemetry["gps_lat"] += self.drift_n / m_lat
            t_telemetry["gps_lon"] += self.drift_e / m_lon
            t_telemetry["gps_vn"] += drift_dir_n * drift_rate
            t_telemetry["gps_ve"] += drift_dir_e * drift_rate
            compromised["gps"] = 1

        elif self.attack_id == 2:
            # 2. GNSS Jamming (Broadband RF noise, satellite loss, HDOP spike)
            t_telemetry["gps_sats"] = int(np.random.choice([0, 1, 2]))
            t_telemetry["gps_hdop"] = float(np.random.uniform(16.0, 24.0))
            t_telemetry["gps_cn0"] = float(np.random.uniform(14.0, 22.0))
            # Large position wander under jamming
            t_telemetry["gps_vn"] += np.random.normal(0, 4.5)
            t_telemetry["gps_ve"] += np.random.normal(0, 4.5)
            compromised["gps"] = 1

        elif self.attack_id == 3:
            # 3. Acoustic MEMS Gyroscope Resonance
            # High intensity ultrasound matches primary gyro resonant frequency (9.25 kHz)
            dc_shift = self.params.get("dc_bias_shift_rad_s", 3.4)  # ~195 deg/s false rate
            t_telemetry["gyro_p"] += dc_shift + np.random.normal(0, 1.8)
            t_telemetry["gyro_q"] += -dc_shift * 0.7 + np.random.normal(0, 1.5)
            t_telemetry["gyro_r"] += np.random.normal(0, 2.2)
            # Notice: gyro2 (secondary IMU) is unaffected because it's isolated/dissimilar!
            compromised["gyro_primary"] = 1

        elif self.attack_id == 4:
            # 4. Optical Seeker Blinding (Laser Dazzler / Dense Aerosol)
            t_telemetry["optical_entropy"] = float(np.random.uniform(0.01, 0.05))
            t_telemetry["optical_quality"] = float(np.random.uniform(0.0, 0.03))
            t_telemetry["optical_flow_vx"] = 0.0
            t_telemetry["optical_flow_vy"] = 0.0
            compromised["optical"] = 1

        elif self.attack_id == 5:
            # 5. Pitot Tube Obstruction / Icing Fault
            if self.frozen_airspeed is None:
                self.frozen_airspeed = t_telemetry["pitot_ias"]
            mode = self.params.get("mode", "freeze")
            if mode == "freeze":
                t_telemetry["pitot_ias"] = self.frozen_airspeed + np.random.normal(0, 0.05)
            else:
                t_telemetry["pitot_ias"] = float(np.random.uniform(0.0, 2.0))  # Near-zero collapse
            compromised["pitot"] = 1

        elif self.attack_id == 6:
            # 6. Radar Altimeter Jamming
            false_alt_offset = self.params.get("false_alt_offset_m", 48.0)
            t_telemetry["radar_alt_agl"] += false_alt_offset + np.random.normal(0, 2.5)
            t_telemetry["radar_reflectivity"] = float(np.random.uniform(0.05, 0.15))
            compromised["radar_alt"] = 1

        elif self.attack_id == 7:
            # 7. Magnetometer Interference
            mult = self.params.get("field_mult", 3.4)
            t_telemetry["mag_x"] *= mult
            t_telemetry["mag_y"] += 1.2
            t_telemetry["mag_norm"] = float(np.sqrt(t_telemetry["mag_x"]**2 + t_telemetry["mag_y"]**2 + t_telemetry["mag_z"]**2))
            compromised["mag"] = 1

        elif self.attack_id == 8:
            # 8. C2 Link Jamming / Buffer Exhaustion DoS
            t_telemetry["c2_loss_rate"] = float(np.clip(np.random.uniform(0.80, 0.98), 0.0, 1.0))
            t_telemetry["c2_jitter_ms"] = float(np.random.uniform(140.0, 260.0))
            t_telemetry["c2_rssi_dbm"] = float(np.random.uniform(-110.0, -100.0))
            compromised["c2_link"] = 1

        return t_telemetry, active_id, compromised
