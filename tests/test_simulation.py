"""
Unit tests for 6-DOF simulation engine and attack injection.
"""

import pytest
import numpy as np
import pandas as pd

from sim.airframe import LoiteringMunitionSim, AirframeParams
from sim.sensors import SensorSuite
from sim.attacks import AttackInjector, SENSOR_KEYS, ATTACK_NAMES
from sim.generator import MissionGenerator


def test_airframe_physics_integration():
    """Verify 6-DOF airframe dynamics step forward stably."""
    sim = LoiteringMunitionSim(dt=0.02)
    initial_alt = sim.pos_ned[2]
    
    # Step for 2 seconds (100 steps)
    for _ in range(100):
        sim.step(target_roll=0.0, target_pitch=0.0, target_speed=32.0)
        
    state = sim.get_state_dict()
    assert state["time"] == pytest.approx(2.0, abs=1e-3)
    assert 20.0 < state["speed_m_s"] < 45.0
    assert 100.0 < state["altitude_m"] < 250.0
    assert np.isfinite(state["accel_x_m_s2"])
    assert np.isfinite(state["gyro_p_rad_s"])


def test_sensor_suite_generation():
    """Verify all sensor modalities produce finite readings."""
    sim = LoiteringMunitionSim(dt=0.02)
    sensors = SensorSuite()
    
    sim.step()
    state = sim.get_state_dict()
    telemetry = sensors.step(state, dt=0.02)
    
    required_keys = [
        "accel_x", "accel_y", "accel_z", "gyro_p", "gyro_q", "gyro_r",
        "accel2_x", "accel2_y", "accel2_z", "gyro2_p", "gyro2_q", "gyro2_r",
        "gps_lat", "gps_lon", "gps_alt", "gps_vn", "gps_ve", "gps_vd",
        "pitot_ias", "optical_entropy", "radar_alt_agl", "mag_norm", "c2_loss_rate"
    ]
    for key in required_keys:
        assert key in telemetry
        assert np.isfinite(telemetry[key])


def test_attack_injections():
    """Verify all 8 attack types activate and set compromised flags correctly."""
    sim = LoiteringMunitionSim(dt=0.02)
    sensors = SensorSuite()
    sim.step()
    state = sim.get_state_dict()
    nominal_telem = sensors.step(state, dt=0.02)
    nominal_telem["time"] = 15.0  # Set to active time window
    
    # Test each attack
    for attack_id in range(1, len(ATTACK_NAMES)):
        injector = AttackInjector(attack_id=attack_id, start_time=10.0, duration=10.0)
        mod_telem, active_id, comp = injector.apply(nominal_telem, dt=0.02)
        
        assert active_id == attack_id
        # At least one sensor must be marked compromised
        assert sum(comp.values()) >= 1
        
        # Attack-specific assertions
        if attack_id == 1:  # GNSS Spoofing
            assert comp["gps"] == 1
        elif attack_id == 3:  # Gyro Resonance
            assert comp["gyro_primary"] == 1
            assert comp["gyro_secondary"] == 0  # Secondary must remain healthy!
        elif attack_id == 4:  # Optical Blinding
            assert comp["optical"] == 1
            assert mod_telem["optical_entropy"] < 0.1
        elif attack_id == 7:  # Mag Interference
            assert comp["mag"] == 1
            assert mod_telem["mag_norm"] > 1.0
        elif attack_id == 8:  # C2 Jamming
            assert comp["c2_link"] == 1
            assert mod_telem["c2_loss_rate"] > 0.5


def test_mission_generator():
    """Verify single mission generator produces complete dataframe with all columns."""
    gen = MissionGenerator()
    df = gen.generate_single_mission(mission_id=1, attack_type=3, duration_s=4.0, dt=0.02, attack_start_s=1.0, attack_duration_s=2.0)
    
    assert len(df) == 200  # 4.0s / 0.02s
    assert "attack_id" in df.columns
    assert "compromised_gps" in df.columns
    assert "compromised_gyro_primary" in df.columns
    assert 3 in df["attack_id"].values
    assert 0 in df["attack_id"].values
