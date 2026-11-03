"""
Mission Dataset Generator with Domain Randomization.
Simulates diverse combat missions across all attack vectors and exports
publication-grade datasets (Parquet / CSV).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

from sim.airframe import LoiteringMunitionSim, AirframeParams
from sim.sensors import SensorSuite
from sim.attacks import AttackInjector, SENSOR_KEYS, ATTACK_NAMES


class MissionGenerator:
    """Generates synthetic, physics-grounded flight missions with domain randomization."""
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.sensor_config_path = str(self.config_dir / "sensors_config.yaml")

    def generate_single_mission(
        self,
        mission_id: int,
        attack_type: int = 0,
        duration_s: float = 40.0,
        dt: float = 0.02,
        attack_start_s: Optional[float] = None,
        attack_duration_s: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Runs one complete flight mission with domain randomization.
        """
        # 1. Domain Randomization: Airframe & Environment
        mass = float(np.random.uniform(3.5, 4.9))  # Mass variation (+/- 18%)
        max_thrust = float(np.random.uniform(35.0, 42.0))
        cruise_speed = float(np.random.uniform(28.0, 36.0))
        params = AirframeParams(mass_kg=mass, max_thrust_n=max_thrust, cruise_speed_m_s=cruise_speed)
        
        # Wind domain randomization
        wind_speed = float(np.random.uniform(1.5, 6.5))
        wind_angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        steady_wind = np.array([wind_speed * np.cos(wind_angle), wind_speed * np.sin(wind_angle), 0.0])
        
        sim = LoiteringMunitionSim(params=params, dt=dt, steady_wind=steady_wind)
        sensors = SensorSuite(config_path=self.sensor_config_path)
        
        # 2. Attack scheduling
        if attack_type != 0:
            if attack_start_s is None:
                attack_start_s = float(np.random.uniform(10.0, 16.0))
            if attack_duration_s is None:
                attack_duration_s = float(np.random.uniform(15.0, 20.0))
            injector = AttackInjector(attack_id=attack_type, start_time=attack_start_s, duration=attack_duration_s)
        else:
            injector = AttackInjector(attack_id=0, start_time=999.0, duration=0.0)

        # 3. Flight maneuver schedule (Loiter turns, pitch undulations)
        # Randomize roll/pitch cycle frequencies
        roll_freq = float(np.random.uniform(0.06, 0.12))  # Period 8-16s
        pitch_freq = float(np.random.uniform(0.04, 0.08))
        target_roll_amp = float(np.radians(np.random.uniform(12.0, 25.0)))
        target_pitch_amp = float(np.radians(np.random.uniform(3.0, 7.0)))
        
        steps = int(duration_s / dt)
        records = []

        for step in range(steps):
            t = sim.time
            # Dynamic flight control targets
            target_roll = target_roll_amp * np.sin(2.0 * np.pi * roll_freq * t)
            target_pitch = target_pitch_amp * np.cos(2.0 * np.pi * pitch_freq * t)
            
            # Step physical dynamics
            sim.step(target_roll=target_roll, target_pitch=target_pitch, target_speed=cruise_speed)
            state = sim.get_state_dict()
            
            # Generate raw noisy sensor telemetry
            raw_telemetry = sensors.step(state, dt=dt)
            
            # Inject cyber-physical attack if active
            mod_telemetry, active_attack_id, compromised_mask = injector.apply(raw_telemetry, dt=dt)
            
            # Combine record
            record = {
                "mission_id": mission_id,
                "time": t,
                "attack_id": active_attack_id,
                "attack_name": ATTACK_NAMES[active_attack_id],
                # Truth state (for benchmark validation)
                "truth_alt": state["altitude_m"],
                "truth_tas": state["true_airspeed_m_s"],
                "truth_roll": state["roll_deg"],
                "truth_pitch": state["pitch_deg"],
                "truth_yaw": state["yaw_deg"],
            }
            # Add telemetry channels
            record.update(mod_telemetry)
            # Add sensor compromise labels
            for k, v in compromised_mask.items():
                record[f"compromised_{k}"] = v
                
            records.append(record)

        return pd.DataFrame(records)

    def generate_dataset_split(
        self,
        output_dir: str,
        split_name: str,
        missions_per_attack: int = 10,
        mission_duration_s: float = 35.0,
    ) -> Path:
        """
        Generates a balanced dataset across all attack classes for a given split (train/val/test).
        """
        out_path = Path(output_dir) / split_name
        out_path.mkdir(parents=True, exist_ok=True)
        
        all_dfs = []
        mission_counter = 0

        print(f"Generating {split_name} set ({len(ATTACK_NAMES)} classes, {missions_per_attack} missions/class)...")
        for attack_id in range(len(ATTACK_NAMES)):
            for i in range(missions_per_attack):
                df = self.generate_single_mission(
                    mission_id=mission_counter,
                    attack_type=attack_id,
                    duration_s=mission_duration_s,
                    dt=0.02
                )
                all_dfs.append(df)
                mission_counter += 1

        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save as Parquet (compressed, fast IO) and sample CSV
        parquet_file = out_path / f"{split_name}_telemetry.parquet"
        combined_df.to_parquet(parquet_file, index=False)
        
        # Save first 5000 rows as human-readable CSV sample for inspection
        csv_sample = out_path / f"{split_name}_sample.csv"
        combined_df.head(5000).to_csv(csv_sample, index=False)
        
        print(f"[{split_name.upper()}] Generated {len(combined_df):,} samples ({mission_counter} missions) -> {parquet_file}")
        return parquet_file
