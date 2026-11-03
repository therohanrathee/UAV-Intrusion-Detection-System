"""
Test Real-time Intrusion Detection on Raw Unlabeled Flight Telemetry.
Verifies that the ONNX model and Physics Invariant Engine successfully detect attacks
and isolate compromised sensors without requiring any ground-truth labels.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from runtime.inference import UAVInferenceEngine
from pipeline.residuals import compute_physics_residuals, FEATURE_COLUMNS


def test_unlabeled_telemetry_attack_detection():
    unlabeled_csv = Path("tests/unlabeled_flight_telemetry.csv")
    assert unlabeled_csv.exists(), "tests/unlabeled_flight_telemetry.csv must exist"

    df = pd.read_csv(unlabeled_csv)

    # 1. Assert that the CSV is strictly unlabeled (zero attack/compromised metadata)
    assert not any("attack" in col.lower() for col in df.columns), "File must have zero attack columns"
    assert not any("compromised" in col.lower() for col in df.columns), "File must have zero compromised sensor columns"
    assert len(df.columns) == 44, "Expected 44 raw flight channels"

    # 2. Compute physics residuals on raw telemetry
    df_feat = compute_physics_residuals(df)
    feat_matrix = df_feat[FEATURE_COLUMNS].values
    assert feat_matrix.shape[0] == len(df)
    assert feat_matrix.shape[1] == len(FEATURE_COLUMNS)

    # 3. Load edge ONNX runtime engine
    engine = UAVInferenceEngine(
        onnx_model_path="weights/uav_ids_model.onnx",
        scaler_path="checkpoints/scaler.joblib",
        manifest_path="weights/model_manifest.json"
    )

    # 4. Phase 1: Nominal Flight Window (t ~ 2.0s to 3.0s, timestep 100:150)
    nominal_window = feat_matrix[100:150]
    res_nom = engine.predict(nominal_window)
    assert res_nom["attack_detected"] is False, "Early flight should be classified as Normal"
    assert res_nom["attack_type"] == "Normal"
    assert res_nom["confidence"] > 0.90
    assert all(res_nom["sensor_health"].values()), "All sensors should be healthy during nominal phase"

    # 5. Phase 2: Active Cyber Attack Window (t ~ 16.0s to 17.0s, timestep 800:850)
    attack_window = feat_matrix[800:850]
    res_att = engine.predict(attack_window)
    assert res_att["attack_detected"] is True, "Attack should be detected on raw unlabeled data"
    assert res_att["attack_type"] == "GNSS_Spoofing"
    assert res_att["confidence"] > 0.95
    assert res_att["sensor_health"]["gps"] is False, "GPS sensor should be flagged as compromised"
    assert res_att["latency_ms"] < 2.0, "Inference must execute in < 2 ms"
