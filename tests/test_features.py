"""
Unit tests for physics-informed residual computations and windowing pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from pipeline.residuals import compute_physics_residuals, FEATURE_COLUMNS
from pipeline.windowing import create_sliding_windows, extract_tabular_window_features, TARGET_SENSOR_COLUMNS
from sim.generator import MissionGenerator


def test_compute_physics_residuals():
    """Verify residual extractor adds all required feature channels without NaNs."""
    gen = MissionGenerator()
    df = gen.generate_single_mission(mission_id=1, attack_type=1, duration_s=6.0, dt=0.02)
    
    feat_df = compute_physics_residuals(df, window_size=25)
    
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
        assert not feat_df[col].isna().any(), f"Column {col} contains NaNs"


def test_sliding_windows_and_tabular_extraction():
    """Verify temporal window segmentation and tabular extraction dimensions."""
    gen = MissionGenerator()
    df1 = gen.generate_single_mission(mission_id=1, attack_type=0, duration_s=5.0, dt=0.02)
    df2 = gen.generate_single_mission(mission_id=2, attack_type=3, duration_s=5.0, dt=0.02)
    df = pd.concat([df1, df2], ignore_index=True)
    
    feat_df = compute_physics_residuals(df, window_size=25)
    X, y_attack, y_sensors = create_sliding_windows(feat_df, window_size=50, stride=10)
    
    assert X.ndim == 3
    assert X.shape[1] == 50  # window_size
    assert X.shape[2] == len(FEATURE_COLUMNS)
    assert len(X) == len(y_attack) == len(y_sensors)
    assert y_sensors.shape[1] == len(TARGET_SENSOR_COLUMNS)
    
    # Check tabular features
    X_tab = extract_tabular_window_features(X)
    assert X_tab.ndim == 2
    assert X_tab.shape[0] == len(X)
    assert X_tab.shape[1] == len(FEATURE_COLUMNS) * 6  # mean, std, min, max, last, delta
    assert not np.isnan(X_tab).any()
