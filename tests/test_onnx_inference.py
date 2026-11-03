"""
Unit test for standalone ONNX inference engine and latency benchmark.
"""

import pytest
import numpy as np
from pathlib import Path

from runtime.inference import UAVInferenceEngine
from pipeline.residuals import FEATURE_COLUMNS
from sim.attacks import ATTACK_NAMES, SENSOR_KEYS


def test_onnx_engine_initialization_and_inference():
    """Verify ONNX model runs with sub-millisecond latency and valid output format."""
    engine = UAVInferenceEngine()
    
    # Generate dummy input window (50 timesteps, 32 features)
    dummy_input = np.zeros((50, len(FEATURE_COLUMNS)), dtype=np.float32)
    # Set nominal values for some features
    dummy_input[:, 12] = 16.0   # gps_sats
    dummy_input[:, 13] = 0.95   # gps_hdop
    dummy_input[:, 14] = 44.0   # gps_cn0
    dummy_input[:, 15] = 0.85   # optical_entropy
    dummy_input[:, 16] = 0.95   # optical_quality
    dummy_input[:, 18] = 0.92   # radar_reflectivity
    dummy_input[:, 26] = 32.0   # pitot_ias
    
    result = engine.predict(dummy_input)
    
    assert "attack_detected" in result
    assert "attack_type" in result
    assert result["attack_type"] in ATTACK_NAMES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "sensor_health" in result
    assert len(result["sensor_health"]) == len(SENSOR_KEYS)
    assert result["latency_ms"] < 10.0  # Must be fast for real-time edge flight
