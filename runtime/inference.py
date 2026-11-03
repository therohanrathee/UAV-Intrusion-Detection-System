"""
Standalone Edge Inference Runtime.
Uses ONNX Runtime for ultra-fast, zero-dependency inference on embedded companion computers
(Raspberry Pi, Nvidia Jetson, x86/ARM Linux, macOS, Windows).
"""

import numpy as np
import onnxruntime as ort
import joblib
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> str:
    path = Path(p)
    if not path.is_absolute() and not path.exists():
        cand = PROJECT_ROOT / p
        if cand.exists():
            return str(cand)
    return str(path)


class UAVInferenceEngine:
    """
    Edge inference engine for real-time cyber-physical UAV intrusion detection.
    Computes invariant residuals and evaluates multi-task ONNX model in < 1 ms.
    """
    def __init__(
        self,
        onnx_model_path: str = "weights/uav_ids_model.onnx",
        scaler_path: str = "checkpoints/scaler.joblib",
        manifest_path: str = "weights/model_manifest.json"
    ):
        self.onnx_path = _resolve(onnx_model_path)
        self.scaler = joblib.load(_resolve(scaler_path))
        
        with open(_resolve(manifest_path), "r") as f:
            self.manifest = json.load(f)
            
        self.attack_classes = self.manifest["attack_classes"]
        self.sensor_channels = self.manifest["sensor_channels"]
        self.feature_names = self.manifest["feature_names"]
        self.window_size = self.manifest["input_shape"][1]
        self.in_features = self.manifest["input_shape"][2]

        # Initialize ONNX Runtime session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, window_features: np.ndarray) -> Dict[str, Any]:
        """
        Runs real-time inference on a temporal sliding window.
        Args:
            window_features: (window_size, in_features) or (1, window_size, in_features) numpy array
        Returns:
            Structured diagnostic dictionary
        """
        t0 = time.perf_counter()
        
        if window_features.ndim == 2:
            window_features = np.expand_dims(window_features, axis=0)

        # Standardize features
        B, T, F = window_features.shape
        scaled_feat = self.scaler.transform(window_features.reshape(-1, F)).reshape(B, T, F).astype(np.float32)

        # Run ONNX inference
        ort_inputs = {self.input_name: scaled_feat}
        attack_logits, sensor_logits = self.session.run(None, ort_inputs)

        # Softmax for attack classification
        exp_att = np.exp(attack_logits[0] - np.max(attack_logits[0]))
        attack_probs = exp_att / np.sum(exp_att)
        pred_attack_idx = int(np.argmax(attack_probs))
        pred_attack_name = self.attack_classes[pred_attack_idx]
        attack_confidence = float(attack_probs[pred_attack_idx])

        # Sigmoid for sensor compromise
        sensor_probs = 1.0 / (1.0 + np.exp(-sensor_logits[0]))
        sensor_health = {}
        for i, s_name in enumerate(self.sensor_channels):
            # 1 = healthy (not compromised), 0 = compromised
            is_compromised = bool(sensor_probs[i] > 0.5)
            sensor_health[s_name] = not is_compromised

        # Ensure sensor health matches detected attack type
        if pred_attack_idx != 0:
            attack_primary_sensor = {
                "GNSS_Spoofing": "gps",
                "GNSS_Jamming": "gps",
                "Acoustic_Gyro_Resonance": "gyro_primary",
                "Optical_Seeker_Blinding": "optical",
                "Pitot_Tube_Fault": "pitot",
                "Radar_Altimeter_Jamming": "radar_alt",
                "Magnetometer_Interference": "mag",
                "C2_Link_Jamming_DoS": "c2_link",
            }
            primary = attack_primary_sensor.get(pred_attack_name)
            if primary and primary in sensor_health:
                sensor_health[primary] = False

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "attack_detected": pred_attack_idx != 0,
            "attack_type": pred_attack_name,
            "attack_id": pred_attack_idx,
            "confidence": attack_confidence,
            "attack_probabilities": {name: float(attack_probs[i]) for i, name in enumerate(self.attack_classes)},
            "sensor_health": sensor_health,
            "sensor_compromise_probs": {name: float(sensor_probs[i]) for i, name in enumerate(self.sensor_channels)},
            "latency_ms": round(dt_ms, 3),
        }
