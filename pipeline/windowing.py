"""
Temporal Sliding Window Segmentation & Batching.
Constructs sequences for:
1. Deep Sequence Modeling: (N_windows, sequence_length, n_features)
2. Gradient Boosted Trees: Tabular statistical summaries over windows
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any, Optional

from pipeline.residuals import FEATURE_COLUMNS
from sim.attacks import SENSOR_KEYS, ATTACK_NAMES

TARGET_SENSOR_COLUMNS = [f"compromised_{k}" for k in SENSOR_KEYS]


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 50,      # 1.0 second @ 50 Hz
    stride: int = 10,           # 0.2 second step (80% overlap)
    feature_cols: List[str] = FEATURE_COLUMNS,
    sensor_cols: List[str] = TARGET_SENSOR_COLUMNS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates temporal sliding windows grouped strictly by mission_id (zero leakage across missions).
    Returns:
        X: (N, window_size, n_features) float32
        y_attack: (N,) int64
        y_sensors: (N, n_sensors) float32
    """
    X_list = []
    y_attack_list = []
    y_sensors_list = []

    for mission_id, group in df.groupby("mission_id", sort=False):
        features = group[feature_cols].values.astype(np.float32)
        attacks = group["attack_id"].values.astype(np.int64)
        sensors = group[sensor_cols].values.astype(np.float32)
        n_samples = len(group)

        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            window_feat = features[start:end]
            # Use label of the latest timestep in the window (real-time detection)
            attack_label = attacks[end - 1]
            sensor_labels = sensors[end - 1]

            X_list.append(window_feat)
            y_attack_list.append(attack_label)
            y_sensors_list.append(sensor_labels)

    X = np.array(X_list, dtype=np.float32)
    y_attack = np.array(y_attack_list, dtype=np.int64)
    y_sensors = np.array(y_sensors_list, dtype=np.float32)
    return X, y_attack, y_sensors


def extract_tabular_window_features(
    X_windows: np.ndarray,
    feature_cols: List[str] = FEATURE_COLUMNS
) -> np.ndarray:
    """
    Computes statistical moments (mean, std, min, max, delta) over each window
    for gradient boosted tree models (LightGBM / XGBoost).
    """
    # X_windows shape: (N, T, F)
    mean_feat = np.mean(X_windows, axis=1)    # (N, F)
    std_feat = np.std(X_windows, axis=1)      # (N, F)
    min_feat = np.min(X_windows, axis=1)      # (N, F)
    max_feat = np.max(X_windows, axis=1)      # (N, F)
    last_feat = X_windows[:, -1, :]           # (N, F) latest sample
    delta_feat = X_windows[:, -1, :] - X_windows[:, 0, :]  # (N, F) change over window

    tabular_X = np.hstack([mean_feat, std_feat, min_feat, max_feat, last_feat, delta_feat])
    return tabular_X.astype(np.float32)


class TelemetrySequenceDataset(Dataset):
    """PyTorch Dataset for multi-task telemetry sequences."""
    def __init__(self, X: np.ndarray, y_attack: np.ndarray, y_sensors: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_attack = torch.tensor(y_attack, dtype=torch.long)
        self.y_sensors = torch.tensor(y_sensors, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_attack[idx], self.y_sensors[idx]
