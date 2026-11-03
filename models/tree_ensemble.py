"""
Gradient Boosted Tree Ensembles (LightGBM & XGBoost).
Serves as high-speed tabular baseline and benchmark comparison.
"""

import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from typing import Dict, Any, Tuple
import joblib


class UAVTreeEnsemble:
    """Multi-task Gradient Boosted Tree suite for tabular window features."""
    def __init__(self, n_estimators: int = 120, max_depth: int = 6, learning_rate: float = 0.05):
        # Multi-class attack classifier
        self.attack_model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=-1
        )
        # Multi-label sensor compromise classifier
        self.sensor_model = MultiOutputClassifier(
            lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                verbosity=-1
            )
        )

    def fit(self, X: np.ndarray, y_attack: np.ndarray, y_sensors: np.ndarray):
        """Fits both tree models on tabular window features."""
        print("Training LightGBM Attack Classifier...")
        self.attack_model.fit(X, y_attack)
        
        print("Training LightGBM Multi-Label Sensor Model...")
        self.sensor_model.fit(X, y_sensors)
        print("Tree ensemble training complete.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predicted attack labels and sensor binary flags."""
        pred_attacks = self.attack_model.predict(X)
        pred_sensors = self.sensor_model.predict(X)
        return pred_attacks, pred_sensors

    def save(self, filepath: str):
        joblib.dump({"attack": self.attack_model, "sensors": self.sensor_model}, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        ensemble = cls()
        ensemble.attack_model = data["attack"]
        ensemble.sensor_model = data["sensors"]
        return ensemble
