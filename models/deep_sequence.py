"""
Multi-Task Deep Sequence Network with Conv1D, Bi-GRU, and Temporal Self-Attention.
Jointly predicts:
1. Multi-class Attack Vector (9 classes: Normal + 8 attacks)
2. Multi-label Sensor Compromise Vector (10 sensor flags)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class UAVMultiTaskDeepNet(nn.Module):
    """
    State-of-the-art multi-task deep neural network for UAV cyber-physical telemetry.
    Combines multi-scale 1D temporal convolutions with bidirectional recurrent memory
    and self-attention pooling.
    """
    def __init__(
        self,
        in_features: int = 29,
        num_attacks: int = 9,
        num_sensors: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.num_attacks = num_attacks
        self.num_sensors = num_sensors
        
        # 1. Multi-scale 1D Temporal Convolutions (transient & vibration feature extraction)
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 2. Bidirectional GRU (long-term drift & integration dynamics)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Temporal Self-Attention Pooling (identifies anomaly onset in the window)
        self.attn_linear = nn.Linear(hidden_dim, 1)
        
        # 4. Shared Representation Layer
        self.shared_dense = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. Multi-Task Output Heads
        # Head A: Attack Type (Logits for Softmax)
        self.head_attack = nn.Linear(64, num_attacks)
        # Head B: Sensor Health Trust (Logits for Sigmoid)
        self.head_sensors = nn.Linear(64, num_sensors)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (Batch, Timesteps, Features) float32 tensor
        Returns:
            attack_logits: (Batch, Num_Attacks)
            sensor_logits: (Batch, Num_Sensors)
        """
        # Transpose for Conv1d: (B, F, T)
        x_c = x.transpose(1, 2)
        h = F.relu(self.bn1(self.conv1(x_c)))
        h = F.relu(self.bn2(self.conv2(h)))
        
        # Transpose back for GRU: (B, T, hidden_dim)
        h = h.transpose(1, 2)
        gru_out, _ = self.gru(h)
        
        # Self-Attention weights across timesteps
        attn_weights = F.softmax(self.attn_linear(gru_out), dim=1)  # (B, T, 1)
        context = torch.sum(gru_out * attn_weights, dim=1)          # (B, hidden_dim)
        
        # Shared features
        features = self.shared_dense(context)
        
        # Output heads
        attack_logits = self.head_attack(features)
        sensor_logits = self.head_sensors(features)
        
        return attack_logits, sensor_logits
