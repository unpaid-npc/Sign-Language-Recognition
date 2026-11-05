"""
Bidirectional LSTM with Kinematic Velocity Features and Temporal Attention Pooling
for Isolated Sign Language Sequence Classification.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    FC_DROPOUT,
    FEATURE_DIM,
    INPUT_PROJECTION_DIM,
    LSTM_DROPOUT,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    NUM_CLASSES,
)


class TemporalAttentionPool(nn.Module):
    """
    Learnable Temporal Attention mechanism over BiLSTM hidden states (B, T, 2*H).
    Computes a normalized attention score for each frame and returns the weighted sum.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(
        self, lstm_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: Tensor of shape (B, T, D) where D = 2 * lstm_hidden_dim.

        Returns:
            Tuple of:
              - context_vector: Tensor of shape (B, D)
              - attn_weights: Tensor of shape (B, T) summing to 1.0 across T
        """
        scores = self.attention_net(lstm_out).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(scores, dim=-1)           # (B, T)
        context_vector = torch.bmm(
            attn_weights.unsqueeze(1), lstm_out
        ).squeeze(1)                                       # (B, D)
        return context_vector, attn_weights


class SignBiLSTMClassifier(nn.Module):
    """
    End-to-End Bidirectional LSTM Sequence Classifier for WLASL-100.

    Pipeline:
      1. Input (B, T=30, F=144) -> Compute velocity delta_x (B, T, 144) -> Concat (B, T, 288)
      2. Spatial Feature Projection: Linear(288 -> 128) + LayerNorm + GELU + Dropout
      3. Temporal Backbone: 2-Layer BiLSTM (128 hidden per direction -> 256-D per frame)
      4. Dual Temporal Aggregation: Concat(TemporalAttentionPool(256), GlobalMaxPool(256)) -> 512-D
      5. Classification Head: Linear(512 -> 128) + LayerNorm + GELU + Dropout + Linear(128 -> num_classes)
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        projection_dim: int = INPUT_PROJECTION_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        lstm_dropout: float = LSTM_DROPOUT,
        fc_dropout: float = FC_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Position (144) + First-order velocity (144) = 288
        kinematic_dim = input_dim * 2

        self.input_projection = nn.Sequential(
            nn.Linear(kinematic_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(p=lstm_dropout * 0.5),
        )

        self.bilstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        bilstm_out_dim = hidden_dim * 2  # 256
        self.temporal_attention = TemporalAttentionPool(bilstm_out_dim)

        # Concatenate Attention Context (256) + Temporal Max Pool (256) = 512
        pooled_dim = bilstm_out_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear and recurrent weights for stable gradient flow."""
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # Set forget gate bias to 1.0 to help capture longer temporal dependencies
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def compute_kinematic_features(x: torch.Tensor) -> torch.Tensor:
        """
        Compute frame-to-frame velocity dx_t = x_t - x_{t-1} and concatenate with x_t.

        Args:
            x: Landmark sequence tensor of shape (B, T, 144).

        Returns:
            Combined kinematic tensor of shape (B, T, 288).
        """
        velocity = torch.zeros_like(x)
        velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return torch.cat([x, velocity], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class logits.

        Args:
            x: Input sequence tensor of shape (B, T, 144).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input of shape (B, T, {self.input_dim}), got {tuple(x.shape)}"
            )

        kinematic_x = self.compute_kinematic_features(x)       # (B, T, 288)
        proj_x = self.input_projection(kinematic_x)            # (B, T, 128)
        lstm_out, _ = self.bilstm(proj_x)                      # (B, T, 256)

        attn_context, _ = self.temporal_attention(lstm_out)    # (B, 256)
        max_context, _ = torch.max(lstm_out, dim=1)            # (B, 256)

        combined = torch.cat([attn_context, max_context], dim=-1)  # (B, 512)
        logits = self.classifier(combined)                     # (B, num_classes)
        return logits

    def get_config(self) -> Dict[str, int | float]:
        """Return serializable architecture config for checkpoint saving."""
        return {
            "input_dim": self.input_dim,
            "projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
        }