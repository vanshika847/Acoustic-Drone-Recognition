"""Binary classifier for acoustic drone detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class DroneClassifier(nn.Module):
    """Small regularised binary classification head returning raw logits."""

    def __init__(
        self,
        fused_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        if fused_dim <= 0 or hidden_dim <= 0:
            raise ValueError("fused_dim and hidden_dim must be > 0.")
        if num_classes != 2:
            raise ValueError(
                f"This detector is binary and requires num_classes=2, "
                f"got {num_classes}."
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "Expected classifier input shape (batch, fused_dim)."
            )
        return self.classifier(features)


__all__ = ["DroneClassifier"]
