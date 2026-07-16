"""Binary classifier for acoustic drone detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class DroneClassifier(nn.Module):
    """
    Binary classifier operating on the fused feature embedding.

    Input
    -----
    (batch_size, fused_dim)

    Output
    ------
    (batch_size, num_classes)

    Notes
    -----
    The output consists of raw logits.
    Do NOT apply Softmax here because CrossEntropyLoss
    performs it internally.
    """

    def __init__(
        self,
        fused_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(features)