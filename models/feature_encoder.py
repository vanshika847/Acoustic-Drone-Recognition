"""Feature encoder blocks for acoustic drone recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Generic CNN encoder for frame-wise acoustic features.

    Expected input shape:
        (batch_size, feature_dim, time_steps)

    Output:
        (batch_size, embedding_dim)
    """

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(

            # ---------------- Block 1 ----------------

            nn.Conv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2),

            # ---------------- Block 2 ----------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2),

            # ---------------- Block 3 ----------------

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, embedding_dim),

            nn.BatchNorm1d(embedding_dim),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Shape (batch, feature_dim, time)

        Returns:
            Shape (batch, embedding_dim)
        """

        x = self.features(x)

        x = self.embedding(x)

        return x