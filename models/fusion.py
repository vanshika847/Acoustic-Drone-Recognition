"""Feature fusion network."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Fuse multiple feature embeddings into a single representation.

    Input
    -----
    (batch_size, num_features, embedding_dim)

    Output
    ------
    (batch_size, fused_dim)
    """

    def __init__(
        self,
        num_features: int = 6,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        fused_dim: int = 256,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fused_dim = fused_dim

        input_dim = num_features * embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        if features.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, num_features, embedding_dim)."
            )

        if features.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features "
                f"but received {features.shape[1]}."
            )

        if features.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"got {features.shape[2]}."
            )

        batch_size = features.shape[0]

        features = features.reshape(
            batch_size,
            self.num_features * self.embedding_dim,
        )

        fused = self.fusion(features)

        return fused