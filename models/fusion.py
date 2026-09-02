"""Feature fusion network."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Fuse six attention-weighted feature embeddings.

    In addition to the flattened embeddings, mean and max statistics are
    provided to the fusion MLP. LayerNorm keeps the network stable with
    small batches.
    """

    def __init__(
        self,
        num_features: int = 6,
        embedding_dim: int = 128,
        hidden_dim: int = 384,
        fused_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        if num_features <= 0 or embedding_dim <= 0:
            raise ValueError("num_features and embedding_dim must be > 0.")

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.fused_dim = fused_dim

        input_dim = (
            num_features * embedding_dim
            + 2 * embedding_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, num_features, embedding_dim)."
            )

        if features.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, "
                f"got {features.shape[1]}."
            )

        if features.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {features.shape[2]}."
            )

        flattened = features.flatten(start_dim=1)
        mean_features = features.mean(dim=1)
        max_features = features.amax(dim=1)

        fusion_input = torch.cat(
            [
                flattened,
                mean_features,
                max_features,
            ],
            dim=1,
        )

        return self.fusion(fusion_input)


__all__ = ["FeatureFusion"]
