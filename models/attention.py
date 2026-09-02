"""Context-aware attention over acoustic feature families."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """
    Learn per-sample importance for MFCC, Mel, spectral, chroma, ZCR, energy.

    The previous attention scored each feature independently. This version
    also supplies the mean feature context to the scorer, so the importance
    of one feature can depend on what the other feature families contain.

    Attention weights sum to one across feature families.
    Weighted embeddings are scaled by num_features so attention does not
    accidentally shrink the representation by roughly 1/6.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be > 0.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.score_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, num_features, embedding_dim)."
            )

        if features.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {features.shape[-1]}."
            )

        context = features.mean(dim=1, keepdim=True)
        context = context.expand(
            -1,
            features.shape[1],
            -1,
        )

        scorer_input = torch.cat(
            [features, context],
            dim=-1,
        )

        scores = self.score_network(
            scorer_input
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        scale = float(features.shape[1])
        weighted_features = (
            features
            * (weights * scale).unsqueeze(-1)
        )

        return weighted_features, weights


__all__ = ["FeatureAttention"]
