"""Feature attention module."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """
    Learns an importance weight for each feature embedding.

    Input
    -----
    (batch_size, num_features, embedding_dim)

    Output
    ------
    weighted_features:
        (batch_size, num_features, embedding_dim)

    attention_weights:
        (batch_size, num_features)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.score_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.softmax = nn.Softmax(dim=1)

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
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"got {features.shape[-1]}."
            )

        scores = self.score_network(features)

        scores = scores.squeeze(-1)

        weights = self.softmax(scores)

        weighted_features = (
            features * weights.unsqueeze(-1)
        )

        return weighted_features, weights