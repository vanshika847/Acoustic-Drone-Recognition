"""Test the feature attention module."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.attention import FeatureAttention


def main() -> None:
    batch_size = 8
    num_features = 6
    embedding_dim = 128

    attention = FeatureAttention(
        embedding_dim=embedding_dim,
        hidden_dim=64,
    )

    features = torch.randn(
        batch_size,
        num_features,
        embedding_dim,
    )

    weighted_features, weights = attention(features)

    print("\nAttention Test")
    print("-" * 40)

    print(f"Input Shape            : {tuple(features.shape)}")
    print(f"Weighted Shape         : {tuple(weighted_features.shape)}")
    print(f"Attention Shape        : {tuple(weights.shape)}")

    print("\nAttention Weights (sample 0):")
    print(weights[0])

    print(
        "\nWeight Sum (sample 0):",
        weights[0].sum().item(),
    )


if __name__ == "__main__":
    main()