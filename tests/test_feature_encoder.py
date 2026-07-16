"""..."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.feature_encoder import FeatureEncoder

def test_encoder(
    feature_name: str,
    feature_dim: int,
    time_steps: int = 126,
) -> None:
    """
    Create a random batch and verify encoder output.

    Args:
        feature_name: Name of feature being tested.
        feature_dim: Number of feature rows.
        time_steps: Number of frames.
    """

    model = FeatureEncoder(input_channels=feature_dim)

    x = torch.randn(
        8,
        feature_dim,
        time_steps,
    )

    y = model(x)

    print(f"\n{feature_name}")
    print("-" * 40)
    print(f"Input Shape : {tuple(x.shape)}")
    print(f"Output Shape: {tuple(y.shape)}")


def main() -> None:

    test_encoder("MFCC", 120)

    test_encoder("Mel", 128)

    test_encoder("Spectral", 12)

    test_encoder("Chroma", 12)

    test_encoder("ZCR", 1)

    test_encoder("Energy", 1)


if __name__ == "__main__":
    main()