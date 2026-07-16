"""Test the drone classifier."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.classifier import DroneClassifier


def main() -> None:

    classifier = DroneClassifier()

    fused = torch.randn(8, 256)

    logits = classifier(fused)

    print("\nClassifier Test")
    print("-" * 40)

    print(f"Input Shape : {tuple(fused.shape)}")
    print(f"Output Shape: {tuple(logits.shape)}")

    print("\nSample logits:")
    print(logits[0])


if __name__ == "__main__":
    main()