"""Test complete acoustic drone network."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_loader import create_dataloader
from models.network import AcousticDroneNet


def main() -> None:

    loader = create_dataloader(
        "outputs/features/train_feature_manifest.csv",
        batch_size=8,
        shuffle=True,
    )

    batch = next(iter(loader))

    model = AcousticDroneNet()

    logits, attention = model(batch)

    print("\nComplete Network Test")
    print("-" * 50)

    print(f"Logits Shape    : {tuple(logits.shape)}")

    print(f"Attention Shape : {tuple(attention.shape)}")

    print("\nExample logits:")

    print(logits[0])

    print("\nAttention:")

    print(attention[0])


if __name__ == "__main__":
    main()