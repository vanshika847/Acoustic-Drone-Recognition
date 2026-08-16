"""Smoke test for the acoustic drone ML pipeline."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.feature_dataset import FeatureDataset
from models.acoustic_drone_model import AcousticDroneModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MANIFEST_PATH = Path(
    "outputs/features/train_feature_manifest.csv"
)

BATCH_SIZE = 4

FEATURE_NAMES = (
    "mfcc",
    "mel",
    "spectral",
    "chroma",
    "zcr",
    "energy",
)


def main() -> None:
    print("=" * 60)
    print("ACOUSTIC DRONE MODEL PIPELINE SMOKE TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    print("\n[1] Loading feature dataset...")

    dataset = FeatureDataset(
        MANIFEST_PATH,
        validate_features=False,
    )

    print(f"Samples: {len(dataset):,}")

    # -----------------------------------------------------
    # 2. Inspect one sample
    # -----------------------------------------------------

    print("\n[2] Inspecting first sample...")

    sample = dataset[0]

    for name in FEATURE_NAMES:
        print(
            f"{name:10s}: "
            f"shape={tuple(sample[name].shape)}, "
            f"dtype={sample[name].dtype}"
        )

    print(f"label     : {sample['label'].item()}")
    print(f"segment_id: {sample['segment_id']}")

    # -----------------------------------------------------
    # 3. Create DataLoader
    # -----------------------------------------------------

    print("\n[3] Creating DataLoader...")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nBatch shapes:")

    for name in FEATURE_NAMES:
        print(
            f"{name:10s}: "
            f"{tuple(batch[name].shape)}"
        )

    print(
        f"label     : "
        f"{tuple(batch['label'].shape)}"
    )

    # -----------------------------------------------------
    # 4. Create model
    # -----------------------------------------------------

    print("\n[4] Creating model...")

    model = AcousticDroneModel(
        embedding_dim=128,
        fused_dim=256,
        num_classes=2,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {parameter_count:,}")

    # -----------------------------------------------------
    # 5. Forward pass
    # -----------------------------------------------------

    print("\n[5] Running forward pass...")

    features = {
        name: batch[name]
        for name in FEATURE_NAMES
    }

    logits, attention_weights = model(features)

    print(
        f"logits            : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"attention_weights : "
        f"{tuple(attention_weights.shape)}"
    )

    # -----------------------------------------------------
    # 6. Validate output shapes
    # -----------------------------------------------------

    assert logits.shape == (
        BATCH_SIZE,
        2,
    ), (
        f"Unexpected logits shape: "
        f"{tuple(logits.shape)}"
    )

    assert attention_weights.shape == (
        BATCH_SIZE,
        6,
    ), (
        f"Unexpected attention shape: "
        f"{tuple(attention_weights.shape)}"
    )

    # Attention weights should sum to 1.
    attention_sums = attention_weights.sum(dim=1)

    assert torch.allclose(
        attention_sums,
        torch.ones_like(attention_sums),
        atol=1e-5,
    ), "Attention weights do not sum to 1."

    print("Output shape checks: OK")

    # -----------------------------------------------------
    # 7. Loss
    # -----------------------------------------------------

    print("\n[6] Testing CrossEntropyLoss...")

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        batch["label"],
    )

    print(f"Initial loss: {loss.item():.6f}")

    # -----------------------------------------------------
    # 8. Backward pass
    # -----------------------------------------------------

    print("\n[7] Testing backward pass...")

    loss.backward()

    print("Backward pass: OK")

    # -----------------------------------------------------
    # Success
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()