"""PyTorch DataLoader utilities."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from dataset.feature_dataset import FeatureDataset


def create_dataloader(
    manifest_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader from a feature manifest."""

    dataset = FeatureDataset(manifest_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )