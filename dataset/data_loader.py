"""DataLoader helpers for shard-based acoustic features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset.feature_dataset import (
    FEATURE_NAMES,
    FeatureDataset,
)


def _collate_fixed_features(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack feature tensors and labels while retaining segment IDs."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    output: dict[str, Any] = {}

    for name in FEATURE_NAMES:
        tensors = [sample[name] for sample in batch]
        output[name] = torch.stack(
            tensors,
            dim=0,
        )

    output["label"] = torch.stack(
        [sample["label"] for sample in batch],
        dim=0,
    )

    output["segment_id"] = [
        sample["segment_id"]
        for sample in batch
    ]

    return output


def create_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    validate_features: bool = True,
) -> DataLoader:
    """Create a normal non-sampled DataLoader."""

    dataset = FeatureDataset(
        manifest_path,
        validate_features=validate_features,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fixed_features,
    )


__all__ = [
    "_collate_fixed_features",
    "create_dataloader",
]
