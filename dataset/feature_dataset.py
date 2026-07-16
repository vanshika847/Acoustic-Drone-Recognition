"""PyTorch dataset for loading extracted audio features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Dataset that loads extracted feature files from a manifest."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self.manifest = pd.read_csv(self.manifest_path)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):

        row = self.manifest.iloc[index]

        sample = {
            "mfcc": torch.from_numpy(
                np.load(row["mfcc_path"]).astype(np.float32)
            ),
            "mel": torch.from_numpy(
                np.load(row["mel_path"]).astype(np.float32)
            ),
            "spectral": torch.from_numpy(
                np.load(row["spectral_path"]).astype(np.float32)
            ),
            "chroma": torch.from_numpy(
                np.load(row["chroma_path"]).astype(np.float32)
            ),
            "zcr": torch.from_numpy(
                np.load(row["zcr_path"]).astype(np.float32)
            ),
            "energy": torch.from_numpy(
                np.load(row["energy_path"]).astype(np.float32)
            ),
            "label": torch.tensor(
                int(row["binary_label"]),
                dtype=torch.long,
            ),
            "segment_id": row["segment_id"],
        }

        return sample