"""Dataset for shard-based acoustic feature arrays."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = (
    "mfcc",
    "mel",
    "spectral",
    "chroma",
    "zcr",
    "energy",
)

REQUIRED_COLUMNS = {
    "split",
    "segment_id",
    "binary_label",
    "shard_path",
    "shard_index",
}


class FeatureDataset(Dataset):
    """Load one sample at a time from NPZ feature shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        validate_features: bool = True,
        project_root: str | Path | None = None,
        max_cached_shards: int = 2,
    ) -> None:
        self.manifest_path = Path(
            manifest_path
        ).resolve()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Shard manifest not found: {self.manifest_path}"
            )

        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else self.manifest_path.parent.parent.parent
        )

        self.dataframe = pd.read_csv(
            self.manifest_path,
            dtype=str,
            keep_default_na=False,
        )

        missing = REQUIRED_COLUMNS.difference(
            self.dataframe.columns
        )
        if missing:
            raise ValueError(
                "Shard manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )

        if self.dataframe.empty:
            raise ValueError(
                f"Shard manifest is empty: {self.manifest_path}"
            )

        labels = pd.to_numeric(
            self.dataframe["binary_label"],
            errors="coerce",
        )
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(
                "binary_label must contain only 0 or 1."
            )

        self.dataframe["binary_label"] = labels.astype(int)

        self.dataframe["shard_index"] = pd.to_numeric(
            self.dataframe["shard_index"],
            errors="raise",
        ).astype(int)

        self.validate_features = bool(validate_features)
        self.max_cached_shards = max(
            1,
            int(max_cached_shards),
        )

        self._cache: OrderedDict[Path, Any] = OrderedDict()

        if self.validate_features:
            self._validate_shards()

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_shard_path(self, value: str) -> Path:
        raw = Path(value)

        if raw.is_absolute() and raw.is_file():
            return raw

        if raw.is_absolute():
            parts = list(raw.parts)
            if "features" in parts:
                idx = parts.index("features")
                candidate = (
                    self.project_root
                    / Path(*parts[idx:])
                )
                if candidate.is_file():
                    return candidate

        candidate = (
            self.project_root / raw
        ).resolve()

        if candidate.is_file():
            return candidate

        raise FileNotFoundError(
            f"Feature shard not found: '{value}'"
        )

    def _load_shard(self, shard_path: Path) -> Any:
        shard_path = shard_path.resolve()

        if shard_path in self._cache:
            shard = self._cache.pop(shard_path)
            self._cache[shard_path] = shard
            return shard

        shard = np.load(
            shard_path,
            allow_pickle=False,
        )

        for name in FEATURE_NAMES:
            if name not in shard:
                shard.close()
                raise ValueError(
                    f"Shard is missing feature '{name}': {shard_path}"
                )

        for required in ("labels", "segment_ids"):
            if required not in shard:
                shard.close()
                raise ValueError(
                    f"Shard is missing '{required}': {shard_path}"
                )

        self._cache[shard_path] = shard

        while len(self._cache) > self.max_cached_shards:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass

        return shard

    def _validate_shards(self) -> None:
        checked: set[Path] = set()

        for _, row in self.dataframe.iterrows():
            shard_path = self._resolve_shard_path(
                row["shard_path"]
            )

            if shard_path in checked:
                continue

            shard = self._load_shard(shard_path)
            labels = shard["labels"]

            if labels.ndim != 1:
                raise ValueError(
                    f"Invalid labels shape {labels.shape}: {shard_path}"
                )

            if len(shard["segment_ids"]) != len(labels):
                raise ValueError(
                    f"segment_ids and labels disagree: {shard_path}"
                )

            for name in FEATURE_NAMES:
                array = shard[name]

                if array.ndim != 3:
                    raise ValueError(
                        f"{name} must be (batch, channels, time), "
                        f"got {array.shape}: {shard_path}"
                    )

                if array.shape[0] != len(labels):
                    raise ValueError(
                        f"{name} and labels have different batch sizes: "
                        f"{shard_path}"
                    )

                if array.shape[1] == 0 or array.shape[2] == 0:
                    raise ValueError(
                        f"{name} contains an empty dimension: {shard_path}"
                    )

                if not np.isfinite(array).all():
                    raise ValueError(
                        f"{name} contains NaN/Inf: {shard_path}"
                    )

            checked.add(shard_path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]

        shard_path = self._resolve_shard_path(
            row["shard_path"]
        )
        shard = self._load_shard(shard_path)

        shard_index = int(row["shard_index"])

        if shard_index < 0 or shard_index >= len(shard["labels"]):
            raise IndexError(
                f"shard_index {shard_index} is outside shard "
                f"bounds for {shard_path}"
            )

        manifest_label = int(row["binary_label"])
        shard_label = int(shard["labels"][shard_index])

        if manifest_label != shard_label:
            raise ValueError(
                f"Label mismatch at manifest index {index}: "
                f"manifest={manifest_label}, shard={shard_label}"
            )

        sample: dict[str, Any] = {}

        for name in FEATURE_NAMES:
            array = np.asarray(
                shard[name][shard_index],
                dtype=np.float32,
            )

            if array.ndim != 2:
                raise ValueError(
                    f"{name} sample must be 2-D, got {array.shape}"
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"{name} contains NaN/Inf at index {index}"
                )

            sample[name] = torch.from_numpy(
                np.ascontiguousarray(array)
            )

        sample["label"] = torch.tensor(
            manifest_label,
            dtype=torch.long,
        )
        sample["segment_id"] = row["segment_id"]

        return sample

    def __del__(self) -> None:
        for shard in getattr(self, "_cache", {}).values():
            try:
                shard.close()
            except Exception:
                pass


__all__ = [
    "FEATURE_NAMES",
    "FeatureDataset",
]
