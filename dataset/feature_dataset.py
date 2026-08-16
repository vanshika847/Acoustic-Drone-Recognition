"""PyTorch dataset for loading extracted audio features.

Purpose
-------
Load feature matrices referenced by a feature manifest and return them as
PyTorch tensors suitable for model training.

Expected manifest columns
-------------------------
mfcc_path
mel_path
spectral_path
chroma_path
zcr_path
energy_path
binary_label
segment_id

Each feature file must contain a two-dimensional NumPy array with shape:

    (feature_channels, time_frames)

The dataset does not modify the feature values. It only loads, validates,
and converts them to PyTorch tensors.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = (
    "mfcc_path",
    "mel_path",
    "spectral_path",
    "chroma_path",
    "zcr_path",
    "energy_path",
)

FEATURE_NAMES = (
    "mfcc",
    "mel",
    "spectral",
    "chroma",
    "zcr",
    "energy",
)

EXPECTED_CHANNELS = {
    "mfcc": 120,
    "mel": 128,
    "spectral": 12,
    "chroma": 12,
    "zcr": 1,
    "energy": 1,
}

REQUIRED_COLUMNS = frozenset(
    {
        *FEATURE_COLUMNS,
        "binary_label",
        "segment_id",
    }
)


class FeatureDatasetError(RuntimeError):
    """Raised when a feature dataset cannot be loaded safely."""


class FeatureDataset(Dataset):
    """Dataset that loads extracted feature files from a manifest.

    Each item contains six feature tensors, a binary class label, and the
    original segment identifier.

    Feature tensor shapes are expected to be:

        MFCC     -> (120, frames)
        Mel      -> (128, frames)
        Spectral -> (12, frames)
        Chroma   -> (12, frames)
        ZCR      -> (1, frames)
        Energy   -> (1, frames)

    The number of time frames must be identical across all feature families.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        validate_features: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            manifest_path:
                Path to the feature manifest CSV.

            validate_features:
                Whether to validate every feature file when the dataset is
                created. Disable only when working with a very large dataset
                and validation has already been performed separately.

        Raises:
            FileNotFoundError:
                If the manifest does not exist.

            FeatureDatasetError:
                If the manifest is malformed or required columns are missing.
        """

        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Feature manifest not found: {self.manifest_path}"
            )

        try:
            self.manifest = pd.read_csv(
                self.manifest_path,
                dtype={
                    "segment_id": str,
                    "binary_label": str,
                },
                keep_default_na=False,
            )
        except Exception as exc:
            raise FeatureDatasetError(
                f"Failed to read feature manifest "
                f"'{self.manifest_path}': {exc}"
            ) from exc

        missing_columns = REQUIRED_COLUMNS.difference(self.manifest.columns)

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise FeatureDatasetError(
                f"Feature manifest is missing required columns: "
                f"{missing_text}"
            )

        if self.manifest.empty:
            raise FeatureDatasetError(
                f"Feature manifest contains no samples: "
                f"{self.manifest_path}"
            )

        if validate_features:
            self._validate_manifest_features()

    def __len__(self) -> int:
        """Return the number of available feature samples."""

        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return one feature sample.

        Args:
            index:
                Zero-based dataset index.

        Returns:
            Dictionary containing six feature tensors, the label, and
            segment ID.

        Raises:
            FeatureDatasetError:
                If the selected feature files are invalid.
        """

        if not 0 <= index < len(self):
            raise IndexError(
                f"FeatureDataset index {index} is out of range."
            )

        row = self.manifest.iloc[index]

        try:
            mfcc = self._load_feature(
                row["mfcc_path"],
                "mfcc",
            )

            mel = self._load_feature(
                row["mel_path"],
                "mel",
            )

            spectral = self._load_feature(
                row["spectral_path"],
                "spectral",
            )

            chroma = self._load_feature(
                row["chroma_path"],
                "chroma",
            )

            zcr = self._load_feature(
                row["zcr_path"],
                "zcr",
            )

            energy = self._load_feature(
                row["energy_path"],
                "energy",
            )

            self._validate_frame_alignment(
                {
                    "mfcc": mfcc,
                    "mel": mel,
                    "spectral": spectral,
                    "chroma": chroma,
                    "zcr": zcr,
                    "energy": energy,
                }
            )

            label = self._parse_label(
                row["binary_label"]
            )

            segment_id = str(row["segment_id"])

            return {
                "mfcc": torch.from_numpy(mfcc),
                "mel": torch.from_numpy(mel),
                "spectral": torch.from_numpy(spectral),
                "chroma": torch.from_numpy(chroma),
                "zcr": torch.from_numpy(zcr),
                "energy": torch.from_numpy(energy),
                "label": torch.tensor(
                    label,
                    dtype=torch.long,
                ),
                "segment_id": segment_id,
            }

        except FeatureDatasetError:
            raise

        except Exception as exc:
            raise FeatureDatasetError(
                f"Failed to load sample at index {index} "
                f"(segment '{row['segment_id']}'): {exc}"
            ) from exc

    def _validate_manifest_features(self) -> None:
        """Validate all feature files referenced by the manifest."""

        for index in range(len(self.manifest)):
            row = self.manifest.iloc[index]

            features: dict[str, NDArray[np.float32]] = {}

            for feature_name, column_name in zip(
                FEATURE_NAMES,
                FEATURE_COLUMNS,
            ):
                features[feature_name] = self._load_feature(
                    row[column_name],
                    feature_name,
                )

            self._validate_frame_alignment(features)

            self._parse_label(row["binary_label"])

    def _load_feature(
        self,
        feature_path: str | Path,
        feature_name: str,
    ) -> NDArray[np.float32]:
        """Load and validate one feature array."""

        path = Path(str(feature_path))

        if not path.is_file():
            raise FeatureDatasetError(
                f"{feature_name} feature file not found: '{path}'"
            )

        try:
            feature = np.load(
                path,
                allow_pickle=False,
            )
        except Exception as exc:
            raise FeatureDatasetError(
                f"Failed to load {feature_name} feature file "
                f"'{path}': {exc}"
            ) from exc

        feature = np.asarray(
            feature,
            dtype=np.float32,
        )

        if feature.ndim != 2:
            raise FeatureDatasetError(
                f"{feature_name} feature must be two-dimensional. "
                f"Got shape {feature.shape} from '{path}'."
            )
        expected_channels = EXPECTED_CHANNELS[feature_name]

        if feature.shape[0] != expected_channels:
            raise FeatureDatasetError(
                f"{feature_name} feature has the wrong number of channels. "
                f"Expected {expected_channels}, "
                f"got {feature.shape[0]} "
                f"from '{path}'."
            )

        if feature.shape[0] <= 0 or feature.shape[1] <= 0:
            raise FeatureDatasetError(
                f"{feature_name} feature has an invalid shape "
                f"{feature.shape} in '{path}'."
            )

        if not np.all(np.isfinite(feature)):
            raise FeatureDatasetError(
                f"{feature_name} feature contains NaN or Inf values "
                f"in '{path}'."
            )

        return np.ascontiguousarray(
            feature,
            dtype=np.float32,
        )

    @staticmethod
    def _validate_frame_alignment(
        features: dict[str, NDArray[np.float32]],
    ) -> None:
        """Ensure every feature family has the same number of frames."""

        frame_counts = {
            name: array.shape[1]
            for name, array in features.items()
        }

        unique_frame_counts = set(
            frame_counts.values()
        )

        if len(unique_frame_counts) != 1:
            details = ", ".join(
                f"{name}={frames}"
                for name, frames in frame_counts.items()
            )

            raise FeatureDatasetError(
                "Feature frame counts are not aligned: "
                f"{details}"
            )

    @staticmethod
    def _parse_label(value: object) -> int:
        """Validate and convert a binary label to an integer."""

        try:
            label = int(value)
        except (TypeError, ValueError) as exc:
            raise FeatureDatasetError(
                f"Invalid binary label: '{value}'."
            ) from exc

        if label not in (0, 1):
            raise FeatureDatasetError(
                f"Binary label must be 0 or 1, got {label}."
            )

        return label