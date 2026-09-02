"""
Pack existing per-segment feature .npy files into compact .npz shards.

This does NOT re-extract features.
It reads the already-created .npy files from features/ and packs them
into larger shard files so the project no longer contains ~130,000
individual feature files.

Usage
-----
python pack_features.py
python pack_features.py --shard-size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

FEATURES_ROOT = PROJECT_ROOT / "features"
MANIFEST_ROOT = PROJECT_ROOT / "outputs" / "features"

FEATURE_NAMES = (
    "mfcc",
    "mel",
    "spectral",
    "chroma",
    "zcr",
    "energy",
)

SPLITS = (
    "train",
    "validation",
    "test",
)


def load_feature(path: Path, feature_name: str) -> np.ndarray:
    """Load and validate one feature array."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {feature_name} feature file: {path}"
        )

    array = np.load(
        path,
        allow_pickle=False,
    )

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{feature_name} must be 2-D, got {array.shape}: {path}"
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"{feature_name} is empty: {path}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{feature_name} contains NaN/Inf: {path}"
        )

    return np.ascontiguousarray(array)


def feature_path_from_manifest(
    value: str,
    segment_id: str,
    feature_name: str,
) -> Path:
    """
    Resolve an existing feature path.

    Handles both absolute paths stored in the manifest and
    paths relative to the repository.
    """

    raw = Path(value)

    if raw.is_absolute() and raw.is_file():
        return raw

    if raw.is_absolute():
        parts = list(raw.parts)

        if "features" in parts:
            index = parts.index("features")

            candidate = (
                PROJECT_ROOT
                / Path(*parts[index:])
            )

            if candidate.is_file():
                return candidate

    candidate = (
        PROJECT_ROOT / raw
    ).resolve()

    if candidate.is_file():
        return candidate

    # Final canonical fallback.
    canonical = (
        FEATURES_ROOT
        / feature_name
        / f"{segment_id.replace(':', '_')}.npy"
    )

    if canonical.is_file():
        return canonical

    raise FileNotFoundError(
        f"Could not find {feature_name} feature for "
        f"segment '{segment_id}'."
    )


def pack_split(
    split: str,
    shard_size: int,
) -> None:
    """Pack one split into .npz shards."""

    manifest_path = (
        MANIFEST_ROOT
        / f"{split}_feature_manifest.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Feature manifest not found: {manifest_path}"
        )

    dataframe = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
    )

    if "status" in dataframe.columns:
        dataframe = dataframe[
            dataframe["status"].isin(
                ("success", "skipped")
            )
        ].copy()

    dataframe = dataframe.reset_index(drop=True)

    if dataframe.empty:
        raise ValueError(
            f"No usable feature rows found for {split}."
        )

    output_directory = (
        FEATURES_ROOT
        / "shards"
        / split
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_rows: list[dict[str, str | int]] = []

    total = len(dataframe)

    print()
    print("=" * 60)
    print(f"PACKING {split.upper()}")
    print("=" * 60)
    print(f"Samples: {total}")
    print(f"Shard size: {shard_size}")
    print()

    shard_number = 0

    for start in range(
        0,
        total,
        shard_size,
    ):
        end = min(
            start + shard_size,
            total,
        )

        batch = dataframe.iloc[start:end]

        feature_batches: dict[str, list[np.ndarray]] = {
            name: []
            for name in FEATURE_NAMES
        }

        labels: list[int] = []
        segment_ids: list[str] = []

        print(
            f"Packing shard {shard_number:04d} "
            f"({start + 1}-{end}/{total})..."
        )

        for _, row in batch.iterrows():

            segment_id = row["segment_id"]

            for feature_name in FEATURE_NAMES:

                path = feature_path_from_manifest(
                    row[f"{feature_name}_path"],
                    segment_id,
                    feature_name,
                )

                array = load_feature(
                    path,
                    feature_name,
                )

                feature_batches[
                    feature_name
                ].append(array)

            labels.append(
                int(row["binary_label"])
            )

            segment_ids.append(
                segment_id
            )

        # Stack into:
        #
        # (batch, channels, time)
        #
        arrays = {
            feature_name: np.stack(
                values,
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            )
            for feature_name, values
            in feature_batches.items()
        }

        labels_array = np.asarray(
            labels,
            dtype=np.int64,
        )

        segment_ids_array = np.asarray(
            segment_ids,
            dtype=np.str_,
        )

        shard_path = (
            output_directory
            / f"shard_{shard_number:04d}.npz"
        )

        np.savez_compressed(
            shard_path,
            mfcc=arrays["mfcc"],
            mel=arrays["mel"],
            spectral=arrays["spectral"],
            chroma=arrays["chroma"],
            zcr=arrays["zcr"],
            energy=arrays["energy"],
            labels=labels_array,
            segment_ids=segment_ids_array,
        )

        for local_index, segment_id in enumerate(
            segment_ids
        ):
            shard_rows.append(
                {
                    "split": split,
                    "segment_id": segment_id,
                    "binary_label": labels[
                        local_index
                    ],
                    "shard_path": str(
                        shard_path
                    ),
                    "shard_index": local_index,
                }
            )

        shard_number += 1

    metadata_path = (
        MANIFEST_ROOT
        / f"{split}_shard_manifest.csv"
    )

    metadata = pd.DataFrame(
        shard_rows
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    print()
    print(
        f"Created {shard_number} shards."
    )
    print(
        f"Shard manifest: {metadata_path}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help=(
            "Number of segments per shard. "
            "Default: 256."
        ),
    )

    return parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    if arguments.shard_size <= 0:
        raise ValueError(
            "--shard-size must be greater than zero."
        )

    print()
    print("=" * 60)
    print("ACOUSTIC DRONE FEATURE SHARD BUILDER")
    print("=" * 60)

    for split in SPLITS:
        pack_split(
            split=split,
            shard_size=arguments.shard_size,
        )

    print()
    print("=" * 60)
    print("SHARD BUILD COMPLETE")
    print("=" * 60)
    print()
    print(
        "Your original .npy files have NOT been deleted."
    )
    print(
        "We will verify the new dataset before deleting them."
    )


if __name__ == "__main__":
    main()