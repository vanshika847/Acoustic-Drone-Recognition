"""Merge labelled audio manifests and build leakage-safe dataset splits.

Purpose
-------
Convert the verified positive and background manifests into one canonical
dataset index and deterministic train, validation, and test CSV manifests.
Exact duplicate audio is removed by SHA-256; any identical audio assigned
conflicting labels is quarantined and never reaches a training split.

Inputs
------
``datasets/metadata/supervised_manifest.csv`` and
``datasets/metadata/background_manifest.csv``.

Outputs
-------
``datasets/metadata/combined_manifest.csv``, ``manifest_conflicts.csv``, and
the three split manifests under ``datasets/processed/manifests``.

Dependencies
------------
Python standard library and :mod:`configs.config`.

Algorithm
---------
Rows are normalised into one schema and grouped by SHA-256.  A hash with a
single binary label is represented once; a hash with conflicting labels is
quarantined.  Remaining files are grouped by their original recording before
a seeded, label-aware greedy allocator assigns whole groups to splits.  This
prevents identical recordings and sequential chunks from crossing split
boundaries.  Runtime is ``O(n log n)`` for grouping and sorting; memory is
``O(n)`` for the manifest rows.

Usage
-----
From the repository root::

    python -m preprocessing.prepare_training_manifests
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Support direct execution (``python preprocessing/prepare_training_manifests.py``)
# as well as the preferred module form.  Direct execution otherwise omits the
# repository root from Python's module search path.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import METADATA_DIR, PROCESSED_DATASET_DIR


LOGGER = logging.getLogger(__name__)
DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
POSITIVE_MANIFEST_NAME = "supervised_manifest.csv"
BACKGROUND_MANIFEST_NAME = "background_manifest.csv"
COMBINED_MANIFEST_NAME = "combined_manifest.csv"
CONFLICT_MANIFEST_NAME = "manifest_conflicts.csv"
SPLIT_MANIFEST_DIRECTORY_NAME = "manifests"
SPLIT_NAMES = tuple(DEFAULT_SPLIT_RATIOS)
POSITIVE_CHUNK_SUFFIX = re.compile(r"_\d+_?$")
CANONICAL_COLUMNS = (
    "split",
    "dataset",
    "relative_path",
    "file_name",
    "binary_label",
    "label_origin",
    "source_category",
    "source_fold",
    "recording_group_id",
    "size_bytes",
    "sha256",
)
CONFLICT_COLUMNS = CANONICAL_COLUMNS + ("conflict_reason",)


class ManifestPreparationError(RuntimeError):
    """Raised when manifests cannot safely form a trainable dataset."""


@dataclass(frozen=True, slots=True)
class ManifestPreparationSummary:
    """Counts and locations resulting from one training-manifest preparation.

    Attributes:
        combined_manifest_path: Canonical deduplicated manifest location.
        conflict_manifest_path: Rows excluded due to contradictory labels.
        split_manifest_paths: Mapping of split name to its CSV location.
        input_rows: Total rows read from the two source manifests.
        retained_rows: Rows available after deduplication and conflict removal.
        duplicate_rows_removed: Same-label duplicate rows removed by hash.
        conflict_rows_quarantined: Rows excluded because one hash had both labels.
        split_counts: Retained row count per output split.
    """

    combined_manifest_path: Path
    conflict_manifest_path: Path
    split_manifest_paths: Mapping[str, Path]
    input_rows: int
    retained_rows: int
    duplicate_rows_removed: int
    conflict_rows_quarantined: int
    split_counts: Mapping[str, int]


def _read_csv_rows(manifest_path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read and validate a CSV manifest with UTF-8 encoding.

    Args:
        manifest_path: Source manifest path.
        required_columns: Header names required by the caller.

    Returns:
        Manifest rows in their source order.

    Raises:
        ManifestPreparationError: If the file or required columns are missing.
    """

    if not manifest_path.is_file():
        raise ManifestPreparationError(f"Manifest was not found: '{manifest_path}'.")

    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        available_columns = set(reader.fieldnames or ())
        missing_columns = required_columns.difference(available_columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ManifestPreparationError(
                f"Manifest '{manifest_path}' is missing columns: {missing_text}."
            )
        return list(reader)


def _derive_positive_group_id(
    dataset: str,
    relative_path: str,
    file_name: str,
) -> str:
    """Derive a leakage-safe recording group for positive recordings.

    Grouping rules are dataset-specific so that chunks from the same
    physical recording never cross train/validation/test boundaries.
    """

    relative = Path(relative_path)

    # ---------------------------------------------------------
    # Al-Emadi
    # ---------------------------------------------------------
    # Each labelled audio file is treated as its own recording.
    if dataset == "al_emadi":
        return f"{dataset}:{relative.as_posix()}"

    # ---------------------------------------------------------
    # UaVirBASE
    # ---------------------------------------------------------
    # Each recording has its own directory.
    if dataset == "uavirbase":
        return f"{dataset}:{relative.parent.as_posix()}"

    # ---------------------------------------------------------
    # DDL
    # ---------------------------------------------------------
    # DDL filenames contain:
    #
    # ...<flight_session>-<recording_session>-<sequence>
    #
    # Example:
    # 20210329141240MINI0030240312886R290321-T004-005236.wav
    #
    # 290321 = flight session
    # T004   = recording session
    # 005236 = individual audio sequence/chunk
    #
    # All sequence chunks belonging to the same recording session
    # must stay in the same train/validation/test split.
        # ---------------------------------------------------------
    # DDL
    # ---------------------------------------------------------
    # DDL contains many sequential WAV chunks belonging to the
    # same recording/session directory. Keep the entire directory
    # in one split so chunks from the same recording can never
    # leak between train/validation/test.
    if dataset == "ddl":
        parent = relative.parent

        if str(parent) not in {"", "."}:
            return f"{dataset}:{parent.as_posix()}"

        # Safety fallback if a DDL file is directly under datasets/raw/ddl.
        return f"{dataset}:{Path(file_name).stem}"

    # ---------------------------------------------------------
    # Fallback for datasets without a dedicated grouping rule.
    # ---------------------------------------------------------
    stem = Path(file_name).stem

    group_stem = POSITIVE_CHUNK_SUFFIX.sub(
        "",
        stem,
    )

    return f"{dataset}:{group_stem or stem}"

    # ---------------------------------------------------------
    # Default / other datasets
    # ---------------------------------------------------------
    # Remove a trailing numeric chunk index when present.
    stem = Path(file_name).stem

    group_stem = POSITIVE_CHUNK_SUFFIX.sub(
        "",
        stem,
    )

    return f"{dataset}:{group_stem or stem}"


def _normalise_positive_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Map the positive manifest schema into the canonical split schema.

    Args:
        rows: Rows from ``supervised_manifest.csv``.

    Returns:
        Canonical positive rows with a derived recording group.

    Raises:
        ManifestPreparationError: If a row is not explicitly drone-positive.
    """

    normalised_rows: list[dict[str, str]] = []
    for row in rows:
        if row["binary_label"] != "1":
            raise ManifestPreparationError(
                "The supervised manifest contains a non-positive row: "
                f"'{row['file_name']}'."
            )
        normalised_rows.append(
            {
                "split": "",
                "dataset": row["dataset"],
                "relative_path": row["relative_path"],
                "file_name": row["file_name"],
                "binary_label": "1",
                "label_origin": row["label_origin"],
                "source_category": "drone",
                "source_fold": "",
                "recording_group_id": _derive_positive_group_id(
                    row["dataset"],
                    row["relative_path"],
                    row["file_name"],
                ),
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
        )
    return normalised_rows


def _normalise_background_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Map the background manifest schema into the canonical split schema.

    Args:
        rows: Rows from ``background_manifest.csv``.

    Returns:
        Canonical no-drone rows retaining their official group IDs.

    Raises:
        ManifestPreparationError: If a row is not explicitly no-drone.
    """

    normalised_rows: list[dict[str, str]] = []
    for row in rows:
        if row["binary_label"] != "0":
            raise ManifestPreparationError(
                "The background manifest contains a non-background row: "
                f"'{row['file_name']}'."
            )
        normalised_rows.append(
            {
                "split": "",
                "dataset": row["dataset"],
                "relative_path": row["relative_path"],
                "file_name": row["file_name"],
                "binary_label": "0",
                "label_origin": row["label_origin"],
                "source_category": row["source_category"],
                "source_fold": row["source_fold"],
                "recording_group_id": row["recording_group_id"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
        )
    return normalised_rows


def _deduplicate_by_hash(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    """Deduplicate same-label audio and quarantine conflicting hash labels.

    Args:
        rows: Canonical positive and background rows.

    Returns:
        Tuple of retained rows, conflict rows, and count of same-label rows
        removed as exact duplicates.
    """

    rows_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_hash[row["sha256"]].append(row)

    retained_rows: list[dict[str, str]] = []
    conflict_rows: list[dict[str, str]] = []
    duplicate_rows_removed = 0
    for audio_hash in sorted(rows_by_hash):
        matching_rows = rows_by_hash[audio_hash]
        labels = {row["binary_label"] for row in matching_rows}
        if len(labels) > 1:
            for row in matching_rows:
                conflict_rows.append(
                    {
                        **row,
                        "conflict_reason": "same_sha256_has_conflicting_binary_labels",
                    }
                )
            continue
        retained_rows.append(matching_rows[0])
        duplicate_rows_removed += len(matching_rows) - 1
    return retained_rows, conflict_rows, duplicate_rows_removed


def _validate_ratios(split_ratios: Mapping[str, float]) -> None:
    """Validate output split names and proportions.

    Args:
        split_ratios: Requested mapping from split name to fraction.

    Raises:
        ManifestPreparationError: If names or fractions are invalid.
    """

    if tuple(split_ratios) != SPLIT_NAMES:
        raise ManifestPreparationError(
            f"Split names must be exactly {SPLIT_NAMES} in that order."
        )
    if any(ratio <= 0.0 for ratio in split_ratios.values()):
        raise ManifestPreparationError("Every split ratio must be greater than zero.")
    if abs(sum(split_ratios.values()) - 1.0) > 1e-9:
        raise ManifestPreparationError("Split ratios must sum to 1.0.")


def _assign_groups_to_splits(
    rows: Sequence[dict[str, str]],
    split_ratios: Mapping[str, float],
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Assign whole recording groups to balanced, deterministic data splits.

    Args:
        rows: Canonical, deduplicated rows.
        split_ratios: Train/validation/test output proportions.
        seed: Random seed controlling tie ordering only.

    Returns:
        Mapping from split name to rows assigned without group leakage.

    Raises:
        ManifestPreparationError: If grouping or class coverage is unsafe.
    """

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["recording_group_id"]].append(row)

    labels_per_group = {
        group_id: {row["binary_label"] for row in group_rows}
        for group_id, group_rows in groups.items()
    }
    mixed_groups = [
        group_id for group_id, labels in labels_per_group.items() if len(labels) != 1
    ]
    if mixed_groups:
        raise ManifestPreparationError(
            "A recording group contains both binary labels. Use dataset-specific "
            "group identifiers before splitting. Example group: "
            f"'{mixed_groups[0]}'."
        )

    groups_by_label: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group_id, group_rows in groups.items():
        label = next(iter(labels_per_group[group_id]))
        groups_by_label[label].append((group_id, group_rows))

    for label in ("0", "1"):
        if len(groups_by_label[label]) < len(split_ratios):
            raise ManifestPreparationError(
                f"Label {label} has only {len(groups_by_label[label])} recording "
                f"groups; at least {len(split_ratios)} are required for split coverage."
            )

    random_generator = random.Random(seed)
    split_rows = {split_name: [] for split_name in split_ratios}
    assigned_count = {
        split_name: {"0": 0, "1": 0} for split_name in split_ratios
    }

    for label in ("0", "1"):
        label_groups = groups_by_label[label]
        random_generator.shuffle(label_groups)
        # Allocate larger groups first, preserving deterministic shuffled order
        # for groups of equal size.
        label_groups.sort(key=lambda item: len(item[1]), reverse=True)
        total_label_rows = sum(len(group_rows) for _, group_rows in label_groups)
        target_count = {
            split_name: total_label_rows * ratio
            for split_name, ratio in split_ratios.items()
        }
        for group_index, (_, group_rows) in enumerate(label_groups):
            # Reserve one complete recording group for every split before
            # optimising the remaining rows. This guarantees both labels are
            # represented even when a dataset has only a few source groups.
            if group_index < len(split_ratios):
                best_split = SPLIT_NAMES[group_index]
            else:
                best_split = max(
                    split_ratios,
                    key=lambda split_name: (
                        target_count[split_name] - assigned_count[split_name][label],
                        -assigned_count[split_name][label],
                    ),
                )
            split_rows[best_split].extend(group_rows)
            assigned_count[best_split][label] += len(group_rows)

    for split_name, assigned_rows in split_rows.items():
        labels = {row["binary_label"] for row in assigned_rows}
        if labels != {"0", "1"}:
            raise ManifestPreparationError(
                f"Split '{split_name}' lacks complete binary label coverage."
            )
        for row in assigned_rows:
            row["split"] = split_name
        assigned_rows.sort(key=lambda row: (row["dataset"], row["relative_path"]))
    return split_rows


def _write_csv_atomically(
    destination: Path, field_names: Sequence[str], rows: Iterable[Mapping[str, str]]
) -> None:
    """Write a CSV through a same-directory temporary file and atomic rename.

    Args:
        destination: Output CSV path.
        field_names: Ordered CSV header fields.
        rows: Rows to write.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.stem}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        writer = csv.DictWriter(temporary_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def prepare_training_manifests(
    metadata_directory: Path = METADATA_DIR,
    processed_datasets_directory: Path = PROCESSED_DATASET_DIR,
    split_ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> ManifestPreparationSummary:
    """Create combined and split manifests from verified source manifests.

    Args:
        metadata_directory: Directory containing and receiving metadata CSVs.
        processed_datasets_directory: Root for processed split manifests.
        split_ratios: Ordered train/validation/test proportions totalling one.
        seed: Deterministic seed used during group-allocation tie breaking.

    Returns:
        Summary of retained, removed, quarantined, and split rows.

    Raises:
        ManifestPreparationError: If manifests are invalid or splitting is unsafe.
    """

    _validate_ratios(split_ratios)
    positive_rows = _read_csv_rows(
        metadata_directory / POSITIVE_MANIFEST_NAME,
        {"dataset", "relative_path", "file_name", "binary_label", "label_origin", "size_bytes", "sha256"},
    )
    background_rows = _read_csv_rows(
        metadata_directory / BACKGROUND_MANIFEST_NAME,
        {
            "dataset", "relative_path", "file_name", "binary_label", "label_origin",
            "source_category", "source_fold", "recording_group_id", "size_bytes", "sha256",
        },
    )
    input_rows = len(positive_rows) + len(background_rows)
    all_rows = _normalise_positive_rows(positive_rows) + _normalise_background_rows(
        background_rows
    )
    retained_rows, conflict_rows, duplicate_rows_removed = _deduplicate_by_hash(all_rows)
    split_rows = _assign_groups_to_splits(retained_rows, split_ratios, seed)

    combined_manifest_path = metadata_directory / COMBINED_MANIFEST_NAME
    conflict_manifest_path = metadata_directory / CONFLICT_MANIFEST_NAME
    split_directory = processed_datasets_directory / SPLIT_MANIFEST_DIRECTORY_NAME
    split_manifest_paths = {
        split_name: split_directory / f"{split_name}.csv" for split_name in split_ratios
    }
    _write_csv_atomically(combined_manifest_path, CANONICAL_COLUMNS, retained_rows)
    _write_csv_atomically(conflict_manifest_path, CONFLICT_COLUMNS, conflict_rows)
    for split_name, manifest_path in split_manifest_paths.items():
        _write_csv_atomically(manifest_path, CANONICAL_COLUMNS, split_rows[split_name])

    summary = ManifestPreparationSummary(
        combined_manifest_path=combined_manifest_path,
        conflict_manifest_path=conflict_manifest_path,
        split_manifest_paths=split_manifest_paths,
        input_rows=input_rows,
        retained_rows=len(retained_rows),
        duplicate_rows_removed=duplicate_rows_removed,
        conflict_rows_quarantined=len(conflict_rows),
        split_counts={name: len(rows) for name, rows in split_rows.items()},
    )
    LOGGER.info("Prepared training manifests: %s", summary)
    return summary


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line options for reproducible split generation.

    Returns:
        Namespace containing the requested seed.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic split seed (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the combined-manifest and leakage-safe split command-line task."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    arguments = _parse_arguments()
    prepare_training_manifests(seed=arguments.seed)


if __name__ == "__main__":
    main()
