"""Build an auditable no-drone manifest from official background metadata.

Purpose
-------
Produce a deterministic manifest of recordings with a verified ``0``
(``no_drone``) target.  The module currently supports ESC-50 and UrbanSound8K,
whose local CSV annotation files explicitly identify every audio clip.

Inputs
------
``datasets/raw/esc50/esc50_labels.csv`` and
``datasets/raw/urbansound8k/UrbanSound8K.csv``, alongside their audio files.

Outputs
-------
``datasets/metadata/background_manifest.csv``.  The raw files are never
modified.  Each manifest row retains the source category and a recording group
identifier to prevent source-recording leakage during model evaluation.

Dependencies
------------
Python standard library and :mod:`configs.config`.

Algorithm
---------
The adapter streams each official CSV in row order, resolves its documented
audio path, verifies that the file exists, calculates a streaming SHA-256 hash,
then writes to a temporary CSV.  The temporary file is atomically renamed only
after successful completion.  The processing cost is ``O(n * file_size)`` for
hashing and constant additional memory.

Usage
-----
From the repository root::

    python -m preprocessing.build_background_manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# Support direct execution (``python preprocessing/build_background_manifest.py``)
# as well as the preferred module form (``python -m preprocessing.build_background_manifest``).
# Direct execution otherwise exposes only the ``preprocessing`` directory to the
# import system, preventing imports from the sibling ``configs`` package.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import METADATA_DIR, RAW_DATASET_DIR


LOGGER = logging.getLogger(__name__)
BINARY_NO_DRONE_LABEL = 0
DEFAULT_MANIFEST_FILENAME = "background_manifest.csv"
HASH_CHUNK_SIZE_BYTES = 1024 * 1024
MANIFEST_COLUMNS = (
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


class BackgroundManifestError(RuntimeError):
    """Raised when a background dataset cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class BackgroundRecord:
    """A validated, labelled background recording ready for CSV output.

    Attributes:
        dataset: Stable source dataset identifier.
        audio_path: Existing location of the raw audio file.
        relative_path: Path relative to the source dataset directory.
        category: Official source category.
        source_fold: Official dataset fold identifier.
        recording_group_id: Original recording identifier for leakage-safe splits.
    """

    dataset: str
    audio_path: Path
    relative_path: Path
    category: str
    source_fold: str
    recording_group_id: str


@dataclass(frozen=True, slots=True)
class BackgroundManifestSummary:
    """Summary of a completed no-drone manifest build."""

    manifest_path: Path
    included_files: int
    missing_audio_files: int
    datasets: tuple[str, ...]
    created_at_utc: str


def _sha256_file(file_path: Path) -> str:
    """Return the SHA-256 digest of a file using bounded memory.

    Args:
        file_path: Audio file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with file_path.open("rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(
    field_names: Sequence[str] | None,
    required_columns: set[str],
    annotation_path: Path,
) -> None:
    """Validate that an official annotation CSV has its required columns.

    Args:
        field_names: Header row returned by :class:`csv.DictReader`.
        required_columns: Required header names for one adapter.
        annotation_path: File being checked for diagnostic context.

    Raises:
        BackgroundManifestError: If the file has no header or required columns.
    """

    available_columns = set(field_names or ())
    missing_columns = required_columns.difference(available_columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise BackgroundManifestError(
            f"Annotation file '{annotation_path}' is missing required columns: "
            f"{missing_text}."
        )


def iter_esc50_records(dataset_directory: Path) -> Iterator[BackgroundRecord]:
    """Yield no-drone records from the official ESC-50 label CSV.

    Args:
        dataset_directory: Local ``esc50`` dataset root.

    Yields:
        Records with paths resolved beneath ``wav_files``.

    Raises:
        BackgroundManifestError: If the expected metadata format is unavailable.
    """

    annotation_path = dataset_directory / "esc50_labels.csv"
    if not annotation_path.is_file():
        raise BackgroundManifestError(
            f"ESC-50 annotation file was not found: '{annotation_path}'."
        )

    with annotation_path.open(encoding="utf-8", newline="") as annotation_file:
        reader = csv.DictReader(annotation_file)
        _require_columns(
            reader.fieldnames,
            {"filename", "fold", "category", "src_file"},
            annotation_path,
        )
        for row in reader:
            filename = row["filename"].strip()
            audio_path = dataset_directory / "wav_files" / filename
            yield BackgroundRecord(
                dataset="esc50",
                audio_path=audio_path,
                relative_path=audio_path.relative_to(dataset_directory),
                category=row["category"].strip(),
                source_fold=row["fold"].strip(),
                recording_group_id=f"esc50:{row['src_file'].strip()}",
            )


def iter_urbansound8k_records(dataset_directory: Path) -> Iterator[BackgroundRecord]:
    """Yield no-drone records from the official UrbanSound8K metadata CSV.

    Args:
        dataset_directory: Local ``urbansound8k`` dataset root.

    Yields:
        Records with paths resolved beneath their official ``foldN`` directory.

    Raises:
        BackgroundManifestError: If the expected metadata format is unavailable.
    """

    annotation_path = dataset_directory / "UrbanSound8K.csv"
    if not annotation_path.is_file():
        raise BackgroundManifestError(
            f"UrbanSound8K annotation file was not found: '{annotation_path}'."
        )

    with annotation_path.open(encoding="utf-8", newline="") as annotation_file:
        reader = csv.DictReader(annotation_file)
        _require_columns(
            reader.fieldnames,
            {"slice_file_name", "fsID", "fold", "class"},
            annotation_path,
        )
        for row in reader:
            fold = row["fold"].strip()
            filename = row["slice_file_name"].strip()
            audio_path = dataset_directory / f"fold{fold}" / filename
            yield BackgroundRecord(
                dataset="urbansound8k",
                audio_path=audio_path,
                relative_path=audio_path.relative_to(dataset_directory),
                category=row["class"].strip(),
                source_fold=fold,
                recording_group_id=f"urbansound8k:{row['fsID'].strip()}",
            )


def _records_for_dataset(
    dataset_name: str, raw_datasets_directory: Path
) -> Iterable[BackgroundRecord]:
    """Return the registered adapter iterable for one background dataset.

    Args:
        dataset_name: Supported dataset identifier.
        raw_datasets_directory: Root directory containing raw datasets.

    Returns:
        Iterable of official background records.

    Raises:
        BackgroundManifestError: If no adapter is registered for the name.
    """

    adapters = {
        "esc50": iter_esc50_records,
        "urbansound8k": iter_urbansound8k_records,
    }
    try:
        return adapters[dataset_name](raw_datasets_directory / dataset_name)
    except KeyError as error:
        supported = ", ".join(sorted(adapters))
        raise BackgroundManifestError(
            f"Unsupported background dataset '{dataset_name}'. Supported: {supported}."
        ) from error


def _manifest_row(record: BackgroundRecord) -> dict[str, str | int]:
    """Convert an existing background record into a CSV-compatible row.

    Args:
        record: Validated source record with an existing audio file.

    Returns:
        Manifest row with a verified no-drone target.
    """

    return {
        "dataset": record.dataset,
        "relative_path": record.relative_path.as_posix(),
        "file_name": record.audio_path.name,
        "binary_label": BINARY_NO_DRONE_LABEL,
        "label_origin": "official_metadata:no_drone",
        "source_category": record.category,
        "source_fold": record.source_fold,
        "recording_group_id": record.recording_group_id,
        "size_bytes": record.audio_path.stat().st_size,
        "sha256": _sha256_file(record.audio_path),
    }


def build_background_manifest(
    raw_datasets_directory: Path = RAW_DATASET_DIR,
    manifest_path: Path | None = None,
    dataset_names: Sequence[str] = ("esc50", "urbansound8k"),
) -> BackgroundManifestSummary:
    """Build a labelled no-drone manifest from official source annotations.

    Args:
        raw_datasets_directory: Root containing the raw source datasets.
        manifest_path: Optional CSV destination; defaults to project metadata.
        dataset_names: Registered adapters to include in deterministic order.

    Returns:
        Summary of the completed manifest build.

    Raises:
        BackgroundManifestError: If no datasets were selected or a source CSV is
            structurally invalid.
        OSError: If source audio or the destination cannot be read or written.
    """

    selected_datasets = tuple(dataset_names)
    if not selected_datasets:
        raise BackgroundManifestError("At least one background dataset is required.")

    destination = manifest_path or METADATA_DIR / DEFAULT_MANIFEST_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    included_files = 0
    missing_audio_files = 0

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
        writer = csv.DictWriter(temporary_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        try:
            for dataset_name in selected_datasets:
                for record in _records_for_dataset(
                    dataset_name, raw_datasets_directory
                ):
                    if not record.audio_path.is_file():
                        missing_audio_files += 1
                        LOGGER.warning(
                            "Skipping annotation row with missing audio: %s",
                            record.audio_path,
                        )
                        continue
                    writer.writerow(_manifest_row(record))
                    included_files += 1
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(destination)
    summary = BackgroundManifestSummary(
        manifest_path=destination,
        included_files=included_files,
        missing_audio_files=missing_audio_files,
        datasets=selected_datasets,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    LOGGER.info(
        "Built background manifest at %s with %d files; %d annotation rows "
        "referenced missing audio.",
        summary.manifest_path,
        summary.included_files,
        summary.missing_audio_files,
    )
    return summary


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the background manifest builder.

    Returns:
        Parsed arguments for output location and source dataset selection.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination CSV path; defaults to datasets/metadata.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("esc50", "urbansound8k"),
        help="Background dataset adapters to include.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point for the background manifest builder."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    arguments = _parse_arguments()
    summary = build_background_manifest(
        manifest_path=arguments.output,
        dataset_names=arguments.datasets,
    )
    LOGGER.info("Background manifest build completed: %s", summary)


if __name__ == "__main__":
    main()
