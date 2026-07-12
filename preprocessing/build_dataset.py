"""Build a reproducible supervised-audio manifest from registered datasets.

Purpose
-------
Create a deterministic CSV index of files that are safe to use for binary
drone-presence training.  The manifest is an immutable *reference* to raw
audio; this module never copies, resamples, or modifies a raw recording.

Inputs
------
Enabled :class:`configs.dataset_rules.DatasetRule` entries and their source
directories below ``datasets/raw``.

Outputs
-------
``datasets/metadata/supervised_manifest.csv`` containing one row per explicitly
labelled supported audio file, plus a structured build summary in Python.

Dependencies
------------
Python standard library and the project's ``configs`` package.

Algorithm
---------
For each enabled dataset, the builder walks files in deterministic order,
matches each path against the dataset's approved directory label mapping,
computes a streaming SHA-256 digest, and writes the row to a temporary CSV.
The temporary file is atomically replaced at completion so an interrupted run
never leaves a partial manifest.  Time is ``O(n * file_size)`` because hashes
must read every included file; memory is ``O(1)`` excluding path strings.

Usage
-----
From the repository root::

    python -m preprocessing.build_dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Sequence

# Support direct execution (``python preprocessing/build_dataset.py``) as well
# as the preferred module form (``python -m preprocessing.build_dataset``).
# Direct execution otherwise exposes only the ``preprocessing`` directory to
# Python, which prevents imports from the sibling ``configs`` package.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import METADATA_DIR, RAW_DATASET_DIR, SUPPORTED_AUDIO_EXTENSIONS
from configs.dataset_rules import DatasetRule, LabelSource, get_enabled_dataset_rules


LOGGER = logging.getLogger(__name__)
DEFAULT_MANIFEST_FILENAME = "supervised_manifest.csv"
HASH_CHUNK_SIZE_BYTES = 1024 * 1024
MANIFEST_COLUMNS = (
    "dataset",
    "relative_path",
    "file_name",
    "extension",
    "binary_label",
    "label_origin",
    "size_bytes",
    "sha256",
)


class ManifestBuildError(RuntimeError):
    """Raised when a supervised dataset manifest cannot be built safely."""


@dataclass(frozen=True, slots=True)
class ManifestBuildSummary:
    """Counts and location produced by one manifest build.

    Attributes:
        manifest_path: Completed CSV manifest path.
        included_files: Labelled audio files written to the manifest.
        skipped_unlabelled_files: Supported files intentionally excluded because
            no approved label mapping matched their path.
        missing_enabled_datasets: Enabled registry entries whose raw directory
            was unavailable when the build ran.
        created_at_utc: ISO-8601 creation timestamp in UTC.
    """

    manifest_path: Path
    included_files: int
    skipped_unlabelled_files: int
    missing_enabled_datasets: tuple[str, ...]
    created_at_utc: str


def _iter_audio_files(dataset_directory: Path) -> Iterator[Path]:
    """Yield supported audio files in deterministic directory order.

    Args:
        dataset_directory: Root directory for one registered raw dataset.

    Yields:
        Supported audio files below ``dataset_directory``.
    """

    for current_root, directory_names, file_names in os.walk(dataset_directory):
        directory_names.sort()
        for file_name in sorted(file_names):
            candidate = Path(current_root) / file_name
            if candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                yield candidate


def _resolve_directory_label(
    file_path: Path, dataset_directory: Path, rule: DatasetRule
) -> int | None:
    """Resolve a binary label using a rule's approved relative directories.

    Args:
        file_path: Candidate audio file inside the rule's dataset directory.
        dataset_directory: Dataset root used for this build.
        rule: Dataset policy defining the approved label directories.

    Returns:
        ``0`` or ``1`` when a mapping matches, otherwise ``None``.

    Raises:
        ManifestBuildError: If the file is not located below the dataset root.
    """

    try:
        relative_file = file_path.relative_to(dataset_directory)
    except ValueError as error:
        raise ManifestBuildError(
            f"File '{file_path}' is outside registered dataset '{rule.name}'."
        ) from error

    path_parts = relative_file.parts[:-1]
    for directory, label in sorted(
        rule.directory_label_map().items(),
        key=lambda item: len(Path(item[0]).parts),
        reverse=True,
    ):
        directory_parts = Path(directory).parts
        if path_parts[: len(directory_parts)] == directory_parts:
            return label
    return None


def _sha256_file(file_path: Path) -> str:
    """Calculate a SHA-256 digest without loading an entire file into memory.

    Args:
        file_path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with file_path.open("rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest_row(
    file_path: Path, dataset_directory: Path, rule: DatasetRule, binary_label: int
) -> dict[str, str | int]:
    """Create one manifest row for a labelled directory-based audio file.

    Args:
        file_path: Labelled audio file below the dataset root.
        dataset_directory: Dataset root used for this build.
        rule: Source dataset rule.
        binary_label: Previously validated binary target.

    Returns:
        CSV-compatible manifest row.

    Raises:
        ManifestBuildError: If the configured label source is unsupported.
    """

    if rule.label_source is not LabelSource.DIRECTORY:
        raise ManifestBuildError(
            f"Dataset '{rule.name}' uses '{rule.label_source}', which requires "
            "a dedicated annotation adapter before it can build a manifest."
        )

    relative_path = file_path.relative_to(dataset_directory)
    return {
        "dataset": rule.name,
        "relative_path": relative_path.as_posix(),
        "file_name": file_path.name,
        "extension": file_path.suffix.lower(),
        "binary_label": binary_label,
        "label_origin": f"directory:{relative_path.parts[0]}",
        "size_bytes": file_path.stat().st_size,
        "sha256": _sha256_file(file_path),
    }


def build_supervised_manifest(
    raw_datasets_directory: Path = RAW_DATASET_DIR,
    manifest_path: Path | None = None,
    dataset_rules: Sequence[DatasetRule] | None = None,
) -> ManifestBuildSummary:
    """Build a deterministic manifest from explicitly enabled datasets.

    Args:
        raw_datasets_directory: Root directory holding all raw dataset folders.
        manifest_path: Target CSV path. Defaults to the project metadata folder.
        dataset_rules: Optional rules for testing or a scoped experiment. When
            omitted, the enabled global registry entries are used.

    Returns:
        Summary of the completed build.

    Raises:
        ManifestBuildError: If no dataset rule is supplied or a selected rule
            declares a label source that this generic builder cannot handle.
        OSError: If files cannot be read or the manifest cannot be written.
    """

    selected_rules = (
        get_enabled_dataset_rules()
        if dataset_rules is None
        else tuple(dataset_rules)
    )
    if not selected_rules:
        raise ManifestBuildError("No enabled dataset rules were supplied.")

    destination = manifest_path or METADATA_DIR / DEFAULT_MANIFEST_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    included_files = 0
    skipped_unlabelled_files = 0
    missing_enabled_datasets: list[str] = []

    # Keep temporary output beside the final CSV so Path.replace is atomic on
    # the same filesystem.
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
            for rule in selected_rules:
                dataset_directory = raw_datasets_directory / rule.raw_directory_name
                if not dataset_directory.is_dir():
                    LOGGER.warning(
                        "Enabled dataset directory is missing: %s", dataset_directory
                    )
                    missing_enabled_datasets.append(rule.name)
                    continue

                if rule.label_source is not LabelSource.DIRECTORY:
                    raise ManifestBuildError(
                        f"Dataset '{rule.name}' is enabled but uses "
                        f"'{rule.label_source}'. Implement its annotation adapter "
                        "before enabling it."
                    )

                for audio_file in _iter_audio_files(dataset_directory):
                    label = _resolve_directory_label(
                        audio_file, dataset_directory, rule
                    )

                    if label is None:
                        skipped_unlabelled_files += 1
                        continue

                    row = _build_manifest_row(
                        audio_file, dataset_directory, rule, label
                    )
                    writer.writerow(row)
                    included_files += 1
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(destination)
    created_at_utc = datetime.now(UTC).isoformat()
    summary = ManifestBuildSummary(
        manifest_path=destination,
        included_files=included_files,
        skipped_unlabelled_files=skipped_unlabelled_files,
        missing_enabled_datasets=tuple(missing_enabled_datasets),
        created_at_utc=created_at_utc,
    )
    LOGGER.info(
        "Built supervised manifest at %s with %d labelled files; skipped %d "
        "unlabelled files.",
        summary.manifest_path,
        summary.included_files,
        summary.skipped_unlabelled_files,
    )
    return summary


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the manifest builder.

    Returns:
        Parsed namespace containing an optional output manifest path.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination CSV path; defaults to datasets/metadata.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the manifest builder command-line entry point.

    Raises:
        ManifestBuildError: If the enabled registry is unsafe to build.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    arguments = _parse_arguments()
    summary = build_supervised_manifest(manifest_path=arguments.output)
    LOGGER.info("Manifest build completed: %s", summary)


if __name__ == "__main__":
    main()
