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
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

# Support direct execution (``python preprocessing/build_dataset.py``) as well
# as the preferred module form (``python -m preprocessing.build_dataset``).
# Direct execution otherwise exposes only the ``preprocessing`` directory to
# Python, which prevents imports from the sibling ``configs`` package.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    METADATA_DIR,
    RAW_DATASET_DIR,
)
from configs.dataset_rules import (
    DatasetRule,
    get_enabled_dataset_rules,
)
from preprocessing.adapters.drone_audio import DroneAudioAdapter
from preprocessing.adapters.uavirbase import UAVirBaseAdapter


LOGGER = logging.getLogger(__name__)
DEFAULT_MANIFEST_FILENAME = "supervised_manifest.csv"
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
# ==========================================================
# Dataset Adapters
# ==========================================================

DATASET_ADAPTERS = {
    "drone_audio": DroneAudioAdapter,
    "uavirbase": UAVirBaseAdapter,
}


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

                adapter_class = DATASET_ADAPTERS.get(rule.name)
                if adapter_class is None:
                    raise ManifestBuildError(
                          f"No adapter registered for dataset '{rule.name}'."
                    )

                adapter = adapter_class()

                for row in adapter.build_rows(
                    dataset_directory,
                    rule,
                ):
                    writer.writerow(row)
                    included_files += 1
                    
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
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
