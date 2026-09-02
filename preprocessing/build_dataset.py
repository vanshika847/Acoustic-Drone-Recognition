"""
Build a reproducible supervised-audio manifest from registered datasets.

Purpose
-------
Create a deterministic CSV index of files that are explicitly labelled as
drone-positive (binary_label = 1).

Background recordings (binary_label = 0) are intentionally NOT written to
this manifest. They are handled by build_background_manifest.py.

The manifest is an immutable reference to raw audio; this module never copies,
resamples, or modifies a raw recording.

Outputs
-------
datasets/metadata/supervised_manifest.csv
"""

from __future__ import annotations
from preprocessing.adapters.ddl import DDLAdapter
import argparse
import csv
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Support direct execution as well as module execution
# ---------------------------------------------------------------------------

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from configs.config import (
    METADATA_DIR,
    RAW_DATASET_DIR,
)

from configs.dataset_rules import (
    DatasetRule,
    get_enabled_dataset_rules,
)

from preprocessing.adapters.al_emadi import AlEmadiAdapter
from preprocessing.adapters.drone_audio import DroneAudioAdapter
from preprocessing.adapters.kaist import KaistAdapter
from preprocessing.adapters.uavirbase import UAVirBaseAdapter


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset adapter registry
# ---------------------------------------------------------------------------

DATASET_ADAPTERS = {
    "drone_audio": DroneAudioAdapter,
    "al_emadi": AlEmadiAdapter,
    "kaist": KaistAdapter,
    "uavirbase": UAVirBaseAdapter,
    "ddl": DDLAdapter,
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

    # Drone information
    "drone_type",
    "movement",
    "rotation",

    # Position
    "distance_m",
    "height_m",
    "azimuth_deg",

    # Recording
    "recording_start",
    "recording_end",

    # Environment
    "temperature_c",
    "humidity_percent",
    "wind_speed_ms",
    "wind_direction_deg",

    # GPS
    "latitude",
    "longitude",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ManifestBuildError(RuntimeError):
    """Raised when a supervised dataset manifest cannot be built safely."""


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ManifestBuildSummary:
    """Summary produced by one supervised manifest build."""

    manifest_path: Path
    included_files: int
    skipped_non_positive_files: int
    missing_enabled_datasets: tuple[str, ...]
    created_at_utc: str


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_supervised_manifest(
    raw_datasets_directory: Path = RAW_DATASET_DIR,
    manifest_path: Path | None = None,
    dataset_rules: Sequence[DatasetRule] | None = None,
) -> ManifestBuildSummary:
    """
    Build a deterministic drone-positive manifest.

    Only rows with binary_label == 1 are written.

    Background rows with binary_label == 0 are intentionally excluded because
    they belong in background_manifest.csv.
    """

    selected_rules = (
        get_enabled_dataset_rules()
        if dataset_rules is None
        else tuple(dataset_rules)
    )

    if not selected_rules:
        raise ManifestBuildError(
            "No enabled dataset rules were supplied."
        )

    destination = (
        manifest_path
        or METADATA_DIR / DEFAULT_MANIFEST_FILENAME
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    included_files = 0
    skipped_non_positive_files = 0
    missing_enabled_datasets: list[str] = []

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            writer = csv.DictWriter(
                temporary_file,
                fieldnames=MANIFEST_COLUMNS,
            )

            writer.writeheader()

            for rule in selected_rules:

                dataset_directory = (
                    raw_datasets_directory
                    / rule.raw_directory_name
                )

                if not dataset_directory.is_dir():

                    LOGGER.warning(
                        "Enabled dataset directory is missing: %s",
                        dataset_directory,
                    )

                    missing_enabled_datasets.append(
                        rule.name
                    )

                    continue

                adapter_class = DATASET_ADAPTERS.get(
                    rule.name
                )

                if adapter_class is None:
                    raise ManifestBuildError(
                        f"No adapter registered for dataset "
                        f"'{rule.name}'."
                    )

                LOGGER.info(
                    "Processing dataset: %s",
                    rule.name,
                )

                adapter = adapter_class()

                for row in adapter.build_rows(
                    dataset_directory,
                    rule,
                ):

                    binary_label = row.get(
                        "binary_label"
                    )

                    # --------------------------------------------------
                    # supervised_manifest contains ONLY positives.
                    # --------------------------------------------------

                    if str(binary_label) != "1":

                        skipped_non_positive_files += 1

                        LOGGER.debug(
                            "Skipping non-positive row: %s",
                            row.get("relative_path"),
                        )

                        continue

                    writer.writerow(row)

                    included_files += 1

    except Exception:

        # Close the temporary file before attempting deletion.
        # This avoids Windows WinError 32.
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except PermissionError:
                LOGGER.warning(
                    "Could not immediately remove temporary file: %s",
                    temporary_path,
                )

        raise

    # -----------------------------------------------------------------------
    # Atomic replacement
    # -----------------------------------------------------------------------

    if temporary_path is None:
        raise ManifestBuildError(
            "Temporary manifest file was not created."
        )

    try:
        temporary_path.replace(
            destination
        )

    except Exception:

        try:
            temporary_path.unlink(
                missing_ok=True
            )
        except PermissionError:
            LOGGER.warning(
                "Could not remove temporary manifest: %s",
                temporary_path,
            )

        raise

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    created_at_utc = datetime.now(
        UTC
    ).isoformat()

    summary = ManifestBuildSummary(
        manifest_path=destination,
        included_files=included_files,
        skipped_non_positive_files=(
            skipped_non_positive_files
        ),
        missing_enabled_datasets=(
            tuple(missing_enabled_datasets)
        ),
        created_at_utc=created_at_utc,
    )

    LOGGER.info(
        "Built supervised manifest at %s with %d "
        "positive files; skipped %d non-positive files.",
        summary.manifest_path,
        summary.included_files,
        summary.skipped_non_positive_files,
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional destination CSV path; "
            "defaults to datasets/metadata."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the supervised manifest builder."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    arguments = _parse_arguments()

    summary = build_supervised_manifest(
        manifest_path=arguments.output
    )

    LOGGER.info(
        "Manifest build completed: %s",
        summary,
    )


if __name__ == "__main__":
    main()