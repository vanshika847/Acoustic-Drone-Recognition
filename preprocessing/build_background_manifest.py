"""
Build an auditable no-drone manifest from official background metadata.

Purpose
-------
Produce a deterministic manifest of recordings with a verified binary label
of 0 (no_drone).

Supported datasets:

- ESC-50
- UrbanSound8K
- UAViBase

The raw files are never modified.

Outputs
-------
datasets/metadata/background_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# Support direct execution.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

from configs.config import (
    METADATA_DIR,
    RAW_DATASET_DIR,
)


LOGGER = logging.getLogger(__name__)

BINARY_NO_DRONE_LABEL = 0

DEFAULT_MANIFEST_FILENAME = (
    "background_manifest.csv"
)

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
    """Raised when a background manifest cannot be built safely."""


@dataclass(frozen=True, slots=True)
class BackgroundRecord:
    """Validated background recording."""

    dataset: str
    audio_path: Path
    relative_path: Path
    category: str
    source_fold: str
    recording_group_id: str


@dataclass(frozen=True, slots=True)
class BackgroundManifestSummary:
    """Summary of a completed background manifest build."""

    manifest_path: Path
    included_files: int
    missing_audio_files: int
    datasets: tuple[str, ...]
    created_at_utc: str


def _sha256_file(file_path: Path) -> str:
    """Calculate SHA-256 using bounded memory."""

    digest = hashlib.sha256()

    with file_path.open("rb") as audio_file:

        for chunk in iter(
            lambda: audio_file.read(
                HASH_CHUNK_SIZE_BYTES
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _require_columns(
    field_names: Sequence[str] | None,
    required_columns: set[str],
    annotation_path: Path,
) -> None:
    """Validate required CSV columns."""

    available_columns = set(
        field_names or ()
    )

    missing_columns = (
        required_columns
        - available_columns
    )

    if missing_columns:

        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise BackgroundManifestError(
            f"Annotation file '{annotation_path}' "
            f"is missing required columns: "
            f"{missing_text}."
        )


# ============================================================
# ESC-50
# ============================================================

def iter_esc50_records(
    dataset_directory: Path,
) -> Iterator[BackgroundRecord]:
    """Yield background recordings from ESC-50."""

    annotation_path = (
        dataset_directory
        / "esc50_labels.csv"
    )

    if not annotation_path.is_file():
        raise BackgroundManifestError(
            f"ESC-50 annotation file was not found: "
            f"'{annotation_path}'."
        )

    with annotation_path.open(
        encoding="utf-8",
        newline="",
    ) as annotation_file:

        reader = csv.DictReader(
            annotation_file
        )

        _require_columns(
            reader.fieldnames,
            {
                "filename",
                "fold",
                "category",
                "src_file",
            },
            annotation_path,
        )

        for row in reader:

            filename = row[
                "filename"
            ].strip()

            audio_path = (
                dataset_directory
                / "wav_files"
                / filename
            )

            yield BackgroundRecord(
                dataset="esc50",
                audio_path=audio_path,
                relative_path=(
                    audio_path.relative_to(
                        dataset_directory
                    )
                ),
                category=row[
                    "category"
                ].strip(),
                source_fold=row[
                    "fold"
                ].strip(),
                recording_group_id=(
                    f"esc50:"
                    f"{row['src_file'].strip()}"
                ),
            )


# ============================================================
# UrbanSound8K
# ============================================================

def iter_urbansound8k_records(
    dataset_directory: Path,
) -> Iterator[BackgroundRecord]:
    """Yield background recordings from UrbanSound8K."""

    annotation_path = (
        dataset_directory
        / "UrbanSound8K.csv"
    )

    if not annotation_path.is_file():
        raise BackgroundManifestError(
            f"UrbanSound8K annotation file was not found: "
            f"'{annotation_path}'."
        )

    with annotation_path.open(
        encoding="utf-8",
        newline="",
    ) as annotation_file:

        reader = csv.DictReader(
            annotation_file
        )

        _require_columns(
            reader.fieldnames,
            {
                "slice_file_name",
                "fsID",
                "fold",
                "class",
            },
            annotation_path,
        )

        for row in reader:

            fold = row[
                "fold"
            ].strip()

            filename = row[
                "slice_file_name"
            ].strip()

            audio_path = (
                dataset_directory
                / f"fold{fold}"
                / filename
            )

            yield BackgroundRecord(
                dataset="urbansound8k",
                audio_path=audio_path,
                relative_path=(
                    audio_path.relative_to(
                        dataset_directory
                    )
                ),
                category=row[
                    "class"
                ].strip(),
                source_fold=fold,
                recording_group_id=(
                    f"urbansound8k:"
                    f"{row['fsID'].strip()}"
                ),
            )
# ============================================================
# Al-Emadi
# ============================================================

def iter_al_emadi_records(
    dataset_directory: Path,
) -> Iterator[BackgroundRecord]:
    """
    Yield explicitly labelled background recordings from Al-Emadi.

    Dataset structure:

        Binary_Drone_Audio/
            yes_drone/
            unknown/

    Only the documented ``unknown`` directory is treated as
    no-drone/background audio.

    The multiclass directory is intentionally NOT used here.
    """

    background_directory = (
        dataset_directory
        / "Binary_Drone_Audio"
        / "unknown"
    )

    if not background_directory.is_dir():
        raise BackgroundManifestError(
            "Al-Emadi background directory was not found: "
            f"'{background_directory}'."
        )

    audio_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    }

    for audio_path in sorted(
        background_directory.rglob("*")
    ):

        if not audio_path.is_file():
            continue

        if audio_path.suffix.lower() not in audio_extensions:
            continue

        relative_path = (
            audio_path.relative_to(
                dataset_directory
            )
        )

        yield BackgroundRecord(
            dataset="al_emadi",
            audio_path=audio_path,
            relative_path=relative_path,
            category="unknown",
            source_fold="",
            recording_group_id=(
                f"al_emadi:"
                f"{audio_path.stem}"
            ),
        )        


# ============================================================
# UAViBase
# ============================================================

def _extract_number(value):
    """Extract numeric portion from a metadata value."""

    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        return None

    text = value.strip()

    if not text:
        return None

    number = ""

    for character in text:

        if (
            character.isdigit()
            or character in ".-"
        ):
            number += character

        elif number:
            break

    return number or None


def iter_uavirbase_records(
    dataset_directory: Path,
) -> Iterator[BackgroundRecord]:
    """
    Yield only explicitly labelled UAViBase no-drone recordings.

    UAViBase stores:

        recording_directory/
            label.json
            output.wav

    Only sound_source values explicitly identifying ambient/background
    recordings are accepted.
    """

    label_files = sorted(
        dataset_directory.rglob(
            "label.json"
        )
    )

    for label_path in label_files:

        recording_directory = (
            label_path.parent
        )

        try:
            with label_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ):

            LOGGER.warning(
                "Skipping invalid UAViBase metadata: %s",
                label_path,
            )

            continue

        if not isinstance(
            metadata,
            dict,
        ):
            continue

        drone = metadata.get(
            "drone",
            {},
        )

        if not isinstance(
            drone,
            dict,
        ):
            drone = {}

        sound_source = drone.get(
            "sound_source"
        )

        if isinstance(
            sound_source,
            str,
        ):
            source_normalized = (
                sound_source
                .strip()
                .lower()
            )
        else:
            source_normalized = ""

        # --------------------------------------------------------
        # ONLY explicit no-drone labels
        # --------------------------------------------------------

        if source_normalized not in {
            "ambient noise",
            "ambient",
            "background",
        }:
            continue

        # --------------------------------------------------------
        # Find audio
        # --------------------------------------------------------

        audio_files = sorted(
            path
            for path in recording_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in {
                    ".wav",
                    ".mp3",
                    ".flac",
                    ".ogg",
                    ".m4a",
                }
            )
        )

        if not audio_files:
            continue

        output_wav = next(
            (
                path
                for path in audio_files
                if path.name.lower()
                == "output.wav"
            ),
            audio_files[0],
        )

        try:
            relative_path = (
                output_wav.relative_to(
                    dataset_directory
                )
            )

        except ValueError:
            continue

        # The recording directory itself is the original recording group.
        recording_group_id = (
            "uavirbase:"
            f"{label_path.parent.relative_to(dataset_directory).as_posix()}"
        )

        yield BackgroundRecord(
            dataset="uavirbase",
            audio_path=output_wav,
            relative_path=relative_path,
            category=source_normalized,
            source_fold="",
            recording_group_id=recording_group_id,
        )


# ============================================================
# Dataset adapter selection
# ============================================================

def _records_for_dataset(
    dataset_name: str,
    raw_datasets_directory: Path,
) -> Iterable[BackgroundRecord]:
    """Return records for a registered background dataset."""

    adapters = {
    "esc50": iter_esc50_records,
    "urbansound8k": iter_urbansound8k_records,
    "uavirbase": iter_uavirbase_records,
    "al_emadi": iter_al_emadi_records,
    }

    try:
        adapter = adapters[
            dataset_name
        ]

    except KeyError as error:

        supported = ", ".join(
            sorted(adapters)
        )

        raise BackgroundManifestError(
            f"Unsupported background dataset "
            f"'{dataset_name}'. Supported: "
            f"{supported}."
        ) from error

    return adapter(
        raw_datasets_directory
        / dataset_name
    )


# ============================================================
# Manifest row
# ============================================================

def _manifest_row(
    record: BackgroundRecord,
) -> dict[str, str | int]:
    """Convert a background record into CSV format."""

    return {
        "dataset": record.dataset,

        "relative_path": (
            record.relative_path
            .as_posix()
        ),

        "file_name": (
            record.audio_path.name
        ),

        "binary_label": (
            BINARY_NO_DRONE_LABEL
        ),

        "label_origin": (
            "official_metadata:no_drone"
        ),

        "source_category": (
            record.category
        ),

        "source_fold": (
            record.source_fold
        ),

        "recording_group_id": (
            record.recording_group_id
        ),

        "size_bytes": (
            record.audio_path.stat()
            .st_size
        ),

        "sha256": _sha256_file(
            record.audio_path
        ),
    }


# ============================================================
# Build
# ============================================================

def build_background_manifest(
    raw_datasets_directory: Path = RAW_DATASET_DIR,
    manifest_path: Path | None = None,

    dataset_names: Sequence[str] = (
    "esc50",
    "urbansound8k",
    "uavirbase",
    "al_emadi",
    ),

) -> BackgroundManifestSummary:
    """Build the complete no-drone manifest."""

    selected_datasets = tuple(
        dataset_names
    )

    if not selected_datasets:
        raise BackgroundManifestError(
            "At least one background dataset is required."
        )

    destination = (
        manifest_path
        or METADATA_DIR
        / DEFAULT_MANIFEST_FILENAME
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        temporary_path = Path(
            temporary_file.name
        )

        writer = csv.DictWriter(
            temporary_file,
            fieldnames=MANIFEST_COLUMNS,
        )

        writer.writeheader()

        try:

            for dataset_name in selected_datasets:

                for record in _records_for_dataset(
                    dataset_name,
                    raw_datasets_directory,
                ):

                    if not record.audio_path.is_file():

                        missing_audio_files += 1

                        LOGGER.warning(
                            "Skipping annotation row "
                            "with missing audio: %s",
                            record.audio_path,
                        )

                        continue

                    writer.writerow(
                        _manifest_row(record)
                    )

                    included_files += 1

        except Exception:

            temporary_path.unlink(
                missing_ok=True
            )

            raise

    # Atomic replacement.
    temporary_path.replace(
        destination
    )

    summary = BackgroundManifestSummary(
        manifest_path=destination,
        included_files=included_files,
        missing_audio_files=missing_audio_files,
        datasets=selected_datasets,
        created_at_utc=(
            datetime.now(UTC)
            .isoformat()
        ),
    )

    LOGGER.info(
        "Built background manifest at %s "
        "with %d files; %d annotation rows "
        "referenced missing audio.",
        summary.manifest_path,
        summary.included_files,
        summary.missing_audio_files,
    )

    return summary


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

    parser.add_argument(
        "--datasets",
        nargs="+",

        default=(
            "esc50",
            "urbansound8k",
            "uavirbase",
            "al_emadi",
        ),

        help=(
            "Background dataset adapters to include."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the background manifest builder."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    arguments = _parse_arguments()

    summary = build_background_manifest(
        manifest_path=arguments.output,
        dataset_names=arguments.datasets,
    )

    LOGGER.info(
        "Background manifest build completed: %s",
        summary,
    )


if __name__ == "__main__":
    main()