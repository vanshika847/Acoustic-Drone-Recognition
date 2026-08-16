"""
ESC-50 dataset adapter.

ESC-50 is used in this project as a verified background/no-drone
dataset.

The adapter reads the official ESC-50 metadata file and converts
each recording into the common project manifest representation.

Dataset structure
-----------------
datasets/raw/esc50/
    wav_files/
        *.wav
    esc50_labels.csv

The adapter never infers a label from the filename. The dataset's
official metadata is used to identify the recording and its source
category.

All ESC-50 recordings are assigned:

    binary_label = 0

because ESC-50 is being used as environmental background audio
for the acoustic drone detection task.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from configs.dataset_rules import DatasetRule

from preprocessing.adapters.base import DatasetAdapter


class ESC50Adapter(DatasetAdapter):
    """
    Adapter for the ESC-50 environmental sound dataset.

    Converts ESC-50 recordings into the common manifest format
    expected by the project.
    """

    ANNOTATION_FILE = "esc50_labels.csv"
    AUDIO_DIRECTORY = "wav_files"

    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int]]:
        """
        Yield one manifest row for each valid ESC-50 recording.

        Parameters
        ----------
        dataset_directory:
            Root directory of the ESC-50 dataset.

        rule:
            Registered DatasetRule for ESC-50.

        Yields
        ------
        dict
            Manifest-compatible background row.
        """

        annotation_path = (
            dataset_directory
            / self.ANNOTATION_FILE
        )

        audio_directory = (
            dataset_directory
            / self.AUDIO_DIRECTORY
        )

        # ------------------------------------------------------
        # Validate dataset structure
        # ------------------------------------------------------

        if not annotation_path.is_file():
            raise FileNotFoundError(
                "ESC-50 annotation file was not found: "
                f"{annotation_path}"
            )

        if not audio_directory.is_dir():
            raise FileNotFoundError(
                "ESC-50 audio directory was not found: "
                f"{audio_directory}"
            )

        # ------------------------------------------------------
        # Read official metadata
        # ------------------------------------------------------

        with annotation_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as annotation_file:

            reader = csv.DictReader(
                annotation_file
            )

            required_columns = {
                "filename",
                "fold",
                "category",
                "src_file",
            }

            available_columns = set(
                reader.fieldnames or ()
            )

            missing_columns = (
                required_columns
                - available_columns
            )

            if missing_columns:
                raise ValueError(
                    "ESC-50 annotation file is missing "
                    "required columns: "
                    + ", ".join(
                        sorted(missing_columns)
                    )
                )

            # --------------------------------------------------
            # Build manifest rows
            # --------------------------------------------------

            for row in reader:

                filename = (
                    row["filename"]
                    .strip()
                )

                if not filename:
                    continue

                audio_path = (
                    audio_directory
                    / filename
                )

                # ----------------------------------------------
                # Never create a manifest row for missing audio
                # ----------------------------------------------

                if not audio_path.is_file():
                    continue

                relative_path = (
                    audio_path.relative_to(
                        dataset_directory
                    )
                )

                category = (
                    row["category"]
                    .strip()
                )

                fold = (
                    row["fold"]
                    .strip()
                )

                source_file = (
                    row["src_file"]
                    .strip()
                )

                # ----------------------------------------------
                # ESC-50 recording identity
                # ----------------------------------------------

                recording_group_id = (
                    f"esc50:{source_file}"
                )

                # ----------------------------------------------
                # Common manifest representation
                # ----------------------------------------------

                yield {
                    "dataset": rule.name,

                    "relative_path": (
                        relative_path.as_posix()
                    ),

                    "file_name": filename,

                    "extension": (
                        audio_path.suffix.lower()
                    ),

                    "binary_label": 0,

                    "label_origin": (
                        "official_metadata:esc50_background"
                    ),

                    "size_bytes": (
                        audio_path.stat().st_size
                    ),

                    "sha256": "",

                    "drone_type": "",
                    "movement": "",
                    "rotation": "",

                    "distance_m": "",
                    "height_m": "",
                    "azimuth_deg": "",

                    "recording_start": "",
                    "recording_end": "",

                    "temperature_c": "",
                    "humidity_percent": "",
                    "wind_speed_ms": "",
                    "wind_direction_deg": "",

                    "latitude": "",
                    "longitude": "",

                    # Extra metadata useful to the background
                    # manifest adapter.
                    "source_category": category,
                    "source_fold": fold,
                    "recording_group_id": (
                        recording_group_id
                    ),
                }