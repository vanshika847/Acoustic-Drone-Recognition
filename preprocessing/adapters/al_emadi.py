"""
Adapter for the Sara Al-Emadi Drone Audio Dataset.

Dataset structure:

al_emadi/
├── Binary_Drone_Audio/
│   ├── yes_drone/
│   └── unknown/
│
└── Multiclass_Drone_Audio/
    ├── bebop_1/
    ├── membo_1/
    └── unknown/

For the current binary experiment:

yes_drone -> binary_label = 1
unknown   -> binary_label = 0

The supervised manifest builder keeps only binary_label == 1.
The background manifest builder handles binary_label == 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from configs.dataset_rules import DatasetRule
from utils.hashing import sha256_file

from .base import DatasetAdapter


class AlEmadiAdapter(DatasetAdapter):
    """Build manifest rows from the Al-Emadi dataset."""

    AUDIO_EXTENSIONS = {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".m4a",
    }

    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int]]:
        """
        Yield labelled audio rows from the Al-Emadi dataset.

        Labels are taken only from the directories explicitly
        declared in DatasetRule.
        """

        label_map = rule.directory_label_map()

        for relative_directory, binary_label in label_map.items():

            source_directory = (
                dataset_directory
                / Path(relative_directory)
            )

            if not source_directory.is_dir():
                continue

            for audio_path in sorted(
                source_directory.rglob("*")
            ):

                if not audio_path.is_file():
                    continue

                if (
                    audio_path.suffix.lower()
                    not in self.AUDIO_EXTENSIONS
                ):
                    continue

                try:
                    relative_path = (
                        audio_path.relative_to(
                            dataset_directory
                        )
                    )
                except ValueError:
                    continue

                yield {
                    "dataset": rule.name,
                    "relative_path": relative_path.as_posix(),
                    "file_name": audio_path.name,
                    "extension": audio_path.suffix.lower(),
                    "binary_label": binary_label,
                    "label_origin": (
                        f"directory:{relative_directory}"
                    ),
                    "size_bytes": audio_path.stat().st_size,
                    "sha256": sha256_file(audio_path),

                    # Drone information
                    "drone_type": "",
                    "movement": "",
                    "rotation": "",

                    # Position
                    "distance_m": "",
                    "height_m": "",
                    "azimuth_deg": "",

                    # Recording
                    "recording_start": "",
                    "recording_end": "",

                    # Environment
                    "temperature_c": "",
                    "humidity_percent": "",
                    "wind_speed_ms": "",
                    "wind_direction_deg": "",

                    # GPS
                    "latitude": "",
                    "longitude": "",
                }