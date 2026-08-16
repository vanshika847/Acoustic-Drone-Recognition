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

The supervised manifest builder will keep only label 1.
The background manifest builder is responsible for label 0.
"""
from utils.hashing import sha256_file
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from configs.dataset_rules import DatasetRule


class AlEmadiAdapter:
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
    ) -> Iterator[dict]:
        """
        Yield labelled audio rows from the Al-Emadi dataset.

        Only directories explicitly declared in the DatasetRule are used.
        No label is inferred from the dataset name.
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

                relative_path = audio_path.relative_to(
                    dataset_directory
                )

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