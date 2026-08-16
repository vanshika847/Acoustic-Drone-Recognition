"""
Adapter for directory-labelled drone audio datasets.
"""

from __future__ import annotations
from utils.hashing import sha256_file
import os
from pathlib import Path
from typing import Iterator

from configs.config import SUPPORTED_AUDIO_EXTENSIONS
from configs.dataset_rules import DatasetRule

from .base import DatasetAdapter



class DroneAudioAdapter(DatasetAdapter):
    """
    Adapter for datasets whose labels are encoded in the
    directory structure.
    """

    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int]]:

        label_map = rule.directory_label_map()

        for current_root, directory_names, file_names in os.walk(dataset_directory):

            directory_names.sort()

            for file_name in sorted(file_names):

                file_path = Path(current_root) / file_name

                if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                    continue

                try:
                    relative_path = file_path.relative_to(dataset_directory)
                except ValueError:
                    continue

                label = None

                path_parts = relative_path.parts[:-1]

                for directory, binary_label in sorted(
                    label_map.items(),
                    key=lambda item: len(Path(item[0]).parts),
                    reverse=True,
                ):

                    directory_parts = Path(directory).parts

                    if path_parts[: len(directory_parts)] == directory_parts:
                        label = binary_label
                        break

                if label is None:
                    continue

                yield {
                    "dataset": rule.name,
                    "relative_path": relative_path.as_posix(),
                    "file_name": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "binary_label": label,
                    "label_origin": f"directory:{relative_path.parts[0]}",
                    "size_bytes": file_path.stat().st_size,
                    "sha256": sha256_file(file_path),

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
}

    