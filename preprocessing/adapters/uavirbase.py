"""
Adapter for the UaVirBASE dataset.

UaVirBASE stores each recording in its own directory containing:
    - label.json
    - output.wav

The adapter converts the dataset-specific JSON metadata into the
project's common supervised manifest format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from configs.config import SUPPORTED_AUDIO_EXTENSIONS
from configs.dataset_rules import DatasetRule
from utils.hashing import sha256_file

from .base import DatasetAdapter


class UAVirBaseAdapter(DatasetAdapter):
    """
    Adapter for UaVirBASE recordings.

    Each recording directory is expected to contain a label.json file
    and one supported audio file, normally output.wav.
    """

    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int | float | None]]:
        """
        Read UaVirBASE recordings and yield normalized manifest rows.

        Parameters
        ----------
        dataset_directory
            Root directory of the UaVirBASE dataset.

        rule
            Dataset registration rule.

        Yields
        ------
        dict
            One standardized manifest row per valid recording.
        """

        # Search deterministically for all label files.
        label_files = sorted(dataset_directory.rglob("label.json"))

        for label_path in label_files:

            recording_directory = label_path.parent

            # ----------------------------------------------------------
            # Read label.json
            # ----------------------------------------------------------

            try:
                with label_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    metadata = json.load(file)

            except (OSError, json.JSONDecodeError):
                # Invalid metadata cannot safely enter supervised training.
                continue

            if not isinstance(metadata, dict):
                continue

            # ----------------------------------------------------------
            # Find corresponding audio file
            # ----------------------------------------------------------

            audio_files = sorted(
                path
                for path in recording_directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
            )

            if not audio_files:
                continue

            # UaVirBASE normally contains output.wav.
            # Prefer it when multiple supported files are present.
            output_wav = next(
                (
                    path
                    for path in audio_files
                    if path.name.lower() == "output.wav"
                ),
                audio_files[0],
            )

            # ----------------------------------------------------------
            # Drone metadata
            # ----------------------------------------------------------

            drone = metadata.get("drone", {})

            if not isinstance(drone, dict):
                drone = {}

            sound_source = drone.get("sound_source")

            # Normalize the sound source for the binary task.
            if isinstance(sound_source, str):
                source_normalized = sound_source.strip().lower()
            else:
                source_normalized = ""

            if source_normalized == "drone":
                binary_label = 1

            elif source_normalized in {
                "ambient noise",
                "ambient",
                "background",
            }:
                binary_label = 0

            else:
                # Never guess a supervised label.
                continue

            # ----------------------------------------------------------
            # Helper for optional numeric values
            # ----------------------------------------------------------

            def to_float(value):
                if value is None:
                    return None

                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            # ----------------------------------------------------------
            # Drone information
            # ----------------------------------------------------------

            drone_type = drone.get("type")
            movement = drone.get("movement")
            rotation = drone.get("rotation")

            distance_m = to_float(drone.get("distance"))
            height_m = to_float(drone.get("height"))
            azimuth_deg = to_float(drone.get("azimuth"))

            # ----------------------------------------------------------
            # Recording information
            # ----------------------------------------------------------

            recording_start = metadata.get("start_recording_time")
            recording_end = metadata.get("end_recording_time")

            # ----------------------------------------------------------
            # Weather information
            # ----------------------------------------------------------

            weather_data = metadata.get("weather_data", {})

            if not isinstance(weather_data, dict):
                weather_data = {}

            measurements = weather_data.get("measurements", {})

            if not isinstance(measurements, dict):
                measurements = {}

            temperature_c = to_float(
                self._extract_number(
                    measurements.get("air temperature")
                )
            )

            humidity_percent = to_float(
                self._extract_number(
                    measurements.get("air humidity")
                )
            )

            wind_speed_ms = to_float(
                self._extract_number(
                    measurements.get("wind speed")
                )
            )

            wind_direction_deg = to_float(
                self._extract_number(
                    measurements.get("wind direction")
                )
            )

            # ----------------------------------------------------------
            # Microphone array / GPS
            # ----------------------------------------------------------

            microphone_array = metadata.get(
                "microphone_array",
                {},
            )

            if not isinstance(microphone_array, dict):
                microphone_array = {}

            latitude = to_float(
                microphone_array.get("center_latitude")
            )

            longitude = to_float(
                microphone_array.get("center_longitude")
            )

            # ----------------------------------------------------------
            # Relative path
            # ----------------------------------------------------------

            try:
                relative_path = output_wav.relative_to(
                    dataset_directory
                )
            except ValueError:
                continue

            # ----------------------------------------------------------
            # Manifest row
            # ----------------------------------------------------------

            yield {
                "dataset": rule.name,
                "relative_path": relative_path.as_posix(),
                "file_name": output_wav.name,
                "extension": output_wav.suffix.lower(),

                "binary_label": binary_label,

                "label_origin": (
                    "annotation_file:"
                    f"{label_path.relative_to(dataset_directory).as_posix()}"
                ),

                "size_bytes": output_wav.stat().st_size,
                "sha256": sha256_file(output_wav),

                # Drone information
                "drone_type": drone_type,
                "movement": movement,
                "rotation": rotation,

                # Position
                "distance_m": distance_m,
                "height_m": height_m,
                "azimuth_deg": azimuth_deg,

                # Recording
                "recording_start": recording_start,
                "recording_end": recording_end,

                # Environment
                "temperature_c": temperature_c,
                "humidity_percent": humidity_percent,
                "wind_speed_ms": wind_speed_ms,
                "wind_direction_deg": wind_direction_deg,

                # GPS
                "latitude": latitude,
                "longitude": longitude,
            }

    @staticmethod
    def _extract_number(value):
        """
        Extract the numeric portion from metadata strings.

        Examples
        --------
        "4.6 C" -> 4.6
        "90 % RH" -> 90
        "1.7 m/s" -> 1.7
        "240 degrees" -> 240
        """

        if value is None:
            return None

        if isinstance(value, (int, float)):
            return value

        if not isinstance(value, str):
            return None

        text = value.strip()

        if not text:
            return None

        # Keep digits, decimal point, and minus sign.
        number = ""

        for character in text:
            if character.isdigit() or character in ".-":
                number += character
            elif number:
                break

        return number or None