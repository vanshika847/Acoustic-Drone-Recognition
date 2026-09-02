"""
Adapter for the DDL (Drone Detection and Localization) dataset.

DDL filename example:

20210329141240MINI0030240312886R290321-T004-005236.wav

The filename contains:
    timestamp
    drone type
    bearing
    distance/range
    altitude
    temperature
    recording type
    flight/session information
    sequence number

Drone types:
    MINI -> DJI Mini 2
    PRO4 -> DJI Phantom 4 Pro
    XXXX -> no-drone/background recording
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

from configs.config import SUPPORTED_AUDIO_EXTENSIONS
from configs.dataset_rules import DatasetRule
from utils.hashing import sha256_file

from .base import DatasetAdapter


class DDLAdapter(DatasetAdapter):
    """Build normalized manifest rows from DDL audio recordings."""

    _PATTERN = re.compile(
        r"^"
        r"(?P<timestamp>\d{14})"
        r"(?P<drone_type>MINI|PRO4|XXXX)"
        r"(?P<azimuth>\d{3})"
        r"(?P<distance>\d{3})"
        r"(?P<height>\d{3})"
        r"(?P<temperature>\d{4})"
        r"(?P<sample_type>[RS])"
        r"(?P<flight_session>\d{6})"
        r"-(?P<recording_session>T\d{3})"
        r"-(?P<sequence>\d{6})"
        r"$",
        re.IGNORECASE,
    )

    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int | float | None]]:
        """Yield one manifest row for every valid DDL recording."""

        for audio_path in sorted(dataset_directory.rglob("*")):

            if not audio_path.is_file():
                continue

            if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue

            metadata = self._parse_filename(audio_path)

            if metadata is None:
                continue

            try:
                relative_path = audio_path.relative_to(
                    dataset_directory
                )
            except ValueError:
                continue

            yield {
                "dataset": rule.name,
                "relative_path": relative_path.as_posix(),
                "file_name": audio_path.name,
                "extension": audio_path.suffix.lower(),

                # XXXX = no drone
                # MINI / PRO4 = drone
                "binary_label": metadata["binary_label"],

                "label_origin": metadata["label_origin"],

                "size_bytes": audio_path.stat().st_size,
                "sha256": sha256_file(audio_path),

                # Drone information
                "drone_type": metadata["drone_type"],
                "movement": "",
                "rotation": "",

                # Localization
                "distance_m": metadata["distance_m"],
                "height_m": metadata["height_m"],
                "azimuth_deg": metadata["azimuth_deg"],

                # Recording
                "recording_start": metadata["recording_start"],
                "recording_end": "",

                # Environment
                "temperature_c": metadata["temperature_c"],
                "humidity_percent": "",
                "wind_speed_ms": "",
                "wind_direction_deg": "",

                # GPS
                "latitude": "",
                "longitude": "",
            }

    @classmethod
    def _parse_filename(
        cls,
        audio_path: Path,
    ) -> dict[str, str | int | float | None] | None:
        """Parse DDL metadata from the filename."""

        stem = audio_path.stem

        match = cls._PATTERN.match(stem)

        if match is None:
            return None

        values = match.groupdict()

        raw_drone_type = values["drone_type"].upper()
        sample_type = values["sample_type"].upper()

        # ---------------------------------------------------------
        # Binary label
        # ---------------------------------------------------------

        if raw_drone_type == "XXXX":
            binary_label = 0
        else:
            binary_label = 1

        # ---------------------------------------------------------
        # Drone type
        # ---------------------------------------------------------

        if raw_drone_type == "MINI":
            drone_type = "DJI Mini 2"

        elif raw_drone_type == "PRO4":
            drone_type = "DJI Phantom 4 Pro"

        else:
            drone_type = "unknown"

        # ---------------------------------------------------------
        # Localization
        # ---------------------------------------------------------

        try:
            azimuth_deg = float(values["azimuth"])
        except (TypeError, ValueError):
            azimuth_deg = None

        try:
            distance_m = float(values["distance"])
        except (TypeError, ValueError):
            distance_m = None

        try:
            height_m = float(values["height"])
        except (TypeError, ValueError):
            height_m = None

        # ---------------------------------------------------------
        # Temperature
        # ---------------------------------------------------------
        # DDL encodes temperature as tenths of Kelvin.
        #
        # Example:
        # 2886 -> 288.6 K -> 15.45 C
        # ---------------------------------------------------------

        try:
            temperature_kelvin = (
                float(values["temperature"]) / 10.0
            )

            temperature_c = (
                temperature_kelvin - 273.15
            )

        except (TypeError, ValueError):
            temperature_c = None

        # ---------------------------------------------------------
        # Recording timestamp
        # ---------------------------------------------------------

        try:
            recording_start = datetime.strptime(
                values["timestamp"],
                "%Y%m%d%H%M%S",
            ).isoformat()

        except (TypeError, ValueError):
            recording_start = ""

        # ---------------------------------------------------------
        # Preserve useful DDL provenance.
        # ---------------------------------------------------------

        label_origin = (
            "ddl:filename_metadata:"
            f"sample_type={sample_type}:"
            f"flight={values['flight_session']}:"
            f"recording={values['recording_session']}:"
            f"sequence={values['sequence']}"
        )

        return {
            "binary_label": binary_label,
            "drone_type": drone_type,
            "distance_m": distance_m,
            "height_m": height_m,
            "azimuth_deg": azimuth_deg,
            "temperature_c": temperature_c,
            "recording_start": recording_start,
            "label_origin": label_origin,
        }