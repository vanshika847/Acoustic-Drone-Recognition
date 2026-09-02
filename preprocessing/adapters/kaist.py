"""
Adapter for the KAIST Drone Sound Dataset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from configs.config import SUPPORTED_AUDIO_EXTENSIONS
from configs.dataset_rules import DatasetRule
from utils.hashing import sha256_file

from .base import DatasetAdapter


class KaistAdapter(DatasetAdapter):
    """Build manifest rows from KAIST drone audio."""

    _PATTERN = re.compile(
        r"""
        ^
        (?P<drone_type>[ABC])
        [_-]
        (?P<movement>CC|F|B|R|L|C)
        [_-]
        (?P<fault>N|MF[1-4]|PC[1-4])
        [_-]
        (?P<drone_index>[^_-]+)
        [_-]
        (?P<background>.+?)
        [_-]
        (?P<background_index>[^_-]+)
        [_-]
        (?P<snr>-?\d+(?:\.\d+)?)
        (?:dB)?
        $
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int]]:
        """Yield one positive manifest row per valid KAIST recording."""

        for audio_path in sorted(
            dataset_directory.rglob("*")
        ):

            if not audio_path.is_file():
                continue

            if (
                audio_path.suffix.lower()
                not in SUPPORTED_AUDIO_EXTENSIONS
            ):
                continue

            metadata = self._parse_metadata(
                audio_path
            )

            if metadata is None:
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

                "relative_path": (
                    relative_path.as_posix()
                ),

                "file_name": audio_path.name,

                "extension": (
                    audio_path.suffix.lower()
                ),

                "binary_label": 1,

                "label_origin": (
                    "kaist:filename_metadata"
                ),

                "size_bytes": (
                    audio_path.stat().st_size
                ),

                "sha256": sha256_file(
                    audio_path
                ),

                "drone_type": metadata[
                    "drone_type"
                ],

                "movement": metadata[
                    "movement"
                ],

                "rotation": (
                    metadata["movement"]
                    if metadata["movement"]
                    in {"C", "CC"}
                    else ""
                ),

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

    @classmethod
    def _parse_metadata(
        cls,
        audio_path: Path,
    ) -> dict[str, str] | None:
        """Parse KAIST metadata from an audio filename."""

        stem = audio_path.stem

        stem = re.sub(
            r"[_-](?:mic[12]|ch[12])$",
            "",
            stem,
            flags=re.IGNORECASE,
        )

        match = cls._PATTERN.match(
            stem
        )

        if match is None:
            return None

        values = match.groupdict()

        return {
            "drone_type": values[
                "drone_type"
            ].upper(),

            "movement": values[
                "movement"
            ].upper(),

            "fault": values[
                "fault"
            ].upper(),

            "drone_index": values[
                "drone_index"
            ],

            "background": values[
                "background"
            ],

            "background_index": values[
                "background_index"
            ],

            "snr_db": values[
                "snr"
            ],
        }