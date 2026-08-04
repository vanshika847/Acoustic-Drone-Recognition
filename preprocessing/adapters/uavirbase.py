"""
Adapter for the UAVirBase dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from configs.config import SUPPORTED_AUDIO_EXTENSIONS
from configs.dataset_rules import DatasetRule
from utils.hashing import sha256_file

from .base import DatasetAdapter


class UAVirBaseAdapter(DatasetAdapter):

    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int]]:

        pass