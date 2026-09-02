"""
Base interface for dataset adapters.

Each supported dataset implements this interface so that the
generic manifest builder can ingest audio without knowing the
dataset's internal structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from configs.dataset_rules import DatasetRule




class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    Every dataset adapter converts a dataset-specific structure
    into a common manifest format.
    """

    @abstractmethod
    def build_rows(
        self,
        dataset_directory: Path,
        rule: DatasetRule,
    ) -> Iterator[dict[str, str | int]]:
        """
        Yield one manifest row for each valid audio sample.

        Parameters
        ----------
        dataset_directory
            Root directory of the dataset.

        rule
            Dataset registration rule.

        Yields
        ------
        dict
            One manifest row compatible with the project
            supervised manifest.
        """
        raise NotImplementedError