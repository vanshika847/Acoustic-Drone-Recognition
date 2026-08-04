"""Dataset registry and labelling policy for acoustic drone recognition.

Purpose
-------
This module is the declarative source of truth for every supported dataset.
It defines where a dataset is expected under ``datasets/raw``, whether it may
participate in the current experiment, and how its labels must be resolved.

The manifest builder will consume these definitions.  It must never infer a
drone/non-drone label from a dataset name: unlabelled or unsupported samples
are deliberately excluded from supervised training.

Inputs
------
No runtime inputs.  Paths are resolved from :mod:`configs.config`.

Outputs
-------
Immutable :class:`DatasetRule` objects and helper functions for retrieving
the active rules.

Dependencies
------------
Python standard library only.

Algorithm
---------
Rules are stored in a tuple to make their ordering deterministic.  Lookup is
performed through a small dictionary constructed on demand, which is
``O(number_of_datasets)`` for the current registry and negligible in practice.

Example
-------
>>> active_names = [rule.name for rule in get_enabled_dataset_rules()]
>>> "drone_audio" in active_names
True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LabelSource(StrEnum):
    """Supported sources for supervised labels.

    ``DIRECTORY`` is appropriate only when a dataset's folder structure is a
    documented label source.  ``ANNOTATION_FILE`` requires an official
    metadata file to be parsed by the dataset adapter.  ``NONE`` prevents a
    dataset from entering supervised training until labels are supplied.
    """

    DIRECTORY = "directory"
    ANNOTATION_FILE = "annotation_file"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class DatasetRule:
    """Configuration and safety policy for one raw audio dataset.

    Attributes:
        name: Stable internal dataset identifier used in manifests.
        raw_directory_name: Directory name below ``datasets/raw``.
        enabled: Whether the current experiment may ingest this dataset.
        label_source: Approved source for labels; never infer labels otherwise.
        annotation_file_name: Optional official annotation filename relative to
            the dataset root.
        directory_binary_labels: Relative-folder-to-binary-label mapping for
            explicitly labelled directory datasets. ``1`` means drone present
            and ``0`` means no drone present.
        description: Human-readable notes for experiment reproducibility.
    """

    name: str
    raw_directory_name: str
    enabled: bool
    label_source: LabelSource
    annotation_file_name: str | None = None
    directory_binary_labels: tuple[tuple[str, int], ...] = ()
    description: str = ""

    def resolve_raw_directory(self, raw_datasets_directory: Path) -> Path:
        """Return this dataset's location without creating or modifying it.

        Args:
            raw_datasets_directory: Project directory containing raw datasets.

        Returns:
            Absolute or project-relative path to this dataset's raw directory.
        """

        return raw_datasets_directory / self.raw_directory_name

    def directory_label_map(self) -> dict[str, int]:
        """Return a copy of the approved directory-to-label mapping.

        Returns:
            Mapping from POSIX-style relative directory to binary label.
        """

        return dict(self.directory_binary_labels)


# Keep this registry conservative.  An available dataset is not automatically
# a training dataset: only samples with an explicit, validated label source are
# eligible for supervised learning.
DATASET_RULES: tuple[DatasetRule, ...] = (
    DatasetRule(
        name="drone_audio",
        raw_directory_name="drone_audio",
        enabled=True,
        label_source=LabelSource.DIRECTORY,
        directory_binary_labels=(("yes_drone", 1),),
        description=(
            "Initial labelled positive set. The 'unknown' directory is "
            "intentionally excluded until its source labels are verified."
        ),
    ),
    DatasetRule(
        name="aerosonicdb",
        raw_directory_name="aerosonicdb",
        enabled=False,
        label_source=LabelSource.ANNOTATION_FILE,
        annotation_file_name="sample_meta.csv",
        description=(
            "Enable only after the AeroSonicDB annotation adapter maps its "
            "official metadata to the project taxonomy."
        ),
    ),
    DatasetRule(
        name="esc50",
        raw_directory_name="esc50",
        enabled=False,
        label_source=LabelSource.ANNOTATION_FILE,
        description="Enable after the ESC-50 metadata adapter is implemented.",
    ),
    DatasetRule(
        name="urbansound8k",
        raw_directory_name="urbansound8k",
        enabled=False,
        label_source=LabelSource.ANNOTATION_FILE,
        description=(
            "Enable after UrbanSound8K official fold metadata is mapped to "
            "background/no-drone classes."
        ),
    ),
    DatasetRule(
        name="ieee_spcup2019",
        raw_directory_name="ieee_spcup2019",
        enabled=False,
        label_source=LabelSource.ANNOTATION_FILE,
        description="Enable after its official labels and recording groups are parsed.",
    ),
    DatasetRule(
        name="ddl",
        raw_directory_name="ddl",
        enabled=False,
        label_source=LabelSource.ANNOTATION_FILE,
        description="Pending download and official annotation adapter.",
    ),
    DatasetRule(
        name="uavirbase",
        raw_directory_name="uavirbase",
        enabled=True,
        label_source=LabelSource.ANNOTATION_FILE,
        annotation_file_name="label.json",
        description=(
        "UaVirBASE multichannel UAV recordings. Labels and metadata "
        "are read from each recording's label.json file."
        ),
    ),
    DatasetRule(
        name="audioset",
        raw_directory_name="audioset",
        enabled=False,
        label_source=LabelSource.NONE,
        description="Excluded from this project phase by design.",
    ),
    DatasetRule(
        name="freesound",
        raw_directory_name="freesound",
        enabled=False,
        label_source=LabelSource.NONE,
        description="Excluded from this project phase by design.",
    ),
)


def get_dataset_rule(dataset_name: str) -> DatasetRule:
    """Retrieve a registered dataset rule by its stable internal name.

    Args:
        dataset_name: Dataset identifier, for example ``"drone_audio"``.

    Returns:
        The matching immutable dataset rule.

    Raises:
        KeyError: If ``dataset_name`` has not been registered.
    """

    rules_by_name = {rule.name: rule for rule in DATASET_RULES}
    try:
        return rules_by_name[dataset_name]
    except KeyError as error:
        available_names = ", ".join(sorted(rules_by_name))
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available datasets: "
            f"{available_names}."
        ) from error


def get_enabled_dataset_rules() -> tuple[DatasetRule, ...]:
    """Return all datasets explicitly enabled for the current experiment.

    Returns:
        Deterministically ordered enabled dataset rules.
    """

    return tuple(rule for rule in DATASET_RULES if rule.enabled)
