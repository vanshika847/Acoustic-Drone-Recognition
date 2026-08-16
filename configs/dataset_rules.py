"""
Dataset registry and labelling policy for acoustic drone recognition.

This module is the declarative source of truth for supported datasets.

Important:
    A dataset is not automatically a training dataset merely because it
    exists on disk. Only datasets with an explicitly defined and validated
    label source may participate in supervised training.

Current experiment:
    - drone_audio
    - al_emadi
    - uavirbase

Background-only datasets such as ESC-50 and UrbanSound8K are handled by
the dedicated background manifest builder and therefore remain disabled
in this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


# ============================================================
# Label Source
# ============================================================

class LabelSource(StrEnum):
    """Supported sources for supervised labels."""

    DIRECTORY = "directory"
    ANNOTATION_FILE = "annotation_file"
    NONE = "none"


# ============================================================
# Dataset Rule
# ============================================================

@dataclass(frozen=True, slots=True)
class DatasetRule:
    """
    Configuration and safety policy for one raw audio dataset.

    Attributes
    ----------
    name:
        Stable internal dataset identifier.

    raw_directory_name:
        Directory name below datasets/raw.

    enabled:
        Whether the dataset participates in the current experiment.

    label_source:
        Approved source used to determine labels.

    annotation_file_name:
        Optional annotation file relative to the dataset root.

    directory_binary_labels:
        Explicit directory-to-binary-label mapping.

        1 = drone
        0 = no-drone/background

    description:
        Human-readable experiment documentation.
    """

    name: str

    raw_directory_name: str

    enabled: bool

    label_source: LabelSource

    annotation_file_name: str | None = None

    directory_binary_labels: tuple[
        tuple[str, int],
        ...
    ] = ()

    description: str = ""

    def resolve_raw_directory(
        self,
        raw_datasets_directory: Path,
    ) -> Path:
        """
        Return the dataset directory.

        This function does not create or modify anything.
        """

        return (
            raw_datasets_directory
            / self.raw_directory_name
        )

    def directory_label_map(
        self,
    ) -> dict[str, int]:
        """
        Return the approved directory-to-label mapping.
        """

        return dict(
            self.directory_binary_labels
        )


# ============================================================
# Dataset Registry
# ============================================================

# Keep this registry conservative.
#
# IMPORTANT:
# Dataset availability does NOT automatically make a dataset eligible
# for supervised training.
#
# Only explicitly validated labels are accepted.

DATASET_RULES: tuple[DatasetRule, ...] = (

    # ========================================================
    # DroneAudio
    # ========================================================

    DatasetRule(
        name="drone_audio",

        raw_directory_name="drone_audio",

        enabled=True,

        label_source=LabelSource.DIRECTORY,

        directory_binary_labels=(
            ("yes_drone", 1),
        ),

        description=(
            "Directory-labelled drone audio dataset. "
            "The yes_drone directory contains explicitly "
            "labelled drone recordings. Other directories are "
            "not automatically assigned a label."
        ),
    ),

    # ========================================================
    # Al-Emadi
    # ========================================================

    DatasetRule(
        name="al_emadi",

        raw_directory_name="al_emadi",

        enabled=True,

        label_source=LabelSource.DIRECTORY,

        directory_binary_labels=(
            (
                "Binary_Drone_Audio/yes_drone",
                1,
            ),
            (
                "Binary_Drone_Audio/unknown",
                0,
            ),
        ),

        description=(
            "Sara Al-Emadi drone audio dataset. "
            "Binary_Drone_Audio/yes_drone contains drone "
            "recordings. Binary_Drone_Audio/unknown contains "
            "documented background/noise recordings. "
            "The Multiclass_Drone_Audio directory is reserved "
            "for future drone-category classification."
        ),
    ),

    # ========================================================
    # UAViBase
    # ========================================================

    DatasetRule(
        name="uavirbase",

        raw_directory_name="uavirbase",

        enabled=True,

        label_source=LabelSource.ANNOTATION_FILE,

        annotation_file_name="label.json",

        description=(
            "UaVirBASE multichannel UAV recordings. "
            "Labels and recording metadata are read from "
            "each recording's label.json file."
        ),
    ),

    # ========================================================
    # ESC-50
    # ========================================================

    DatasetRule(
        name="esc50",

        raw_directory_name="esc50",

        enabled=False,

        label_source=LabelSource.ANNOTATION_FILE,

        annotation_file_name="esc50_labels.csv",

        description=(
            "Background-only dataset for the current binary "
            "experiment. It is handled by "
            "build_background_manifest.py rather than the "
            "supervised positive manifest registry."
        ),
    ),

    # ========================================================
    # UrbanSound8K
    # ========================================================

    DatasetRule(
        name="urbansound8k",

        raw_directory_name="urbansound8k",

        enabled=False,

        label_source=LabelSource.ANNOTATION_FILE,

        annotation_file_name="UrbanSound8K.csv",

        description=(
            "Background-only dataset for the current binary "
            "experiment. It is handled by "
            "build_background_manifest.py."
        ),
    ),

    # ========================================================
    # DDL
    # ========================================================

    DatasetRule(
        name="ddl",

        raw_directory_name="ddl",

        enabled=False,

        label_source=LabelSource.ANNOTATION_FILE,

        description=(
            "DDL dataset. Disabled until its official "
            "annotations and recording-group structure are "
            "mapped into the project taxonomy."
        ),
    ),

    # ========================================================
    # KAIST
    # ========================================================

    DatasetRule(
        name="kaist",

        raw_directory_name="kaist",

        enabled=False,

        label_source=LabelSource.ANNOTATION_FILE,

        description=(
            "KAIST drone acoustic dataset. Disabled until "
            "the downloaded version is inspected and its "
            "official annotations are mapped."
        ),
    ),

    # ========================================================
    # AeroSonicDB
    # ========================================================

    DatasetRule(
        name="aerosonicdb",

        raw_directory_name="aerosonicdb",

        enabled=False,

        label_source=LabelSource.ANNOTATION_FILE,

        annotation_file_name="sample_meta.csv",

        description=(
            "AeroSonicDB. Enable only after its official "
            "metadata is mapped to the project binary "
            "taxonomy and recording groups."
        ),
    ),

    # ========================================================
    # IEEE SPCUP 2019
    # ========================================================

    DatasetRule(
        name="ieee_spcup2019",

        raw_directory_name="ieee_spcup2019",

        enabled=False,

        label_source=LabelSource.ANNOTATION_FILE,

        description=(
            "IEEE SPCUP 2019 acoustic dataset. Disabled until "
            "official labels and recording groups are parsed."
        ),
    ),

    # ========================================================
    # AudioSet
    # ========================================================

    DatasetRule(
        name="audioset",

        raw_directory_name="audioset",

        enabled=False,

        label_source=LabelSource.NONE,

        description=(
            "Excluded from the current project phase."
        ),
    ),

    # ========================================================
    # FreeSound
    # ========================================================

    DatasetRule(
        name="freesound",

        raw_directory_name="freesound",

        enabled=False,

        label_source=LabelSource.NONE,

        description=(
            "Excluded from the current project phase."
        ),
    ),
)


# ============================================================
# Lookup
# ============================================================

def get_dataset_rule(
    dataset_name: str,
) -> DatasetRule:
    """
    Retrieve a dataset rule by its stable internal name.

    Raises
    ------
    KeyError
        If the dataset has not been registered.
    """

    rules_by_name = {
        rule.name: rule
        for rule in DATASET_RULES
    }

    try:
        return rules_by_name[dataset_name]

    except KeyError as error:

        available_names = ", ".join(
            sorted(rules_by_name)
        )

        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available_names}."
        ) from error


# ============================================================
# Enabled Rules
# ============================================================

def get_enabled_dataset_rules() -> tuple[DatasetRule, ...]:
    """
    Return datasets explicitly enabled for the current experiment.

    The order is deterministic because DATASET_RULES is a tuple.
    """

    return tuple(
        rule
        for rule in DATASET_RULES
        if rule.enabled
    )