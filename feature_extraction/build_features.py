"""Build feature matrices for every segmented audio file in split manifests.

Purpose
-------
Read the segment-level manifests produced by
:mod:`preprocessing.preprocess_training_audio`, load each referenced WAV,
extract MFCC, mel, spectral, chroma, ZCR, and RMS energy features, and persist
one ``.npy`` file per segment per feature family.

Inputs
------
``datasets/processed/manifests/{split}_segments.csv`` for each requested split.

Outputs
-------
Feature arrays under ``features/<feature_name>/`` and a per-split metadata CSV
under ``outputs/features/``.

Dependencies
------------
NumPy, pandas, tqdm, librosa, and the project's ``configs`` and ``utils``
packages.

Usage
-----
From the repository root::

    python -m feature_extraction.build_features --splits train validation test
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import OUTPUT_DIR, PROCESSED_DATASET_DIR, PROJECT_ROOT, TARGET_SAMPLE_RATE
from configs.preprocessing_config import NORMALIZE_AUDIO, TARGET_PEAK
from feature_extraction.chroma import ChromaConfig, extract_chroma
from feature_extraction.energy import EnergyConfig, extract_energy
from feature_extraction.mel import MelSpectrogramConfig, extract_mel_spectrogram
from feature_extraction.mfcc import MFCCConfig, extract_mfcc
from feature_extraction.spectral import SPECTRAL_FEATURE_NAMES, SpectralConfig, extract_spectral_features
from feature_extraction.zcr import ZCRConfig, extract_zcr
from utils.audio_loader import load_audio
from utils.audio_processor import normalize_audio, remove_dc_offset
from utils.logger import setup_logger


LOGGER = logging.getLogger("feature_extraction.log")
SPLIT_NAMES = ("train", "validation", "test")
FEATURES_ROOT = PROJECT_ROOT / "features"
FEATURE_METADATA_DIR = OUTPUT_DIR / "features"
SEGMENT_MANIFEST_SUFFIX = "_segments.csv"
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "split",
        "segment_id",
        "processed_relative_path",
        "binary_label",
        "source_sha256",
        "recording_group_id",
        "sample_rate",
    }
)
FEATURE_METADATA_COLUMNS = (
    "split",
    "segment_id",
    "segment_file_name",
    "processed_relative_path",
    "binary_label",
    "source_sha256",
    "recording_group_id",
    "source_dataset",
    "source_relative_path",
    "feature_version",
    "status",
    "mfcc_path",
    "mel_path",
    "spectral_path",
    "chroma_path",
    "zcr_path",
    "energy_path",
    "mfcc_shape",
    "mel_shape",
    "spectral_shape",
    "chroma_shape",
    "zcr_shape",
    "energy_shape",
    "error",
)
FEATURE_DEFINITIONS: tuple[tuple[str, Callable[..., NDArray[np.float32]]], ...] = (
    ("mfcc", extract_mfcc),
    ("mel", extract_mel_spectrogram),
    ("spectral", extract_spectral_features),
    ("chroma", extract_chroma),
    ("zcr", extract_zcr),
    ("energy", extract_energy),
)


class FeatureBuildError(RuntimeError):
    """Raised when feature extraction cannot be started safely."""


@dataclass(frozen=True, slots=True)
class FeatureExtractionSettings:
    """Immutable settings for manifest-driven feature extraction.

    Attributes:
        sample_rate: Target sample rate used when loading segment audio.
        apply_normalization: Whether to peak-normalise loaded audio.
        target_peak: Peak amplitude used during optional normalisation.
        remove_dc_offset_enabled: Whether to subtract the waveform mean.
        skip_existing: Whether to skip segments whose feature files already exist.
        mfcc_config: MFCC extraction parameters.
        mel_config: Mel spectrogram extraction parameters.
        spectral_config: Spectral feature extraction parameters.
        chroma_config: Chroma extraction parameters.
        zcr_config: Zero crossing rate extraction parameters.
        energy_config: RMS energy extraction parameters.
    """

    sample_rate: int = TARGET_SAMPLE_RATE
    apply_normalization: bool = NORMALIZE_AUDIO
    target_peak: float = TARGET_PEAK
    remove_dc_offset_enabled: bool = True
    skip_existing: bool = True
    feature_version: str = "v1"
    mfcc_config: MFCCConfig = MFCCConfig()
    mel_config: MelSpectrogramConfig = MelSpectrogramConfig()
    spectral_config: SpectralConfig = SpectralConfig()
    chroma_config: ChromaConfig = ChromaConfig()
    zcr_config: ZCRConfig = ZCRConfig()
    energy_config: EnergyConfig = EnergyConfig()


@dataclass(frozen=True, slots=True)
class FeatureBuildSummary:
    """Summary of one completed split feature build."""

    split_name: str
    manifest_path: Path
    metadata_path: Path
    total_segments: int
    successful_segments: int
    skipped_segments: int
    failed_segments: int


def resolve_segment_manifest_path(
    split_name: str,
    manifests_directory: Path = PROCESSED_DATASET_DIR / "manifests",
) -> Path:
    """Resolve the segment manifest path for one split.

    Args:
        split_name: Requested split identifier.
        manifests_directory: Directory containing split manifests.

    Returns:
        Existing segment manifest path.

    Raises:
        FeatureBuildError: If the split is unsupported or the manifest is absent.
    """

    if split_name not in SPLIT_NAMES:
        raise FeatureBuildError(f"Unsupported split name '{split_name}'.")

    segment_manifest = manifests_directory / f"{split_name}{SEGMENT_MANIFEST_SUFFIX}"
    if segment_manifest.is_file():
        return segment_manifest

    raise FeatureBuildError(
        f"Segment manifest was not found for split '{split_name}': "
        f"'{segment_manifest}'. Run preprocessing before feature extraction."
    )


def read_segment_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read and validate a segment manifest CSV.

    Args:
        manifest_path: Segment manifest path.

    Returns:
        Manifest rows in source order.

    Raises:
        FeatureBuildError: If required columns are missing.
    """

    dataframe = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_MANIFEST_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise FeatureBuildError(
            f"Manifest '{manifest_path}' is missing required columns: {missing_text}."
        )
    return dataframe.to_dict(orient="records")


def segment_file_name(segment_id: str) -> str:
    """Convert a segment identifier into a filesystem-safe feature basename.

    Args:
        segment_id: Canonical segment identifier from the manifest.

    Returns:
        Safe basename without extension.
    """

    return segment_id.replace(":", "_")


def feature_output_path(feature_name: str, segment_id: str) -> Path:
    """Return the destination ``.npy`` path for one segment and feature family.

    Args:
        feature_name: Feature family directory name.
        segment_id: Canonical segment identifier from the manifest.

    Returns:
        Absolute feature output path.
    """

    return FEATURES_ROOT / feature_name / f"{segment_file_name(segment_id)}.npy"


def all_feature_paths_exist(segment_id: str) -> bool:
    """Return whether every feature file already exists for one segment.

    Args:
        segment_id: Canonical segment identifier from the manifest.

    Returns:
        True when all configured feature files are present.
    """

    return all(
        feature_output_path(feature_name, segment_id).is_file()
        for feature_name, _ in FEATURE_DEFINITIONS
    )


def resolve_segment_audio_path(
    row: Mapping[str, str],
    processed_datasets_directory: Path = PROCESSED_DATASET_DIR,
) -> Path:
    """Resolve the processed WAV path referenced by one manifest row.

    Args:
        row: Segment manifest row.
        processed_datasets_directory: Root directory for processed audio.

    Returns:
        Existing segment WAV path.

    Raises:
        FeatureBuildError: If the manifest path is unsafe or missing.
    """

    relative_path = Path(row["processed_relative_path"])
    if relative_path.is_absolute():
        raise FeatureBuildError(
            f"processed_relative_path must be relative: '{relative_path}'."
        )

    audio_path = (processed_datasets_directory / relative_path).resolve()
    processed_root = processed_datasets_directory.resolve()
    if not audio_path.is_relative_to(processed_root):
        raise FeatureBuildError(
            f"Segment path escapes processed directory: '{relative_path}'."
        )
    if not audio_path.is_file():
        raise FeatureBuildError(f"Segment audio file was not found: '{audio_path}'.")
    return audio_path


def prepare_waveform(
    audio_path: Path,
    settings: FeatureExtractionSettings,
) -> tuple[NDArray[np.float32], int]:
    """Load and optionally condition one segment waveform.

    Args:
        audio_path: Existing processed segment WAV path.
        settings: Feature extraction settings.

    Returns:
        Prepared mono waveform and sample rate.

    Raises:
        FeatureBuildError: If decoding fails or produces unusable audio.
    """

    try:
        waveform, sample_rate, _metadata = load_audio(audio_path, settings.sample_rate)
        if sample_rate != settings.sample_rate:
            raise FeatureBuildError(
                f"Unexpected sample rate {sample_rate}. "
                f"Expected {settings.sample_rate}."
            )
    except Exception as error:
        raise FeatureBuildError(f"Failed to load '{audio_path}': {error}") from error

    prepared = np.asarray(waveform, dtype=np.float32)
    if prepared.ndim != 1 or prepared.size == 0:
        raise FeatureBuildError(f"No usable mono audio decoded from '{audio_path}'.")
    if not np.all(np.isfinite(prepared)):
        raise FeatureBuildError(f"Non-finite audio samples decoded from '{audio_path}'.")

    if settings.remove_dc_offset_enabled:
        prepared = np.asarray(remove_dc_offset(prepared), dtype=np.float32)
    if settings.apply_normalization:
        prepared = np.asarray(
            normalize_audio(prepared, settings.target_peak),
            dtype=np.float32,
        )
    return prepared, int(sample_rate)


def save_feature_array(
    feature_array: NDArray[np.float32],
    destination: Path,
) -> None:
    """
    Save a feature array safely using an atomic write.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = destination.with_suffix(".tmp")

    with open(temp_path, "wb") as file:
        np.save(
            file,
            np.ascontiguousarray(
                feature_array,
                dtype=np.float32,
            ),
        )

    temp_path.replace(destination)


def extract_feature_set(
    waveform: NDArray[np.float32],
    sample_rate: int,
    settings: FeatureExtractionSettings,
) -> dict[str, NDArray[np.float32]]:
    """Extract every configured feature family from one waveform.

    Args:
        waveform: Prepared mono waveform.
        sample_rate: Waveform sample rate in Hertz.
        settings: Feature extraction settings.

    Returns:
        Mapping from feature family name to feature matrix.
    """

    return {
        "mfcc": extract_mfcc(waveform, sample_rate, settings.mfcc_config),
        "mel": extract_mel_spectrogram(waveform, sample_rate, settings.mel_config),
        "spectral": extract_spectral_features(
            waveform,
            sample_rate,
            settings.spectral_config,
        ),
        "chroma": extract_chroma(waveform, sample_rate, settings.chroma_config),
        "zcr": extract_zcr(waveform, sample_rate, settings.zcr_config),
        "energy": extract_energy(waveform, sample_rate, settings.energy_config),
    }


def process_segment_row(
    row: Mapping[str, str],
    settings: FeatureExtractionSettings,
    processed_datasets_directory: Path = PROCESSED_DATASET_DIR,
) -> dict[str, str]:
    """Extract and persist all features for one manifest row.

    Args:
        row: Segment manifest row.
        settings: Feature extraction settings.
        processed_datasets_directory: Root directory for processed audio.

    Returns:
        Metadata row describing success, skip, or failure.
    """

    segment_id = row["segment_id"]
    base_metadata = {
        "split": row["split"],
        "segment_id": segment_id,
        "segment_file_name": segment_file_name(segment_id),
        "processed_relative_path": row["processed_relative_path"],
        "binary_label": row["binary_label"],
        "source_sha256": row["source_sha256"],
        "recording_group_id": row["recording_group_id"],
        "source_dataset": row["source_dataset"],
        "source_relative_path": row["source_relative_path"],
        "feature_version": settings.feature_version,
    }

    if settings.skip_existing and all_feature_paths_exist(segment_id):
        return {
            **base_metadata,
            "status": "skipped",
            "mfcc_path": str(feature_output_path("mfcc", segment_id)),
            "mel_path": str(feature_output_path("mel", segment_id)),
            "spectral_path": str(feature_output_path("spectral", segment_id)),
            "chroma_path": str(feature_output_path("chroma", segment_id)),
            "zcr_path": str(feature_output_path("zcr", segment_id)),
            "energy_path": str(feature_output_path("energy", segment_id)),
            "mfcc_shape": "",
            "mel_shape": "",
            "spectral_shape": "",
            "chroma_shape": "",
            "zcr_shape": "",
            "energy_shape": "",
            "error": "",
        }

    try:
        audio_path = resolve_segment_audio_path(row, processed_datasets_directory)
        waveform, sample_rate = prepare_waveform(audio_path, settings)
        feature_arrays = extract_feature_set(waveform, sample_rate, settings)
        for feature_name, feature_array in feature_arrays.items():
            if not np.all(np.isfinite(feature_array)):
                raise FeatureBuildError(
                    f"{feature_name} contains NaN or Inf values."
                )

        saved_paths: dict[str, str] = {}
        saved_shapes: dict[str, str] = {}
        for feature_name, feature_array in feature_arrays.items():
            destination = feature_output_path(feature_name, segment_id)
            save_feature_array(feature_array, destination)
            saved_paths[f"{feature_name}_path"] = str(destination)
            saved_shapes[f"{feature_name}_shape"] = str(tuple(feature_array.shape))

        return {
            **base_metadata,
            "status": "success",
            **saved_paths,
            **saved_shapes,
            "error": "",
        }
    except Exception as error:
        LOGGER.error(
            "Feature extraction failed for segment '%s' (%s): %s",
            segment_id,
            row.get("processed_relative_path", ""),
            error,
        )
        return {
            **base_metadata,
            "status": "failed",
            "mfcc_path": str(feature_output_path("mfcc", segment_id)),
            "mel_path": str(feature_output_path("mel", segment_id)),
            "spectral_path": str(feature_output_path("spectral", segment_id)),
            "chroma_path": str(feature_output_path("chroma", segment_id)),
            "zcr_path": str(feature_output_path("zcr", segment_id)),
            "energy_path": str(feature_output_path("energy", segment_id)),
            "mfcc_shape": "",
            "mel_shape": "",
            "spectral_shape": "",
            "chroma_shape": "",
            "zcr_shape": "",
            "energy_shape": "",
            "error": str(error),
        }


def write_metadata_csv(
    destination: Path,
    rows: Iterable[Mapping[str, str]],
) -> None:
    """Write feature metadata through a temporary file and atomic rename.

    Args:
        destination: Completed metadata CSV path.
        rows: Metadata rows to write.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.stem}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        writer = csv.DictWriter(temporary_file, fieldnames=FEATURE_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def build_features_for_split(
    split_name: str,
    settings: FeatureExtractionSettings,
    manifests_directory: Path = PROCESSED_DATASET_DIR / "manifests",
    processed_datasets_directory: Path = PROCESSED_DATASET_DIR,
) -> FeatureBuildSummary:
    """Build features for every segment listed in one split manifest.

    Args:
        split_name: Requested split identifier.
        settings: Feature extraction settings.
        manifests_directory: Directory containing segment manifests.
        processed_datasets_directory: Root directory for processed audio.

    Returns:
        Summary of processed, skipped, and failed segments.
    """

    manifest_path = resolve_segment_manifest_path(split_name, manifests_directory)
    manifest_rows = read_segment_manifest(manifest_path)
    metadata_rows: list[dict[str, str]] = []

    for row in tqdm(
        manifest_rows,
        desc=f"Extracting {split_name} features",
        unit="segment",
    ):
        if row["split"] != split_name:
            raise FeatureBuildError(
                f"Manifest '{manifest_path}' contains rows outside split "
                f"'{split_name}'."
            )
        metadata_rows.append(
            process_segment_row(
                row=row,
                settings=settings,
                processed_datasets_directory=processed_datasets_directory,
            )
        )

    metadata_path = FEATURE_METADATA_DIR / f"{split_name}_feature_manifest.csv"
    write_metadata_csv(metadata_path, metadata_rows)

    successful_segments = sum(row["status"] == "success" for row in metadata_rows)
    skipped_segments = sum(row["status"] == "skipped" for row in metadata_rows)
    failed_segments = sum(row["status"] == "failed" for row in metadata_rows)
    summary = FeatureBuildSummary(
        split_name=split_name,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        total_segments=len(metadata_rows),
        successful_segments=successful_segments,
        skipped_segments=skipped_segments,
        failed_segments=failed_segments,
    )
    LOGGER.info("Completed feature build for split '%s': %s", split_name, summary)
    return summary


def build_features_for_splits(
    split_names: Sequence[str] = SPLIT_NAMES,
    settings: FeatureExtractionSettings | None = None,
) -> tuple[FeatureBuildSummary, ...]:
    """Build features for one or more dataset splits.

    Args:
        split_names: Requested split identifiers.
        settings: Feature extraction settings.

    Returns:
        Ordered summaries for each requested split.
    """

    extraction_settings = settings or FeatureExtractionSettings()
    return tuple(
        build_features_for_split(split_name, extraction_settings)
        for split_name in split_names
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line options for feature extraction.

    Returns:
        Parsed namespace containing split selection and overwrite behaviour.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=SPLIT_NAMES,
        help="One or more split manifests to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and replace existing feature files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run manifest-driven feature extraction from the command line."""

    global LOGGER
    LOGGER = setup_logger("feature_extraction.log")
    for feature_name, _ in FEATURE_DEFINITIONS:
        (FEATURES_ROOT / feature_name).mkdir(
            parents=True,
            exist_ok=True,
        )
    
    arguments = _parse_arguments()
    settings = FeatureExtractionSettings(skip_existing=not arguments.overwrite)
    LOGGER.info("Feature Extraction Configuration")

    LOGGER.info("MFCC: %s", settings.mfcc_config)
    LOGGER.info("Mel: %s", settings.mel_config)
    LOGGER.info("Spectral: %s", settings.spectral_config)
    LOGGER.info("Chroma: %s", settings.chroma_config)
    LOGGER.info("ZCR: %s", settings.zcr_config)
    LOGGER.info("Energy: %s", settings.energy_config)
    summaries = build_features_for_splits(
        split_names=arguments.splits,
        settings=settings,
    )
    for summary in summaries:
        LOGGER.info(
            "Split=%s total=%d success=%d skipped=%d failed=%d metadata=%s",
            summary.split_name,
            summary.total_segments,
            summary.successful_segments,
            summary.skipped_segments,
            summary.failed_segments,
            summary.metadata_path,
        )
        LOGGER.info(
            "Success Rate: %.2f%%",
            (
                summary.successful_segments
                / max(summary.total_segments, 1)
            ) * 100,
    )
    LOGGER.info(
        "Spectral feature row order: %s",
        ", ".join(SPECTRAL_FEATURE_NAMES),
    )


if __name__ == "__main__":
    main()