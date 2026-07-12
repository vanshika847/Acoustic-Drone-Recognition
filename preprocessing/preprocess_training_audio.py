"""Convert approved split-manifest recordings into model-ready audio segments.

Purpose
-------
Read only the labelled files already assigned to train, validation, and test
splits; resample them to mono 16 kHz audio, remove DC offset, peak-normalise,
segment them into fixed windows, and write a segment-level manifest.  Raw
audio is never modified.

Inputs
------
Split manifests under ``datasets/processed/manifests`` and their referenced
files below ``datasets/raw``.

Outputs
-------
PCM WAV segments under ``datasets/processed/segments/<split>/`` and one
``<split>_segments.csv`` manifest per split.  A report CSV records every source
recording that completed or failed.

Dependencies
------------
NumPy, SciPy, SoundFile, and :mod:`preprocessing.audio_segmentation`.

Algorithm
---------
Each source recording is loaded as mono audio at the target sample rate,
centred to remove DC offset, peak-normalised, and split into fixed overlapping
windows.  A deterministic filename based on the source SHA-256 and segment
index makes the operation idempotent.  Segments retain their source hash,
group ID, label, and start/end timestamps.  Processing is ``O(total input
samples)`` and supports a bounded number of worker threads for I/O-heavy runs.

Usage
-----
From the repository root::

    python -m preprocessing.preprocess_training_audio --workers 4
"""

from __future__ import annotations

import argparse
import csv
import logging
from math import gcd
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Support direct execution as well as ``python -m preprocessing...``.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROCESSED_DATASET_DIR, RAW_DATASET_DIR
from preprocessing.audio_segmentation import AudioSegmenter, SegmentationConfig


LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")
SOURCE_MANIFEST_DIRECTORY_NAME = "manifests"
SEGMENT_DIRECTORY_NAME = "segments"
REPORT_DIRECTORY_NAME = "reports"
SEGMENT_MANIFEST_COLUMNS = (
    "split",
    "segment_id",
    "processed_relative_path",
    "source_dataset",
    "source_relative_path",
    "source_sha256",
    "recording_group_id",
    "binary_label",
    "source_category",
    "source_fold",
    "segment_index",
    "start_sample",
    "end_sample",
    "start_seconds",
    "end_seconds",
    "is_padded",
    "sample_rate",
    "segment_samples",
)
REPORT_COLUMNS = (
    "split",
    "source_dataset",
    "source_relative_path",
    "source_sha256",
    "recording_group_id",
    "binary_label",
    "status",
    "segment_count",
    "error",
)
REQUIRED_SOURCE_COLUMNS = {
    "split",
    "dataset",
    "relative_path",
    "file_name",
    "binary_label",
    "source_category",
    "source_fold",
    "recording_group_id",
    "sha256",
}


class AudioPreprocessingError(RuntimeError):
    """Raised when a source manifest or preprocessing configuration is unsafe."""


@dataclass(frozen=True, slots=True)
class AudioPreprocessingConfig:
    """Immutable settings for loading, normalising, and segmenting audio.

    Attributes:
        sample_rate: Target mono sample rate in Hertz.
        segment_window_seconds: Fixed output segment duration.
        segment_hop_seconds: Time between adjacent segment starts.
        minimum_final_seconds: Shortest tail eligible for zero-padding.
        target_peak: Peak amplitude after normalisation.
        remove_dc_offset: Whether to subtract the waveform mean.
    """

    sample_rate: int = 16_000
    segment_window_seconds: float = 4.0
    segment_hop_seconds: float = 2.0
    minimum_final_seconds: float = 1.0
    target_peak: float = 0.99
    remove_dc_offset: bool = True

    def __post_init__(self) -> None:
        """Validate processing settings at construction time."""

        if self.sample_rate <= 0:
            raise AudioPreprocessingError("sample_rate must be greater than zero.")
        if not 0.0 < self.target_peak <= 1.0:
            raise AudioPreprocessingError("target_peak must be in the range (0, 1].")
        SegmentationConfig(
            sample_rate=self.sample_rate,
            window_seconds=self.segment_window_seconds,
            hop_seconds=self.segment_hop_seconds,
            minimum_final_seconds=self.minimum_final_seconds,
            pad_final_segment=True,
        )

    def segmentation_config(self) -> SegmentationConfig:
        """Create the segmenter's configuration from preprocessing settings.

        Returns:
            Valid fixed-window segmentation configuration.
        """

        return SegmentationConfig(
            sample_rate=self.sample_rate,
            window_seconds=self.segment_window_seconds,
            hop_seconds=self.segment_hop_seconds,
            minimum_final_seconds=self.minimum_final_seconds,
            pad_final_segment=True,
        )


@dataclass(frozen=True, slots=True)
class SourceProcessingRequest:
    """All information needed to process one source manifest row."""

    row: Mapping[str, str]
    split_name: str
    raw_datasets_directory: Path
    processed_datasets_directory: Path
    config: AudioPreprocessingConfig
    overwrite: bool


@dataclass(frozen=True, slots=True)
class SourceProcessingResult:
    """Segment rows and source-level outcome for one input recording."""

    segment_rows: tuple[dict[str, str | int | float | bool], ...]
    report_row: dict[str, str | int]


@dataclass(frozen=True, slots=True)
class SplitPreprocessingSummary:
    """Result summary for one completed split preprocessing task."""

    split_name: str
    source_manifest_path: Path
    segment_manifest_path: Path
    report_path: Path
    source_recordings: int
    successful_recordings: int
    failed_recordings: int
    written_segments: int


def _read_source_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read a split manifest after validating its required columns.

    Args:
        manifest_path: Input split CSV path.

    Returns:
        Source rows in their existing deterministic order.

    Raises:
        AudioPreprocessingError: If the manifest or its required columns are absent.
    """

    if not manifest_path.is_file():
        raise AudioPreprocessingError(f"Split manifest was not found: '{manifest_path}'.")
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = REQUIRED_SOURCE_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise AudioPreprocessingError(
                f"Split manifest '{manifest_path}' is missing columns: {missing_text}."
            )
        return list(reader)


def _resolve_source_path(row: Mapping[str, str], raw_datasets_directory: Path) -> Path:
    """Resolve and validate a source audio path from a manifest row.

    Args:
        row: Source manifest row containing dataset and relative path fields.
        raw_datasets_directory: Root directory holding all raw datasets.

    Returns:
        Existing raw audio path located within its declared dataset directory.

    Raises:
        AudioPreprocessingError: If the row attempts path traversal or the file
            does not exist.
    """

    dataset_name = row["dataset"]
    if Path(dataset_name).name != dataset_name:
        raise AudioPreprocessingError(f"Invalid dataset name in manifest: '{dataset_name}'.")
    dataset_directory = (raw_datasets_directory / dataset_name).resolve()
    source_path = (dataset_directory / row["relative_path"]).resolve()
    if not source_path.is_relative_to(dataset_directory):
        raise AudioPreprocessingError(
            f"Source path escapes dataset directory: '{row['relative_path']}'."
        )
    if not source_path.is_file():
        raise AudioPreprocessingError(f"Source audio file was not found: '{source_path}'.")
    return source_path


def _load_and_standardise(
    source_path: Path, config: AudioPreprocessingConfig
) -> np.ndarray:
    """Load, resample, convert to mono, centre, and peak-normalise audio.

    Args:
        source_path: Existing source audio file.
        config: Target audio processing settings.

    Returns:
        Finite one-dimensional float32 waveform at the configured sample rate.

    Raises:
        AudioPreprocessingError: If decoding produces no usable audio.
    """

    samples, original_sample_rate = sf.read(
        source_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = np.mean(samples, axis=1, dtype=np.float32)
    if original_sample_rate != config.sample_rate:
        common_divisor = gcd(original_sample_rate, config.sample_rate)
        waveform = resample_poly(
            waveform,
            up=config.sample_rate // common_divisor,
            down=original_sample_rate // common_divisor,
        ).astype(np.float32, copy=False)
    if waveform.ndim != 1 or waveform.size == 0:
        raise AudioPreprocessingError(f"No mono audio samples decoded from '{source_path}'.")
    if not np.all(np.isfinite(waveform)):
        raise AudioPreprocessingError(f"Non-finite audio samples decoded from '{source_path}'.")

    if config.remove_dc_offset:
        waveform = waveform - np.mean(waveform, dtype=np.float64)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = waveform * (config.target_peak / peak)
    return np.asarray(waveform, dtype=np.float32)


def _write_segment(
    waveform: np.ndarray, sample_rate: int, destination: Path, overwrite: bool
) -> None:
    """Atomically persist one PCM WAV segment unless an existing file is reused.

    Args:
        waveform: Finite mono segment waveform.
        sample_rate: Segment sample rate in Hertz.
        destination: Final WAV path.
        overwrite: Whether an existing destination may be replaced.
    """

    if destination.is_file() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".wav.tmp")
    sf.write(
        temporary_path,
        waveform,
        sample_rate,
        subtype="PCM_16",
        format="WAV",
    )
    temporary_path.replace(destination)


def _segment_destination(
    request: SourceProcessingRequest, segment_index: int
) -> tuple[Path, Path]:
    """Return absolute and processed-root-relative paths for one segment.

    Args:
        request: Source-processing context.
        segment_index: Zero-based segment index for the source file.

    Returns:
        Pair of final absolute path and path relative to processed-data root.
    """

    row = request.row
    relative_path = (
        Path(SEGMENT_DIRECTORY_NAME)
        / request.split_name
        / f"label_{row['binary_label']}"
        / row["dataset"]
        / f"{row['sha256']}_{segment_index:04d}.wav"
    )
    return (
        request.processed_datasets_directory / relative_path,
        relative_path,
    )


def _process_source_recording(request: SourceProcessingRequest) -> SourceProcessingResult:
    """Process one source recording and return segment plus report rows.

    Args:
        request: Source row and processing configuration.

    Returns:
        Completed segment rows or an error report. Per-recording failures are
        captured so one corrupt input does not discard a full split.
    """

    row = request.row
    report_base: dict[str, str | int] = {
        "split": request.split_name,
        "source_dataset": row["dataset"],
        "source_relative_path": row["relative_path"],
        "source_sha256": row["sha256"],
        "recording_group_id": row["recording_group_id"],
        "binary_label": row["binary_label"],
    }
    try:
        source_path = _resolve_source_path(row, request.raw_datasets_directory)
        waveform = _load_and_standardise(source_path, request.config)
        audio_segments = AudioSegmenter(request.config.segmentation_config()).segment(
            waveform
        )
        if not audio_segments:
            raise AudioPreprocessingError(
                "Audio is shorter than the configured minimum final duration."
            )

        segment_rows: list[dict[str, str | int | float | bool]] = []
        for audio_segment in audio_segments:
            destination, processed_relative_path = _segment_destination(
                request, audio_segment.index
            )
            _write_segment(
                audio_segment.waveform,
                request.config.sample_rate,
                destination,
                request.overwrite,
            )
            segment_rows.append(
                {
                    "split": request.split_name,
                    "segment_id": f"{row['sha256']}:{audio_segment.index}",
                    "processed_relative_path": processed_relative_path.as_posix(),
                    "source_dataset": row["dataset"],
                    "source_relative_path": row["relative_path"],
                    "source_sha256": row["sha256"],
                    "recording_group_id": row["recording_group_id"],
                    "binary_label": row["binary_label"],
                    "source_category": row["source_category"],
                    "source_fold": row["source_fold"],
                    "segment_index": audio_segment.index,
                    "start_sample": audio_segment.start_sample,
                    "end_sample": audio_segment.end_sample,
                    "start_seconds": audio_segment.start_sample / request.config.sample_rate,
                    "end_seconds": audio_segment.end_sample / request.config.sample_rate,
                    "is_padded": audio_segment.is_padded,
                    "sample_rate": request.config.sample_rate,
                    "segment_samples": audio_segment.waveform.size,
                }
            )
        return SourceProcessingResult(
            segment_rows=tuple(segment_rows),
            report_row={
                **report_base,
                "status": "success",
                "segment_count": len(segment_rows),
                "error": "",
            },
        )
    except Exception as error:  # Preserve split progress while reporting failures.
        LOGGER.error("Failed preprocessing %s: %s", row["relative_path"], error)
        return SourceProcessingResult(
            segment_rows=(),
            report_row={
                **report_base,
                "status": "failed",
                "segment_count": 0,
                "error": str(error),
            },
        )


def _write_csv_atomically(
    destination: Path,
    field_names: Sequence[str],
    rows: Iterable[Mapping[str, str | int | float | bool]],
) -> None:
    """Write a CSV through a temporary file and same-filesystem atomic rename.

    Args:
        destination: Completed CSV destination.
        field_names: Ordered output header names.
        rows: Rows to write.
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
        writer = csv.DictWriter(temporary_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def preprocess_split_manifest(
    split_name: str,
    source_manifest_path: Path,
    raw_datasets_directory: Path = RAW_DATASET_DIR,
    processed_datasets_directory: Path = PROCESSED_DATASET_DIR,
    config: AudioPreprocessingConfig = AudioPreprocessingConfig(),
    workers: int = 1,
    overwrite: bool = False,
) -> SplitPreprocessingSummary:
    """Preprocess all approved source recordings in one split manifest.

    Args:
        split_name: Output split name; must be train, validation, or test.
        source_manifest_path: CSV manifest created by the split-preparation step.
        raw_datasets_directory: Root containing source dataset folders.
        processed_datasets_directory: Root for processed segments and reports.
        config: Audio loading, normalisation, and segmentation settings.
        workers: Number of bounded worker threads; use one for fully sequential IO.
        overwrite: Replace existing segment WAV files when true.

    Returns:
        Summary with output locations and success/failure counts.

    Raises:
        AudioPreprocessingError: If split metadata or worker settings are invalid.
    """

    if split_name not in SPLIT_NAMES:
        raise AudioPreprocessingError(f"Unsupported split name '{split_name}'.")
    if workers <= 0:
        raise AudioPreprocessingError("workers must be greater than zero.")

    source_rows = _read_source_manifest(source_manifest_path)
    invalid_split_rows = [row for row in source_rows if row["split"] != split_name]
    if invalid_split_rows:
        raise AudioPreprocessingError(
            f"Manifest '{source_manifest_path}' contains rows outside split "
            f"'{split_name}'."
        )

    requests = (
        SourceProcessingRequest(
            row=row,
            split_name=split_name,
            raw_datasets_directory=raw_datasets_directory,
            processed_datasets_directory=processed_datasets_directory,
            config=config,
            overwrite=overwrite,
        )
        for row in source_rows
    )
    if workers == 1:
        results = [_process_source_recording(request) for request in requests]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_source_recording, requests))

    segment_rows = [row for result in results for row in result.segment_rows]
    segment_rows.sort(key=lambda row: str(row["segment_id"]))
    report_rows = [result.report_row for result in results]
    segment_manifest_path = (
        processed_datasets_directory / SOURCE_MANIFEST_DIRECTORY_NAME / f"{split_name}_segments.csv"
    )
    report_path = (
        processed_datasets_directory / REPORT_DIRECTORY_NAME / f"{split_name}_preprocessing.csv"
    )
    _write_csv_atomically(segment_manifest_path, SEGMENT_MANIFEST_COLUMNS, segment_rows)
    _write_csv_atomically(report_path, REPORT_COLUMNS, report_rows)

    successful_recordings = sum(
        result.report_row["status"] == "success" for result in results
    )
    summary = SplitPreprocessingSummary(
        split_name=split_name,
        source_manifest_path=source_manifest_path,
        segment_manifest_path=segment_manifest_path,
        report_path=report_path,
        source_recordings=len(source_rows),
        successful_recordings=successful_recordings,
        failed_recordings=len(source_rows) - successful_recordings,
        written_segments=len(segment_rows),
    )
    LOGGER.info("Preprocessed split '%s': %s", split_name, summary)
    return summary


def preprocess_all_splits(
    source_manifest_directory: Path = PROCESSED_DATASET_DIR / SOURCE_MANIFEST_DIRECTORY_NAME,
    raw_datasets_directory: Path = RAW_DATASET_DIR,
    processed_datasets_directory: Path = PROCESSED_DATASET_DIR,
    config: AudioPreprocessingConfig = AudioPreprocessingConfig(),
    workers: int = 1,
    overwrite: bool = False,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> tuple[SplitPreprocessingSummary, ...]:
    """Preprocess one or more complete train/validation/test source manifests.

    Args:
        source_manifest_directory: Directory containing split CSV source manifests.
        raw_datasets_directory: Root containing raw source audio.
        processed_datasets_directory: Root for processed output.
        config: Audio preprocessing configuration.
        workers: Per-split worker thread count.
        overwrite: Whether existing segment WAVs are replaced.
        split_names: Requested supported split names.

    Returns:
        Ordered summaries for every requested split.
    """

    return tuple(
        preprocess_split_manifest(
            split_name=split_name,
            source_manifest_path=source_manifest_directory / f"{split_name}.csv",
            raw_datasets_directory=raw_datasets_directory,
            processed_datasets_directory=processed_datasets_directory,
            config=config,
            workers=workers,
            overwrite=overwrite,
        )
        for split_name in split_names
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line controls for a complete preprocessing run.

    Returns:
        Parsed split selection, worker count, and overwrite flag.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=SPLIT_NAMES,
        help="One or more split manifests to preprocess.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded worker thread count (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already-written segment WAV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run manifest-driven preprocessing from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    arguments = _parse_arguments()
    preprocess_all_splits(
        workers=arguments.workers,
        overwrite=arguments.overwrite,
        split_names=arguments.splits,
    )


if __name__ == "__main__":
    main()
